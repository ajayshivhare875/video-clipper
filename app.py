"""ClipKaro - AI video clipper with subscriptions.

Manual clip: upload + start/end -> fast ffmpeg stream-copy cut.
AI auto-clip: Whisper transcript (Groq) -> LLM picks viral moments ->
clips with burned-in captions, optional 9:16 vertical reframe.
Billing: FamBridge UPI subscriptions (auto-verified).
"""
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

import imageio_ffmpeg
import requests
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from db_store import db_enabled, db_load, db_save, _DB_ERROR, _MISSING

# ---------------------------------------------------------------- config
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "")
if not app.secret_key:
    app.secret_key = os.urandom(32)
    print("[clipkaro] WARNING: SECRET_KEY not set, using random (sessions won't survive restarts)", flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
FAMBRIDGE_API_KEY = os.environ.get("FAMBRIDGE_API_KEY", "").strip()
FAMBRIDGE_BASE = os.environ.get("FAMBRIDGE_BASE", "https://fambridge-gateway.onrender.com").rstrip("/")

CLEANUP_AFTER_SEC = 3600   # uploaded/clipped files auto-delete after 1 hour
MAX_CLIP_SEC = 600         # manual clip max length (10 min)
MAX_AI_AUDIO_MIN = 30      # cap source length for AI jobs (Whisper 25MB limit)

PLANS = {
    "free":    {"name": "Free",    "price": 0,   "clips": 10,  "max_mb": 100},
    "basic":   {"name": "Basic",   "price": 299, "clips": 30,  "max_mb": 500},
    "pro":     {"name": "Pro",     "price": 399, "clips": 70,  "max_mb": 1024},
    "premium": {"name": "Premium", "price": 599, "clips": 150, "max_mb": 2048},
}
PLAN_ORDER = ["free", "basic", "pro", "premium"]

CAPTION_STYLES = {
    "bold":    "FontName=DejaVu Sans,FontSize=26,Bold=1,PrimaryColour=&H0000FFFF&,OutlineColour=&H80000000&,BackColour=&H80000000&,Outline=2,Shadow=1,Alignment=2,MarginV=45",
    "minimal": "FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF&,OutlineColour=&H80000000&,BackColour=&H80000000&,Outline=1,Shadow=0,Alignment=2,MarginV=35",
    "neon":    "FontName=DejaVu Sans,FontSize=26,Bold=1,PrimaryColour=&H00FF66FF&,OutlineColour=&H80000000&,BackColour=&H80000000&,Outline=2,Shadow=1,Alignment=2,MarginV=45",
}

USERS_KEY = "clipkaro.users"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# ---------------------------------------------------------------- storage
_users_db_tainted = False


def _load_file_users():
    try:
        with open(USERS_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_file_users(users):
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(users, fh)
    os.replace(tmp, USERS_FILE)


def load_users():
    """Users dict keyed by email. Supabase kv_store first, file fallback."""
    global _users_db_tainted
    if db_enabled():
        doc = db_load(USERS_KEY, _MISSING)
        if doc is _DB_ERROR:
            _users_db_tainted = True   # don't clobber real DB data with file data
            return _load_file_users()
        if doc is not _MISSING:
            _users_db_tainted = False
            return doc if isinstance(doc, dict) else {}
    _users_db_tainted = False
    return _load_file_users()


def save_users(users):
    _save_file_users(users)  # always keep a local backup
    if db_enabled() and not _users_db_tainted:
        db_save(USERS_KEY, users)


def current_month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_user():
    email = session.get("email")
    if not email:
        return None
    return load_users().get(email.lower())


def refresh_plan(user):
    """Downgrade expired paid plans to free. Returns (plan_id, changed)."""
    plan = user.get("plan", "free")
    if plan != "free" and time.time() > user.get("plan_expires_at", 0):
        user["plan"] = "free"
        user["plan_expires_at"] = 0
        return "free", True
    return plan, False


def quota_check(user):
    """Returns (allowed: bool, message: str). Resets monthly counter as needed."""
    plan_id, changed = refresh_plan(user)
    if user.get("quota_month") != current_month():
        user["quota_month"] = current_month()
        user["clips_used"] = 0
        changed = True
    if changed:
        users = load_users()
        users[user["email"]] = user
        save_users(users)
    limit = PLANS[plan_id]["clips"]
    if user.get("clips_used", 0) >= limit:
        return False, f"Monthly limit reached ({limit} clips on {PLANS[plan_id]['name']}). Upgrade your plan."
    return True, ""


def consume_clip(user):
    users = load_users()
    u = users.get(user["email"])
    if not u:
        return
    if u.get("quota_month") != current_month():
        u["quota_month"] = current_month()
        u["clips_used"] = 0
    u["clips_used"] = u.get("clips_used", 0) + 1
    users[user["email"]] = u
    save_users(users)


# ---------------------------------------------------------------- helpers
def parse_ts(s):
    """Parse mm:ss or seconds -> float seconds."""
    s = (s or "").strip()
    if not s:
        return None
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^(?:(\d+):)?([0-5]?\d):([0-5]\d(?:\.\d+)?)$", s)
    if m:
        h = int(m.group(1) or 0)
        return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def fmt_ts(sec):
    sec = max(0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cleanup_old():
    now = time.time()
    for f in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, f)
        try:
            if now - os.path.getmtime(p) > CLEANUP_AFTER_SEC:
                os.remove(p)
        except OSError:
            pass


def cleanup_loop():
    while True:
        time.sleep(600)
        cleanup_old()


threading.Thread(target=cleanup_loop, daemon=True).start()


# ---------------------------------------------------------------- job queue
# One job at a time (512MB Render free tier). Poll via /job/<id>.
job_queue = queue.Queue()
jobs = {}
jobs_lock = threading.Lock()


def job_new(kind):
    jid = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[jid] = {"status": "queued", "stage": "Queued…", "progress": 0,
                     "result": None, "error": None, "kind": kind,
                     "created": time.time()}
    # prune old jobs
    with jobs_lock:
        for k in [k for k, v in jobs.items()
                  if time.time() - v["created"] > CLEANUP_AFTER_SEC]:
            jobs.pop(k, None)
    return jid


def job_update(jid, **kw):
    with jobs_lock:
        if jid in jobs:
            jobs[jid].update(kw)


def ffmpeg_escape_path(p):
    # escape for the subtitles filter filename argument
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("[", "\\[").replace("]", "\\]")


def run_cmd(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def extract_audio(src, dst, max_min=MAX_AI_AUDIO_MIN):
    """Extract mono 16kHz mp3 (Whisper-friendly). Caps length."""
    cmd = [FFMPEG, "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
           "-b:a", "64k", "-t", str(max_min * 60), dst]
    r = run_cmd(cmd, 300)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("audio extraction failed")
    if os.path.getsize(dst) > 24 * 1024 * 1024:
        raise RuntimeError("audio too large for transcription (video too long)")


def whisper_transcribe(audio_path):
    """Groq Whisper large-v3 -> list of {start, end, text} segments."""
    if not GROQ_API_KEY:
        raise RuntimeError("AI features need GROQ_API_KEY (set it in env vars)")
    with open(audio_path, "rb") as fh:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.mp3", fh, "audio/mpeg")},
            data={"model": "whisper-large-v3", "response_format": "verbose_json"},
            timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"transcription failed (HTTP {r.status_code})")
    data = r.json()
    segs = []
    for s in data.get("segments", []):
        t = (s.get("text") or "").strip()
        if t:
            segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": t})
    if not segs and (data.get("text") or "").strip():
        segs.append({"start": 0.0, "end": 60.0, "text": data["text"].strip()})
    if not segs:
        raise RuntimeError("no speech detected in video")
    return segs


def pick_viral_moments(segments, want=4):
    """LLM picks the most viral moments from the transcript."""
    if not GROQ_API_KEY:
        raise RuntimeError("AI features need GROQ_API_KEY (set it in env vars)")
    lines = []
    for s in segments:
        lines.append(f"[{s['start']:.0f}s-{s['end']:.0f}s] {s['text']}")
    transcript = "\n".join(lines)[:12000]
    prompt = (
        "You are a viral short-form video editor. From this transcript, pick the "
        f"{want} most viral-worthy moments for Reels/Shorts/TikTok. Each moment must be "
        "15-60 seconds, self-contained, with a hook in the first 3 seconds.\n"
        "Reply ONLY with a JSON array, no other text:\n"
        '[{"start": <seconds>, "end": <seconds>, "score": <0-100>, '
        '"title": "<short catchy title>", "reason": "<one line why it will go viral>"}]\n\n'
        f"Transcript:\n{transcript}")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.7, "max_tokens": 1500},
        timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"highlight picker failed (HTTP {r.status_code})")
    text = r.json()["choices"][0]["message"]["content"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise RuntimeError("could not parse highlight moments")
    moments = json.loads(m.group(0))
    clean = []
    for mo in moments[:want]:
        try:
            s, e = float(mo["start"]), float(mo["end"])
            if e - s < 5 or e - s > 90 or s < 0:
                continue
            clean.append({"start": s, "end": e,
                          "score": int(mo.get("score", 70)),
                          "title": str(mo.get("title", "Highlight"))[:80],
                          "reason": str(mo.get("reason", ""))[:160]})
        except (ValueError, TypeError, KeyError):
            continue
    if not clean:
        raise RuntimeError("no valid moments found")
    return clean


def segments_to_srt(segments, clip_start, clip_end, max_words=9):
    """Whisper segments -> SRT lines for one clip (times relative to clip)."""
    # collect words in the window, interpolating each word's timing
    # evenly across its segment for better lip-sync
    words = []
    for s in segments:
        if s["end"] <= clip_start or s["start"] >= clip_end:
            continue
        toks = s["text"].split()
        if not toks:
            continue
        dur = s["end"] - s["start"]
        for i, w in enumerate(toks):
            ws = s["start"] + dur * i / len(toks)
            we = s["start"] + dur * (i + 1) / len(toks)
            if we <= clip_start or ws >= clip_end:
                continue
            words.append((max(ws, clip_start) - clip_start,
                          min(we, clip_end) - clip_start, w))
    if not words:
        return ""
    # chunk into caption lines
    lines, cur, cur_s, cur_e = [], [], None, None
    for ws, we, w in words:
        if cur_s is None:
            cur_s = ws
        cur.append(w)
        cur_e = we
        if len(cur) >= max_words:
            lines.append((cur_s, cur_e, " ".join(cur)))
            cur, cur_s, cur_e = [], None, None
    if cur:
        lines.append((cur_s, cur_e, " ".join(cur)))
    out = []
    for i, (a, b, t) in enumerate(lines, 1):
        out.append(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{t}\n")
    return "\n".join(out)


def build_vf(vertical, srt_path, style):
    """Video filter chain: optional 9:16 crop + caption burn-in."""
    vf = []
    if vertical:
        vf.append("crop=ih*9/16:ih:(iw-ow)/2:0")
    if srt_path:
        vf.append(f"subtitles='{ffmpeg_escape_path(srt_path)}':force_style='{style}'")
    return ",".join(vf)


def cut_clip(src, start, dur, out, srt_path=None, style=None, vertical=False):
    """Cut one clip. Stream-copy fast path when no re-encode needed."""
    vf = build_vf(vertical, srt_path, style)
    if not vf:
        cmd = [FFMPEG, "-y", "-ss", str(start), "-i", src, "-t", str(dur),
               "-c", "copy", "-movflags", "+faststart", out]
    else:
        cmd = [FFMPEG, "-y", "-ss", str(start), "-i", src, "-t", str(dur),
               "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-movflags", "+faststart", out]
    r = run_cmd(cmd, 600)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("ffmpeg cut failed: " + r.stderr.decode()[-300:])


def run_ai_job(jid, src, opts):
    job_update(jid, status="processing", stage="Extracting audio…", progress=5)
    workdir = os.path.join(UPLOAD_DIR, f"ai_{jid}")
    os.makedirs(workdir, exist_ok=True)
    try:
        audio = os.path.join(workdir, "audio.mp3")
        extract_audio(src, audio)

        job_update(jid, stage="Transcribing speech…", progress=20)
        segments = whisper_transcribe(audio)

        job_update(jid, stage="Finding viral moments…", progress=45)
        moments = pick_viral_moments(segments, want=opts.get("count", 4))

        results = []
        for i, mo in enumerate(moments):
            job_update(jid, stage=f"Cutting clip {i+1}/{len(moments)}…",
                       progress=50 + int(45 * (i + 1) / len(moments)))
            dur = mo["end"] - mo["start"]
            srt_path = None
            if opts.get("captions", True):
                srt = segments_to_srt(segments, mo["start"], mo["end"])
                if srt.strip():
                    srt_path = os.path.join(workdir, f"cap_{i}.srt")
                    with open(srt_path, "w") as fh:
                        fh.write(srt)
            out = os.path.join(UPLOAD_DIR, f"aiclip_{jid}_{i}.mp4")
            try:
                cut_clip(src, mo["start"], dur, out,
                         srt_path=srt_path,
                         style=CAPTION_STYLES.get(opts.get("style", "bold"), CAPTION_STYLES["bold"]),
                         vertical=bool(opts.get("vertical")))
            except RuntimeError as e:
                # caption burn failed (e.g. no fonts) -> retry plain cut
                if srt_path:
                    cut_clip(src, mo["start"], dur, out, vertical=bool(opts.get("vertical")))
                else:
                    raise
            results.append({"title": mo["title"], "score": mo["score"],
                            "reason": mo["reason"],
                            "download": f"/dl/aiclip_{jid}_{i}",
                            "name": f"clip_{i+1}.mp4"})
        job_update(jid, status="done", stage="Done!", progress=100, result=results)
    except Exception as e:
        job_update(jid, status="error", stage="Failed", error=str(e)[:300])
    finally:
        for p in (src,):
            try:
                os.remove(p)
            except OSError:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def worker_loop():
    while True:
        task = job_queue.get()
        try:
            if task["kind"] == "ai":
                run_ai_job(task["jid"], task["src"], task["opts"])
        except Exception as e:  # never kill the worker
            job_update(task["jid"], status="error", stage="Failed", error=str(e)[:300])
        finally:
            job_queue.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


# ---------------------------------------------------------------- auth
def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*a, **kw):
        if not get_user():
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


@app.route("/signup", methods=["GET", "POST"])
def signup():
    err = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw = request.form.get("password") or ""
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            err = "Enter a valid email."
        elif len(pw) < 6:
            err = "Password must be at least 6 characters."
        else:
            users = load_users()
            if email in users:
                err = "Account already exists. Please log in."
            else:
                users[email] = {"email": email,
                                "pw_hash": generate_password_hash(pw),
                                "plan": "free", "plan_expires_at": 0,
                                "clips_used": 0, "quota_month": current_month(),
                                "created_at": int(time.time())}
                save_users(users)
                session["email"] = email
                return redirect(url_for("dashboard"))
    return render_template("signup.html", err=err)


@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw = request.form.get("password") or ""
        u = load_users().get(email)
        if u and check_password_hash(u.get("pw_hash", ""), pw):
            session["email"] = email
            return redirect(request.args.get("next") or url_for("dashboard"))
        err = "Wrong email or password."
    return render_template("login.html", err=err)


@app.route("/logout")
def logout():
    session.pop("email", None)
    return redirect(url_for("index"))


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", user=get_user())


@app.route("/dashboard")
@login_required
def dashboard():
    user = get_user()
    plan_id, _ = refresh_plan(user)
    ok, msg = quota_check(user)
    plan = PLANS[plan_id]
    used = user.get("clips_used", 0)
    return render_template("dashboard.html", user=user, plan=plan, plan_id=plan_id,
                           used=used, limit=plan["clips"], quota_msg=msg,
                           max_mb=plan["max_mb"], max_min=MAX_CLIP_SEC // 60,
                           ai_ready=bool(GROQ_API_KEY))


@app.route("/health")
def health():
    return jsonify(ok=True, service="clipkaro")


# ---------------------------------------------------------------- manual clip
@app.route("/clip", methods=["POST"])
@login_required
def clip():
    cleanup_old()
    user = get_user()
    ok, msg = quota_check(user)
    if not ok:
        return jsonify(ok=False, error=msg), 402
    plan = PLANS[user.get("plan", "free")]

    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(ok=False, error="No video uploaded"), 400
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > plan["max_mb"] * 1024 * 1024:
        return jsonify(ok=False, error=f"File too big (max {plan['max_mb']}MB on {plan['name']})"), 400
    if size == 0:
        return jsonify(ok=False, error="Empty file"), 400

    start = parse_ts(request.form.get("start", "0"))
    end = parse_ts(request.form.get("end", ""))
    if start is None or start < 0:
        return jsonify(ok=False, error="Bad start time (use mm:ss or seconds)"), 400
    if end is None or end <= start:
        return jsonify(ok=False, error="End must be after start"), 400
    if end - start > MAX_CLIP_SEC:
        return jsonify(ok=False, error=f"Clip too long (max {MAX_CLIP_SEC // 60} min)"), 400

    job = uuid.uuid4().hex[:12]
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        ext = ".mp4"
    src = os.path.join(UPLOAD_DIR, f"src_{job}{ext}")
    out = os.path.join(UPLOAD_DIR, f"clip_{job}.mp4")
    f.save(src)
    try:
        cut_clip(src, start, end - start, out)
    except RuntimeError as e:
        for p in (src, out):
            if os.path.exists(p):
                os.remove(p)
        return jsonify(ok=False, error=str(e)[:200]), 500
    os.remove(src)
    consume_clip(user)
    return jsonify(ok=True, download=f"/dl/{job}", name=f"clip_{job}.mp4")


# ---------------------------------------------------------------- AI auto-clip
@app.route("/ai-clip", methods=["POST"])
@login_required
def ai_clip():
    cleanup_old()
    user = get_user()
    ok, msg = quota_check(user)
    if not ok:
        return jsonify(ok=False, error=msg), 402
    if not GROQ_API_KEY:
        return jsonify(ok=False, error="AI clipping is not enabled yet (GROQ_API_KEY missing)"), 503
    plan = PLANS[user.get("plan", "free")]

    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(ok=False, error="No video uploaded"), 400
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > plan["max_mb"] * 1024 * 1024:
        return jsonify(ok=False, error=f"File too big (max {plan['max_mb']}MB on {plan['name']})"), 400
    if size == 0:
        return jsonify(ok=False, error="Empty file"), 400

    try:
        count = max(1, min(5, int(request.form.get("count", 4))))
    except ValueError:
        count = 4
    opts = {"count": count,
            "captions": request.form.get("captions", "on") == "on",
            "style": request.form.get("style", "bold") if request.form.get("style") in CAPTION_STYLES else "bold",
            "vertical": request.form.get("vertical", "on") == "on"}

    jid = job_new("ai")
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        ext = ".mp4"
    src = os.path.join(UPLOAD_DIR, f"aisrc_{jid}{ext}")
    f.save(src)
    job_queue.put({"kind": "ai", "jid": jid, "src": src, "opts": opts})
    consume_clip(user)  # one AI job = one clip of quota
    return jsonify(ok=True, job_id=jid)


@app.route("/job/<jid>")
@login_required
def job_status(jid):
    with jobs_lock:
        j = jobs.get(jid)
    if not j:
        return jsonify(ok=False, error="job not found"), 404
    return jsonify(ok=True, status=j["status"], stage=j["stage"],
                   progress=j["progress"], result=j.get("result"),
                   error=j.get("error"))


@app.route("/dl/<job>")
def dl(job):
    if not re.match(r"^[a-f0-9]{12}$", job or ""):
        # ai clip ids look like aiclip_<jid>_<i>
        m = re.match(r"^aiclip_([a-f0-9]{12})_(\d+)$", job or "")
        if not m:
            return "bad id", 400
        p = os.path.join(UPLOAD_DIR, f"aiclip_{m.group(1)}_{m.group(2)}.mp4")
        name = f"clip_{int(m.group(2)) + 1}.mp4"
    else:
        p = os.path.join(UPLOAD_DIR, f"clip_{job}.mp4")
        name = f"clip_{job}.mp4"
    if not os.path.exists(p):
        return "expired or not found (files auto-delete after 1 hour)", 404
    return send_file(p, as_attachment=True, download_name=name)


# ---------------------------------------------------------------- plans + billing
@app.route("/plans")
@login_required
def plans():
    user = get_user()
    plan_id, _ = refresh_plan(user)
    return render_template("plans.html", user=user, plans=PLANS,
                           order=PLAN_ORDER, current=plan_id,
                           billing_on=bool(FAMBRIDGE_API_KEY))


def fambridge_create_order(amount, note):
    r = requests.post(f"{FAMBRIDGE_BASE}/api/create_order",
                      json={"api_key": FAMBRIDGE_API_KEY, "amount": amount, "note": note},
                      timeout=25)
    return r.json()


def fambridge_order_status(order_id):
    r = requests.get(f"{FAMBRIDGE_BASE}/api/order_status",
                     params={"api_key": FAMBRIDGE_API_KEY, "order_id": order_id},
                     timeout=20)
    return r.json()


@app.route("/plans/buy/<plan_id>", methods=["GET", "POST"])
@login_required
def buy_plan(plan_id):
    user = get_user()
    if plan_id not in PLANS or plan_id == "free":
        return redirect(url_for("plans"))
    if not FAMBRIDGE_API_KEY:
        return render_template("buy.html", user=user, plan=PLANS[plan_id],
                               plan_id=plan_id, err="Payments are not enabled yet. Please try again later.")
    plan = PLANS[plan_id]
    note = f"clipkaro_{user['email']}_{plan_id}"
    try:
        order = fambridge_create_order(plan["price"], note)
    except Exception as e:
        return render_template("buy.html", user=user, plan=plan, plan_id=plan_id,
                               err=f"Could not create order: {e}")
    if not order.get("order_id"):
        return render_template("buy.html", user=user, plan=plan, plan_id=plan_id,
                               err=f"Order failed: {order.get('error', 'unknown error')}")
    return render_template("buy.html", user=user, plan=plan, plan_id=plan_id,
                           order_id=order["order_id"],
                           qr=order.get("qr_base64", ""),
                           upi_link=order.get("upi_link", ""), err="")


@app.route("/plans/order_status/<order_id>")
@login_required
def buy_status(order_id):
    user = get_user()
    plan_id = request.args.get("plan", "")
    if plan_id not in PLANS or not FAMBRIDGE_API_KEY:
        return jsonify(ok=False), 400
    try:
        st = fambridge_order_status(order_id)
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:120])
    status = (st.get("status") or "").lower()
    if status == "paid":
        users = load_users()
        u = users.get(user["email"])
        if u:
            u["plan"] = plan_id
            u["plan_expires_at"] = time.time() + 30 * 86400
            u["quota_month"] = current_month()
            u["clips_used"] = 0
            users[user["email"]] = u
            save_users(users)
        return jsonify(ok=True, paid=True)
    return jsonify(ok=True, paid=False, status=status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
