#!/usr/bin/env python3
"""HypeCut web dashboard: paste stream links, start/stop clipping, review the clips.

    .venv/bin/python webui.py            # http://127.0.0.1:8765
    .venv/bin/python webui.py --port 9000

Each stream runs as its own `clipper.py` process writing to `clips_<slug>/`.
Stdlib only. Listens on 127.0.0.1 only.
"""
import argparse
import json
import mimetypes
import os
import re
import shutil
import signal
import tempfile
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import groq_client

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "webui"
CLIPPER = ROOT / "clipper.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_NAME = "hypecut.log"       # clipper stdout+stderr, inside the job's out folder
KEPT_NAME = "webui_kept.json"  # names of clips marked Keep
AI_NAME = "webui_ai.json"      # Gemini's keep/reject suggestion per clip
CLIP_RE = re.compile(r"^\d{3}_.*\.mp4$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LAYOUTS = ("auto", "float", "split", "track", "blur")

jobs = {}  # id -> Job
jobs_lock = threading.Lock()
fs_lock = threading.Lock()  # Keep/Reject file moves


# ---------------------------------------------------------------- helpers

def clean_slug(s):
    return re.sub(r"[^a-z0-9_-]+", "-", (s or "").lower()).strip("-_")[:40]


def channel_from_url(url):
    """Channel name when the URL itself names it (@handle, twitch.tv/<name>, ...), else None."""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    parts = [p for p in u.path.split("/") if p]
    if any(p.startswith("@") for p in parts):
        return next(p for p in parts if p.startswith("@"))[1:]
    if parts and host.endswith(("twitch.tv", "kick.com")) and parts[0] not in ("videos", "video"):
        return parts[0]
    if len(parts) >= 2 and parts[0] in ("c", "user"):
        return parts[1]
    return None


def channel_lookup(url, timeout=25):
    """Ask yt-dlp who streams this (for watch?v=... style links). None if it can't tell in time."""
    exe = ROOT / ".venv" / "bin" / "yt-dlp"
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "yt_dlp"]
    try:
        r = subprocess.run(cmd + ["--no-warnings", "--no-playlist", "--skip-download",
                                  "--print", "%(channel,uploader,uploader_id)s", url],
                           capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and ln.strip() != "NA"]
    return lines[0] if r.returncode == 0 and lines else None


def slug_for(url, lookup=True):
    """Folder slug: the channel name, falling back to the video id / last path part."""
    s = clean_slug(channel_from_url(url) or (channel_lookup(url) if lookup else None))
    if s:
        return s
    u = urlparse(url)
    parts = [p for p in u.path.split("/") if p]
    q = parse_qs(u.query)
    if "v" in q:
        s = q["v"][0]
    elif parts:
        s = parts[-1] if parts[-1] not in ("live", "streams") or len(parts) == 1 else parts[-2]
    else:
        host = (u.hostname or "").lower()
        s = host.split(".")[-2] if host.count(".") else host
    return clean_slug(s) or "stream"


def safe_folder(name):
    """A direct child folder of the repo, by name; None if invalid or missing."""
    if not name or not NAME_RE.match(name):
        return None
    p = ROOT / name
    return p if p.is_dir() and p.parent == ROOT else None


def clip_files(folder):
    return sorted((f for f in folder.iterdir()
                   if f.is_file() and CLIP_RE.match(f.name) and not f.name.endswith("_meme.mp4")),
                  key=lambda f: f.name)


def finished_clips(folder):
    """Clips that are done: clipper writes NNN_x.json last, after the clip and its meme version.
    A clip without one counts as done once it hasn't changed for 2 minutes (stray/hand-made videos)."""
    now = time.time()
    return [f for f in clip_files(folder)
            if f.with_suffix(".json").is_file() or now - f.stat().st_mtime > 120]


def output_folders():
    out = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and NAME_RE.match(d.name):
            if (d / "clips.jsonl").exists() or (d / LOG_NAME).exists() or any(
                    CLIP_RE.match(f.name) for f in d.iterdir()):
                out.append({"name": d.name, "clips": len(finished_clips(d))})
    return out


def tail_lines(path, n=200, max_bytes=256 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    lines = []
    for ln in text.split("\n"):
        ln = ln.rsplit("\r", 1)[-1] if "\r" in ln.rstrip("\r") else ln.rstrip("\r")   # progress bars: last update
        if not ln.strip():
            continue
        if lines and ln.startswith("[download]") and lines[-1].startswith("[download]"):
            lines[-1] = ln   # keep only the newest of a run of download-progress lines
        else:
            lines.append(ln)
    return lines[-n:]


def load_kept(folder):
    try:
        return set(json.loads((folder / KEPT_NAME).read_text()))
    except (OSError, ValueError):
        return set()


def save_kept(folder, kept):
    tmp = folder / (KEPT_NAME + ".tmp")
    tmp.write_text(json.dumps(sorted(kept), indent=1))
    tmp.replace(folder / KEPT_NAME)


# ---------------------------------------------------------------- jobs

JOB_NAME = "webui_job.json"     # the job that writes into a folder (so the job list survives restarts)
SETTINGS_NAME = "webui_settings.json"


def own_scope(argv, name):
    """Run a job in its own systemd scope: if the machine runs out of memory, the kernel kills one job, and
    nothing else (the dashboard, the other jobs, or whatever app launched the dashboard) goes down with it."""
    exe = shutil.which("systemd-run")
    if not exe or not os.environ.get("XDG_RUNTIME_DIR"):
        return argv
    try:
        ok = subprocess.run([exe, "--user", "--scope", "--quiet", "--collect", "--", "true"],
                            capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        return argv
    return [exe, "--user", "--scope", "--quiet", "--collect", f"--unit={name}", "-p", "OOMPolicy=continue", "--",
            *argv]   # systemd-run execs the command, so the pid stays the clipper's


# measured 2026-11-19 (1080p60, --memes): recording + analysis ~0.7 GB per job; a render adds ~1.2 GB, and
# renders take turns across jobs (clipper's render queue), so only one render slot is needed in total
JOB_MEM_GB = 0.7
RENDER_MEM_GB = 1.2
LOCAL_CAPTION_GB = 1.5   # --captions without GROQ_API_KEY loads Whisper large-v3-turbo in every job


def mem_available_gb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1048576
    except OSError:
        pass
    return None


def pid_is_clipper(pid, out):
    """True if pid is still a live clipper.py writing to `out` (pids get reused after a reboot)."""
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    args = [c.decode("utf-8", "replace") for c in cmd]
    return any(a.endswith("clipper.py") for a in args) and out in args


class MemoryWarning(ValueError):
    pass


class Job:
    def __init__(self, url, out, argv, state=None):
        """New job (starts clipper), or with `state` a job re-attached from its folder's webui_job.json."""
        self.url, self.out, self.argv = url, out, argv
        self.log_path = ROOT / out / LOG_NAME
        self.proc = None
        if state:
            self.id, self.pid = state["id"], state["pid"]
            self.started, self.ended = state["started"], state.get("ended")
            self.stop_requested = state.get("stop_requested", False)
            self.returncode = state.get("returncode")
            self.attached_ended = self.ended is not None or not pid_is_clipper(self.pid, out)
            if self.attached_ended and self.ended is None:   # it ended while the dashboard was down
                self.ended = time.time()
                self.save()
            if not self.attached_ended:
                threading.Thread(target=self._watch_pid, daemon=True).start()
            return
        self.id = uuid.uuid4().hex[:8]
        self.started = time.time()
        self.ended = None
        self.stop_requested = False
        self.returncode = None
        self.attached_ended = False
        (ROOT / out).mkdir(exist_ok=True)
        log = open(self.log_path, "ab")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} webui: {' '.join(argv)}\n".encode())
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # own session: Ctrl+C in the webui terminal doesn't hit clipper directly; we SIGINT it ourselves.
        # stdout goes straight to a file so the job survives the webui going away.
        unit = f"hypecut-{clean_slug(out)}-{self.id}"
        self.proc = subprocess.Popen(own_scope(argv, unit), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        log.close()
        self.pid = self.proc.pid
        self.save()
        threading.Thread(target=self._wait, daemon=True).start()

    def save(self, hidden=False):
        try:   # one record per folder: never overwrite a newer job's record (e.g. "Remove" on an old job)
            cur = json.loads((ROOT / self.out / JOB_NAME).read_text())
            if cur.get("id") != self.id and cur.get("started", 0) > self.started:
                return
        except (OSError, ValueError):
            pass
        d = {"id": self.id, "url": self.url, "out": self.out, "argv": self.argv, "pid": self.pid,
             "started": self.started, "ended": self.ended, "stop_requested": self.stop_requested,
             "returncode": self.returncode, "hidden": hidden}
        try:
            tmp = ROOT / self.out / (JOB_NAME + ".tmp")
            tmp.write_text(json.dumps(d, indent=1))
            tmp.replace(ROOT / self.out / JOB_NAME)
        except OSError:
            pass

    def _wait(self):
        self.returncode = self.proc.wait()
        self.ended = time.time()
        self.save()

    def _watch_pid(self):   # a clipper from before a dashboard restart: not our child, so poll it
        while pid_is_clipper(self.pid, self.out):
            time.sleep(2)
        self.attached_ended = True
        self.ended = time.time()
        self.save()

    @property
    def running(self):
        if self.proc is not None:
            return self.returncode is None
        return not self.attached_ended

    @property
    def status(self):
        if self.running:
            return "stopping" if self.stop_requested else "running"
        if self.stop_requested:
            return "stopped"
        if self.returncode is None:   # re-attached job: exit code unknown, read the log
            return "finished" if any(" done. " in ln for ln in self.run_lines(20)) else "interrupted"
        return "finished" if self.returncode == 0 else f"failed (exit {self.returncode})"

    def stop(self, force=False):
        if not self.running:
            return
        self.stop_requested = True
        self.save()
        try:
            if force:
                os.killpg(self.pid, signal.SIGKILL)
            else:
                os.kill(self.pid, signal.SIGINT)  # clipper stops recording and finishes renders
        except ProcessLookupError:
            pass

    def wait(self):
        while self.running:
            time.sleep(0.5)

    def run_lines(self, n=80):
        """Log lines of this run only (a folder's log holds every run, separated by '===== ... webui:')."""
        lines = tail_lines(self.log_path, n)
        for k in range(len(lines) - 1, -1, -1):
            if lines[k].startswith("===== ") and " webui: " in lines[k]:
                return lines[k + 1:]
        return lines

    def progress(self):
        """Where the job is, from its log: 'analyzed 10:30 of stream', 'finishing renders', ..."""
        for ln in reversed(self.run_lines()):
            if " done. " in ln:
                if self.running:   # another run's "done." in the same folder log (e.g. a stopped duplicate)
                    continue
                return "done"
            if "finishing queued renders" in ln:
                return "finishing renders"
            m = re.search(r"analyzed seg_\d+\.ts\s+t=\s*([\d.]+)s", ln)
            if m:
                t = float(m.group(1)) + 30
                return f"analyzed {int(t // 60)}:{int(t % 60):02d} of stream"
            if "recorder: recording" in ln:
                return "recording started"
            if "warming up" in ln or "loading whisper" in ln:
                return "starting"
        return "starting"

    def info(self, log_lines=0):
        folder = ROOT / self.out
        d = {"id": self.id, "url": self.url, "out": self.out, "status": self.status,
             "running": self.running, "pid": self.pid,
             "started": self.started, "ended": self.ended,
             "cmd": " ".join(self.argv[1:]), "progress": self.progress() if self.running else "",
             "clips": len(finished_clips(folder)) if folder.is_dir() else 0,
             "rendering": (len(clip_files(folder)) - len(finished_clips(folder))) if folder.is_dir() else 0}
        if log_lines:
            d["log"] = self.run_lines(log_lines)
        return d


def load_jobs(max_age_days=3):
    """Re-create the job list from the folders' webui_job.json after a dashboard restart."""
    for d in ROOT.iterdir():
        f = d / JOB_NAME
        if not (d.is_dir() and NAME_RE.match(d.name) and f.is_file()):
            continue
        try:
            st = json.loads(f.read_text())
            if st.get("hidden") or st.get("out") != d.name:
                continue
            if st.get("ended") and time.time() - st["ended"] > max_age_days * 86400:
                continue
            job = Job(st["url"], d.name, st["argv"], state=st)
            jobs[job.id] = job
        except (OSError, ValueError, KeyError):
            continue
    # adopt any clipper still writing to a repo folder that has no record (or a stale one)
    known = {j.pid for j in jobs.values()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in known:
            continue
        try:
            args = [a.decode("utf-8", "replace") for a in (proc / "cmdline").read_bytes().split(b"\0") if a]
            started = (proc / "cmdline").stat().st_ctime
        except OSError:
            continue
        ci = next((k for k, a in enumerate(args) if a.endswith("clipper.py")), None)
        if ci is None or "--out" not in args[:-1] or ci + 1 >= len(args):
            continue
        out = Path(args[args.index("--out") + 1]).name
        if not safe_folder(out):
            continue
        st = {"id": uuid.uuid4().hex[:8], "pid": int(proc.name), "started": started, "ended": None}
        job = Job(args[ci + 1], out, args, state=st)
        job.save()
        jobs[job.id] = job
    for d in ROOT.iterdir():
        try:
            if d.is_dir() and json.loads((d / SETTINGS_NAME).read_text()).get("auto_classify"):
                auto_folders.add(d.name)
        except (OSError, ValueError):
            pass


def set_auto(folder_name, on):
    (auto_folders.add if on else auto_folders.discard)(folder_name)
    try:
        (ROOT / folder_name / SETTINGS_NAME).write_text(json.dumps({"auto_classify": bool(on)}))
    except OSError:
        pass


def build_argv(url, out, opt):
    argv = [str(PYTHON if PYTHON.exists() else sys.executable), str(CLIPPER), url, "--out", out]
    for flag in ("memes", "captions", "ai_cam"):
        if opt.get(flag):
            argv.append("--" + flag.replace("_", "-"))
    layout = opt.get("layout") or "auto"
    if layout not in LAYOUTS:
        raise ValueError(f"bad layout {layout!r}")
    if layout != "auto":
        argv += ["--layout", layout]
    for key, flag, kind, lo, hi in (("target_per_hour", "--target-per-hour", float, 1, 1000),
                                    ("duration", "--duration", float, 0, 7 * 24 * 60),
                                    ("min_height", "--min-height", int, 0, 4320)):
        v = opt.get(key)
        if v in (None, ""):
            continue
        try:
            v = kind(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not lo <= v <= hi:
            label = {"target_per_hour": "Target clips / hour", "duration": "Stop after (minutes)",
                     "min_height": "Min source height"}[key]
            raise ValueError(f"{label} must be between {lo} and {hi} (leave it empty for the default)")
        argv += [flag, str(v)]
    return argv


def start_jobs(text, opt):
    # one per line, or separated by spaces/commas; drop stray punctuation around pasted links
    urls = [u.strip("\"'<>()[],;") for u in re.split(r"[\s,]+", text or "")]
    urls = list(dict.fromkeys(u for u in urls if u))
    if not urls:
        raise ValueError("paste at least one stream URL")
    for u in urls:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise ValueError(f"not an http(s) URL: {u}")
    free = mem_available_gb()
    per_job = JOB_MEM_GB + (LOCAL_CAPTION_GB if opt.get("captions") and not groq_client.api_key() else 0)
    need = per_job * len(urls) + (0 if any(j.running for j in jobs.values()) else RENDER_MEM_GB)
    if free is not None and free < need and not opt.get("force"):
        raise MemoryWarning(f"Only {free:.1f} GB of memory is free; {len(urls)} job(s) need about {need:.1f} GB. "
                            "If the computer runs out, the system kills a job. Close some apps or stop a job "
                            "first, or start anyway.")
    # channel lookups hit the network: run them in parallel, before taking the lock
    slugs = {}
    threads = [threading.Thread(target=lambda u=u: slugs.__setitem__(u, slug_for(u))) for u in set(urls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    started = []
    with jobs_lock:
        for u in urls:
            busy = {j.out for j in jobs.values() if j.running}
            base = "clips_" + slugs[u]
            out, n = base, 2
            while out in busy:
                out, n = f"{base}-{n}", n + 1
            job = Job(u, out, build_argv(u, out, opt))
            if opt.get("auto_classify"):
                set_auto(out, True)
            jobs[job.id] = job
            started.append(job.info())
    return started


# ---------------------------------------------------------------- clips

def list_clips(folder):
    kept = load_kept(folder)
    ai = load_ai(folder)
    clips = []
    done = finished_clips(folder)
    for f in reversed(done):  # newest first
        meta = {}
        try:
            meta = json.loads(f.with_suffix(".json").read_text())
        except (OSError, ValueError):
            pass
        meme = meta.get("meme_file") or f.stem + "_meme.mp4"
        clips.append({
            "file": f.name,
            "meme": meme if (folder / meme).is_file() else None,
            "kept": f.name in kept,
            "ai": ai.get(f.name),
            "size": f.stat().st_size,
            "mtime": f.stat().st_mtime,
            **{k: meta.get(k) for k in ("score", "ai_score", "reasons", "emotion", "title", "hook", "edges",
                                         "duration", "layout", "captions", "transcript", "created")},
        })
    rejected = folder / "rejected"
    n_rej = len(clip_files(rejected)) if rejected.is_dir() else 0
    return {"folder": folder.name, "clips": clips, "rejected": n_rej,
            "rendering": len(clip_files(folder)) - len(done),
            "classify": classify_status(folder.name), "gemini": ai_ready()}


def clip_group(folder, name):
    """The clip's files: NNN_x.mp4, NNN_x.json, NNN_x_meme.mp4, and any other NNN_x.<ext>."""
    if not CLIP_RE.match(name) or name.endswith("_meme.mp4") or not (folder / name).is_file():
        raise ValueError("no such clip")
    stem = name[:-4]
    return [f for f in folder.iterdir() if f.is_file() and (
        f.name.startswith(stem + ".") or f.name.startswith(stem + "_meme."))]


def reject_clip(folder, name):
    with fs_lock:
        files = clip_group(folder, name)
        dest = folder / "rejected"
        dest.mkdir(exist_ok=True)
        for f in files:
            target, n = dest / f.name, 2
            while target.exists():  # never overwrite an earlier reject
                target, n = dest / f"{f.stem}.{n}{f.suffix}", n + 1
            shutil.move(str(f), str(target))
        kept = load_kept(folder)
        if name in kept:
            kept.discard(name)
            save_kept(folder, kept)
        return [f.name for f in files]


def apply_suggestions(folder):
    """Gemini's keep -> Keep; its reject -> rejected/ (clips you marked Keep yourself are never moved)."""
    ai, kept = load_ai(folder), load_kept(folder)
    n_keep = n_rej = 0
    for f in finished_clips(folder):
        v = ai.get(f.name)
        if not v or "error" in v:
            continue
        if v["keep"] and f.name not in kept:
            keep_clip(folder, f.name)
            n_keep += 1
        elif not v["keep"] and f.name not in kept:
            reject_clip(folder, f.name)
            n_rej += 1
    return {"kept": n_keep, "rejected": n_rej}


def keep_clip(folder, name, keep=True):
    with fs_lock:
        clip_group(folder, name)
        kept = load_kept(folder)
        (kept.add if keep else kept.discard)(name)
        save_kept(folder, kept)


# ---------------------------------------------------------------- storage cleanup
# clipper keeps its raw recording (.work/run_*/segments, VOD downloads, chat logs) and per-clip render files
# (.work/<clip>/) and never deletes them at the end of a run; after a crash they include half-downloaded files.
# Cleanup only ever deletes inside <folder>/.work (never clips, .json, rejected/), keeps the small meme_assets,
# and skips folders a running clipper is writing to.

KEEP_IN_WORK = {"meme_assets"}


def tree_size(p):
    total = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def leftover_items(folder):
    work = folder / ".work"
    if not work.is_dir() or work.is_symlink():
        return []
    return [p for p in work.iterdir() if p.name not in KEEP_IN_WORK]


def clipper_outs():
    """--out folder names of every clipper.py running on this machine (dashboard jobs or started by hand)."""
    outs = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = [a.decode("utf-8", "replace") for a in (d / "cmdline").read_bytes().split(b"\0")]
        except OSError:
            continue
        if any(a.endswith("clipper.py") for a in args) and "--out" in args[:-1]:
            outs.add(Path(args[args.index("--out") + 1]).name)
    return outs


def tmp_leftovers():
    return [p for p in Path(tempfile.gettempdir()).glob("hypecut-ai-*") if p.is_dir() and not p.is_symlink()]


def storage_report():
    busy = clipper_outs()
    rows = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and NAME_RE.match(d.name) and not d.is_symlink():
            items = leftover_items(d)
            if items:
                rows.append({"folder": d.name, "bytes": sum(tree_size(p) for p in items), "in_use": d.name in busy})
    return {"folders": rows, "tmp_bytes": sum(tree_size(p) for p in tmp_leftovers())}


def clean_storage(names=None):
    """Delete leftovers in the given folders (None = all not in use) plus stale /tmp previews. Returns bytes freed."""
    busy = clipper_outs()
    freed, skipped = 0, []
    targets = [safe_folder(n) for n in names] if names else [d for d in ROOT.iterdir() if d.is_dir() and NAME_RE.match(d.name)]
    with fs_lock:
        for folder in targets:
            if folder is None:
                continue
            if folder.name in busy:
                skipped.append(folder.name)
                continue
            for p in leftover_items(folder):
                size = tree_size(p)
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                freed += size - (tree_size(p) if p.exists() else 0)
        with cls_lock:
            classifying = cls_current is not None
        if not classifying:   # a preview being uploaded right now lives there
            for p in tmp_leftovers():
                freed += tree_size(p)
                shutil.rmtree(p, ignore_errors=True)
    return {"freed": freed, "skipped_in_use": skipped}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def clean_cli(yes=False):
    rep_ = storage_report()
    if not rep_["folders"] and not rep_["tmp_bytes"]:
        print("nothing to clean")
        return
    total = 0
    for r in rep_["folders"]:
        print(f"  {human(r['bytes']):>9}  {r['folder']}/.work{'   (in use by a running clipper - skipped)' if r['in_use'] else ''}")
        total += 0 if r["in_use"] else r["bytes"]
    if rep_["tmp_bytes"]:
        print(f"  {human(rep_['tmp_bytes']):>9}  dashboard previews in {tempfile.gettempdir()}")
        total += rep_["tmp_bytes"]
    print(f"Deletes raw recordings and render files only (clips, .json and rejected/ stay): {human(total)}")
    if not yes and input("Delete them? [y/N] ").strip().lower() not in ("y", "yes"):
        print("nothing deleted")
        return
    r = clean_storage()
    print(f"freed {human(r['freed'])}" + (f"; skipped (in use): {', '.join(r['skipped_in_use'])}" if r["skipped_in_use"] else ""))


# ---------------------------------------------------------------- auto-classify (Gemini watches the clip)
# Suggestions only: verdicts are stored in <folder>/webui_ai.json and shown on the cards;
# nothing is kept or moved until the user clicks (one by one, or "Apply suggestions").

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
JUDGE_PROMPT = """You are the editor of a YouTube Shorts / TikTok channel that posts clips from live streams.
Watch and listen to this vertical clip (gameplay and/or the streamer's facecam). The streamer may speak English,
Hindi/Urdu or Hinglish. Decide if it is worth posting.

KEEP only if a viewer scrolling past, with no context, would stop and watch to the end:
- a real, visible or audible reaction (shock, rage, laughing fit, scream) or a funny / wild moment,
- something actually happens (a payoff), not just loud talking or background noise,
- it makes sense on its own.
REJECT if it is mostly talking, menus, loading screens, dead air, music, a reaction to something viewers can't
see or understand, or the "hype" is only chat spam.

Be strict: around half of the clips from a stream should be rejected.
Return keep (true/false), score 1-10 (how strongly a Shorts viewer would watch it), a reason of at most 15 words
naming what happens in the clip, and a catchy title of at most 6 words in the streamer's language style.

The clip cutter's own notes (may be wrong): {notes}"""

auto_folders = set()          # folders whose new clips get classified as they finish
model_busy_until = {}         # model -> time it may be tried again
cls_lock = threading.Condition()
cls_queue = []                # (folder name, clip name)
cls_current = None


def ai_ready():
    return bool(env_key() or groq_client.api_key())


def env_key():
    """GEMINI_API_KEY from the environment or .env, re-read each time so adding it needs no restart."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v.strip() and not line.lstrip().startswith("#"):
                return v.strip().strip('"\'')
    except OSError:
        pass
    return None


def load_ai(folder):
    try:
        return json.loads((folder / AI_NAME).read_text())
    except (OSError, ValueError):
        return {}


def save_ai_entry(folder, name, entry):
    with fs_lock:
        d = load_ai(folder)
        d[name] = entry
        tmp = folder / (AI_NAME + ".tmp")
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        tmp.replace(folder / AI_NAME)


def preview_copy(src, dst):
    """Small 360p copy for the upload (~1-2 MB for 20 s instead of ~30 MB)."""
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", "90",
                        "-vf", "scale=360:-2,fps=15", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                        "-c:a", "aac", "-b:a", "64k", "-ac", "1", str(dst)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr.strip().splitlines() or ["failed"])[-1])


def gemini_judge(video_bytes, notes):
    """Returns (dict(keep, score, reason, title), model). Raises if no model answered."""
    key = env_key()
    if not key:
        raise RuntimeError("no GEMINI_API_KEY in .env")
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class Verdict(BaseModel):
        keep: bool
        score: int
        reason: str
        title: str

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120000))
    part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Verdict,
                                      temperature=0)
    errors = []
    now = time.time()
    models = [m for m in GEMINI_MODELS if model_busy_until.get(m, 0) <= now] or GEMINI_MODELS
    for model in models:
        try:
            o = client.models.generate_content(model=model, contents=[part, JUDGE_PROMPT.format(notes=notes)],
                                               config=cfg).parsed
        except Exception as e:
            msg = str(e)
            if "API key" in msg or "PERMISSION_DENIED" in msg or "401" in msg:
                raise RuntimeError("Gemini rejected the API key") from None
            model_busy_until[model] = time.time() + 300   # overloaded / rate-limited: skip it for 5 minutes
            errors.append(f"{model}: {e.__class__.__name__}")
            continue
        if o is not None:
            return {"keep": bool(o.keep), "score": max(1, min(10, int(o.score))),
                    "reason": o.reason.strip()[:200], "title": o.title.strip()[:80]}, model
    raise RuntimeError("no Gemini model answered (" + "; ".join(errors) + ")")


def classify_one(folder, name):
    import tempfile
    clip = folder / name
    meta = {}
    try:
        meta = json.loads(clip.with_suffix(".json").read_text())
    except (OSError, ValueError):
        pass
    notes = "; ".join(meta.get("reasons") or []) or "none"
    if meta.get("transcript"):
        notes += f". Transcript (automatic, rough): {meta['transcript'][:600]}"
    errors = []
    with tempfile.TemporaryDirectory(prefix="hypecut-ai-") as td:
        if env_key():
            try:
                small = Path(td) / "clip.mp4"
                preview_copy(clip, small)
                verdict, model = gemini_judge(small.read_bytes(), notes)
                return {**verdict, "model": model, "at": time.time()}
            except RuntimeError as e:
                errors.append(f"Gemini: {e}")
        if groq_client.api_key():
            try:
                sheet = Path(td) / "frames.jpg"
                contact_sheet(clip, sheet, meta.get("duration"))
                verdict, model = groq_judge(sheet.read_bytes(), notes)
                return {**verdict, "model": "groq " + model.split("/")[-1], "at": time.time()}
            except (RuntimeError, groq_client.GroqError) as e:
                errors.append(f"Groq: {e}")
    raise RuntimeError("; ".join(errors) or "no GEMINI_API_KEY or GROQ_API_KEY in .env")


JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["keep", "score", "reason", "title"],
                "properties": {"keep": {"type": "boolean"}, "score": {"type": "integer"},
                               "reason": {"type": "string"}, "title": {"type": "string"}}}


def contact_sheet(src, dst, duration=None):
    """6 frames spread over the clip, tiled 3x2 into one JPEG (Groq's vision model takes at most 3 images)."""
    if not duration:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(src)], capture_output=True, text=True, timeout=30)
        duration = float(r.stdout.strip() or 20)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-vf", f"fps={6 / max(float(duration), 1):.4f},scale=320:-2,tile=3x2",
                        "-frames:v", "1", "-q:v", "4", str(dst)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not Path(dst).exists():
        raise RuntimeError("ffmpeg frames: " + (r.stderr.strip().splitlines() or ["failed"])[-1])


def groq_judge(jpeg, notes):
    prompt = (JUDGE_PROMPT.format(notes=notes).replace("Watch and listen to this vertical clip",
                                                       "Judge this vertical clip")
              + "\n\nYou cannot watch or hear it: the image shows 6 frames spread over the clip, in time order "
                "(left to right, then top to bottom). Judge from the frames and the transcript.")
    v, model = groq_client.chat_json([{"type": "text", "text": prompt}, groq_client.image_part(jpeg)],
                                     JUDGE_SCHEMA, models=groq_client.VISION_MODELS, name="verdict", timeout=90)
    return {"keep": bool(v["keep"]), "score": max(1, min(10, int(v["score"]))),
            "reason": str(v["reason"]).strip()[:200], "title": str(v["title"]).strip()[:80]}, model


def enqueue(folder, names):
    with cls_lock:
        n = 0
        for name in names:
            item = (folder.name, name)
            if item not in cls_queue and item != cls_current:
                cls_queue.append(item)
                n += 1
        cls_lock.notify()
        return n


def unclassified(folder, retry_errors=False):
    ai = load_ai(folder)
    return [f.name for f in finished_clips(folder)
            if f.name not in ai or (retry_errors and "error" in ai[f.name])]


def classify_worker():
    global cls_current
    while True:
        with cls_lock:
            while not cls_queue:
                cls_lock.wait()
            cls_current = cls_queue.pop(0)
        fname, name = cls_current
        folder = safe_folder(fname)
        try:
            if folder and (folder / name).is_file():   # may have been rejected/replaced meanwhile
                try:
                    entry = classify_one(folder, name)
                except Exception as e:
                    entry = {"error": str(e)[:300], "at": time.time()}
                save_ai_entry(folder, name, entry)
        finally:
            with cls_lock:
                cls_current = None


def auto_watcher():
    """New clips in auto folders get queued as they finish (only clips without a verdict, errors not retried)."""
    while True:
        time.sleep(5)
        if not ai_ready():
            continue
        for fname in list(auto_folders):
            folder = safe_folder(fname)
            if folder:
                enqueue(folder, unclassified(folder))


def classify_status(folder_name):
    with cls_lock:
        return {"queued": sum(1 for f, _ in cls_queue if f == folder_name),
                "current": cls_current[1] if cls_current and cls_current[0] == folder_name else None,
                "auto": folder_name in auto_folders}


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "HypeCut"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        if os.environ.get("HYPECUT_WEBUI_DEBUG"):
            super().log_message(fmt, *a)

    # -- guards: localhost Host header (DNS rebinding) and JSON-only POSTs (cross-site forms)
    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, code, msg):
        self.send_json({"error": msg}, code)

    def do_GET(self):
        if not self._host_ok():
            return self.error(403, "bad host")
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html", cache=False)
        if path == "/api/state":
            with jobs_lock:
                js_jobs = list(jobs.values())
                js = [j.info() for j in sorted(js_jobs, key=lambda j: -j.started)]
            free = mem_available_gb()
            return self.send_json({"jobs": js, "folders": output_folders(),
                                   "mem_free_gb": round(free, 1) if free is not None else None,
                                   "job_mem_gb": JOB_MEM_GB,
                                   "render_mem_gb": 0 if any(j.running for j in js_jobs) else RENDER_MEM_GB})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
        if m:
            job = jobs.get(m.group(1))
            if not job:
                return self.error(404, "no such job")
            return self.send_json(job.info(log_lines=int(q.get("n", ["200"])[0] or 200)))
        if path == "/api/clips":
            folder = safe_folder(q.get("folder", [""])[0])
            if not folder:
                return self.error(404, "no such folder")
            return self.send_json(list_clips(folder))
        if path == "/api/storage":
            return self.send_json(storage_report())
        if path == "/api/log":
            folder = safe_folder(q.get("folder", [""])[0])
            if not folder:
                return self.error(404, "no such folder")
            return self.send_json({"log": tail_lines(folder / LOG_NAME, 200)})
        m = re.fullmatch(r"/media/([^/]+)/([^/]+)", path)
        if m:
            folder = safe_folder(m.group(1))
            name = m.group(2)
            if not folder or not CLIP_RE.match(name):
                return self.error(404, "not found")
            return self.send_file(folder / name, cache=True)
        self.error(404, "not found")

    def do_POST(self):
        if not self._host_ok():
            return self.error(403, "bad host")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self.error(415, "JSON only")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n <= 1 << 20 else None
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.error(400, "bad JSON")
        path = urlparse(self.path).path
        try:
            if path == "/api/jobs":
                try:
                    return self.send_json({"started": start_jobs(body.get("urls"), body.get("options") or {})})
                except MemoryWarning as e:
                    return self.send_json({"error": str(e), "low_memory": True}, 409)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/(stop|kill|forget)", path)
            if m:
                job = jobs.get(m.group(1))
                if not job:
                    return self.error(404, "no such job")
                if m.group(2) == "forget":
                    if job.running:
                        return self.error(409, "job still running")
                    with jobs_lock:
                        jobs.pop(job.id, None)
                    job.save(hidden=True)
                    return self.send_json({"ok": True})
                job.stop(force=m.group(2) == "kill")
                return self.send_json(job.info())
            if path == "/api/storage/clean":
                if body.get("all") is True:
                    return self.send_json(clean_storage(None))
                names = body.get("folders")
                if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
                    return self.error(400, "pass folders: [names] or all: true")
                return self.send_json(clean_storage(names))
            if path in ("/api/classify", "/api/classify/auto", "/api/clips/apply"):
                folder = safe_folder(body.get("folder", ""))
                if not folder:
                    return self.error(404, "no such folder")
                if path == "/api/clips/apply":
                    return self.send_json(apply_suggestions(folder))
                if path == "/api/classify/auto":
                    set_auto(folder.name, body.get("on"))
                    return self.send_json(classify_status(folder.name))
                if not ai_ready():
                    return self.error(400, "add GEMINI_API_KEY or GROQ_API_KEY to .env first (both have free keys)")
                return self.send_json({"queued": enqueue(folder, unclassified(folder, retry_errors=True))})
            m = re.fullmatch(r"/api/clips/(keep|unkeep|reject)", path)
            if m:
                folder = safe_folder(body.get("folder", ""))
                name = body.get("file", "")
                if not folder or not isinstance(name, str):
                    return self.error(404, "no such folder")
                if m.group(1) == "reject":
                    return self.send_json({"moved": reject_clip(folder, name)})
                keep_clip(folder, name, m.group(1) == "keep")
                return self.send_json({"ok": True})
        except ValueError as e:
            return self.error(400, str(e))
        self.error(404, "not found")

    def send_file(self, path, cache):
        try:
            f = open(path, "rb")
        except OSError:
            return self.error(404, "not found")
        with f:
            size = os.fstat(f.fileno()).st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            length = end - start + 1
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            f.seek(start)
            try:
                while length > 0:
                    chunk = f.read(min(1 << 16, length))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    length -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled (normal while seeking)


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)  # browsers drop connections all the time


def say(msg):
    try:
        print(msg, flush=True)
    except OSError:   # the terminal window was closed
        pass


def single_instance(port):
    """Only one dashboard per repo: a second one would re-attach the same clipper jobs and stop them on exit."""
    import fcntl
    import hashlib
    run = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    path = Path(run) / f"hypecut-webui-{hashlib.md5(str(ROOT).encode()).hexdigest()[:8]}.lock"
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        other = f.read().strip() or "?"
        say(f"HypeCut dashboard is already running (http://127.0.0.1:{other}) - not starting a second one.")
        return None
    f.seek(0)
    f.truncate()
    f.write(str(port))
    f.flush()
    return f   # keep the file open: the lock lasts as long as this process


def main():
    ap = argparse.ArgumentParser(description="HypeCut web dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--clean", action="store_true",
                    help="delete leftover raw recordings and render files (in each clip folder's .work) and exit")
    ap.add_argument("--yes", action="store_true", help="with --clean: don't ask")
    args = ap.parse_args()
    if args.clean:
        return clean_cli(args.yes)
    lock = single_instance(args.port)
    if lock is None:
        return
    try:
        srv = Server(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"can't listen on 127.0.0.1:{args.port} ({e.strerror}); is another program using that port? "
                 f"Try --port {args.port + 1}")
    srv.daemon_threads = True

    leave_jobs = []

    def on_hangup(*_):   # terminal window closed: shut down like Ctrl+C (stops the jobs)
        raise KeyboardInterrupt

    def on_term(*_):     # killed from outside (systemd, logout): jobs run in their own scopes, leave them be;
        leave_jobs.append(1)   # the next dashboard start re-attaches them
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)   # even if started with SIGINT ignored
    signal.signal(signal.SIGHUP, on_hangup)
    signal.signal(signal.SIGTERM, on_term)
    load_jobs()
    threading.Thread(target=classify_worker, daemon=True).start()
    threading.Thread(target=auto_watcher, daemon=True).start()
    say(f"HypeCut dashboard: http://127.0.0.1:{args.port}   (Ctrl+C to quit)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    running = [j for j in jobs.values() if j.running]
    if running and leave_jobs:
        say(f"\ndashboard stopped from outside; {len(running)} job(s) keep running and reappear on the next start")
        running = []
    if running:
        say(f"\nstopping {len(running)} clipping job(s); they finish their renders first "
            f"(Ctrl+C again to kill them)...")
        for j in running:
            j.stop()
        try:
            for j in running:
                j.wait()
        except KeyboardInterrupt:
            for j in running:
                j.stop(force=True)
    say("bye")


if __name__ == "__main__":
    main()
