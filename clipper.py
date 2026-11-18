#!/usr/bin/env python3
"""
HypeCut - record a live stream and keep cutting vertical "viral" clips while it runs.

Pipeline (all local unless a flag says otherwise, runs continuously):
  1. Recorder   : yt-dlp resolves the stream, ffmpeg records it into 30s .ts segments.
  2. Chat       : YouTube live chat (via yt-dlp) or Twitch IRC -> volume and hype per second.
  3. Analyzer   : per segment -> loudness, screams, motion, face emotion, Whisper transcript.
  4. Selector   : excitement score per second, each signal measured against the stream's own
                  last 5 minutes; picks peaks, merges neighbours, snaps edges to pauses.
  5. AI review  : (optional, --ai) Claude rates each candidate and writes a title + hook.
  6. Renderer   : 1080x1920 vertical. Facecam detected -> gameplay on top, face at bottom.
                  Big face (IRL) -> 9:16 crop that follows the face. No face -> blurred fill.
                  (optional, --captions) Gemini-verified word-by-word captions.

Usage:
  python clipper.py "https://www.youtube.com/watch?v=XXXX"           # live stream
  python clipper.py "https://www.twitch.tv/somechannel"
  python clipper.py recording.mp4                                      # existing file / VOD
"""
import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACE_MODEL = HERE / "models" / "face_detection_yunet_2023mar.onnx"
OUT_W, OUT_H = 1080, 1920
OUT_FPS = 30  # raised to 60 automatically for 60fps sources
SR = 16000

HYPE_WORDS = [
    # english
    "no way", "oh my god", "omg", "what the", "wtf", "let's go", "lets go", "bro", "dude", "insane",
    "crazy", "holy", "yes", "no no", "what", "wait", "oh no", "run", "help", "clip that", "clip it",
    "unbelievable", "are you serious", "hahaha", "haha", "laugh", "scream", "damn", "sheesh",
    # hindi / urdu / hinglish
    "bhai", "arre", "kya", "oye", "yaar", "mar gaya", "bachao", "bhaago", "chal", "abey", "abe",
    "kya baat", "pagal", "mazaa", "maza",
]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def even(x):
    x = int(round(x))
    return x - (x % 2)


def slug(s, n=48):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", s)[:n] or "clip"


def probe_duration(p):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def pick_encoder():
    encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in encs:
        return ["-c:v", "h264_videotoolbox", "-b:v", "18M", "-maxrate", "24M", "-profile:v", "high"]
    if "h264_nvenc" in encs:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "10M"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "17"]


# --------------------------------------------------------------------------------------
# Source resolution + recording
# --------------------------------------------------------------------------------------

def resolve_stream(url, max_h):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True,
            "format": f"bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]/best"}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    fmts = info.get("requested_formats") or [info]
    return info, [f["url"] for f in fmts]


class Recorder(threading.Thread):
    """Keeps an ffmpeg process recording the live stream into numbered segments.
    Re-resolves and reconnects when ffmpeg drops (URLs expire, network blips)."""

    def __init__(self, url, segdir, seg_time, max_h):
        super().__init__(daemon=True)
        self.url, self.segdir, self.seg_time, self.max_h = url, Path(segdir), seg_time, max_h
        self.stop_evt = threading.Event()
        self.proc = None
        self.ended = False

    def _next_index(self):
        nums = [int(p.stem.split("_")[1]) for p in self.segdir.glob("seg_*.ts")]
        return max(nums) + 1 if nums else 0

    def run(self):
        fails = 0
        while not self.stop_evt.is_set():
            try:
                info, urls = resolve_stream(self.url, self.max_h)
            except Exception as e:  # stream ended / offline / network
                fails += 1
                log(f"recorder: could not resolve stream ({e.__class__.__name__}); retry {fails}/6")
                if fails >= 6:
                    break
                time.sleep(10)
                continue
            if fails and not info.get("is_live"):
                log("recorder: stream is no longer live")
                break
            # -copyts keeps the platform's shared clock: video and audio arrive as separate streams that
            # start at different points, and without it ffmpeg zeroes each one separately (lip-sync drift)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts"]
            for u in urls:
                if "m3u8" in u or "hls_playlist" in u:
                    cmd += ["-extension_picky", "0"]  # YouTube HLS audio segments use odd extensions
                cmd += ["-i", u]
            cmd += (["-map", "0:v:0", "-map", "1:a:0"] if len(urls) > 1 else ["-map", "0:v:0", "-map", "0:a:0?"])
            cmd += ["-c", "copy", "-f", "segment", "-segment_time", str(self.seg_time),
                    "-segment_start_number", str(self._next_index()),
                    str(self.segdir / "seg_%06d.ts")]
            log(f"recorder: recording '{info.get('title', '?')}' ({info.get('height', '?')}p)")
            self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
            _, err = self.proc.communicate()
            if self.stop_evt.is_set():
                break
            errs = [l for l in (err or "").splitlines() if not l.startswith("[hls")][-2:]
            log("recorder: ffmpeg exited" + (f" ({' | '.join(errs)})" if errs else "") + "; reconnecting")
            fails += 1
            if fails >= 6:
                break
            time.sleep(3)
        self.ended = True

    def stop(self):
        self.stop_evt.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin and self.proc.stdin.close()
            except Exception:
                pass
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------------------
# Chat activity
# --------------------------------------------------------------------------------------

CHAT_HYPE = re.compile(
    r"(\blo+l+\b|\blmf?ao+\b|\bomg+\b|\bw+\b|\bwtf\b|\bclip\b|\bpog|\bkek|\bhaha|\bxd+\b|\bno way\b|\bbruh+\b|"
    r"\bbhai\b|\bkya\b|\bsheesh|\binsane|!!+|\?\?+|😂|🤣|💀|😭|🔥|😱|🤯|😳|:face-|:fire:)", re.I)


def chat_hype(text):
    """0..3 intensity of a chat message: hype tokens, ALL CAPS shouting, emote spam."""
    if not text:
        return 0.0
    h = min(2, len(CHAT_HYPE.findall(text)))
    letters = [c for c in text if c.isalpha() and c.isascii()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        h += 1
    return float(min(3, h))


class ChatCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.walls = []  # (wall-clock timestamp, hype 0..3, paid?)

    def add(self, ts, text="", paid=False):
        with self.lock:
            self.walls.append((ts, chat_hype(text), paid))

    def snapshot(self):
        with self.lock:
            return list(self.walls)


def _find_renderers(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer") and isinstance(v, dict):
                out.append((k, v))
            else:
                _find_renderers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_renderers(v, out)
    return out


def youtube_chat_thread(url, workdir, counter, stop_evt):
    """yt-dlp writes live chat progressively; we tail the file and grab message timestamps."""
    base = Path(workdir) / "chat"
    proc = subprocess.Popen([sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--skip-download",
                             "--write-subs", "--sub-langs", "live_chat", "-o", str(base), url],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    offsets = {}
    seen = set()
    try:
        while not stop_evt.is_set():
            for f in Path(workdir).glob("chat.live_chat.json*"):
                off = offsets.get(f, 0)
                try:
                    with open(f, "rb") as fh:
                        fh.seek(off)
                        data = fh.read()
                except OSError:
                    continue
                cut = data.rfind(b"\n") + 1
                offsets[f] = off + cut
                for line in data[:cut].splitlines():
                    if b"liveChatTextMessageRenderer" not in line and b"liveChatPaidMessageRenderer" not in line:
                        continue
                    try:
                        rends = _find_renderers(json.loads(line), [])
                    except ValueError:
                        continue
                    for kind, r in rends:
                        mid = r.get("id") or r.get("timestampUsec")
                        if not r.get("timestampUsec") or mid in seen:
                            continue
                        seen.add(mid)
                        runs = (r.get("message") or {}).get("runs") or []
                        text = " ".join(x.get("text") or ":" + str(((x.get("emoji") or {}).get("shortcuts") or ["e"])[0])
                                        for x in runs)
                        counter.add(int(r["timestampUsec"]) / 1e6, text, kind == "liveChatPaidMessageRenderer")
            time.sleep(2)
    finally:
        proc.terminate()


def twitch_chat_thread(channel, counter, stop_evt):
    """Anonymous read-only Twitch IRC connection."""
    while not stop_evt.is_set():
        try:
            s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=30)
            s.sendall(f"NICK justinfan{np.random.randint(10000, 99999)}\r\nJOIN #{channel.lower()}\r\n".encode())
            buf = b""
            while not stop_evt.is_set():
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\r\n")
                for l in lines:
                    if l.startswith(b"PING"):
                        s.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif b" PRIVMSG #" in l:
                        msg = l.split(b" PRIVMSG #", 1)[1].split(b" :", 1)
                        counter.add(time.time(), msg[1].decode("utf-8", "ignore") if len(msg) > 1 else "",
                                    b"bits=" in l.split(b" ", 1)[0])
        except Exception:
            time.sleep(5)


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------

@dataclass
class Segment:
    idx: int
    path: str
    t0: float      # start on the global timeline (seconds since recording start)
    dur: float
    wall0: float   # wall clock time of the segment's first frame (approx.)


@dataclass
class Word:
    t0: float
    t1: float
    text: str


def decode_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
# how much each facial emotion counts as "clip-worthy excitement"
EMO_WEIGHT = np.array([0.8, 0.5, 1.0, 0.9, 0.0, 0.4, 1.0])
FER_MODEL = HERE / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
_FER_STD = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class EmotionReader:
    """YuNet face + landmarks -> aligned 112x112 crop -> MobileFaceNet FER (7 emotions)."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.ok = FER_MODEL.exists() and FACE_MODEL.exists()
        if self.ok:
            self.det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.6, 0.3, 20)
            self.net = cv2.dnn.readNet(str(FER_MODEL))

    def read(self, img):
        cv2 = self.cv2
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, fs = self.det.detect(img)
        if fs is None or not len(fs):
            return None
        f = max(fs, key=lambda r: r[2] * r[3])
        M, _ = cv2.estimateAffinePartial2D(f[4:14].reshape(5, 2).astype(np.float32), _FER_STD)
        if M is None:
            return None
        face = cv2.warpAffine(img, M, (112, 112))
        x = (cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5) / 0.5
        self.net.setInput(cv2.dnn.blobFromImage(x), "data")
        out = self.net.forward(["label"])[0][0]
        p = np.exp(out - out.max())
        return p / p.sum()


class Timeline:
    """Per-second signal store for the whole stream (global timeline = seconds since recording start)."""

    EMO_FPS = 2

    MOTION_FPS = 5
    MW, MH = 160, 90

    def __init__(self, whisper_model, language, chat_lag):
        self.segs: list[Segment] = []
        self.words: list[Word] = []
        self.audio_bins: dict[int, list] = {}   # bin -> [(db, centroid_hz, flux), ...]
        self.motion_bins: dict[int, list] = {}  # bin -> [(game_motion, face_motion), ...]
        self.end = 0.0
        self.whisper = whisper_model
        self.language = language
        self.chat_lag = chat_lag
        self.emo_bins: dict[int, list] = {}     # bin -> [7 emotion probabilities, ...]
        self.heat = []                          # YouTube "most replayed": [(start, end, value 0..1)]
        self.layout = None                      # facecam layout (set once detected)
        self.emotion = EmotionReader()
        self._z = {}

    def _emotion_features(self, seg):
        lay = self.layout
        if not self.emotion.ok or not lay or lay.get("mode") == "blur":
            return
        W, H = lay["W"], lay["H"]
        if lay["mode"] == "split":
            cx, cy, cw, ch = lay["cam"]
            pad = 0.1
            x0, y0 = max(0, cx - cw * pad), max(0, cy - ch * pad)
            x1, y1 = min(W, cx + cw * (1 + pad)), min(H, cy + ch * (1 + pad))
            ow = 320
            oh = even(ow * (y1 - y0) / max(1, x1 - x0))
            vf = f"crop={even(x1 - x0)}:{even(y1 - y0)}:{even(x0)}:{even(y0)},scale={ow}:{oh}"
        else:
            ow, oh = 640, even(640 * H / W)
            vf = f"scale={ow}:{oh}"
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", seg.path, "-an", "-vf",
                              f"fps={self.EMO_FPS},{vf},format=bgr24", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        fsz = ow * oh * 3
        for k in range(len(raw) // fsz):
            img = np.frombuffer(raw[k * fsz:(k + 1) * fsz], np.uint8).reshape(oh, ow, 3)
            p = self.emotion.read(img)
            if p is not None:
                self.emo_bins.setdefault(int(seg.t0 + (k + 0.5) / self.EMO_FPS), []).append(p)

    # ---------------- per-segment analysis ----------------
    def _audio_features(self, seg, audio):
        hop = int(SR * 0.25)
        win = np.hanning(hop).astype(np.float32)
        freqs = np.fft.rfftfreq(hop, 1 / SR)
        prev = None
        for i in range(0, max(0, len(audio) - hop + 1), hop):
            fr = audio[i:i + hop]
            rms = float(np.sqrt(np.mean(fr * fr)))
            db = 20 * math.log10(rms + 1e-6)
            mag = np.abs(np.fft.rfft(fr * win))
            tot = float(mag.sum()) + 1e-9
            centroid = float((mag * freqs).sum() / tot)
            norm = mag / tot
            flux = float(np.abs(norm - prev).sum()) if prev is not None else 0.0
            prev = norm
            self.audio_bins.setdefault(int(seg.t0 + i / SR), []).append((db, centroid, flux))

    def _motion_features(self, seg):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", seg.path, "-an", "-vf",
                              f"fps={self.MOTION_FPS},scale={self.MW}:{self.MH},format=gray",
                              "-f", "rawvideo", "-"], capture_output=True).stdout
        n = len(raw) // (self.MW * self.MH)
        if n < 2:
            return
        frames = np.frombuffer(raw[:n * self.MW * self.MH], np.uint8).reshape(n, self.MH, self.MW).astype(np.int16)
        face_mask = None
        lay = self.layout
        if lay and lay.get("mode") == "split":
            sx, sy = self.MW / lay["W"], self.MH / lay["H"]
            cx, cy, cw, ch = lay["cam"]
            face_mask = np.zeros((self.MH, self.MW), bool)
            face_mask[int(cy * sy):int((cy + ch) * sy) + 1, int(cx * sx):int((cx + cw) * sx) + 1] = True
        diffs = np.abs(np.diff(frames, axis=0)).astype(np.float32)
        for k in range(diffs.shape[0]):
            d = diffs[k]
            if face_mask is not None and face_mask.any():
                game, face = float(d[~face_mask].mean()), float(d[face_mask].mean())
            elif lay and lay.get("mode") == "track":
                game, face = float(d.mean()), float(d.mean())
            else:
                game, face = float(d.mean()), 0.0
            self.motion_bins.setdefault(int(seg.t0 + (k + 1) / self.MOTION_FPS), []).append((game, face))

    def add_segment(self, idx, path, wall_end):
        dur = probe_duration(path)
        if dur <= 0.2:
            return None
        seg = Segment(idx, str(Path(path).resolve()), self.end, dur, wall_end - dur)
        audio = decode_audio(path)
        self._audio_features(seg, audio)
        self._motion_features(seg)
        self._emotion_features(seg)
        if self.whisper is not None and len(audio) > SR:
            try:
                segs, _ = self.whisper.transcribe(audio, word_timestamps=True, vad_filter=True, beam_size=1,
                                                  language=self.language, condition_on_previous_text=False)
                for s in segs:
                    for w in (s.words or []):
                        txt = w.word.strip()
                        if txt:
                            self.words.append(Word(seg.t0 + w.start, seg.t0 + w.end, txt))
            except Exception as e:
                log(f"whisper failed on segment {idx}: {e}")
        self.segs.append(seg)
        self.end = seg.t0 + dur
        return seg

    def wall_to_t(self, wall):
        for s in reversed(self.segs):
            if s.wall0 <= wall < s.wall0 + s.dur + 1:
                return s.t0 + (wall - s.wall0)
        return None

    # ---------------- features ----------------
    def _robust_z(self, name, arr, window=300, cache=True):
        """z-score of each bin vs. the trailing `window` seconds (median/MAD) - adapts to each streamer."""
        c = self._z.setdefault(name, []) if cache else []
        stable = max(0, len(arr) - 3)       # last few bins may still change
        del c[stable:]
        for i in range(len(c), len(arr)):
            lo = max(0, i - window)
            base = arr[lo:i] if i - lo >= 30 else arr[:max(i, 30)]
            med = float(np.median(base))
            mad = float(np.median(np.abs(base - med))) * 1.4826 + 1e-3
            c.append((arr[i] - med) / mad)
        return np.array(c[:len(arr)])

    @staticmethod
    def _smooth(x, k):
        return np.convolve(x, np.ones(k) / k, mode="same")

    def features(self, chat_msgs):
        n = int(self.end)
        if n < 5:
            return None
        loud = np.full(n, -90.0)
        bright = np.zeros(n)
        flux = np.zeros(n)
        for b in range(n):
            v = self.audio_bins.get(b)
            if v:
                a = np.array(v)
                loud[b] = np.percentile(a[:, 0], 90)
                bright[b] = a[:, 1].max()
                flux[b] = a[:, 2].mean()
        game = np.zeros(n)
        face = np.zeros(n)
        for b in range(n):
            v = self.motion_bins.get(b)
            if v:
                a = np.array(v)
                game[b] = np.median(a[:, 0])
                face[b] = a[:, 1].mean()
        emo = np.zeros(n)                 # 0..1 facial excitement
        emo_label = [""] * n
        for b in range(n):
            v = self.emo_bins.get(b)
            if v:
                a = np.array(v)
                sc = a @ EMO_WEIGHT
                k = int(np.argmax(sc))
                emo[b] = float(sc[k])
                if emo[b] > 0.45:
                    emo_label[b] = EMOTIONS[int(np.argmax(a[k] * (EMO_WEIGHT > 0)))]
        heat = np.zeros(n)
        for h in self.heat:
            heat[int(h["start_time"]):min(n, int(h["end_time"]) + 1)] = h["value"]
        if heat.any():
            heat = np.clip((heat - np.median(heat)) / (heat.max() - np.median(heat) + 1e-6), 0, 1)
        chat = np.zeros(n)
        chat_h = np.zeros(n)
        for w, hype, paid in chat_msgs:
            t = self.wall_to_t(w)
            if t is not None:
                b = int(t - self.chat_lag)
                if 0 <= b < n:
                    chat[b] += 1 + (2 if paid else 0)
                    chat_h[b] += hype
        kw = np.zeros(n)
        speech = np.zeros(n)
        txt_by_bin = {}
        for w in self.words:
            b = int(w.t0)
            if 0 <= b < n:
                speech[b] += 1
                txt_by_bin.setdefault(b, []).append(w.text.lower())
        for b, toks in txt_by_bin.items():
            joined = " " + " ".join(re.sub(r"[^\w\s']", "", t) for t in toks) + " "
            hits = sum(1 for h in HYPE_WORDS if f" {h} " in joined)
            if any(t.endswith("!") for t in toks):
                hits += 0.5
            kw[b] = hits

        z = {}
        z["loud"] = self._robust_z("loud", loud)
        z["bright"] = self._robust_z("bright", bright)
        z["flux"] = self._robust_z("flux", flux)
        z["game"] = self._robust_z("game", self._smooth(game, 3))
        z["face"] = self._robust_z("face", self._smooth(face, 3)) if face.any() else np.zeros(n)
        z["speech"] = self._robust_z("speech", self._smooth(speech, 4))
        emo_s = self._smooth(emo, 3)
        z["emo"] = self._robust_z("emo", emo_s) if emo.any() else np.zeros(n)
        # chat arrives late, so its z-scores are recomputed every pass
        z["chat"] = self._robust_z("chat", self._smooth(chat, 5), cache=False) if chat.any() else np.zeros(n)
        z["chat_h"] = self._robust_z("chat_h", self._smooth(chat_h, 5), cache=False) if chat_h.any() else np.zeros(n)

        pos = lambda x, cap=4.0: np.clip(x, 0, cap)
        scream = pos(z["loud"]) * (pos(z["bright"], 3) / 3) * (z["loud"] > 1)   # loud AND high-pitched
        # build-up -> payoff: energy now vs. the calm 8-25s before it
        lz = self._smooth(pos(z["loud"]), 3)
        calm = np.array([lz[max(0, i - 25):max(1, i - 8)].mean() for i in range(n)])
        payoff = np.clip(lz - calm, 0, 4)
        e = (1.00 * pos(z["loud"])
             + 0.60 * scream
             + 0.30 * pos(z["flux"], 3)
             + 0.90 * pos(z["chat"])
             + 0.70 * pos(z["chat_h"])
             + 0.60 * np.clip(self._smooth(kw, 3) * 3, 0, 3)
             + 0.45 * pos(z["face"], 3)       # streamer physically reacting
             + 0.30 * pos(z["game"], 3)       # explosions / chases / action on screen
             + 0.30 * pos(z["speech"], 3)     # rapid talking
             + 0.40 * payoff
             + 1.20 * np.clip(emo_s - 0.25, 0, 1) / 0.75   # surprised / laughing / scared face
             + 0.40 * pos(z["emo"], 3)                      # unusually expressive for this streamer
             + 3.00 * heat)                                 # YouTube viewers rewatched this part
        # multi-signal agreement bonus: moments where several channels fire together are the real ones
        agree = ((z["loud"] > 1.5).astype(float) + (z["chat"] > 1.5) + (z["face"] > 1.5)
                 + (z["game"] > 1.5) + (kw > 0) + (z["chat_h"] > 1.5) + (emo_s > 0.5))
        e = e * (1 + 0.15 * np.clip(agree - 1, 0, 5))
        e_s = self._smooth(e, 5)
        return {"E": e_s, "raw": e, "z": z, "kw": kw, "chat": chat, "agree": agree, "payoff": payoff,
                "emo": emo_s, "emo_label": emo_label, "heat": heat}

    def words_between(self, a, b):
        return [w for w in self.words if w.t1 > a and w.t0 < b]


# --------------------------------------------------------------------------------------
# Clip selection
# --------------------------------------------------------------------------------------

@dataclass
class Clip:
    start: float
    end: float
    peak: float
    score: float
    reasons: list = field(default_factory=list)
    title: str = ""
    hook: str = ""
    ai_score: int | None = None
    emotion: str = ""
    file: str = ""
    replaces: object = None   # earlier Clip this one supersedes (two moments merged into one clip)


def snap_edges(words, start, end, peak, cmin, cmax):
    """Move clip edges into pauses in speech so we never start/stop mid-word."""
    ws = sorted(words, key=lambda w: w.t0)
    starts, ends = [], []
    for a, b in zip(ws, ws[1:]):
        if b.t0 - a.t1 >= 0.35:
            starts.append(b.t0 - 0.12)
            ends.append(a.t1 + 0.3)
    if ws:
        starts.append(ws[0].t0 - 0.12)
        ends.append(ws[-1].t1 + 0.3)
    c = [s for s in starts if start - 4 <= s <= min(start + 4, peak - 3)]
    if c:
        start = min(c, key=lambda s: abs(s - start))
    c = [e for e in ends if max(end - 3, peak + 2) <= e <= end + 5]
    if c:
        end = min(c, key=lambda e: abs(e - end))
    # never end while someone is mid-word
    for w in ws:
        if w.t0 < end < w.t1:
            end = w.t1 + 0.2
    if end - start < cmin:
        end = start + cmin
    if end - start > cmax:
        # keep the payoff: trim the setup first
        start = max(start, end - cmax)
    return max(0.0, start), end


def explain(feats, i):
    z = feats["z"]
    sl = slice(max(0, i - 3), i + 4)
    r = []
    if feats["emo_label"] and any(feats["emo_label"][sl]):
        labs = [l for l in feats["emo_label"][sl] if l]
        r.append(f"{max(set(labs), key=labs.count)} face")
    if feats["heat"][sl].max() > 0.5:
        r.append("YouTube most-replayed")
    if z["loud"][sl].max() > 2:
        r.append("scream" if (z["bright"][sl].max() > 1.5) else "volume spike")
    if z["chat"][sl].max() > 1.8:
        r.append("chat explodes")
    if z["chat_h"][sl].max() > 1.8:
        r.append("chat hyped (LOL/W/emotes)")
    if feats["kw"][sl].sum() > 0:
        r.append("hype words")
    if z["face"][sl].max() > 2:
        r.append("big physical reaction")
    if z["game"][sl].max() > 2:
        r.append("on-screen action")
    if feats["payoff"][sl].max() > 1.5:
        r.append("calm-to-chaos payoff")
    return r


def find_clips(tl, feats, frontier, avail, last_end, args, percentile, prev=None):
    E = feats["E"]
    if avail - frontier < 3:
        return [], frontier
    thr = args.threshold
    if avail > 90:
        thr = max(thr, float(np.percentile(E[:avail], percentile)))
    clips = []
    i = max(frontier, 6)
    while i < avail:
        lo, hi = max(0, i - 8), min(len(E), i + 9)
        is_peak = E[i] >= thr and E[i] == E[lo:hi].max()
        gap_ok = i - last_end >= args.min_gap
        # a second moment right after the previous clip: regenerate ONE clip covering both
        merge = (is_peak and not gap_ok and prev is not None and i > prev.end
                 and (i + args.aftermath + 2) - prev.start <= args.max_len)
        if is_peak and (gap_ok or merge):
            # how long does the moment last? (energy stays above 55% of the peak)
            a = i
            while a > max(0, i - 40) and E[a - 1] >= 0.55 * E[i]:
                a -= 1
            b = i
            while b < len(E) - 1 and b < i + 40 and E[b + 1] >= 0.55 * E[i]:
                b += 1
            start = a - args.setup            # context before the moment
            end = b + args.aftermath          # let the reaction land
            if end - start < args.min_len:
                pad = args.min_len - (end - start)
                start -= pad * 0.6
                end += pad * 0.4
            if merge:
                start = prev.start
                end = min(end, prev.start + args.max_len)
            else:
                start = max(start, last_end + 0.5, 0)
            end = min(end, tl.end)
            words = tl.words_between(start - 5, end + 6)
            s, e = snap_edges(words, start, end, float(i), args.min_len, args.max_len)
            s = prev.start if merge else max(s, last_end + 0.5)
            e = min(e, tl.end)
            if e - s >= args.min_len * 0.75:
                bins = slice(int(s), max(int(s) + 1, int(e)))
                score = float(E[bins].mean() + E[i])
                emo = ""
                labs = [l for l in feats["emo_label"][max(0, i - 4):i + 5] if l]
                if labs:
                    emo = max(set(labs), key=labs.count)
                c = Clip(s, e, float(i), score, explain(feats, i), emotion=emo)
                if merge:
                    if E[int(prev.peak)] > E[i]:
                        c.peak = prev.peak
                    c.reasons = list(dict.fromkeys(prev.reasons + c.reasons + ["two moments merged"]))
                    c.emotion = c.emotion or prev.emotion
                    c.replaces = prev
                    if prev in clips:          # not submitted yet: just swap it
                        clips.remove(prev)
                        c.replaces = prev.replaces
                clips.append(c)
                prev = c
                last_end = e
                i = int(e) + 1
                continue
        i += 1
    return clips, avail


# --------------------------------------------------------------------------------------
# AI review (optional)
# --------------------------------------------------------------------------------------

def ai_review(client, model, tl, clip, args):
    from pydantic import BaseModel

    class Verdict(BaseModel):
        viral_score: int
        title: str
        hook: str
        start: float
        end: float
        reason: str

    c0, c1 = max(0.0, clip.start - 15), min(tl.end, clip.end + 15)
    lines, cur, cur_t = [], [], None
    for w in tl.words_between(c0, c1):
        if cur_t is None:
            cur_t = w.t0
        cur.append(w.text)
        if w.text.endswith((".", "?", "!")) or len(cur) >= 14:
            lines.append(f"[{cur_t - c0:6.1f}s] {' '.join(cur)}")
            cur, cur_t = [], None
    if cur:
        lines.append(f"[{cur_t - c0:6.1f}s] {' '.join(cur)}")
    transcript = "\n".join(lines) or "(no speech detected - reaction may be non-verbal)"
    prompt = (
        "You are a short-form video editor picking moments from a live stream for TikTok/Shorts/Reels.\n"
        f"Below is a transcript of a {c1 - c0:.0f}s window (timestamps relative to window start). "
        f"Our detector flagged a peak at {clip.peak - c0:.1f}s because of: {', '.join(clip.reasons) or 'overall energy'}.\n"
        f"Proposed clip: {clip.start - c0:.1f}s to {clip.end - c0:.1f}s.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Rate how likely this works as a standalone viral short (1-10; reactions, jokes, rage, surprise, "
        "wholesome moments and clear payoffs score high; dead air, menus, mid-sentence fragments score low). "
        f"Choose start/end (seconds, window-relative) so the clip opens on a hook and ends right after the payoff, "
        f"{args.min_len:.0f}-{args.max_len:.0f}s long. "
        "Write a punchy title (max 60 chars) and a hook overlay (max 6 words) in the stream's language/style."
    )
    kw = dict(model=model, max_tokens=8000, output_config={"effort": "low"},
              messages=[{"role": "user", "content": prompt}], output_format=Verdict)
    try:
        resp = client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    except TypeError:
        resp = client.messages.parse(**kw)
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    v = resp.parsed_output
    s, e = c0 + v.start, c0 + v.end
    if args.min_len * 0.8 <= e - s <= args.max_len * 1.2 and c0 <= s < e <= c1:
        clip.start, clip.end = s, e
    clip.ai_score, clip.title, clip.hook = v.viral_score, v.title, v.hook
    clip.reasons.append("AI: " + v.reason)
    return v


# --------------------------------------------------------------------------------------
# Layout detection (facecam)
# --------------------------------------------------------------------------------------

_face_det = None


def face_detector():
    global _face_det
    if _face_det is None:
        import cv2
        _face_det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.7, 0.3, 50)
    return _face_det


def sample_frames(paths, n=10):
    import cv2
    frames = []
    per = max(1, math.ceil(n / len(paths)))
    for p in paths:
        dur = probe_duration(p) or 10
        for k in range(per):
            png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur * (k + 0.5) / per:.2f}", "-i", str(p),
                                  "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                                 capture_output=True).stdout
            if png:
                fr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if fr is not None:
                    frames.append(fr)
    return frames


def detect_layout(frames):
    """Returns dict(mode='split'|'track'|'blur', W, H, face=(x,y,w,h), cam=(x,y,w,h)) in source pixels."""
    import cv2
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    scale = 960.0 / W
    sw, sh = int(W * scale), int(H * scale)
    det = face_detector()
    det.setInputSize((sw, sh))
    faces, grays = [], []
    for fr in frames:
        small = cv2.resize(fr, (sw, sh))
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
        _, f = det.detect(small)
        if f is not None and len(f):
            best = max(f, key=lambda r: r[2] * r[3] * r[14])
            faces.append(best[:4])
    if len(faces) < max(2, len(frames) // 3):
        return {"mode": "blur", "W": W, "H": H}
    fx, fy, fw, fh = np.median(np.array(faces), axis=0)
    spread = np.std(np.array(faces)[:, 0]) / max(fw, 1)
    area = fw * fh / (sw * sh)
    if area > 0.04:
        # big face = IRL / just-chatting / full-cam: follow the face with a 9:16 crop
        return {"mode": "track", "W": W, "H": H, "face": tuple(v / scale for v in (fx, fy, fw, fh))}

    # small face = facecam overlay. Find its box from edges that stay put across frames.
    ex = np.mean([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) for g in grays], axis=0)
    ey = np.mean([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) for g in grays], axis=0)

    def strongest(profile, lo, hi):
        lo, hi = int(max(0, lo)), int(min(len(profile), hi))
        if hi - lo < 3:
            return None
        seg = profile[lo:hi]
        k = int(np.argmax(seg))
        return lo + k if seg[k] > 3.0 * (np.median(seg) + 1e-3) and seg[k] > 12 else None

    def first_strong(profile, lo, hi):
        """nearest clear edge below the face (the cam border comes before any chat/overlay text)"""
        lo, hi = int(max(0, lo)), int(min(len(profile), hi))
        if hi - lo < 3:
            return None
        seg = profile[lo:hi]
        thr = max(2.5 * (np.median(seg) + 1e-3), 8)
        idx = np.nonzero(seg > thr)[0]
        return lo + int(idx[0]) if len(idx) else None

    r0, r1 = int(fy), int(min(sh, fy + fh))
    c0, c1 = int(fx), int(min(sw, fx + fw))
    colp = ex[r0:r1].mean(axis=0)
    rowp = ey[:, c0:c1].mean(axis=1)
    near = 1.3
    left = 0 if fx < fw * near else strongest(colp, fx - 2.5 * fw, fx - 0.1 * fw)
    right = sw if sw - (fx + fw) < fw * near else strongest(colp, fx + 1.1 * fw, fx + fw + 2.5 * fw)
    top = 0 if fy < fh * near else strongest(rowp, fy - 2.0 * fh, fy - 0.1 * fh)
    bottom = sh if sh - (fy + fh) < fh * 0.4 else first_strong(rowp, fy + 0.95 * fh, fy + fh + 1.6 * fh)
    left = left if left is not None else max(0, fx - 1.0 * fw)
    right = right if right is not None else min(sw, fx + 2.0 * fw)
    top = top if top is not None else max(0, fy - 0.7 * fh)
    bottom = bottom if bottom is not None else min(sh, fy + 1.3 * fh)
    cam = tuple(v / scale for v in (left, top, right - left, bottom - top))
    return {"mode": "split", "W": W, "H": H, "face": tuple(v / scale for v in (fx, fy, fw, fh)),
            "cam": cam, "jitter": float(spread)}


def build_filter(layout, ass_name, src="[0:v]", zoom=None, dur=None, progress=False):
    W, H = layout["W"], layout["H"]
    mode = layout["mode"]
    if mode == "split":
        cx, cy, cw, ch = layout["cam"]
        fx, fy, fw, fh = layout["face"]
        cam_aspect = cw / max(ch, 1)
        bot_h = even(min(1000, max(640, OUT_W / max(cam_aspect, 0.5))))
        top_h = OUT_H - bot_h
        # facecam crop with the bottom panel's aspect, centred on the face, inside the cam box
        asp = OUT_W / bot_h
        w = min(cw, ch * asp)
        h = w / asp
        x = min(max(fx + fw / 2 - w / 2, cx), cx + cw - w)
        y = min(max(fy + fh * 0.6 - h / 2, cy), cy + ch - h)
        # gameplay crop for the top panel, steering clear of the facecam when possible
        gasp = OUT_W / top_h
        gh = H
        gw = min(W, gh * gasp)
        if gw >= W:
            gw, gh = W, W / gasp
        gx = (W - gw) / 2
        if cx + cw / 2 > W / 2 and gx + gw > cx and cx - gw >= W * 0.08:
            gx = cx - gw
        elif cx + cw / 2 <= W / 2 and gx < cx + cw and cx + cw + gw <= W * 0.92:
            gx = cx + cw
        gy = (H - gh) / 2
        fc = (f"{src}split=2[g][c];"
              f"[g]crop={even(gw)}:{even(gh)}:{even(gx)}:{even(gy)},scale={OUT_W}:{top_h}:flags=lanczos[top];"
              f"[c]crop={even(w)}:{even(h)}:{even(x)}:{even(y)},scale={OUT_W}:{bot_h}:flags=lanczos[bot];"
              f"[top][bot]vstack=inputs=2,setsar=1,fps={OUT_FPS},format=yuv420p")
        cap_y = top_h
    elif mode == "track":
        fx, fy, fw, fh = layout["face"]
        cw = H * 9 / 16
        x = min(max(fx + fw / 2 - cw / 2, 0), W - cw)
        fc = f"{src}crop={even(cw)}:{even(H)}:{even(x)}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,fps={OUT_FPS},format=yuv420p"
        cap_y = int(OUT_H * 0.74)
    else:
        fg_h = even(OUT_W * 3 / 4)
        fc = (f"{src}split=2[a][b];"
              f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
              f"gblur=sigma=40,eq=brightness=-0.12[bg];"
              f"[b]crop=ih*4/3:ih,scale={OUT_W}:{fg_h}:flags=lanczos[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={OUT_FPS},format=yuv420p")
        cap_y = int(OUT_H / 2 + fg_h / 2 + 140)
    if zoom:  # punch-in zoom at the big moments
        fc += (f",scale=w='2*trunc({OUT_W // 2}*({zoom}))':h='2*trunc({OUT_H // 2}*({zoom}))':eval=frame,"
               f"crop={OUT_W}:{OUT_H}")
    if progress and dur:  # retention progress bar along the top edge
        fc += (f"[lay];color=c=0xFFD400:s={OUT_W}x14:r={OUT_FPS}[bar];"
               f"[lay][bar]overlay=x='-w+w*t/{dur:.3f}':y=0:shortest=1")
    if ass_name:
        fc += f",ass={ass_name}"
    return fc + "[v]", cap_y


# --------------------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------------------

def ass_time(t):
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_ass(path, words, clip_start, clip_dur, cap_y, hook, font):
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Hook,{font},64,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(3.5, clip_dur))},Hook,,0,0,0,,"
                  f"{{\\fad(150,250)}}{ass_escape(hook)}")
    ws = [Word(w.t0 - clip_start, w.t1 - clip_start, w.text.strip()) for w in words]
    ws = sorted((w for w in ws if w.t1 > 0 and w.t0 < clip_dur and w.text.strip(" ,.")), key=lambda w: w.t0)
    # clean timings: strictly increasing starts, no overlaps, sane word lengths
    for k, w in enumerate(ws):
        if k and w.t0 < ws[k - 1].t0 + 0.06:
            w.t0 = ws[k - 1].t0 + 0.06
        w.t1 = min(max(w.t1, w.t0 + 0.12), w.t0 + 1.2)
    for k in range(len(ws) - 1):
        ws[k].t1 = min(ws[k].t1, ws[k + 1].t0)
    # phrases of up to 3 words / 16 letters, broken at pauses and punctuation
    chunks, cur = [], []
    for w in ws:
        if cur and (len(cur) >= 3 or w.t0 - cur[-1].t1 > 0.5 or cur[-1].text.endswith((".", "?", "!", ","))
                    or sum(len(c.text) for c in cur) + len(w.text) > 16):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    upper = lambda t: t.upper()
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0].t0 if ci + 1 < len(chunks) else clip_dur
        c0 = ch[0].t0
        # hold the phrase until the next one starts if the gap is short (no blink); otherwise fade out
        c1 = nxt if nxt - ch[-1].t1 < 0.45 else min(ch[-1].t1 + 0.35, nxt)
        c1 = min(max(c1, c0 + 0.45), nxt, clip_dur)
        letters = sum(len(x.text) for x in ch) + len(ch) - 1
        fs = "" if letters <= 16 else f"\\fs{max(50, int(84 * 16 / letters))}"
        for k, w in enumerate(ch):
            t0 = c0 if k == 0 else w.t0
            t1 = ch[k + 1].t0 if k + 1 < len(ch) else c1
            if t1 - t0 < 0.02:
                continue
            # same text, same size, same position for the whole phrase: only the colour of one word changes
            parts = [("{\\c&H00E5FF&}" + ass_escape(upper(x.text.strip(" ,."))) + "{\\c&HFFFFFF&}") if j == k
                     else ass_escape(upper(x.text.strip(" ,."))) for j, x in enumerate(ch)]
            ev.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Cap,,0,0,0,,"
                      f"{{\\pos({OUT_W // 2},{cap_y})\\q2{fs}}}{' '.join(parts)}")
    Path(path).write_text(hdr + "\n".join(ev) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Roman (Latin-letter) captions: Hindi/Urdu speech is transcribed as Hindi and romanised.
# --------------------------------------------------------------------------------------
_DEV_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
             "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
             "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
             "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g",
             "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y"}
_DEV_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
_DEV_VOW = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o",
            "औ": "au", "ऋ": "ri", "ऑ": "o", "ऍ": "e"}
_DEV_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
              "ृ": "ri", "ॉ": "o", "ॅ": "e"}
_DEV_FIX = {"hai": "hai", "nahin": "nahi", "men": "mein", "main": "main", "han": "haan", "yah": "yeh",
            "vah": "woh", "vo": "woh", "ham": "hum", "hamne": "humne", "kahin": "kahin", "yahan": "yahan",
            "vahan": "wahan", "sirph": "sirf", "bat": "baat", "achchha": "acha", "achchhi": "achi", "kyon": "kyun",
            "par": "par", "tha": "tha", "aur": "aur", "kya": "kya", "bhai": "bhai"}


def romanize(word):
    """Devanagari -> Roman Hindi/Urdu the way people type it. Returns '' if anything non-Latin is left."""
    if word.isascii():
        return word
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        if ch in _DEV_CONS:
            c = _DEV_CONS[ch]
            if nxt == "़":
                c = _DEV_NUKTA.get(ch, c)
                i += 1
                nxt = word[i + 1] if i + 1 < n else ""
            out.append(c)
            if nxt in _DEV_MATRA:
                out.append(_DEV_MATRA[nxt])
                i += 1
            elif nxt == "्":
                i += 1
            else:
                rest = word[i + 1:]
                at_end = not any(r in _DEV_CONS or r in _DEV_VOW for r in rest)
                # inherent 'a' is silent at the end of a word, and before a final consonant+vowel
                medial = (len(rest) >= 2 and rest[0] in _DEV_CONS and rest[1] in _DEV_MATRA and len(out) >= 3
                          and not any(r in _DEV_CONS for r in rest[2:]))
                if not at_end and not medial:
                    out.append("a")
        elif ch in _DEV_VOW:
            out.append(_DEV_VOW[ch])
        elif ch in _DEV_MATRA:
            out.append(_DEV_MATRA[ch])
        elif ch in "ंँ":
            out.append("n")
        elif ch == "ः":
            out.append("h")
        elif ch in "।॥":
            out.append(".")
        elif ch.isascii():
            out.append(ch)
        elif ch in "़्‍‌":
            pass
        else:
            return ""
        i += 1
    r = "".join(out)
    core = r.strip(".,!?")
    return r.replace(core, _DEV_FIX.get(core, core)) if core else r


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

class Renderer:
    def __init__(self, outdir, args):
        self.outdir = Path(outdir)
        self.args = args
        self.enc = pick_encoder()
        self.last_layout = None
        self.count = len(list(self.outdir.glob("*.mp4")))

    def plan_edit(self, clip, tl, feats):
        """Source intervals in output order. Removes dead air (jump cuts) and prepends a teaser of the peak."""
        a = self.args
        cuts = [(clip.start, clip.end)]
        if not a.no_jump_cuts and feats is not None:
            loudz = feats["z"]["loud"]
            ws = sorted(tl.words_between(clip.start, clip.end), key=lambda w: w.t0)
            keep, cur = [], clip.start
            if ws and ws[0].t0 - clip.start > a.dead_air + 0.5:      # silent intro
                g1 = ws[0].t0 - 0.25
                if loudz[int(clip.start):int(g1) + 1].max(initial=0) < 1.0 and not (clip.start < clip.peak < g1 + 1):
                    cur = g1
            for w1, w2 in zip(ws, ws[1:]):
                g0, g1 = w1.t1 + 0.18, w2.t0 - 0.12
                if g1 - g0 >= a.dead_air and not (g0 - 2 < clip.peak < g1 + 2) and g0 > cur:
                    seg_l = loudz[int(g0):int(g1) + 1]
                    if len(seg_l) and seg_l.max() < 1.0:   # truly quiet, not a non-verbal reaction
                        keep.append((cur, g0))
                        cur = g1
            keep.append((cur, clip.end))
            cuts = [(x, y) for x, y in keep if y - x > 0.3] or cuts
        teaser = 0.0
        if not a.no_teaser and clip.peak - cuts[0][0] > 5:
            t0, t1 = clip.peak - 0.5, min(clip.end, clip.peak + 1.5)
            cuts = [(t0, t1)] + cuts
            teaser = t1 - t0
        return cuts, teaser

    # ---------------- captions (2nd pass, timed on the finished clip's own audio) ----------------
    def _caption_model(self):
        if getattr(self, "_cap_model", None) is None:
            from faster_whisper import WhisperModel
            log(f"loading caption model '{self.args.caption_model}' (first time downloads ~1.5 GB)...")
            self._cap_model = WhisperModel(self.args.caption_model, device="cpu", compute_type="int8")
        return self._cap_model

    GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"]

    def _gemini(self):
        if getattr(self, "_gem", None) is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                self._gem = False
            else:
                from google import genai
                from google.genai import types
                self._gem = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000))
                self._gem_ok = {}
        return self._gem

    def gemini_fix(self, audio, rough):
        """Gemini listens to the clip and corrects each word (by index, so Whisper's timings stay attached).
        Returns list of corrected words (same length, '' = drop) or None when it isn't sure."""
        import io
        import wave
        from google.genai import types
        from pydantic import BaseModel

        class Item(BaseModel):
            i: int
            w: str

        class Out(BaseModel):
            confident: bool
            title: str
            words: list[Item]

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        prompt = (
            "This audio is from a live streamer. They may speak English, Hindi/Urdu, or Hinglish (both mixed in one "
            "sentence). Below is a rough automatic word list, one word per index, in time order.\n"
            "Listen to the audio and return the corrected word for EVERY index (same index numbers), written the way "
            "people type casually: English words in normal English spelling (brightness, unlock, Vice City, road, "
            "mission), Hindi/Urdu words in Roman letters (nahi, hai, kya, bhai, mere khyal se). Latin letters only - "
            "never Devanagari or Urdu script. Do not translate. Do not add, merge or reorder indexes. If an index was "
            "not actually spoken, return an empty string for it.\n"
            "Set confident=false if the speech is unclear, drowned out by game audio or music, or you cannot verify "
            "most of the words - wrong captions are worse than none.\n"
            "Also give a short catchy title for this clip (max 6 words, same language style).\n\n"
            + "\n".join(f"{i}: {w}" for i, w in enumerate(rough)))
        client = self._gemini()
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav")
        cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Out, temperature=0)
        now = time.time()
        for model in self.GEMINI_MODELS:
            if self._gem_ok.get(model, 0) > now:      # recently overloaded: skip for a few minutes
                continue
            try:
                o = client.models.generate_content(model=model, contents=[part, prompt], config=cfg).parsed
            except Exception as e:
                self._gem_ok[model] = now + 90
                log(f"gemini {model} unavailable ({e.__class__.__name__}); trying next")
                continue
            if o is None or not o.confident:
                return None, "", model
            fixed = [None] * len(rough)
            for it in o.words:
                if 0 <= it.i < len(rough) and fixed[it.i] is None:
                    fixed[it.i] = it.w.strip()
            if sum(f is None for f in fixed) > 0.1 * len(rough):
                return None, "", model
            return ["" if f is None else f for f in fixed], o.title.strip(), model
        return None, "", "none"

    def make_captions(self, audio, work, cap_y, dur, clip):
        """Captions are only burned in when BOTH Whisper (timing) and Gemini (listening check) are confident.
        Anything doubtful -> clean clip with no captions."""
        a = self.args
        words, info = [], "off"
        if a.captions and audio is not None and len(audio) > SR:
            if not self._gemini():
                return None, "off (no GEMINI_API_KEY)"

            def run(lang):
                segs, det = self._caption_model().transcribe(
                    audio, word_timestamps=True, vad_filter=True, beam_size=3, language=lang,
                    condition_on_previous_text=False)
                return [w for sg in segs for w in (sg.words or []) if w.word.strip()], det.language
            try:
                raw, det_lang = run(a.caption_lang or getattr(self, "_cap_lang", None))
                # Urdu comes out in Arabic script, which can't be romanised (no vowels): re-listen as Hindi.
                if any(re.search(r"[؀-ۿ]", w.word) for w in raw):
                    self._cap_lang = "hi"
                    raw, det_lang = run("hi")
            except Exception as e:
                log(f"caption transcription failed: {e}")
                raw, det_lang = [], "?"
            conf = float(np.mean([w.probability for w in raw])) if raw else 0.0
            if not raw:
                info = "no speech"
            elif conf < a.caption_conf:
                info = f"skipped (timing confidence {conf:.2f} < {a.caption_conf})"
            else:
                rough = [romanize(w.word.strip()) or "?" for w in raw]
                try:
                    fixed, title, model = self.gemini_fix(audio, rough)
                except Exception as e:
                    log(f"gemini error: {e.__class__.__name__}: {str(e)[:120]}")
                    fixed, title, model = None, "", "error"
                if fixed is None:
                    info = f"skipped (Gemini not confident / unavailable: {model})"
                else:
                    pairs = [(w, f) for w, f in zip(raw, fixed) if f]
                    if (len(pairs) < 0.7 * len(raw) or not all(f.isascii() for _, f in pairs)):
                        info = "skipped (too many unverified words)"
                    else:
                        words = [Word(w.start, w.end, f) for w, f in pairs]
                        info = f"{model} verified, {len(words)} words"
                        if title and not clip.title and title.isascii():
                            clip.title = title
        if not words and not clip.hook:
            return None, info
        write_ass(work / "subs.ass", words, 0.0, dur, cap_y, clip.hook, a.font)
        return "subs.ass", info

    # ---------------- background worker so rendering never blocks the live analysis ----------------
    def start_worker(self, on_done):
        import queue
        self.q = queue.Queue()
        self.on_done = on_done

        def loop():
            while True:
                item = self.q.get()
                if item is None:
                    break
                clip, tl, feats = item
                try:
                    out = self.render(clip, tl, feats)
                    self.on_done(clip, out)
                except Exception as e:
                    log(f"render error: {e.__class__.__name__}: {e}")
                finally:
                    self.q.task_done()
        self.worker = threading.Thread(target=loop, daemon=True)
        self.worker.start()

    def submit(self, clip, tl, feats):
        self.q.put((clip, tl, feats))

    def finish(self):
        self.q.join()
        self.q.put(None)

    @staticmethod
    def map_time(t, cuts, first):
        off = sum(b - a for a, b in cuts[:first])
        for a, b in cuts[first:]:
            if a <= t <= b:
                return off + (t - a)
            off += b - a
        return None

    def render(self, clip, tl, feats=None):
        segs = [s for s in tl.segs if s.t0 < clip.end + 0.5 and s.t0 + s.dur > clip.start - 1]
        segs = [s for s in segs if Path(s.path).exists()]
        if not segs:
            return None
        self.count += 1
        name = f"{self.count:03d}_{slug(clip.title or (clip.emotion + '-' if clip.emotion else '') + time.strftime('%H%M%S'))}"
        work = self.outdir / ".work" / name
        work.mkdir(parents=True, exist_ok=True)
        (work / "list.txt").write_text("".join(f"file '{s.path}'\n" for s in segs))

        layout = None
        if self.args.layout != "blur":
            layout = detect_layout(sample_frames([s.path for s in segs], n=10))
            known = self.last_layout or tl.layout
            if layout and self.args.layout in ("split", "track") and layout["mode"] != self.args.layout:
                layout = known if known and known["mode"] == self.args.layout else layout
            if (layout is None or layout["mode"] == "blur") and known and known["mode"] == "split":
                layout = known  # streamer looked away; keep the known facecam box
        if layout is None:
            wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=width,height", "-of", "csv=p=0", segs[0].path],
                                capture_output=True, text=True).stdout.strip().split(",")
            layout = {"mode": "blur", "W": int(wh[0]) if wh[0] else 1920, "H": int(wh[1]) if len(wh) > 1 else 1080}
        if layout["mode"] != "blur":
            self.last_layout = layout

        has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                         "stream=index", "-of", "csv=p=0", segs[0].path],
                                        capture_output=True, text=True).stdout.strip())
        cuts, teaser = self.plan_edit(clip, tl, feats)
        first_main = 1 if teaser else 0
        out_dur = sum(b - a for a, b in cuts)

        zoom = None
        if not self.args.no_zoom:
            wins = []
            if teaser:
                wins.append((0.0, teaser))
            tp = self.map_time(clip.peak, cuts, first_main)
            if tp is not None:
                wins.append((tp - 0.15, tp + 1.7))
            if wins:
                amt = 0.10 + 0.04 * min(1.0, clip.score / 6)
                terms = [f"min(1,max(0,(t-{a0:.2f})/0.18))*min(1,max(0,({b0:.2f}-t)/0.35))" for a0, b0 in wins]
                expr = terms[0]
                for t_ in terms[1:]:
                    expr = f"max({expr},{t_})"
                zoom = f"1+{amt:.3f}*{expr}"

        global OUT_FPS
        fr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=r_frame_rate", "-of", "csv=p=0", segs[0].path],
                            capture_output=True, text=True).stdout.strip().splitlines()
        fr = (fr[0] if fr else "30/1").strip().split("/")
        try:
            src_fps = float(fr[0]) / float(fr[1] if len(fr) > 1 and fr[1] else 1)
        except ValueError:
            src_fps = 30
        OUT_FPS = 60 if src_fps >= 50 else 30

        base = segs[0].t0
        n = len(cuts)
        vparts, aparts = [], []
        vl = "".join(f"[s{k}]" for k in range(n))
        vparts.append(f"[0:v]split={n}{vl}" if n > 1 else "[0:v]null[s0]")
        # aresample=async fills any gap in the recorded audio with silence instead of sliding it out of sync
        al = "".join(f"[r{k}]" for k in range(n))
        aparts.append(f"[0:a]aresample=async=1:first_pts=0," + (f"asplit={n}{al}" if n > 1 else "anull[r0]"))
        for k, (a0, b0) in enumerate(cuts):
            ra, rb = max(0.0, a0 - base), b0 - base
            ln = rb - ra
            vparts.append(f"[s{k}]trim=start={ra:.3f}:end={rb:.3f},setpts=PTS-STARTPTS[tv{k}]")
            aparts.append(f"[r{k}]atrim=start={ra:.3f}:end={rb:.3f},asetpts=PTS-STARTPTS,"
                          f"afade=t=in:d=0.03,afade=t=out:st={max(0, ln - 0.04):.3f}:d=0.04[ta{k}]")
        vparts.append("".join(f"[tv{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[cv]")
        aparts.append("".join(f"[ta{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[ca]")

        _, cap_y = build_filter(layout, None)
        ass, cap_info = None, "off"
        if self.args.captions or clip.hook:
            audio = None
            if has_audio and self.args.captions:
                (work / "afilter.txt").write_text(";".join(aparts))
                raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", "list.txt",
                                      "-filter_complex_script", "afilter.txt", "-map", "[ca]", "-ac", "1",
                                      "-ar", str(SR), "-f", "f32le", "-"], cwd=work, capture_output=True).stdout
                audio = np.frombuffer(raw, dtype=np.float32).copy()
            ass, cap_info = self.make_captions(audio, work, cap_y, out_dur, clip)

        lay_fc, _ = build_filter(layout, ass, src="[cv]", zoom=zoom, dur=out_dur, progress=not self.args.no_progress)
        parts = vparts + [lay_fc]
        if has_audio:
            parts += aparts + ["[ca]loudnorm=I=-14:TP=-1.5:LRA=11[ao]"]
        (work / "filter.txt").write_text(";".join(parts))

        out = self.outdir / f"{name}.mp4"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", "list.txt", "-filter_complex_script", "filter.txt", "-map", "[v]"]
        if has_audio:
            cmd += ["-map", "[ao]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        cmd += [*self.enc, "-movflags", "+faststart", str(out.resolve())]
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"render failed: {r.stderr.strip()[-500:]}")
            return None
        clip.file = out.name
        old = clip.replaces
        clip.replaces = old.file if old is not None else ""
        if old is not None and old.file:   # regenerated as one longer clip -> drop the overlapping older one
            for ext in (".mp4", ".json"):
                (self.outdir / old.file).with_suffix(ext).unlink(missing_ok=True)
            log(f"merged with previous moment -> replaced {old.file}")
        meta = {**asdict(clip), "layout": layout["mode"], "duration": round(out_dur, 2),
                "captions": cap_info, "fps": OUT_FPS,
                "source_span": [round(clip.start, 2), round(clip.end, 2)], "jump_cuts": n - 1 - first_main,
                "teaser": round(teaser, 2),
                "transcript": " ".join(w.text for w in tl.words_between(clip.start, clip.end)),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.outdir / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        with open(self.outdir / "clips.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False, default=str) + "\n")
        return out


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Record a live stream and keep cutting vertical viral clips.")
    ap.add_argument("source", help="live stream URL (YouTube/Twitch/Kick/...), VOD URL, or local video file")
    ap.add_argument("--out", default="clips", help="output folder for clips")
    ap.add_argument("--min-len", type=float, default=12)
    ap.add_argument("--max-len", type=float, default=45)
    ap.add_argument("--min-gap", type=float, default=12, help="min seconds between clips")
    ap.add_argument("--target-per-hour", type=float, default=30,
                    help="rate controller: lowers the bar when clips come slower than this")
    ap.add_argument("--merge-off", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setup", type=float, default=7, help="seconds of context before a moment")
    ap.add_argument("--aftermath", type=float, default=4, help="seconds kept after a moment")
    ap.add_argument("--dead-air", type=float, default=0.7, help="silences longer than this get jump-cut")
    ap.add_argument("--jump-cuts", dest="no_jump_cuts", action="store_false", default=True,
                    help="cut silent gaps out of clips (off by default: makes the picture jump)")
    ap.add_argument("--no-teaser", action="store_true", help="don't open with a 2s flash of the peak")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--threshold", type=float, default=1.0, help="absolute engagement floor")
    ap.add_argument("--percentile", type=float, default=85,
                    help="only peaks above this percentile of the stream so far become clips")
    ap.add_argument("--layout", choices=["auto", "split", "track", "blur"], default="auto")
    ap.add_argument("--whisper", default="base", help="faster-whisper model (tiny/base/small/medium) or 'none'")
    ap.add_argument("--language", default=None, help="force transcript language, e.g. en, hi, ur")
    ap.add_argument("--captions", action="store_true",
                    help="burn captions - only when the transcript is confident, otherwise the clip stays clean")
    ap.add_argument("--caption-conf", type=float, default=0.65, help="min Whisper timing confidence before Gemini checks the words")
    ap.add_argument("--caption-model", default="large-v3-turbo",
                    help="Whisper model used only for the final captions (accuracy matters here)")
    ap.add_argument("--caption-lang", default=None, help="force caption language (e.g. en, hi, ur)")
    ap.add_argument("--caption-min-prob", type=float, default=0.5, help="hide words below this confidence")
    ap.add_argument("--font", default="Arial Black")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--chat-lag", type=float, default=6, help="seconds chat reacts after a moment")
    ap.add_argument("--ai", action="store_true", help="have Claude rate/title/trim every candidate")
    ap.add_argument("--ai-model", default="claude-opus-5-5")
    ap.add_argument("--ai-min", type=int, default=6, help="drop candidates Claude scores below this")
    ap.add_argument("--seg-time", type=int, default=30)
    ap.add_argument("--max-height", type=int, default=4320, help="cap source resolution (default: best available)")
    ap.add_argument("--keep-minutes", type=float, default=20,
                    help="live mode: delete raw segments older than this (0 = keep everything)")
    ap.add_argument("--duration", type=float, default=0, help="live mode: stop after N minutes (0 = until stream ends)")
    args = ap.parse_args()
    import shutil
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. Install ffmpeg first (macOS: brew install ffmpeg, Ubuntu: sudo apt install ffmpeg).")
    if not FACE_MODEL.exists():
        sys.exit("Face models are missing from models/. Run ./setup.sh once.")
    envf = HERE / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = outdir / ".work" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    segdir = run_dir / "segments"
    segdir.mkdir(parents=True, exist_ok=True)

    is_file = Path(args.source).exists()
    live = False
    stop_evt = threading.Event()
    recorder = None
    chat = ChatCounter()
    heatmap = []

    if not is_file:
        info, _ = resolve_stream(args.source, args.max_height)
        live = bool(info.get("is_live"))
        log(f"source: {info.get('title')} | {info.get('uploader') or info.get('channel')} | live={live}")
        if not live:
            heatmap = info.get("heatmap") or []
            log("not live -> downloading the video first"
                + (f" (YouTube most-replayed data: {len(heatmap)} points)" if heatmap else ""))
            import yt_dlp
            dst = run_dir / "source.%(ext)s"
            with yt_dlp.YoutubeDL({"quiet": True, "outtmpl": str(dst), "merge_output_format": "mp4",
                                   "format": f"bestvideo[height<={args.max_height}]+bestaudio/best"}) as y:
                y.download([args.source])
            args.source = str(next(run_dir.glob("source.*")))
            is_file = True

    if is_file:
        log("splitting file into segments...")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", args.source, "-map", "0:v:0",
                        "-map", "0:a:0?", "-c", "copy", "-f", "segment", "-segment_time", str(args.seg_time),
                        str(segdir / "seg_%06d.ts")], check=True)
    else:
        recorder = Recorder(args.source, segdir, args.seg_time, args.max_height)
        recorder.start()
        if not args.no_chat:
            host = args.source.lower()
            if "youtube.com" in host or "youtu.be" in host:
                threading.Thread(target=youtube_chat_thread, args=(args.source, run_dir, chat, stop_evt),
                                 daemon=True).start()
                log("chat: tracking YouTube live chat")
            elif "twitch.tv" in host:
                ch = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", args.source)
                if ch:
                    threading.Thread(target=twitch_chat_thread, args=(ch.group(1), chat, stop_evt),
                                     daemon=True).start()
                    log(f"chat: tracking Twitch #{ch.group(1)}")

    whisper = None
    if args.whisper != "none":
        try:
            from faster_whisper import WhisperModel
            log(f"loading whisper '{args.whisper}'...")
            whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8")
        except Exception as e:
            log(f"whisper unavailable ({e}); continuing without transcripts/captions")

    ai_client = None
    if args.ai:
        try:
            import anthropic
            ai_client = anthropic.Anthropic()
            log(f"AI review on ({args.ai_model})")
        except Exception as e:
            log(f"AI review disabled: {e}")

    tl = Timeline(whisper, args.language, args.chat_lag)
    tl.heat = heatmap
    renderer = Renderer(outdir, args)
    made = 0

    def on_done(c, out):
        nonlocal made
        if out:
            made += 0.5 if not c.replaces else 0
            log(f"CLIP {out.name}  [{c.start:.0f}s-{c.end:.0f}s] score={c.score:.2f} "
                f"{'ai=' + str(c.ai_score) + ' ' if c.ai_score else ''}| {'; '.join(c.reasons) or 'overall energy'}")

    renderer.start_worker(on_done)
    done_idx = set()
    frontier, last_end = 0, -1e9
    prev_clip = None
    started = time.time()
    lookahead = 45  # seconds of future needed to see how a moment ends
    layout_tries = 0

    def shutdown(*_):
        stop_evt.set()
        if recorder:
            log("stopping recorder...")
            recorder.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log(f"clips -> {outdir.resolve()}")
    while True:
        files = sorted(segdir.glob("seg_*.ts"), key=lambda p: int(p.stem.split("_")[1]))
        recording = recorder is not None and not recorder.ended and not stop_evt.is_set()
        complete = files[:-1] if recording else files
        new = [p for p in complete if p.name not in done_idx]
        for p in new:
            if tl.layout is None and layout_tries < 6 or (len(done_idx) % 20 == 19):
                layout_tries += 1
                lay = detect_layout(sample_frames([p], n=8))
                if lay and lay["mode"] != "blur":
                    if tl.layout is None:
                        log(f"layout: {lay['mode']} (facecam/face found -> emotion + reaction tracking on)")
                    tl.layout = lay
            seg = tl.add_segment(int(p.stem.split("_")[1]), p, p.stat().st_mtime)
            done_idx.add(p.name)
            if seg:
                log(f"analyzed {p.name}  t={seg.t0:7.1f}s  words={len(tl.words)}  chat_msgs={len(chat.walls)}")

        finished = not recording and len(done_idx) >= len(files)
        feats = tl.features(chat.snapshot()) if (new or finished) else None
        if feats is not None:
            avail = len(feats["E"]) - (0 if finished else lookahead)
            expected = args.target_per_hour * tl.end / 3600
            perc = float(np.clip(args.percentile - 4 * (expected - made), 60, 97))
            cands, frontier = find_clips(tl, feats, frontier, max(avail, frontier), last_end, args, perc,
                                         prev_clip)
            for c in cands:
                c.end = min(c.end, tl.end)
                if ai_client:
                    try:
                        v = ai_review(ai_client, args.ai_model, tl, c, args)
                        if v and v.viral_score < args.ai_min:
                            log(f"AI skipped {c.start:.0f}-{c.end:.0f}s (score {v.viral_score}): {v.reason}")
                            last_end = max(last_end, c.end)
                            continue
                    except Exception as e:
                        log(f"AI review error ({e.__class__.__name__}: {e}); keeping clip")
                renderer.submit(c, tl, feats)
                prev_clip = c
                last_end = max(last_end, c.end)
                if c.replaces is None:
                    made += 0.5  # count queued clips half until rendered, so the rate controller doesn't overshoot

        # disk housekeeping for long streams
        if not is_file and args.keep_minutes > 0:
            cutoff = min(frontier, last_end) - args.keep_minutes * 60
            for s in tl.segs:
                if s.t0 + s.dur < cutoff and Path(s.path).exists():
                    Path(s.path).unlink()

        if args.duration and recorder and not stop_evt.is_set() and time.time() - started > args.duration * 60:
            log("duration reached")
            shutdown()
        if finished:
            break
        time.sleep(2)

    log("finishing queued renders...")
    renderer.finish()
    stop_evt.set()
    n = len(list(outdir.glob("*.mp4")))
    log(f"done. {n} clip(s) in {outdir.resolve()}")


if __name__ == "__main__":
    main()
