"""Tiny Groq client (OpenAI-compatible REST API, stdlib only). Free key: https://console.groq.com/keys

Used by clipper.py (pick clip start/end from the transcript) and webui.py (Keep/Reject fallback when Gemini
is overloaded). Models that are rate-limited or overloaded are skipped for a while instead of retried.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.groq.com/openai/v1/chat/completions"
TEXT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
VISION_MODELS = ["qwen/qwen3.8-27b"]   # max 3 images per request
_busy_until = {}


class GroqError(RuntimeError):
    pass


def api_key():
    """GROQ_API_KEY from the environment or .env next to this file (re-read every call: no restart needed)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        for line in (Path(__file__).resolve().parent / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "GROQ_API_KEY" and v.strip() and not line.lstrip().startswith("#"):
                return v.strip().strip("\"'")
    except OSError:
        pass
    return None


ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ASR_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]


def transcribe(wav_bytes, language=None, prompt=None, timeout=60):
    """Whisper on Groq with word timestamps. Returns (words, segments, model): words = [{word, start, end}],
    segments = [{start, end, text, avg_logprob, no_speech_prob}]. Max 25 MB of audio (a 75 s 16 kHz WAV is ~2.4 MB)."""
    key = api_key()
    if not key:
        raise GroqError("no GROQ_API_KEY in .env")
    errors = []
    for model in [m for m in ASR_MODELS if _busy_until.get(m, 0) <= time.time()]:
        fields = [("model", model), ("response_format", "verbose_json"), ("temperature", "0"),
                  ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
        if language:
            fields.append(("language", language))
        if prompt:
            fields.append(("prompt", prompt[:800]))
        boundary = "----hypecut" + os.urandom(8).hex()
        body = b""
        for k, v in fields:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(ASR_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "hypecut/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            return d.get("words") or [], d.get("segments") or [], model
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            if e.code == 401 or (e.code == 403 and "invalid" in detail.lower()):
                raise GroqError("Groq rejected the API key") from None
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 60) if e.headers else 60
                _busy_until[model] = time.time() + min(max(wait, 10), 900)
            errors.append(f"{model}: HTTP {e.code} {detail[:120]}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            errors.append(f"{model}: {e.__class__.__name__}")
    raise GroqError("; ".join(errors) or "all Groq speech models are rate-limited right now")


def image_part(jpeg_bytes):
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}}


def chat_json(content, schema, models=TEXT_MODELS, name="answer", timeout=60):
    """Send one user message (str or list of parts); returns (parsed JSON matching `schema`, model id).
    `schema` must be strict-mode compatible: every property required, additionalProperties false."""
    key = api_key()
    if not key:
        raise GroqError("no GROQ_API_KEY in .env")
    now = time.time()
    todo = [m for m in models if _busy_until.get(m, 0) <= now]
    if not todo:
        raise GroqError("all Groq models are rate-limited right now")
    errors = []
    for model in todo:
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": name, "strict": True, "schema": schema}}}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "hypecut/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            text = data["choices"][0]["message"]["content"]
            return json.loads(text), model
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            if e.code == 401 or (e.code == 403 and "invalid" in detail.lower()):
                raise GroqError("Groq rejected the API key") from None
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 60) if e.headers else 60
                _busy_until[model] = time.time() + min(max(wait, 10), 900)
            errors.append(f"{model}: HTTP {e.code} {detail[:120]}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(f"{model}: {e.__class__.__name__}")
        except (KeyError, IndexError, ValueError) as e:
            errors.append(f"{model}: bad reply ({e.__class__.__name__})")
    raise GroqError("; ".join(errors))
