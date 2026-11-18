# HypeCut

Point it at a live stream and it keeps cutting vertical (9:16) clips of the most exciting moments **while the stream is still running**. Works with YouTube, Twitch, anything else yt-dlp can open, and local video files.

- Finds moments from face emotion, chat hype, screams, motion and more, each measured against the stream's own baseline
- Puts gameplay on top and the streamer's face at the bottom (or follows the face for IRL streams)
- Exports 1080x1920 clips at up to 60 fps, ready for Shorts, Reels and TikTok
- Optional word-by-word captions for Hinglish and English, left off whenever the transcript is in doubt

## Quick start
You need **Python 3.10+** and **ffmpeg** (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu).

```bash
git clone https://github.com/Zura1z/hypecut.git
cd hypecut
./setup.sh
.venv/bin/python clipper.py "https://www.youtube.com/watch?v=<live-stream-id>"
```

Other sources:

```bash
.venv/bin/python clipper.py "https://www.twitch.tv/<channel>"
.venv/bin/python clipper.py my_vod.mp4
```

Press **Ctrl+C** to stop. Clips go to `clips/` as `NNN_emotion-time.mp4`, each with a `.json` file (score, reasons, transcript), plus a running `clips.jsonl` log.

The first run downloads the Whisper speech model (about 150 MB; about 1.6 GB more the first time you use `--captions`).

### Captions and AI review (optional)
`setup.sh` creates a `.env` file. Add keys there only for the features you want:

| Feature | Flag | Key in `.env` |
|---|---|---|
| Captions | `--captions` | `GEMINI_API_KEY` (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |
| Claude scores and titles each clip | `--ai` | `ANTHROPIC_API_KEY` |

Without keys everything else still works; clips simply come out without captions.

### Windows
`setup.sh` is for macOS and Linux. On Windows use WSL, or do the same steps by hand: create a virtual environment, `pip install -r requirements.txt`, copy `.env.example` to `.env`, and make sure `ffmpeg` is on your PATH.

## How it picks moments
Every second of the stream gets an excitement score. Each signal is compared with the streamer's own last 5 minutes, so "loud" or "busy chat" always means *more than usual for this stream*:
- **Face emotion**: surprised, laughing, scared or angry expressions on the facecam (local model, 2 checks per second)
- **Chat volume** and **chat hype**: messages per second, plus LOL / W / emote / ALL-CAPS spam; paid messages count extra
- **Volume spikes and screams** (loud and high-pitched), plus sudden calm-to-chaos jumps
- **Physical reactions** (movement in the facecam) and **on-screen action** (movement in the gameplay)
- **Hype words** and fast talking
- **YouTube "most replayed"** data, for finished videos only (YouTube doesn't publish it for a stream that is still live)
- Moments where several signals fire together get a bonus
- **Optional `--ai`**: Claude scores each candidate 1-10, drops weak ones, and writes a title and hook

A rate controller aims for `--target-per-hour` clips (default 30) and lowers the bar when clips come slower than that. When a second moment lands right after a clip, the two are regenerated as one longer clip and the older file is deleted, so clips never overlap.

## Editing
- Opens with a 2-second flash of the best moment, then plays the full clip (`--no-teaser` to turn off)
- Slight zoom-in at the peak (`--no-zoom`) and a progress bar on top (`--no-progress`)
- Records the best quality the stream offers and exports 1080x1920 at up to 60 fps, ~16 Mbps
- **Captions (`--captions`)** need a Google Gemini key in a `.env` file next to `clipper.py` (`GEMINI_API_KEY=...`). Whisper times each word, then Gemini listens to the clip and corrects the words into Hinglish written in Roman letters (English words in English spelling, never Urdu/Hindi script). If Whisper or Gemini is not confident, or Gemini is unreachable, the clip gets no captions at all.
- Jump cuts over silent gaps are off by default (`--jump-cuts` to enable).

## Layouts (detected automatically)
| Stream type | Output |
|---|---|
| Gameplay + facecam overlay | gameplay on top, **face at bottom** |
| Full-cam / IRL / just chatting | 9:16 crop that follows the face |
| No face | 4:3 center on a blurred background |

### Useful options
| Flag | Default | Meaning |
|---|---|---|
| `--target-per-hour` | 30 | how many clips to aim for per hour |
| `--min-len / --max-len` | 12 / 45 | clip length in seconds |
| `--min-gap` | 12 | seconds between separate clips (closer moments get merged) |
| `--captions` | off | Gemini-verified Hinglish captions; skipped when in doubt |
| `--ai` | off | Claude review (needs `ANTHROPIC_API_KEY`) |
| `--layout` | auto | force `split`, `track` or `blur` |
| `--keep-minutes` | 20 | deletes raw recording older than this (about 3 GB/hour at 1080p60) |
| `--duration` | 0 | stop after N minutes |

Run `.venv/bin/python clipper.py --help` for the full list.

## Notes
- Tested on macOS (Apple Silicon) with Python 3.14. Linux should work the same way but is not yet tested. Encoding uses Apple VideoToolbox or NVIDIA NVENC when ffmpeg has them, otherwise x264.
- Raw recording takes about 3 GB per hour at 1080p60; old footage is deleted after `--keep-minutes`.
- YouTube live chat is read through yt-dlp and Twitch chat anonymously, so no login is needed.
- The `--ai` review has not been tested end to end yet.
- Face models in `models/` come from the OpenCV Model Zoo (see `models/README.md`).
- Only clip streams you own or have permission to repost.

## License
MIT
