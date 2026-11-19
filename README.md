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
| Captions | `--captions` | `GEMINI_API_KEY` (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)); `GROQ_API_KEY` makes them more accurate |
| Smarter clip start/end (reads the transcript) | on automatically, `--edges` | `GROQ_API_KEY` (free at [console.groq.com/keys](https://console.groq.com/keys)) |
| Claude scores and titles each clip | `--ai` | `ANTHROPIC_API_KEY` |

Without keys everything else still works. `--captions` without a Gemini key falls back to local Whisper + a built-in romanizer (Roman Hindi/Urdu); those words are not double-checked, and low-confidence words or clips are still left uncaptioned.

### Windows
`setup.sh` is for macOS and Linux. On Windows use WSL, or do the same steps by hand: create a virtual environment, `pip install -r requirements.txt`, copy `.env.example` to `.env`, and make sure `ffmpeg` is on your PATH.

## Web dashboard
Prefer clicking to typing commands? Run `./start.sh`: it sets HypeCut up on the first run, starts the dashboard and opens <http://127.0.0.1:8765> in your browser (if it's already running, it just opens the page).

```bash
./start.sh
```

To start it with a double-click instead (Linux desktops), create a **HypeCut** icon on your Desktop and in the app menu once:

```bash
./start.sh --install
```

Keep the window that opens while you clip; closing it (or Ctrl+C) stops the dashboard and its running jobs cleanly. If the dashboard is closed some other way, running jobs keep going and show up again the next time you start it. You can also run the dashboard directly with `.venv/bin/python webui.py`.

- Paste each stream link into its own row (**+ Add stream** for more), pick the options (layout, captions, clips per hour, stop after N minutes, minimum and maximum resolution, live watch quality) and press that row's **Start**. Each stream runs as its own `clipper.py` job writing to `clips_<channel>/`; while it runs its row button is **Stop**, so you can stop it, change the options and start it again. Meme versions aren't offered in the dashboard.
- Each job shows its status, clip count and live log. **Stop** works like Ctrl+C: recording stops and clips already queued still finish rendering.
- The header shows HypeCut's own **bandwidth** today (↓ downloads / ↑ uploads); click it for the last 7 days, this month and each day. It counts the dashboard, every clipper and their yt-dlp/ffmpeg helpers (not your browser or other apps) while the dashboard runs, and is kept in `~/.local/share/hypecut/bandwidth.json`.
- Pick any clip folder to watch its clips (and meme versions, if a folder has any), see why each was picked, and mark them **Keep** or **Reject**. Reject moves the clip, its `.json` and its meme version into `rejected/` inside that folder; nothing is deleted.
- **Auto-classify (optional, needs `GEMINI_API_KEY` and/or `GROQ_API_KEY`):** Gemini watches each clip and suggests Keep or Reject with a 1-10 score, a one-line reason and a title idea. When Gemini is overloaded or has no key, Groq judges instead from 6 frames of the clip plus its transcript (it can't hear the audio, so it is a weaker judge); each card says which model decided. Press **Classify with AI** for a folder, turn on **Auto** (or tick *auto-classify clips* when starting a job) to check new clips as they finish, then **Apply suggestions** once you agree. Nothing is kept or moved until you click, and clips you marked Keep yourself are never rejected. Each clip is uploaded as a small 360p copy (about 1-2 MB).

The dashboard only listens on your own computer (127.0.0.1). Each job's log is saved as `hypecut.log` in its clip folder, Keep marks in `webui_kept.json` and Gemini's suggestions in `webui_ai.json`.

## How it picks moments
Every second of the stream gets an excitement score. Each signal is compared with the streamer's own last 5 minutes, so "loud" or "busy chat" always means *more than usual for this stream*:
- **Face emotion**: surprised, laughing, scared or angry expressions on the facecam (local model, 2 checks per second), measured against the streamer's own usual face, so a resting face that the model reads as "happy" doesn't count
- **Chat volume** and **chat hype**: messages per second, plus LOL / W / emote / ALL-CAPS spam; paid messages count extra; each viewer counts at most once per 8 seconds, so a few spammers can't fake a moment
- **Volume spikes and screams** (loud and high-pitched), plus sudden calm-to-chaos jumps
- **Physical reactions** (movement in the facecam) and **on-screen action** (movement in the gameplay)
- **Hype words** and fast talking
- **YouTube "most replayed"** data, for finished videos only (YouTube doesn't publish it for a stream that is still live)
- Moments where several signals fire together get a bonus
- **Optional `--ai`**: Claude scores each candidate 1-10, drops weak ones, and writes a title and hook

A clip also needs the streamer to react (louder than usual, a scream, hype words, a clearly unusual expression or a big physical reaction); chat or gameplay alone isn't enough. Nothing is clipped during the first `--warmup` seconds (default 180) while HypeCut learns the stream's normal, and calm stretches produce no clips at all (`--threshold`, default 3.5). A rate controller aims for `--target-per-hour` clips (default 12) and lowers the bar when clips come slower than that, but never below `--min-percentile` (default 85). When a second moment lands right after a clip, the two are regenerated as one longer clip and the older file is deleted, so clips never overlap.

**Low-data live mode (YouTube):** a YouTube live stream is watched at `--watch-height` (default 480p, ~0.4-0.5 GB/hour instead of ~2.5 GB/hour plus playlist overhead at 1080p60) to find the moments, recorded by fetching the stream's 2-5 s pieces directly (ffmpeg's HLS reader re-downloads a ~1 MB playlist every few seconds, which was most of the data). When a clip is found, that stretch is fetched at the stream's best resolution from YouTube's rewind buffer (the numbered 2-second pieces a live stream keeps for hours) and lined up with the recording by matching the audio, so clips still come out at max resolution. Twitch/Kick have no rewind, so they are recorded at full resolution as before; `--live-lowres off` does that for YouTube too. A lower watch resolution saves more data but makes face/emotion detection on a small facecam weaker. YouTube needs a JavaScript runtime for this (node or deno on your PATH).

**Uploaded videos (VODs)** of 20 minutes or more are not downloaded whole. HypeCut downloads only low-bitrate audio (~0.25 GB for an 11 h stream vs. ~16-21 GB of 1080p video; kept in `~/.cache/hypecut/audio` so a restart doesn't download it again), scores every second of it (loudness vs. the streamer's normal, screams, calm-to-loud payoffs, plus YouTube's most-replayed data when the video has it), then downloads ~105 s of video around the strongest moments (at the best resolution, or `--max-height`), best first, and clips each one with the full analysis (face, emotion, motion, transcript). Clip quality is the same; what the audio scan can miss are quiet-but-funny moments. `--vod-clips` (default 20) sets how many clips, `--vod-mode full` restores the old full download, and each clip's `.json` records where it is in the original video (`vod_time`). Chat replay isn't used for VODs: YouTube serves it at about 2 minutes of stream per minute.

Clip length follows the moment: a clip starts where the build-up began (plus `--setup` seconds of context) and ends once the reaction has died down (excitement back to calm for 3 seconds, plus `--aftermath`), between `--min-len` and `--max-len` (70-90 s for the finished clip, hook included). With a free `GROQ_API_KEY` in `.env`, Groq then reads the timestamped transcript around the moment and moves the start and end so the clip opens on the setup and ends right after the payoff, at sentence boundaries. Only text is sent (the transcript and the excitement curve), and if Groq is unreachable the signal-based edges are used. Turn it off with `--edges signals`.

## Editing
- Opens with a 7-second hook of the best moment (from just before the peak through the reaction; `--teaser-len`), then plays the full clip (`--no-teaser` to turn off)
- Slight zoom-in at the peak (`--no-zoom`) and a progress bar on top (`--no-progress`)
- Records the best quality the stream offers and exports 1080x1920 at up to 60 fps, ~16 Mbps
- **Captions (`--captions`)** show what the **streamer** says, plus game or video lines only when they matter for the moment (the setup or punchline the streamer reacts to). Background dialogue, NPC chatter, songs and music are left out. English is written in English, Urdu/Hindi in Roman letters (nahi, kya, bhai), never in Urdu/Hindi script. Whisper times each word: Whisper large-v3 on Groq when `GROQ_API_KEY` is set (more accurate, and no 1.5 GB model in memory; `--caption-asr` to choose), otherwise the local `--caption-model`. Gemini (`GEMINI_API_KEY`) then listens to the clip, corrects the words and says who speaks each one. Without Gemini, or when Whisper or Gemini is not confident, the clip gets no captions at all - wrong captions are worse than none.
- Jump cuts over silent gaps are off by default (`--jump-cuts` to enable).
- **Meme version (`--memes`)**: every clip also gets a `NNN_..._meme.mp4` with emoji pops, meme text, screen shake, color flashes and sound effects (boom, whoosh, ding, sad trombone-style notes), placed only on the seconds that earned them: laughing, shock or screams, rage, big chat explosions or on-screen action. Everything is generated locally; emoji need Pillow and a color emoji font (e.g. Noto Color Emoji).

## Layouts (detected automatically)
| Stream type | Output |
|---|---|
| Gameplay + facecam overlay | `stack`: gameplay on the top ~72% (cropped inside cutscene black bars), a thin yellow divider, and the facecam as a full-width strip underneath (head, shoulders, a bit of the room). A small facecam is enlarged at most `--cam-upscale` (2.4x) and shown over a blurred copy of itself, so it stays sharp |
| Full-cam / IRL / just chatting | 9:16 crop that follows the face |
| No face | 4:3 center on a blurred background |

### Useful options
| Flag | Default | Meaning |
|---|---|---|
| `--target-per-hour` | 12 | how many clips to aim for per hour |
| `--threshold` | 3.5 | absolute excitement floor; calm stretches produce no clips |
| `--warmup` | 180 | seconds to watch before the first clip |
| `--memes` | off | also write a meme-edited version of each clip |
| `--min-height` | 1080 | refuse sources below this resolution instead of recording lower quality |
| `--min-len / --max-len` | 70 / 90 | finished clip length range in seconds, hook included; each clip is sized to its moment |
| `--live-lowres` / `--watch-height` | auto / 480 | YouTube lives: watch low-res, fetch each clip at max res from the rewind |
| `--vod-mode` / `--vod-clips` | scan / 20 | VODs: audio scan + download only the moments, and how many clips |
| `--max-height` | best | cap the resolution that is downloaded for clips (less data) |
| `--edges` | auto | `groq` = Groq picks start/end from the transcript (automatic when `GROQ_API_KEY` is set), `signals` = never |
| `--min-gap` | 12 | seconds between separate clips (closer moments get merged) |
| `--captions` | off | Gemini-verified Hinglish captions; skipped when in doubt |
| `--ai` | off | Claude review (needs `ANTHROPIC_API_KEY`) |
| `--layout` | auto | `stack` (facecam streams' default), `float` (full-screen gameplay + small floating cam), `split` (half/half with a face close-up), `track` or `blur` |
| `--keep-minutes` | 20 | deletes raw recording older than this (about 3 GB/hour at 1080p60) |
| `--duration` | 0 | stop after N minutes |

Run `.venv/bin/python clipper.py --help` for the full list.

## Notes
- Tested on macOS (Apple Silicon) with Python 3.14. Linux should work the same way but is not yet tested. Encoding uses Apple VideoToolbox, NVIDIA NVENC or VAAPI (AMD/Intel on Linux) when a short test encode shows they work on this machine, otherwise x264. Works with ffmpeg 7+ (including 9).
- Raw recording takes about 3 GB per hour at 1080p60; old footage is deleted after `--keep-minutes`.
- YouTube live chat is read through yt-dlp and Twitch chat anonymously, so no login is needed.
- The `--ai` review has not been tested end to end yet.
- Face models in `models/` come from the OpenCV Model Zoo (see `models/README.md`).
- Only clip streams you own or have permission to repost.

## License
MIT
