#!/usr/bin/env bash
# HypeCut: start the dashboard (if it isn't running yet) and open it in the browser.
#
#   ./start.sh              start + open http://127.0.0.1:8765
#   ./start.sh --install    also put a "HypeCut" icon on the Desktop and in the app menu (double-click to start)
#
# Keep the window open while you clip. Closing it (or Ctrl+C) stops the dashboard and its jobs cleanly.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
HERE="$PWD"
PORT="${HYPECUT_PORT:-8765}"
URL="http://127.0.0.1:$PORT"

pause_exit() { trap 'exit 0' INT; echo; read -r -p "Press Enter to close this window" _; exit "${1:-1}"; }
is_up() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

if [ "$1" = "--install" ]; then
    entry="[Desktop Entry]
Type=Application
Name=HypeCut
Comment=Paste a stream link, get vertical clips
Exec=\"$HERE/start.sh\"
Path=$HERE
Icon=camera-video
Terminal=true
Categories=AudioVideo;Video;"
    desk="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    mkdir -p "$HOME/.local/share/applications"
    for f in "$desk/HypeCut.desktop" "$HOME/.local/share/applications/HypeCut.desktop"; do
        [ -d "$(dirname "$f")" ] || continue
        printf '%s\n' "$entry" > "$f" && chmod +x "$f" && echo "launcher: $f"
        command -v gio >/dev/null && gio set "$f" metadata::trusted true 2>/dev/null
    done
    exit 0
fi

if [ ! -x .venv/bin/python ]; then
    echo "First run: installing HypeCut (a few minutes)..."
    ./setup.sh || { echo "setup failed - see the messages above"; pause_exit 1; }
fi

if is_up; then
    echo "HypeCut is already running - opening $URL"
    xdg-open "$URL" >/dev/null 2>&1 &
    sleep 1
    exit 0
fi

# open the browser as soon as the dashboard answers
(
    for _ in $(seq 120); do
        if is_up; then xdg-open "$URL" >/dev/null 2>&1; exit 0; fi
        sleep 0.5
    done
) &

echo "Starting HypeCut... the dashboard opens in your browser."
echo "Keep this window open while clipping. Close it or press Ctrl+C to stop everything."
echo
# own systemd scope: whatever launched this window (or runs out of memory) can't take the dashboard down with it
RUN=()
if command -v systemd-run >/dev/null && systemd-run --user --scope --quiet --collect -- true 2>/dev/null; then
    RUN=(systemd-run --user --scope --quiet --collect "--unit=hypecut-dashboard-$$" -p OOMPolicy=continue --)
fi
"${RUN[@]}" .venv/bin/python webui.py --port "$PORT"
code=$?
# 130 = Ctrl+C, 129 = window closed, 143 = stopped from outside (jobs keep running): normal ways to quit
[ $code -eq 0 ] || [ $code -eq 130 ] || [ $code -eq 129 ] || [ $code -eq 143 ] || { echo "HypeCut stopped unexpectedly (exit $code)"; pause_exit 1; }
