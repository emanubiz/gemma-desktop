# Gemma 4 E4B Desktop

Local voice assistant and chat app powered by Gemma 4 E4B on Apple Silicon.

## Prerequisites

- macOS Sonoma+ on Apple Silicon (M1/M2/M3/M4)
- Rust + Cargo (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew + Node.js (for Tauri)
- Python virtual environments:
  - `~/gemma-env/` with mlx-vlm, sounddevice, soundfile, numpy, requests, websockets
  - `~/kokoro-env/` with Kokoro TTS server
- `~/kokoro-server.py` configured and working
- ffmpeg (`brew install ffmpeg`)

## Project structure

```
gemma-desktop/
├── backend.py               # Python WebSocket server (VAD, Gemma API, TTS)
├── dist/
│   ├── index.html           # Frontend UI (mascot, chat, voice, settings)
│   ├── mascot-ready.png
│   ├── mascot-thinking.png
│   └── mascot-speaking.png
├── src-tauri/
│   ├── src/main.rs          # Tauri Rust app — spawns backend, handles cleanup
│   ├── tauri.conf.json      # Tauri config (window, bundle, entitlements)
│   ├── Cargo.toml
│   ├── entitlements.plist   # macOS permissions (microphone, network)
│   └── Info.plist           # NSMicrophoneUsageDescription
└── README.md
```

## Usage

1. Launch the app (`cargo tauri dev` or open the built `.app`)
2. The backend automatically starts Gemma (port 8080) and Kokoro (port 5050) servers
3. **Voice** tab: toggle "LISTEN ON" to start speaking
4. **Chat** tab: type text messages
5. ⚙️ **Settings** (gear icon):
   - **Mic sensitivity** — adjust VAD threshold
   - **Silence timeout** — how much silence before end of utterance
   - **Max audio duration** — recording limit
   - **System prompt** — customize assistant behavior (free text)
   - **File logging** — save logs to `~/gemma-desktop/logs/backend.log`

## Server indicators

Two badges Gemma ● / Kokoro ● in the header:
- **Green** = server is up
- **Red** = server is offline

## Development

```bash
cd ~/gemma-desktop
cargo tauri dev
```

## Production build

```bash
cd ~/gemma-desktop
cargo tauri build
```

The `.app` bundle is at `src-tauri/target/release/bundle/macos/`.

## Architecture

- **Rust (Tauri)**: macOS window with WebView, spawns and manages backend lifecycle
- **Python (backend.py)**: WebSocket on port 7861, VAD via sounddevice, Gemma API calls (MLX), Kokoro TTS
- **Frontend (index.html)**: Vanilla HTML/CSS/JS, mascot with 3 visual states, mic toggle, chat, settings

The backend communicates with local servers:
- Gemma (MLX): `http://localhost:8080/v1/chat/completions`
- Kokoro TTS: `http://localhost:5050`

On app close:
1. Frontend sends `quit` via WebSocket
2. Backend terminates child servers (SIGTERM → SIGKILL)
3. Rust runs safety sweeps via PID file and `ps aux`
