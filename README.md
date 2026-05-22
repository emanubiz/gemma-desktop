# Gemma 4 E4B Desktop

Assistente vocale e chat locale con Gemma 4 E4B su Apple Silicon.

## Prerequisiti

- macOS Sonoma+ su Apple Silicon (M1/M2/M3/M4)
- Rust + Cargo (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew + Node.js (per Tauri)
- Python ambienti virtuali:
  - `~/gemma-env/` con mlx-vlm, sounddevice, soundfile, numpy, requests, websockets
  - `~/kokoro-env/` con Kokoro TTS server
- `~/kokoro-server.py` configurato e funzionante
- ffmpeg (`brew install ffmpeg`)

## Struttura

```
gemma-desktop/
├── backend.py               # Server WebSocket Python (VAD, Gemma API, TTS)
├── dist/
│   ├── index.html           # Frontend UI (mascotte, chat, voce, impostazioni)
│   ├── mascot-ready.png
│   ├── mascot-thinking.png
│   └── mascot-speaking.png
├── src-tauri/
│   ├── src/main.rs          # App Tauri (Rust) — spawna backend, cleanup server
│   ├── tauri.conf.json      # Config Tauri (finestra, bundle, entitlements)
│   ├── Cargo.toml
│   ├── entitlements.plist   # Permessi macOS (microfono, rete)
│   └── Info.plist           # NSMicrophoneUsageDescription
└── README.md
```

## Utilizzo

1. Avvia l'app (`cargo tauri dev` o apri il `.app` built)
2. Il backend avvia automaticamente i server Gemma (8080) e Kokoro (5050)
3. Tab **Voce**: attiva "ASCOLTO ON" per parlare
4. Tab **Chat**: scrivi messaggi di testo
5. ⚙️ **Impostazioni** (icona ingranaggio):
   - **Sensibilità microfono** — regola la soglia VAD
   - **Pausa per fine frase** — quanto silenzio prima di considerare fine intervento
   - **Durata max audio** — limite massimo registrazione
   - **System prompt** — modifica il comportamento dell'assistente (testo libero)
   - **Log su file** — salva log in `~/gemma-desktop/logs/backend.log`

## Indicatori server

Nell'header sono visibili due badge Gemma ● / Kokoro ●:
- **Verde** = server attivo
- **Rosso** = server offline

## Sviluppo

```bash
cd ~/gemma-desktop
cargo tauri dev
```

## Build produzione

```bash
cd ~/gemma-desktop
cargo tauri build
```

Il bundle `.app` si trova in `src-tauri/target/release/bundle/macos/`.

## Architettura

- **Rust (Tauri)**: finestra macOS con WebView, spawna e gestisce ciclo vita backend
- **Python (backend.py)**: WebSocket su porta 7861, VAD con sounddevice, chiamate API a Gemma (MLX), TTS con Kokoro
- **Frontend (index.html)**: HTML/CSS/JS vanilla, mascotte con 3 stati visivi, toggle microfono, chat, impostazioni

Il backend comunica con i server locali:
- Gemma (MLX): `http://localhost:8080/v1/chat/completions`
- Kokoro TTS: `http://localhost:5050`

Alla chiusura dell'app:
1. Il frontend invia `quit` via WebSocket
2. Il backend termina i server figli (SIGTERM → SIGKILL)
3. Rust esegue passate di sicurezza via PID file e `ps aux`
