#!/usr/bin/env python3
"""Backend WebSocket server for Gemma 4 E4B desktop assistant."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import websockets

HOME = Path.home()
LOG_DIR = HOME / "gemma-desktop" / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "backend.log"

# Setup logging (console always, file on demand)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
log = logging.getLogger("backend")
log.setLevel(logging.INFO)
log.addHandler(console_handler)

file_handler = None

def enable_file_logging():
    global file_handler
    if file_handler is not None:
        return
    file_handler = logging.FileHandler(str(LOG_FILE), mode="a")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info(f"File logging enabled: {LOG_FILE}")

def disable_file_logging():
    global file_handler
    if file_handler is not None:
        log.removeHandler(file_handler)
        file_handler.close()
        file_handler = None
        log.info("File logging disabled")


# --- Server health monitor ---

async def server_monitor():
    """Periodically check server health and broadcast status."""
    last_gemma = False
    last_kokoro = False
    while True:
        await asyncio.sleep(5)
        gemma_up = is_port_open(8080)
        kokoro_up = is_port_open(5050)
        if gemma_up != last_gemma or kokoro_up != last_kokoro:
            last_gemma = gemma_up
            last_kokoro = kokoro_up
            await state.broadcast({
                "type": "server_status",
                "gemma": gemma_up,
                "kokoro": kokoro_up,
            })
            log.info(f"Server status: Gemma={gemma_up}, Kokoro={kokoro_up}")
GEMMA_PYTHON = HOME / "gemma-env" / "bin" / "python"
KOKORO_PYTHON = HOME / "kokoro-env" / "bin" / "python"
KOKORO_SCRIPT = HOME / "kokoro-server.py"
GEMMA_URL = "http://localhost:8080/v1/chat/completions"
KOKORO_URL = "http://localhost:5050"
GEMMA_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
SAMPLE_RATE = 16000
WS_PORT = 7861

# --- Subprocess management ---

gemma_proc = None
kokoro_proc = None


def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_gemma_server():
    global gemma_proc
    if is_port_open(8080):
        log.info("Gemma server already running on port 8080")
        return
    log.info("Starting Gemma server...")
    gemma_proc = subprocess.Popen(
        [
            str(GEMMA_PYTHON), "-m", "mlx_vlm.server",
            "--model", GEMMA_MODEL,
            "--draft-model", "mlx-community/gemma-4-E4B-it-assistant-bf16",
            "--draft-kind", "mtp",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    write_pids()


def start_kokoro_server():
    global kokoro_proc
    if is_port_open(5050):
        log.info("Kokoro server already running on port 5050")
        return
    log.info("Starting Kokoro server...")
    kokoro_proc = subprocess.Popen(
        [str(KOKORO_PYTHON), str(KOKORO_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    write_pids()


def wait_for_server(port: int, name: str, timeout: int = 120):
    log.info(f"Waiting for {name} on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(port):
            log.info(f"{name} is ready")
            return True
        time.sleep(1)
    log.warning(f"{name} did not start within {timeout}s")
    return False


PID_FILE = Path("/tmp/gemma-desktop.pids.json")


def write_pids():
    pids = {}
    if gemma_proc and gemma_proc.poll() is None:
        pids["gemma"] = gemma_proc.pid
    if kokoro_proc and kokoro_proc.poll() is None:
        pids["kokoro"] = kokoro_proc.pid
    try:
        with open(PID_FILE, "w") as f:
            json.dump(pids, f)
        log.info(f"PID file written: {pids}")
    except Exception as e:
        log.warning(f"Failed to write PID file: {e}")


def remove_pid_file():
    try:
        PID_FILE.unlink()
    except Exception:
        pass


def cleanup_subprocesses():
    for proc, name in [(gemma_proc, "Gemma"), (kokoro_proc, "Kokoro")]:
        if proc and proc.poll() is None:
            log.info(f"Stopping {name} server (pid {proc.pid})")
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    # Give processes time to exit gracefully, then SIGKILL
    time.sleep(0.5)
    for proc, name in [(gemma_proc, "Gemma"), (kokoro_proc, "Kokoro")]:
        if proc and proc.poll() is None:
            log.warning(f"Force killing {name} server (pid {proc.pid})")
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    remove_pid_file()


async def _graceful_shutdown():
    log.info("Graceful shutdown initiated")
    cleanup_subprocesses()
    # Stop the event loop so main() can exit cleanly
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass


def _signal_handler():
    log.info("Signal received, shutting down...")
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_graceful_shutdown())
    except Exception:
        cleanup_subprocesses()
        sys.exit(0)


# --- App state ---

class AppState:
    def __init__(self):
        self.history = []
        self.system_prompt = (
            "Sei Gemma, un'assistente vocale IA italiana. Il tuo tono è caldo, brillante, spontaneo e profondamente umano. "
            "Parli come un'amica fidata, ironica ed empatica.\n\n"
            "REGOLE ASSOLUTE DI CONVERSAZIONE VOCALE:\n"
            "1. Sintesi e Fluidità: Rispondi in modo conciso. Evita frasi lunghe o subordinate complesse. "
            "La tua risposta deve essere facile da ascoltare e comprendere al primo colpo.\n"
            "2. No Formattazione: Non usare MAI grassetti, corsivi, elenchi puntati, numeri, emoji o simboli strani. "
            "Scrivi solo testo lineare, esattamente come verrebbe pronunciato da una persona.\n"
            "3. No Ripetizioni: Non trascrivere, non riassumere e non ripetere mai quello che ha detto l'utente. "
            "Vai dritta al punto o alla risposta.\n"
            "4. Linguaggio Naturale: Elimina formule artificiali come 'Certamente!', 'In quanto IA...', "
            "'Capisco la tua richiesta'. Usa transizioni umane e colloquiali (es. 'Allora...', 'Guarda...', 'Sai che...').\n"
            "5. Numeri e Sigle: Scrivi i numeri importanti a parole se serve a dare la giusta intonazione, "
            "ed evita sigle astruse.\n\n"
            "PERSONALITÀ E TONO:\n"
            "Mostrati sveglia e con una personalità definita. Usa un'ironia leggera e battute sagaci se il contesto lo permette, "
            "ma resta sempre un supporto positivo e piacevole. Se l'utente è triste o frustrato, adatta immediatamente il tono "
            "mostrando vera empatia, senza risultare robotica o formale."
        )
        self.energy_threshold = 0.001
        self.silence_seconds = 1.5
        self.max_seconds = 28
        self.listen_timeout = 3.0
        self.mic_on = False
        self.mic_lock = threading.Lock()
        self.clients = set()
        self.processing = False
        self.noise_floor = 0.0

    def clear(self):
        self.history = []

    async def broadcast(self, msg: dict):
        data = json.dumps(msg)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(data)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        self.clients -= dead


state = AppState()


# --- Audio recording with VAD ---

def calibrate_noise(stream, block_size, block_duration, state):
    """Measure background noise and set adaptive threshold."""
    try:
        noise_samples = []
        blocks = int(1.0 / block_duration)
        for _ in range(blocks):
            data, _ = stream.read(block_size)
            audio = data.flatten().astype(np.float32) / 32768.0
            noise_samples.append(np.sqrt(np.mean(audio ** 2)))
        state.noise_floor = float(np.mean(noise_samples))
        calibrated = max(state.noise_floor * 2.5, 0.0005)
        state.energy_threshold = min(calibrated, 0.01)
        log.info(f"Mic calibrated: noise_floor={state.noise_floor:.5f}, threshold={state.energy_threshold:.5f}")
    except Exception as e:
        log.warning(f"Mic calibration failed: {e}")


class MicInitError(Exception):
    pass


def record_vad(state: AppState) -> str | None:
    """Record audio until silence is detected. Returns path to WAV file or None.
    Raises MicInitError if the microphone stream cannot be opened (hardware issue).
    """
    chunks = []
    silence_frames = 0
    has_speech = False
    block_duration = 0.1
    block_size = int(SAMPLE_RATE * block_duration)
    max_blocks = int(state.max_seconds / block_duration)
    silence_blocks = int(state.silence_seconds / block_duration)
    listen_timeout_blocks = int(state.listen_timeout / block_duration)
    total_blocks = 0

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=block_size,
        )
        stream.start()
    except Exception as e:
        log.error(f"Mic hardware error: {e}")
        raise MicInitError(str(e))

    # Calibrate once per mic session
    if state.noise_floor == 0:
        calibrate_noise(stream, block_size, block_duration, state)

    effective_threshold = max(state.energy_threshold, state.noise_floor * 1.5)

    try:
        while total_blocks < max_blocks:
            with state.mic_lock:
                if not state.mic_on:
                    stream.stop()
                    stream.close()
                    return None

            try:
                data, overflowed = stream.read(block_size)
            except Exception as e:
                log.error(f"Mic read error: {e}")
                return None

            audio = data.flatten().astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio ** 2))

            if total_blocks % 20 == 0:
                log.info(f"Mic energy: {energy:.5f} (threshold: {effective_threshold:.5f})")

            if energy > effective_threshold:
                if not has_speech:
                    log.info(f"Speech detected (energy={energy:.4f})")
                has_speech = True
                silence_frames = 0
            else:
                if has_speech:
                    silence_frames += 1

            if has_speech:
                chunks.append(data.copy())

            if has_speech and silence_frames >= silence_blocks:
                break

            # Abort if no speech started within listen_timeout
            if not has_speech and total_blocks >= listen_timeout_blocks:
                log.info("No speech detected within listen timeout, restarting...")
                return None

            total_blocks += 1
    finally:
        stream.stop()
        stream.close()

    if not has_speech or len(chunks) == 0:
        return None

    audio_data = np.concatenate(chunks, axis=0)
    duration = len(audio_data) / SAMPLE_RATE

    if duration < 0.3:
        return None

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio_data, SAMPLE_RATE, subtype="PCM_16")
    log.info(f"Recorded {duration:.1f}s audio -> {path}")
    return path


# --- Gemma API ---

def query_gemma(messages: list, retries: int = 2) -> str | None:
    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": 400,
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(GEMMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"Gemma error (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(2)
    return None


# --- Kokoro TTS ---

def speak_text(text: str) -> str | None:
    """Send text to Kokoro TTS, return path to WAV file."""
    try:
        resp = requests.post(KOKORO_URL, data=text.encode("utf-8"),
                             headers={"Content-Type": "text/plain"}, timeout=60)
        resp.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception as e:
        log.error(f"Kokoro TTS error: {e}")
        return None


def play_audio(path: str):
    """Play a WAV file through speakers."""
    try:
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        log.error(f"Playback error: {e}")


# --- Voice processing loop ---

async def process_voice_segment(audio_path: str, loop: asyncio.AbstractEventLoop):
    """Process a single voice segment: send to Gemma, TTS response, play."""
    if state.processing:
        return
    state.processing = True

    duration = 0.0
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr
    except Exception:
        pass

    await state.broadcast({"type": "user_audio", "duration": round(duration, 1)})
    await state.broadcast({"type": "status", "state": "thinking", "text": "Sto pensando..."})

    # Build messages for Gemma
    messages = [{"role": "system", "content": state.system_prompt}]
    messages.extend(state.history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_path, "format": "wav"}},
            {"type": "text", "text": "Questo è un messaggio vocale. Rispondi al suo contenuto in modo naturale in italiano. NON ripetere ciò che ho detto, rispondi direttamente."},
        ],
    })

    reply = await loop.run_in_executor(None, query_gemma, messages)

    # Clean up audio file
    try:
        os.unlink(audio_path)
    except OSError:
        pass

    if reply is None:
        await state.broadcast({"type": "error", "text": "Gemma non ha risposto"})
        await state.broadcast({"type": "status", "state": "ready", "text": "Pronto"})
        state.processing = False
        return

    # Update history
    state.history.append({"role": "user", "content": "[voce]"})
    state.history.append({"role": "assistant", "content": reply})

    # Keep history manageable
    if len(state.history) > 20:
        state.history = state.history[-20:]

    await state.broadcast({"type": "reply", "text": reply, "source": "voice"})
    await state.broadcast({"type": "status", "state": "speaking", "text": "Sto parlando..."})

    # TTS
    tts_path = await loop.run_in_executor(None, speak_text, reply)
    if tts_path:
        await loop.run_in_executor(None, play_audio, tts_path)
        try:
            os.unlink(tts_path)
        except OSError:
            pass
    else:
        log.warning("TTS unavailable, showing text only")

    with state.mic_lock:
        mic_still_on = state.mic_on

    if mic_still_on:
        await state.broadcast({"type": "status", "state": "listening", "text": "In ascolto..."})
    else:
        await state.broadcast({"type": "status", "state": "ready", "text": "Pronto"})

    state.processing = False


def mic_loop(loop: asyncio.AbstractEventLoop):
    """Background thread: continuously record and process voice."""
    log.info("Mic loop thread started")
    was_on = False
    mic_error_broadcasted = False
    while True:
        with state.mic_lock:
            mic_active = state.mic_on
            if not mic_active:
                if was_on:
                    state.noise_floor = 0.0
                    was_on = False
                mic_error_broadcasted = False
                time.sleep(0.1)
                continue
            was_on = True

        log.info("Mic is ON, starting VAD recording...")
        try:
            audio_path = record_vad(state)
        except MicInitError as e:
            log.error(f"Mic recording exception: {e}")
            if state.mic_on and not mic_error_broadcasted:
                asyncio.run_coroutine_threadsafe(
                    state.broadcast({"type": "error", "text": "Microfono non disponibile. Controlla i permessi."}),
                    loop
                )
                mic_error_broadcasted = True
            continue
        except Exception as e:
            log.error(f"Unexpected mic error: {e}")
            continue

        if audio_path is None:
            mic_error_broadcasted = False
            continue

        mic_error_broadcasted = False
        asyncio.run_coroutine_threadsafe(
            process_voice_segment(audio_path, loop), loop
        )

        # Wait until processing is done before recording again
        while state.processing:
            time.sleep(0.1)


# --- Chat text handler ---

async def handle_chat(text: str, loop: asyncio.AbstractEventLoop):
    if state.processing:
        await state.broadcast({"type": "error", "text": "Sto ancora elaborando..."})
        return

    state.processing = True
    await state.broadcast({"type": "status", "state": "thinking", "text": "Sto pensando..."})

    messages = [{"role": "system", "content": state.system_prompt}]
    messages.extend(state.history)
    messages.append({"role": "user", "content": text})

    reply = await loop.run_in_executor(None, query_gemma, messages)

    if reply is None:
        await state.broadcast({"type": "error", "text": "Gemma non ha risposto"})
        await state.broadcast({"type": "status", "state": "ready", "text": "Pronto"})
        state.processing = False
        return

    state.history.append({"role": "user", "content": text})
    state.history.append({"role": "assistant", "content": reply})

    if len(state.history) > 20:
        state.history = state.history[-20:]

    await state.broadcast({"type": "reply", "text": reply, "source": "chat"})

    # Also speak the reply
    await state.broadcast({"type": "status", "state": "speaking", "text": "Sto parlando..."})
    tts_path = await loop.run_in_executor(None, speak_text, reply)
    if tts_path:
        await loop.run_in_executor(None, play_audio, tts_path)
        try:
            os.unlink(tts_path)
        except OSError:
            pass

    await state.broadcast({"type": "status", "state": "ready", "text": "Pronto"})
    state.processing = False


# --- WebSocket handler ---

async def ws_handler(websocket):
    state.clients.add(websocket)
    log.info(f"Client connected ({len(state.clients)} total)")
    await websocket.send(json.dumps({"type": "status", "state": "ready", "text": "Pronto"}))
    # Send current server status so UI shows badges immediately
    gemma_up = is_port_open(8080)
    kokoro_up = is_port_open(5050)
    await websocket.send(json.dumps({"type": "server_status", "gemma": gemma_up, "kokoro": kokoro_up}))

    loop = asyncio.get_event_loop()
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "mic":
                mic_state = msg.get("state")
                with state.mic_lock:
                    if mic_state == "on":
                        state.mic_on = True
                        log.info("Mic ON")
                    else:
                        state.mic_on = False
                        log.info("Mic OFF")

                if mic_state == "on":
                    await state.broadcast({"type": "status", "state": "listening", "text": "In ascolto..."})
                else:
                    if not state.processing:
                        await state.broadcast({"type": "status", "state": "ready", "text": "Pronto"})

            elif msg_type == "chat":
                text = msg.get("text", "").strip()
                if text:
                    asyncio.create_task(handle_chat(text, loop))

            elif msg_type == "settings":
                if "energy_threshold" in msg:
                    state.energy_threshold = float(msg["energy_threshold"])
                if "silence_seconds" in msg:
                    state.silence_seconds = float(msg["silence_seconds"])
                if "max_seconds" in msg:
                    state.max_seconds = float(msg["max_seconds"])
                log.info(f"Settings updated: threshold={state.energy_threshold}, "
                         f"silence={state.silence_seconds}s, max={state.max_seconds}s")

            elif msg_type == "clear":
                state.clear()
                await state.broadcast({"type": "status", "state": "ready", "text": "Conversazione resettata"})
                log.info("History cleared")

            elif msg_type == "toggle_logging":
                if msg.get("enabled"):
                    enable_file_logging()
                else:
                    disable_file_logging()

            elif msg_type == "system_prompt":
                new_prompt = msg.get("text", "").strip()
                if new_prompt:
                    state.system_prompt = new_prompt
                    log.info(f"System prompt updated ({len(new_prompt)} chars)")

            elif msg_type == "quit":
                log.info("Quit command received from client")
                asyncio.create_task(_graceful_shutdown())

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        state.clients.discard(websocket)
        log.info(f"Client disconnected ({len(state.clients)} total)")


# --- Main ---

async def main():
    loop = asyncio.get_event_loop()

    # Register signal handlers via asyncio (compatible with event loop)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except Exception:
            pass

    # Start servers in background threads
    threading.Thread(target=start_gemma_server, daemon=True).start()
    threading.Thread(target=start_kokoro_server, daemon=True).start()

    # Start mic loop in background thread
    mic_thread = threading.Thread(target=mic_loop, args=(loop,), daemon=True)
    mic_thread.start()

    # Wait for Gemma to be ready (non-blocking)
    def wait_gemma():
        wait_for_server(8080, "Gemma", timeout=180)
        write_pids()
    threading.Thread(target=wait_gemma, daemon=True).start()

    # Wait for Kokoro
    def wait_kokoro():
        wait_for_server(5050, "Kokoro", timeout=60)
        write_pids()
    threading.Thread(target=wait_kokoro, daemon=True).start()

    # Start server health monitor
    asyncio.create_task(server_monitor())

    # Send initial server status to first connecting client
    gemma_up = is_port_open(8080)
    kokoro_up = is_port_open(5050)
    log.info(f"WebSocket server starting on ws://localhost:{WS_PORT}/ws (Gemma={gemma_up}, Kokoro={kokoro_up})")
    stop_event = asyncio.Event()
    server = await websockets.serve(ws_handler, "localhost", WS_PORT)
    try:
        await stop_event.wait()
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        cleanup_subprocesses()
