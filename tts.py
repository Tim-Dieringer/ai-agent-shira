import logging
import multiprocessing as mp
import queue
import socket
import time
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent

import numpy as np
import requests
import sounddevice as sd

import config
from gui.gui_messages import TtsTimeMsg

log = logging.getLogger("shira.tts")

# ---------------------------------------------------------------------------
# Config - host/port and the voice description live in config.py (env vars).
# Start the server yourself first, e.g.:
#   breeze-server.exe breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8083
# ---------------------------------------------------------------------------
HOST = config.TTS_HOST
PORT = config.TTS_PORT
SPEECH_URL = config.TTS_SPEECH_URL

# Voice design: the voice is described in plain text (see SHIRA_TTS_VOICE).
INSTRUCTION = config.TTS_VOICE_INSTRUCTION

SAMPLE_RATE = 24000        # server streams mono 24 kHz signed 16-bit LE PCM
BYTES_PER_SAMPLE = 2
PREBUFFER_SECONDS = 0.2   # audio to buffer before playback starts (avoids underruns)

FRAME_BYTES = int(0.03 * SAMPLE_RATE) * BYTES_PER_SAMPLE  # 30 ms of audio per frame
GAIN = 6.0 # tune so normal speech reaches ~0.6-1.0

VOLUME = 0.5  # 1.0 = unchanged, 0.5 is about -6 dB, 0.25 is about -12 dB

# Local server: never route through a system/env proxy (VPNs and proxy tools
# can answer localhost requests with bogus 404s).
_http = requests.Session()
_http.trust_env = False


def start_tts_process(stop_event: MpEvent, input_q: MpQueue, gui_q: MpQueue) -> mp.Process:
    p = mp.Process(
        target=_tts_process_fn,
        args=(stop_event, input_q, gui_q),
        daemon=True,
        name="tts-process",
    )

    p.start()
    return p


def _form(text: str) -> dict:
    """Multipart fields for the speech endpoint; the instruction is only sent when set."""
    form = {"text": (None, text)}
    if INSTRUCTION:
        form["instruction"] = (None, INSTRUCTION)
    return form


def _port_open() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_server(stop_event: MpEvent) -> bool:
    """Block until the server is listening. Returns False if asked to stop first."""
    last_msg = 0.0
    while not stop_event.is_set():
        if _port_open():
            return True
        if time.time() - last_msg > 5:
            log.info(f"Waiting for breeze-server on {HOST}:{PORT}...")
            last_msg = time.time()
        time.sleep(0.5)
    return False


def _check(resp: requests.Response) -> None:
    """raise_for_status, but include what the server actually said."""
    if not resp.ok:
        body = resp.text[:300].strip()
        raise RuntimeError(
            f"HTTP {resp.status_code} from {resp.url}"
            + (f" - {body}" if body else " (empty body)")
            + f"\n[TTS] Is breeze-server really the thing listening on {HOST}:{PORT}?"
        )

def _play(stream, data: bytes) -> None:
    """Write audio in small frames, sending a mouth level before each one."""
    for i in range(0, len(data), FRAME_BYTES):
        frame = data[i:i + FRAME_BYTES]
        s = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        if VOLUME < 1.0:
            frame = (s * VOLUME * 32768).astype(np.int16).tobytes()
        stream.write(frame)  # blocks, so this paces the levels to playback

def _speak(text: str, stop_event: MpEvent, gui_q: MpQueue) -> None:
    """Stream PCM from the server and play it as it arrives."""
    start_time = time.time()

    # `files=` with (None, value) forces multipart/form-data, which is what the
    # server's --form-string examples use.
    form = _form(text)

    prebuffer_bytes = int(PREBUFFER_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
    stream = None
    prebuf = bytearray()
    pending = b""
    got_first = False

    def open_stream():
        s = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", latency="low"
        )
        s.start()
        return s

    with _http.post(SPEECH_URL, files=form, stream=True, timeout=(5, 120)) as resp:
        _check(resp)
        try:
            for chunk in resp.iter_content(chunk_size=4096):
                if stop_event.is_set():
                    if stream is not None:
                        stream.abort()
                    return
                if not chunk:
                    continue

                # Keep sample alignment: int16 samples are 2 bytes each.
                pending += chunk
                usable = len(pending) - (len(pending) % BYTES_PER_SAMPLE)
                data, pending = pending[:usable], pending[usable:]
                if not data:
                    continue

                if not got_first:
                    got_first = True
                    elapsed = time.time() - start_time
                    log.info(f"TTS first audio in {elapsed:.2f}s")
                    gui_q.put(TtsTimeMsg(time=elapsed))

                if stream is None:
                    prebuf += data
                    if len(prebuf) < prebuffer_bytes:
                        continue
                    stream = open_stream()
                    _play(stream, bytes(prebuf))
                    prebuf.clear()
                else:
                    _play(stream, data)

            # Short utterance that never reached the prebuffer threshold.
            if stream is None and prebuf:
                stream = open_stream()
                _play(stream, bytes(prebuf))

            if stream is not None:
                stream.stop()  # blocks until queued audio has finished playing
        finally:
            if stream is not None:
                stream.close()



def _tts_process_fn(stop_event: MpEvent, input_q: MpQueue, gui_q: MpQueue):
    config.setup_logging()
    log.info("Process started.")

    if not _wait_for_server(stop_event):
        log.info("Process exiting.")
        return

    # Warm-up so the first real line isn't slow. Audio is discarded.
    try:
        with _http.post(
            SPEECH_URL,
            files=_form("Ready."),
            stream=True,
            timeout=(5, 120),
        ) as r:
            _check(r)
            for _ in r.iter_content(chunk_size=65536):
                pass
    except Exception as e:
        log.warning(f"Warm-up failed: {e}")
        log.info("Process exiting.")
        return

    log.info("Server ready.")

    while not stop_event.is_set():
        try:
            text = input_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            _speak(text, stop_event, gui_q)
        except requests.ConnectionError:
            log.warning("Lost connection to breeze-server, waiting for it to come back...")
            if not _wait_for_server(stop_event):
                break
        except Exception as e:
            log.error(f"Playback/synthesis error: {e}")

    log.info("Process exiting.")
