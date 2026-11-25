import logging
import multiprocessing as mp
import os
import queue
import tempfile
import time
from collections import deque
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent

import nemo.collections.asr as nemo_asr
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import load_silero_vad

import config
from gui.gui_messages import (
    AsrRecordingMsg,
    AsrTimeMsg,
    AsrTranscribingMsg,
    ConversationMsg,
    MicVolumeMsg,
    TranscriptMsg,
)

log = logging.getLogger("shira.asr")


QUEUE_MAXSIZE = 400         # avoid unbounded memory growth
SILENCE_TRANSCRIPT_DURATION = 0.5      # seconds of silence to consider the phrase ended
SILENCE_SENDOFF_DURATION = 2.0      # seconds of silence to consider the phrase ended

audio_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
transcribed_msg = ""



VAD_THRESHOLD_START = 0.8
VAD_THRESHOLD_CONTINUE = 0.25

SAMPLE_RATE = 16000
BLOCKSIZE = 512            # smaller blocks -> lower callback latency


# -----------------
# Utility: RMS energy
# -----------------
def rms(samples: np.ndarray):
    return float(np.sqrt(np.mean(np.square(samples), axis=0).mean()))

#----------------
# Utility: VAD probability
#----------------
def vad_prob(vad_model, block: np.ndarray) -> float:
    """block: exactly 512 samples of mono audio at 16 kHz."""
    x = torch.from_numpy(block).flatten()
    return vad_model(x, 16000).item()

# -----------------
# Audio callback (must be non-blocking)
# -----------------
def audio_callback(indata, frames, time, status):
    # indata is float32 in range [-1, 1] by default
    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        pass

def start_asr_process(stop_event: MpEvent, output_q: MpQueue, confirm_q: MpQueue, gui_q: MpQueue) -> mp.Process:
    """Spawn and return the ASR processing thread."""
    p = mp.Process(
        target=processor_thread_fn,
        args=(stop_event, output_q, confirm_q, gui_q),
        daemon=True,
        name="asr-process",
    )
    p.start()
    return p

# -----------------
# Processing thread
# -----------------
def processor_thread_fn(stop_event: MpEvent, output_q: MpQueue, confirm_q: MpQueue, gui_q: MpQueue):
    config.setup_logging()
    log.info("Process started.")
    global transcribed_msg
    device = config.ASR_DEVICE
    if device == "cpu":
        # Hide the GPU from this process so no CUDA context (or VRAM) is ever
        # created for ASR. Safe here because nothing has initialised CUDA yet
        # in this process (importing torch / nemo does not).
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(config.ASR_CPU_THREADS)
        try:
            torch.set_num_interop_threads(config.ASR_CPU_THREADS)
        except RuntimeError:
            pass  # can only be set once, before any parallel work has started

    # map_location makes sure the checkpoint is loaded straight onto the right
    # device, and .to()/.eval() guarantee the final placement and inference mode.
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2", map_location=device
    )
    asr_model = asr_model.to(device).eval()
    log.info(f"Model loaded on {device}.")

    asr_model.transcribe([config.ASR_WARMUP_WAV], verbose=False)  # warm up the model

    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1

    vad_model = load_silero_vad(onnx=True)   # once, at startup

    buffer_frames = []
    ring_buffer = deque(maxlen=5)  # store last 5 blocks of audio (5 * 512 samples = ~160ms at 16kHz)
    last_voice_time = None
    recording_started = False
    not_yet_recorded_voice = False

    log.info("Model ready.")

    with sd.InputStream(blocksize=BLOCKSIZE, callback=audio_callback):
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.1)  # wait up to 100ms for new audio
            except queue.Empty:
                continue

            # VAD: check energy of the incoming block
            block_rms = rms(block)
            vadprob = vad_prob(vad_model, block)
            gui_q.put(MicVolumeMsg(rms=float(block_rms)))
            # Check if Mic picked up sound -> Start recording
            if (not recording_started and vadprob >= VAD_THRESHOLD_START) or (recording_started and vadprob >= VAD_THRESHOLD_CONTINUE):
                recording_started = True
                not_yet_recorded_voice = True
                last_voice_time = time.time()
                if len(buffer_frames) == 0:
                    gui_q.put(AsrRecordingMsg(recording=True))
                    buffer_frames = list(ring_buffer)  # start with the last few blocks to avoid cutting off the start of speech

            # If Recording has started, keep adding blocks to the buffer
            if recording_started:
                buffer_frames.append(block)
                recording_duration = sum(len(b) for b in buffer_frames) / SAMPLE_RATE
                last_voice = time.time() - last_voice_time  # update last voice time on each block with sound

                # If we have not transcribed anything yet and short silence is reached -> transcribe and send
                if transcribed_msg == "" and last_voice >= SILENCE_TRANSCRIPT_DURATION:
                    log.debug(f"Transcribing phrase (duration: {recording_duration:.2f}s, silence: {time.time() - last_voice_time:.2f}s)")
                    do_transcribe(buffer_frames, asr_model, output_q, gui_q)
                    not_yet_recorded_voice = False
                    if transcribed_msg == "":
                        log.info("No transcribed message received, resetting buffer.")
                        buffer_frames, last_voice_time, recording_started = [], None, False
                        vad_model.reset_states()
                        gui_q.put(AsrRecordingMsg(recording=False))

                # If we have transcribed something already, silence duration is between threshold and sendoff, and we have new audio -> transcribe again and send
                elif transcribed_msg != "" and last_voice >= SILENCE_TRANSCRIPT_DURATION and last_voice < SILENCE_SENDOFF_DURATION and not_yet_recorded_voice:
                    log.debug(f"Transcribing phrase again (duration: {recording_duration:.2f}s, silence: {time.time() - last_voice_time:.2f}s)")
                    do_transcribe(buffer_frames, asr_model, output_q, gui_q)
                    not_yet_recorded_voice = False

                # If long silence is reached -> confirm send and reset state
                elif last_voice >= SILENCE_SENDOFF_DURATION and transcribed_msg != "" and audio_q.qsize() == 0:
                    gui_q.put(ConversationMsg(role="user", content=transcribed_msg))
                    confirm_q.put("send")
                    log.info("Long silence reached, sending CONFIRM")
                    gui_q.put(TranscriptMsg(text=transcribed_msg))
                    transcribed_msg, buffer_frames, last_voice_time, recording_started = "", [], None, False
                    vad_model.reset_states()
                    gui_q.put(AsrRecordingMsg(recording=False))

            ring_buffer.append(block)  # keep a rolling buffer of the last few blocks

        log.info("Process exiting.")


# -----------------
# Transcribe helper
# -----------------
def do_transcribe(frames_list, asr_model, output_q: MpQueue, gui_q, sample_rate=SAMPLE_RATE):
    global transcribed_msg
    start_time = time.time()
    gui_q.put(AsrTranscribingMsg(transcribing=True))

    # join numpy frames into single 1-D array
    audio = np.concatenate(frames_list, axis=0)

    # optional: normalize to -1..1 if needed
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / max(1.0, peak)

    # write to a temporary wav file, then call NeMo's transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        sf.write(tmp_name, audio, sample_rate, subtype='PCM_16')   # PCM_16 is typical for ASR
        # run transcription (use no_grad to avoid autograd overhead)
        with torch.no_grad():
            raw_output = asr_model.transcribe([tmp_name], verbose=False)
        if raw_output:
            first = raw_output[0]
            text = getattr(first, "text", first) if first is not None else ""
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            text = text.strip() if isinstance(text, str) else str(text)
        else:
            text = ""

        # If the recording had proper audio
        if text:
            gui_q.put(AsrTimeMsg(time=time.time() - start_time))
            # And we haven't send anything yet for processing
            transcribed_msg = text
            log.debug(f"Sending transcribed message: {transcribed_msg}")
            output_q.put(transcribed_msg)


    except Exception as e:
        log.error(f"Error transcribing: {e}")
    finally:
        gui_q.put(AsrTranscribingMsg(transcribing=False))
        try:
            os.remove(tmp_name)
        except Exception:
            pass
