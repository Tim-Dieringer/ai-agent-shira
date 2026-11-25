import logging
import multiprocessing as mp
import sys
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent

from PySide6.QtWidgets import QApplication

import asr
import config
import llm
import tts
from gui.gui import MainWindow

log = logging.getLogger("shira.main")

# -----------------
# Main
# -----------------
def main(stop_event: MpEvent, asr_llm_q: MpQueue, asr_llm_confirm_q: MpQueue, llm_tts_q: MpQueue, gui_q: MpQueue):
    config.setup_logging()
    log.info("Starting LLM process and loading model...")
    llm_process = llm.start_llm_process(stop_event, asr_llm_q, asr_llm_confirm_q, llm_tts_q, gui_q)
    log.info(f"LLM process started with PID {llm_process.pid}")

    log.info("Starting ASR process")
    asr_process = asr.start_asr_process(stop_event, asr_llm_q, asr_llm_confirm_q, gui_q)
    log.info(f"ASR process started with PID {asr_process.pid}")

    log.info("Starting TTS process")
    tts_process = tts.start_tts_process(stop_event, llm_tts_q, gui_q)
    log.info(f"TTS process started with PID {tts_process.pid}")

    app = QApplication(sys.argv)
    window = MainWindow(gui_q)
    window.show()
    app.exec()          # blocks until window is closed

    # Cleanup runs here after the window closes
    stop_event.set()
    asr_process.join(timeout=3.0)
    tts_process.join(timeout=5.0)
    llm_process.join(timeout=5.0)
    log.info("Exited cleanly")

if __name__ == "__main__":
    stop_event = mp.Event()
    asr_llm_q = mp.Queue()
    asr_llm_confirm_q = mp.Queue()
    llm_tts_q = mp.Queue()
    gui_q = mp.Queue()
    main(stop_event, asr_llm_q, asr_llm_confirm_q, llm_tts_q, gui_q)
