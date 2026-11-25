import logging
import queue

from PySide6.QtCore import QObject, QTimer, Signal

log = logging.getLogger("shira.gui")


class GuiBridge(QObject):
    mic_volume   = Signal(float)          # rms 0.0–1.0
    asr_recording = Signal(bool)          # recording
    asr_transcribing = Signal(bool)       # transcribing
    transcript   = Signal(str)            # raw transcript text
    llm_state    = Signal(bool)           # thinking
    conversation = Signal(str, str, str, list)  # role, content, reasoning_content, tool_calls
    asr_stats_updated = Signal(float, float, float)  # avg, min, max
    tts_stats_updated = Signal(float, float, float)  # avg, min, max
    llm_stats_updated = Signal(float, float, float)  # avg, min, max
    llm_token_updated = Signal(float, float, float)  # avg, min, max tokens
    llm_speed_updated = Signal(float, float, float)  # avg, min, max tokens per second


    def __init__(self, gui_q, parent=None):
        super().__init__(parent)
        self._q = gui_q
        self._timer = QTimer(self)
        self._timer.setInterval(40)       # 40 ms ≈ 25 fps is plenty
        self._timer.timeout.connect(self._poll)
        self._timer.start()

        self._asr_stats = StatsTracker()
        self._tts_stats = StatsTracker()
        self._llm_stats = StatsTracker()
        self._llm_token_stats = StatsTracker()
        self._llm_speed_stats = StatsTracker()


    def _poll(self):
        try:
            while True:                   # drain all pending messages
                msg = self._q.get_nowait()
                self._dispatch(msg)
        except queue.Empty:
            pass

    def _dispatch(self, msg):
        t = msg.type
        if t == "mic_volume":
            self.mic_volume.emit(msg.rms)
        elif t == "asr_recording":
            self.asr_recording.emit(msg.recording)
        elif t == "asr_transcribing":
            self.asr_transcribing.emit(msg.transcribing)
        elif t == "transcript":
            self.transcript.emit(msg.text)
        elif t == "llm_state":
            self.llm_state.emit(msg.thinking)
        elif t == "conversation":
            if msg.role == "assistant":
                if msg.tool_calls:
                    self.conversation.emit(msg.role, msg.content, msg.reasoning_content, msg.tool_calls)
                else:
                    self.conversation.emit(msg.role, msg.content, msg.reasoning_content, [])
            else:
                self.conversation.emit(msg.role, msg.content, "", [])
        elif t == "asr_time":
            self._asr_stats.add(msg.time)
            self.asr_stats_updated.emit(self._asr_stats.avg, self._asr_stats.min_time, self._asr_stats.max_time)
        elif t == "tts_time":
            self._tts_stats.add(msg.time)
            self.tts_stats_updated.emit(self._tts_stats.avg, self._tts_stats.min_time, self._tts_stats.max_time)
        elif t == "llm_time":
            self._llm_stats.add(msg.time)
            self.llm_stats_updated.emit(self._llm_stats.avg, self._llm_stats.min_time, self._llm_stats.max_time)
        elif t == "llm_token_count":
            self._llm_token_stats.add(msg.count)
            self.llm_token_updated.emit(self._llm_token_stats.avg, self._llm_token_stats.min_time, self._llm_token_stats.max_time)
        elif t == "llm_tokens_per_second":
            log.debug(f"Received LLM speed: {msg.tokens_per_second:.2f} tokens/s")
            self._llm_speed_stats.add(msg.tokens_per_second)
            self.llm_speed_updated.emit(self._llm_speed_stats.avg, self._llm_speed_stats.min_time, self._llm_speed_stats.max_time)
        else:
            log.warning(f"Unknown message type: {t}")



class StatsTracker:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0

    def add(self, t: float):
        self.count += 1
        self.total += t
        self.min_time = min(self.min_time, t)
        self.max_time = max(self.max_time, t)

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0
