from dataclasses import dataclass
from typing import Literal


@dataclass
class MicVolumeMsg:
    type: Literal["mic_volume"] = "mic_volume"
    rms: float = 0.0

@dataclass
class AsrTimeMsg:
    type: Literal["asr_time"] = "asr_time"
    time: float = 0.0

@dataclass
class TtsTimeMsg:
    type: Literal["tts_time"] = "tts_time"
    time: float = 0.0

@dataclass
class LlmTimeMsg:
    type: Literal["llm_time"] = "llm_time"
    time: float = 0.0

@dataclass
class LlmTokenCountMsg:
    type: Literal["llm_token_count"] = "llm_token_count"
    count: int = 0

@dataclass
class LlmTokenPerSecondMsg:
    type: Literal["llm_tokens_per_second"] = "llm_tokens_per_second"
    tokens_per_second: float = 0.0

@dataclass
class AsrRecordingMsg:
    type: Literal["asr_recording"] = "asr_recording"
    recording: bool = False

@dataclass
class AsrTranscribingMsg:
    type: Literal["asr_transcribing"] = "asr_transcribing"
    transcribing: bool = False

@dataclass
class TranscriptMsg:
    type: Literal["transcript"] = "transcript"
    text: str = ""

@dataclass
class LlmStateMsg:
    type: Literal["llm_state"] = "llm_state"
    thinking: bool = False

@dataclass
class ConversationMsg:
    type: Literal["conversation"] = "conversation"
    role: str = ""        # "user" or "assistant" or "tool"
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list | None = None
