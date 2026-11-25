import logging
import os
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
SCREENSHOT_DIR = os.path.join(MEMORY_DIR, "screenshots")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

HISTORY_FILE = os.path.join(MEMORY_DIR, "conversation_history.json")
LOG_FILE = os.path.join(MEMORY_DIR, "conversation_log.json")
ASR_WARMUP_WAV = os.path.join(AUDIO_DIR, "asr_warm_up.wav")

# ASR runs on the CPU by default so the GPU stays free for the LLM / TTS.
# Set SHIRA_ASR_DEVICE=cuda to run it on the GPU instead.
ASR_DEVICE = os.environ.get("SHIRA_ASR_DEVICE", "cpu").strip().lower()
ASR_CPU_THREADS = int(os.environ.get("SHIRA_ASR_THREADS", min(8, os.cpu_count() or 1)))

# Names used for {USER_NAME} / {ASSISTANT_NAME} in the prompt files
USER_NAME = os.environ.get("SHIRA_USER_NAME", "User")
ASSISTANT_NAME = os.environ.get("SHIRA_ASSISTANT_NAME", "Shira")

# Local servers (all overridable via environment variables).
# Start them yourself before launching Shira, e.g.:
#   LLM (main chat):          llama-server -m <model.gguf> --port 8080
#   LLM (memory compression): llama-server -m <model.gguf> --port 8081
#   TTS (breeze-server):      breeze-server.exe breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8083
LLM_SERVER_URL = os.environ.get("SHIRA_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
STM_SERVER_URL = os.environ.get("SHIRA_STM_URL", "http://127.0.0.1:8081/v1/chat/completions")
TTS_HOST = os.environ.get("SHIRA_TTS_HOST", "127.0.0.1")
TTS_PORT = int(os.environ.get("SHIRA_TTS_PORT", 8083))
TTS_SPEECH_URL = f"http://{TTS_HOST}:{TTS_PORT}/v1/audio/speech"

# Voice description sent to the TTS server (plain text, e.g. "A calm, deep male voice").
TTS_VOICE_INSTRUCTION = os.environ.get(
    "SHIRA_TTS_VOICE", "A warm, friendly woman with a clear delivery."
)

# LLM behaviour
LLM_MAX_TOKENS = int(os.environ.get("SHIRA_LLM_MAX_TOKENS", 1024))        # max tokens per reply
LLM_REQUEST_TIMEOUT = int(os.environ.get("SHIRA_LLM_TIMEOUT", 60))         # seconds
LLM_COMPRESS_AT_TOKENS = int(os.environ.get("SHIRA_COMPRESS_AT_TOKENS", 60000))  # compress history above this context size
MAX_TOOL_CALLS_PER_TURN = 5       # after this many tool rounds in one turn, tools are disabled
SCREENSHOT_KEEP_TURNS = 7         # how many turns a screenshot stays in the context
STM_MAX_TOKENS = int(os.environ.get("SHIRA_STM_MAX_TOKENS", 5000))         # max tokens for a memory summary
STM_REQUEST_TIMEOUT = int(os.environ.get("SHIRA_STM_TIMEOUT", 300))        # seconds

# Screenshot tool
SCREENSHOT_MAX_SIZE = 1280        # longest side in pixels
SCREENSHOT_JPEG_QUALITY = 80
# Optional free-text hardware notes appended to the screenshot tool description,
# e.g. "left is a 1440p 170Hz gaming screen, right is a 1080p side monitor".
MONITOR_NOTES = os.environ.get("SHIRA_MONITOR_NOTES", "").strip()

# CPU name reported by the system-resources tool (auto-detected unless overridden).
CPU_NAME = os.environ.get("SHIRA_CPU_NAME") or platform.processor() or platform.machine()

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = os.environ.get("SHIRA_LOG_LEVEL", "INFO").strip().upper()


def setup_logging() -> None:
    """Configure logging. Call once at the start of every process (spawned
    processes don't inherit the parent's logging configuration on Windows)."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("{USER_NAME}", USER_NAME).replace("{ASSISTANT_NAME}", ASSISTANT_NAME)
