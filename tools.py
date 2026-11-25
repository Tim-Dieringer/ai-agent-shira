import json
import logging
import os
import time

import mss
import psutil
from PIL import Image

import config

log = logging.getLogger("shira.tools")

SCREENSHOT_DIR = config.SCREENSHOT_DIR
_pending_images: list[str] = []

# ─────────────────────────────────────────────
#  Tool definitions
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": (
                "Returns the current date and time, and the time you woke up this session. "
                f"Call this if {config.USER_NAME} asks what time it is, how long you've been talking, or anything time-related. "
                "Alternatively, call this if you want to know the current time, or how long you've been awake this session."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_resources",
            "description": (
                "Returns current CPU, GPU, and RAM usage. "
                f"Call this if {config.USER_NAME} asks about system performance, whether the PC is struggling, "
                "temperatures, or anything hardware-related. "
                f"Alternatively, call this if you want to know how {config.USER_NAME}'s PC is performing."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_rgb_color",
            "description": (
                f"Returns the current RGB color of {config.USER_NAME}'s Computer. "
                f"Call this when {config.USER_NAME} wants to know the current lighting color of their PC. "
                f"Also call this to know how to modify the color if {config.USER_NAME} wants you to simply dim or brighten the current color."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_rgb_color",
            "description": (
                f"Sets the RGB color of {config.USER_NAME}'s Computer. "
                f"Call this when {config.USER_NAME} wants to change the lighting of their PC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "integer",
                        "description": "Red component of the color (0-255)"
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green component of the color (0-255)"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue component of the color (0-255)"
                    }
                },
                "required": ["r", "g", "b"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "look_at_screen",
        "description": (
            f"Take a screenshot of one of {config.USER_NAME}'s monitors. "
            "'left' is the leftmost monitor, 'right' is the rightmost and 'center' is the one in the middle. "
            "With fewer than three monitors, the nearest available one is used."
            + (f" {config.MONITOR_NOTES}" if config.MONITOR_NOTES else "")
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "which": {
                    "type": "string",
                    "enum": ["left", "center", "right"],
                    "description": "Which screen to capture."
                }
            },
            "required": ["which"]
        }
    }
}
]


# ─────────────────────────────────────────────
#  Tool implementations
# ─────────────────────────────────────────────

_session_start = time.time()
_motherboard = None
_rgb_init_attempted = False


def _get_motherboard():
    """Connect to the OpenRGB server on first use. Returns None if RGB control
    is unavailable (package not installed, server not running, no motherboard)."""
    global _motherboard, _rgb_init_attempted
    if _rgb_init_attempted:
        return _motherboard
    _rgb_init_attempted = True
    try:
        from openrgb import OpenRGBClient
        from openrgb.utils import DeviceType

        client = OpenRGBClient()
        _motherboard = client.get_devices_by_type(DeviceType.MOTHERBOARD)[0]
    except Exception as e:
        log.warning(f"RGB control unavailable (is the OpenRGB server running?): {e}")
    return _motherboard


def set_rgb_color(r: int, g: int, b: int) -> bool:
    motherboard = _get_motherboard()
    if motherboard is None:
        return False
    from openrgb.utils import RGBColor

    motherboard.set_color(RGBColor(r, g, b))
    return True


def get_rgb_color() -> str:
    motherboard = _get_motherboard()
    if motherboard:
        color = motherboard.colors[0]  # Assuming one zone for simplicity
        return json.dumps({"r": color.red, "g": color.green, "b": color.blue})
    else:
        return json.dumps({"error": "RGB client not initialized"})


def get_time() -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    wake_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_session_start))
    elapsed_seconds = int(time.time() - _session_start)
    elapsed_minutes = elapsed_seconds // 60

    return json.dumps({
        "current_time": now,
        "session_start": wake_time,
        "session_duration_minutes": elapsed_minutes
    })


def _get_gpu_info() -> dict:
    """Query the first NVIDIA GPU through NVML (the `nvidia-ml-py` package).
    Returns an {"error": ...} dict instead of raising when no GPU is available."""
    try:
        import pynvml
    except ImportError:
        return {"error": "nvidia-ml-py is not installed"}

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        return {"error": f"No NVIDIA GPU or driver found ({e})"}

    try:
        if pynvml.nvmlDeviceGetCount() == 0:
            return {"error": "No GPU found"}
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):  # older nvidia-ml-py versions return bytes
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": name,
            "load_percent": float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            "memory_used_mb": round(mem.used / (1024 ** 2)),
            "memory_total_mb": round(mem.total / (1024 ** 2)),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_system_resources() -> str:
    gpu_info = _get_gpu_info()

    mem = psutil.virtual_memory()

    return json.dumps({
        "cpu": {
            "name": config.CPU_NAME,
            "usage_percent": psutil.cpu_percent(interval=0.2)
        },
        "gpu": gpu_info,
        "ram": {
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "percent": mem.percent
        }
    })


def look_at_screen(which: str) -> str:
    try:
        which = which.strip().lower()
        if which not in ("left", "center", "right"):
            raise ValueError("which must be 'left', 'center', or 'right'")

        with mss.mss() as sct:  # per-call; mss objects aren't thread-safe
            mon = _resolve_monitor(sct, which)
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        img.thumbnail((config.SCREENSHOT_MAX_SIZE, config.SCREENSHOT_MAX_SIZE))
        path = os.path.join(SCREENSHOT_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{which}.jpg")
        img.save(path, "JPEG", quality=config.SCREENSHOT_JPEG_QUALITY)
        _pending_images.append(path)
        return json.dumps({"result": f"Screenshot of the {which} screen captured. It is attached in the next message."})
    except Exception as e:
        return json.dumps({"error": str(e)})

def _resolve_monitor(sct, which: str) -> dict:
    """Map 'left' / 'center' / 'right' to a physical monitor, ordered by x position."""
    mons = sorted(sct.monitors[1:], key=lambda m: m["left"])  # skip [0] = combined box
    if not mons:
        raise RuntimeError("No monitors found")
    index = {"left": 0, "center": (len(mons) - 1) // 2, "right": len(mons) - 1}[which]
    return mons[index]

def take_pending_images() -> list[str]:
    global _pending_images
    imgs, _pending_images = _pending_images, []
    return imgs


# ─────────────────────────────────────────────
#  Dispatcher — call this from your main file
# ─────────────────────────────────────────────

def dispatch(tool_name: str, arguments: dict) -> str:
    """
    Called by your inference loop when the model requests a tool.
    Returns a string result to inject as the tool role message.
    """
    if tool_name == "get_time":
        return get_time()

    elif tool_name == "get_system_resources":
        return get_system_resources()

    elif tool_name == "get_rgb_color":
        return get_rgb_color()

    elif tool_name == "set_rgb_color":
        try:
            r = int(arguments.get("r"))
            g = int(arguments.get("g"))
            b = int(arguments.get("b"))

            if all(0 <= x <= 255 for x in (r, g, b)):
                if not set_rgb_color(r, g, b):
                    return json.dumps({"error": "RGB client not initialized"})
                return json.dumps({"result": "success"})
            else:
                return json.dumps({"error": "RGB values must be between 0-255."})

        except (TypeError, ValueError):
            return json.dumps({"error": "RGB values must be integers."})

    elif tool_name == "look_at_screen":
        return look_at_screen(arguments.get("which", ""))

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
