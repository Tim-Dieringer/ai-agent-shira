import json
import sys
import types

import pytest

import tools


def _fake_mss(monitors, size=(40, 20)):
    """Build a stand-in for the `mss` module that 'captures' a solid image."""

    class FakeShot:
        def __init__(self):
            self.size = size
            self.bgra = bytes([30, 60, 90, 255]) * (size[0] * size[1])

    class FakeSct:
        def __init__(self):
            self.monitors = monitors

        def grab(self, mon):
            return FakeShot()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return types.SimpleNamespace(mss=lambda: FakeSct())


# ── tool schema ───────────────────────────────────────────────────────────

def test_tool_schemas_are_well_formed():
    names = []
    for tool in tools.TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        required = fn["parameters"].get("required", [])
        assert set(required) <= set(fn["parameters"].get("properties", {}))
        names.append(fn["name"])
    assert len(names) == len(set(names)), "duplicate tool names"


def test_every_declared_tool_is_dispatchable(monkeypatch):
    monkeypatch.setattr(tools, "_get_motherboard", lambda: None)
    monkeypatch.setattr(tools, "_get_gpu_info", lambda: {"error": "No GPU found"})
    for tool in tools.TOOLS:
        result = json.loads(tools.dispatch(tool["function"]["name"], {}))
        assert "Unknown tool" not in result.get("error", "")


def test_unknown_tool_returns_error():
    result = json.loads(tools.dispatch("does_not_exist", {}))
    assert result == {"error": "Unknown tool: does_not_exist"}


# ── get_time / get_system_resources ───────────────────────────────────────

def test_get_time_fields():
    data = json.loads(tools.get_time())
    assert set(data) == {"current_time", "session_start", "session_duration_minutes"}
    assert data["session_duration_minutes"] >= 0


def test_get_system_resources_without_gpu(monkeypatch):
    monkeypatch.setattr(tools, "_get_gpu_info", lambda: {"error": "No GPU found"})
    data = json.loads(tools.get_system_resources())
    assert data["gpu"] == {"error": "No GPU found"}
    assert 0 <= data["cpu"]["usage_percent"] <= 100
    assert data["ram"]["total_gb"] > 0


def test_get_gpu_info_when_nvml_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` raise ImportError
    assert "error" in tools._get_gpu_info()


def test_get_gpu_info_reads_nvml(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    fake = types.SimpleNamespace(
        NVMLError=FakeNVMLError,
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetName=lambda h: b"Test GPU",
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=2 * 1024 ** 2, total=8 * 1024 ** 2),
        nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=42),
        nvmlDeviceGetTemperature=lambda h, sensor: 55,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert tools._get_gpu_info() == {
        "name": "Test GPU",
        "load_percent": 42.0,
        "memory_used_mb": 2,
        "memory_total_mb": 8,
        "temperature_c": 55,
    }


def test_get_gpu_info_reports_nvml_init_failure(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    def boom():
        raise FakeNVMLError("driver not loaded")

    fake = types.SimpleNamespace(NVMLError=FakeNVMLError, nvmlInit=boom)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert "driver not loaded" in tools._get_gpu_info()["error"]


# ── RGB ───────────────────────────────────────────────────────────────────

def test_get_rgb_color_without_openrgb(monkeypatch):
    monkeypatch.setattr(tools, "_get_motherboard", lambda: None)
    assert "error" in json.loads(tools.get_rgb_color())


def test_get_rgb_color_reads_first_zone(monkeypatch):
    color = types.SimpleNamespace(red=1, green=2, blue=3)
    board = types.SimpleNamespace(colors=[color])
    monkeypatch.setattr(tools, "_get_motherboard", lambda: board)
    assert json.loads(tools.get_rgb_color()) == {"r": 1, "g": 2, "b": 3}


def test_set_rgb_color_success(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "set_rgb_color", lambda r, g, b: calls.append((r, g, b)) or True)
    result = json.loads(tools.dispatch("set_rgb_color", {"r": 255, "g": "128", "b": 0}))
    assert result == {"result": "success"}
    assert calls == [(255, 128, 0)]


def test_set_rgb_color_reports_missing_server(monkeypatch):
    monkeypatch.setattr(tools, "set_rgb_color", lambda r, g, b: False)
    result = json.loads(tools.dispatch("set_rgb_color", {"r": 1, "g": 2, "b": 3}))
    assert "error" in result


@pytest.mark.parametrize("bad", [-1, 256, 1000])
def test_set_rgb_color_rejects_out_of_range(monkeypatch, bad):
    monkeypatch.setattr(tools, "set_rgb_color", lambda r, g, b: pytest.fail("must not be called"))
    result = json.loads(tools.dispatch("set_rgb_color", {"r": bad, "g": 0, "b": 0}))
    assert result == {"error": "RGB values must be between 0-255."}


@pytest.mark.parametrize("args", [{}, {"r": "red", "g": 0, "b": 0}, {"r": None, "g": 0, "b": 0}])
def test_set_rgb_color_rejects_non_integers(args):
    result = json.loads(tools.dispatch("set_rgb_color", args))
    assert result == {"error": "RGB values must be integers."}


# ── screenshots ───────────────────────────────────────────────────────────

MONITORS = [
    {"left": 0, "top": 0, "width": 5760, "height": 1080},   # [0] combined box
    {"left": 1920, "top": 0, "width": 1920, "height": 1080},
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},
    {"left": 0, "top": 0, "width": 1920, "height": 1080},
]


@pytest.mark.parametrize("which, expected_left", [("left", 0), ("center", 1920), ("right", 3840)])
def test_resolve_monitor_orders_by_x_position(which, expected_left):
    sct = types.SimpleNamespace(monitors=MONITORS)
    assert tools._resolve_monitor(sct, which)["left"] == expected_left


def test_resolve_monitor_with_fewer_than_three_screens():
    sct = types.SimpleNamespace(monitors=MONITORS[:2])  # combined box + one monitor
    for which in ("left", "center", "right"):
        assert tools._resolve_monitor(sct, which)["left"] == 1920


def test_resolve_monitor_with_two_screens_picks_nearest():
    sct = types.SimpleNamespace(monitors=[MONITORS[0], MONITORS[3], MONITORS[1]])
    assert tools._resolve_monitor(sct, "left")["left"] == 0
    assert tools._resolve_monitor(sct, "center")["left"] == 0
    assert tools._resolve_monitor(sct, "right")["left"] == 1920


def test_resolve_monitor_without_monitors():
    with pytest.raises(RuntimeError):
        tools._resolve_monitor(types.SimpleNamespace(monitors=[MONITORS[0]]), "left")


def test_look_at_screen_saves_image_and_queues_it(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "mss", _fake_mss(MONITORS))
    monkeypatch.setattr(tools, "SCREENSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(tools, "_pending_images", [])

    result = json.loads(tools.look_at_screen(" Left "))
    assert "result" in result

    files = list(tmp_path.glob("*_left.jpg"))
    assert len(files) == 1
    assert tools.take_pending_images() == [str(files[0])]
    assert tools.take_pending_images() == []   # queue is drained


def test_look_at_screen_rejects_invalid_argument(monkeypatch):
    monkeypatch.setattr(tools, "_pending_images", [])
    result = json.loads(tools.look_at_screen("up"))
    assert "error" in result
    assert tools.take_pending_images() == []
