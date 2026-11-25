import base64
import json

import pytest

import llm

# ── helpers ───────────────────────────────────────────────────────────────

class FakeQueue:
    """Minimal stand-in for multiprocessing.Queue (only qsize() is used)."""

    def __init__(self, size=0):
        self._size = size

    def qsize(self):
        return self._size


class FakeResponse:
    """Stand-in for a streaming `requests` response."""

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


def sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode()


def delta(**fields) -> bytes:
    return sse({"choices": [{"delta": fields}]})


def make_image(tmp_path, name="20250101_120000_left.jpg", data=b"fake-jpeg-bytes"):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# ── small helpers ─────────────────────────────────────────────────────────

def test_screen_name_from_screenshot_path():
    assert llm._screen_name("/x/memory/screenshots/20250101_120000_left.jpg") == "left"
    assert llm._screen_name("20250101_120000_center.jpg") == "center"


def test_content_to_text():
    assert llm._content_to_text("hello") == "hello"
    assert llm._content_to_text(None) == ""
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_ref", "path": "x.jpg", "keep_turns": 3},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    assert llm._content_to_text(parts) == "look [screenshot] [screenshot]"


def test_strip_images_returns_text_only_copy():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "shot"}, {"type": "image_ref", "path": "p", "keep_turns": 1}]},
    ]
    stripped = llm._strip_images(history)
    assert stripped[0] == history[0]
    assert stripped[1]["content"] == "shot [screenshot]"
    assert isinstance(history[1]["content"], list), "input must not be modified"


# ── image handling ────────────────────────────────────────────────────────

def _image_msg(path, keep_turns=2):
    return {"role": "user", "content": [
        {"type": "text", "text": "[Screenshot]"},
        {"type": "image_ref", "path": path, "keep_turns": keep_turns},
    ]}


def test_materialize_leaves_plain_messages_alone():
    msg = {"role": "user", "content": "hi"}
    assert llm._materialize(msg, 5) is msg


def test_materialize_embeds_fresh_image_as_base64(tmp_path):
    path = make_image(tmp_path, data=b"abc")
    result = llm._materialize(_image_msg(path, keep_turns=2), turns_after=2)
    image_part = result["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"abc").decode()


def test_materialize_expires_old_images(tmp_path):
    path = make_image(tmp_path)
    result = llm._materialize(_image_msg(path, keep_turns=2), turns_after=3)
    assert result["content"][1] == {"type": "text", "text": "[screenshot outdated]"}


def test_materialize_handles_deleted_file(tmp_path):
    result = llm._materialize(_image_msg(str(tmp_path / "gone.jpg")), turns_after=0)
    assert result["content"][1]["text"] == "[screenshot outdated]"


def test_build_messages_prepends_system_prompt_and_keeps_order(monkeypatch):
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    monkeypatch.setattr(llm, "_conversation_history", history)
    messages = llm.build_messages()
    assert messages[0] == {"role": "system", "content": llm.STATIC_SYSTEM_PROMPT}
    assert [m["content"] for m in messages[1:]] == ["one", "two"]


@pytest.mark.parametrize("keep_turns, expect_image", [(2, True), (1, False)])
def test_build_messages_expires_screenshots_by_user_turns(monkeypatch, tmp_path, keep_turns, expect_image):
    path = make_image(tmp_path)
    history = [
        {"role": "user", "content": "look at my screen"},
        _image_msg(path, keep_turns=keep_turns),
        {"role": "assistant", "content": "I see code"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "welcome"},
        {"role": "user", "content": "next question"},
    ]
    monkeypatch.setattr(llm, "_conversation_history", history)
    image_msg = llm.build_messages()[2]   # [system, user, screenshot, ...]
    kind = image_msg["content"][1]["type"]
    assert kind == ("image_url" if expect_image else "text")
    assert history[1]["content"][1]["type"] == "image_ref", "history must keep the reference"


# ── streaming parser ──────────────────────────────────────────────────────

def test_parse_llm_response_collects_text_reasoning_and_counts():
    lines = [
        b": keep-alive",
        delta(reasoning_content="hmm "),
        delta(content="Hel"),
        delta(content="lo"),
        sse({"choices": []}),                                   # ignored
        sse({"choices": [{"delta": {}}], "timings": {"predicted_n": 12, "n_past": 345}}),
        b"data: [DONE]",
    ]
    reply, reasoning, completion, context, tool_calls = llm.parse_llm_response(
        FakeResponse(lines), FakeQueue(0), FakeQueue(0)
    )
    assert reply == "Hello"
    assert reasoning == "hmm "
    assert (completion, context) == (12, 345)
    assert tool_calls == []


def test_parse_llm_response_reads_openai_style_usage():
    lines = [delta(content="x"), sse({"choices": [{"delta": {}}], "usage": {"completion_tokens": 3, "total_tokens": 99}})]
    _, _, completion, context, _ = llm.parse_llm_response(FakeResponse(lines), FakeQueue(), FakeQueue())
    assert (completion, context) == (3, 99)


def test_parse_llm_response_assembles_streamed_tool_calls():
    lines = [
        delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "set_rgb_", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "color", "arguments": '{"r": 1,'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": ' "g": 2, "b": 3}'}}]),
        delta(tool_calls=[{"index": 1, "id": "call_2", "function": {"name": "get_time", "arguments": "{}"}}]),
    ]
    *_, tool_calls = llm.parse_llm_response(FakeResponse(lines), FakeQueue(), FakeQueue())
    assert [tc["function"]["name"] for tc in tool_calls] == ["set_rgb_color", "get_time"]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"r": 1, "g": 2, "b": 3}
    assert tool_calls[0]["id"] == "call_1"


def test_parse_llm_response_aborts_when_new_input_arrives():
    response = FakeResponse([delta(content="never read")])
    result = llm.parse_llm_response(response, FakeQueue(size=1), FakeQueue(size=0))
    assert result == ("", "", 0, 0, [])
    assert response.closed


def test_parse_llm_response_does_not_abort_while_waiting_for_confirmation():
    # A pending confirmation means the new input is the same utterance, not an interruption.
    response = FakeResponse([delta(content="ok")])
    reply, *_ = llm.parse_llm_response(response, FakeQueue(size=1), FakeQueue(size=1))
    assert reply == "ok"


# ── persistence ───────────────────────────────────────────────────────────

def test_append_message_to_log_keeps_a_valid_json_array(monkeypatch, tmp_path):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(llm, "LOG_FILE", str(log_file))
    messages = [
        {"role": "user", "content": "Hallo, wie geht's? ✓"},
        {"role": "assistant", "content": "Gut."},
        {"role": "tool", "content": '{"result": "ok"}'},
    ]
    for message in messages:
        llm.append_message_to_log(message)
        json.loads(log_file.read_text(encoding="utf-8"))   # valid after every append
    assert json.loads(log_file.read_text(encoding="utf-8")) == messages


def test_history_round_trip(monkeypatch, tmp_path):
    history_file = tmp_path / "history.json"
    monkeypatch.setattr(llm, "HISTORY_FILE", str(history_file))
    history = [{"role": "user", "content": "grüß dich"}, {"role": "assistant", "content": "hi"}]

    monkeypatch.setattr(llm, "_conversation_history", history)
    llm.save_conversation_history()

    monkeypatch.setattr(llm, "_conversation_history", [])
    llm.load_conversation_history()
    assert llm._conversation_history == history


def test_load_history_survives_missing_and_corrupt_files(monkeypatch, tmp_path):
    history_file = tmp_path / "history.json"
    monkeypatch.setattr(llm, "HISTORY_FILE", str(history_file))

    monkeypatch.setattr(llm, "_conversation_history", [])
    llm.load_conversation_history()                 # file does not exist
    assert llm._conversation_history == []

    history_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(llm, "_conversation_history", [{"role": "user", "content": "stale"}])
    llm.load_conversation_history()                 # corrupt file -> start fresh
    assert llm._conversation_history == []
