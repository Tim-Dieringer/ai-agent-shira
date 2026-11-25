import base64
import json
import logging
import multiprocessing as mp
import os
import queue
import time
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent

import requests

import config
import short_term_memory
import tools
from gui.gui_messages import ConversationMsg, LlmStateMsg, LlmTimeMsg, LlmTokenCountMsg, LlmTokenPerSecondMsg

log = logging.getLogger("shira.llm")

LLM_SERVER_URL = config.LLM_SERVER_URL
HISTORY_FILE = config.HISTORY_FILE
LOG_FILE = config.LOG_FILE
SCREENSHOT_KEEP_TURNS = config.SCREENSHOT_KEEP_TURNS

STATIC_SYSTEM_PROMPT = config.load_prompt("system_prompt")
_conversation_history: list[dict] = []

def _screen_name(path: str) -> str:
    """Recover 'left' / 'center' / 'right' from a screenshot filename like 20250101_120000_left.jpg."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]


# Function called from the main process to spawn the LLM background process
def start_llm_process(stop_event: MpEvent, input_q: MpQueue, confirm_q: MpQueue, output_q: MpQueue, gui_q: MpQueue) -> mp.Process:
    """Spawn and return the LLM background process."""
    p = mp.Process(
        target=_llm_process_fn,
        args=(stop_event, input_q, confirm_q, output_q, gui_q),
        daemon=False,
        name="llm-process",
    )
    p.start()
    return p


# The LLM thread, handling start up, sending prompts to inference, and managing conversation history and short-term memory compression.
def _llm_process_fn(stop_event: MpEvent, input_q: MpQueue, confirm_q: MpQueue, output_q: MpQueue, gui_q: MpQueue):
    global _conversation_history
    config.setup_logging()
    stm_request_q = mp.Queue()   # LLM process -> STM process: history to compress
    stm_result_q = mp.Queue()    # STM process -> LLM process: compressed summary (or None on failure)
    stm_process = short_term_memory.start_stm_process(stop_event, stm_request_q, stm_result_q)
    conversation_index = 0

    session = requests.Session()
    load_conversation_history()
    result = json.dumps({"result": f"You awaken at {time.strftime('%Y-%m-%d %H:%M:%S')} after {config.USER_NAME} started your program."})
    _conversation_history.append({"role": "user", "content": result})

    for msg in _conversation_history:
        role = msg.get("role")
        content = _content_to_text(msg.get("content"))
        reasoning_content = msg.get("reasoning_content", "")
        tool_calls = msg.get("tool_calls", [])
        gui_q.put(ConversationMsg(role=role, content=content, reasoning_content=reasoning_content, tool_calls=tool_calls))


    while not stop_event.is_set():
        try:
            prompt = input_q.get(timeout=0.5)
        except queue.Empty:
            continue

        gui_q.put(LlmStateMsg(thinking=True))
        try:
            total_tokens = _run_inference(prompt, input_q, output_q, confirm_q, gui_q, session, stop_event)
        except Exception as e:
            log.error(f"Error during inference: {e}")
            total_tokens = 0
        gui_q.put(LlmStateMsg(thinking=False))

        log.debug(f"Conversation index: {conversation_index}, total tokens: {total_tokens}")

        if total_tokens > config.LLM_COMPRESS_AT_TOKENS and conversation_index == 0:
            conversation_index = len(_conversation_history) // 2
            log.info(f"Conversation history exceeded token limit, sending to STM for compression starting from index {conversation_index}...")
            stm_request_q.put(_strip_images(_conversation_history[:conversation_index]))
            log.debug(f"Sent conversation history to STM for compression. History length: {len(_conversation_history)} messages, starting from index {conversation_index}")

        try:
            compressed_memory = stm_result_q.get_nowait()
            if compressed_memory is None:
                # The STM server failed; keep the full history and retry on a later turn
                log.warning("Short-term memory compression failed, keeping full history.")
            else:
                log.debug(f"Received compressed memory: {compressed_memory}")
                # Replace the older half of the conversation history with the compressed version
                _conversation_history = [{"role": "tool", "content": "<short_term_memory>\n" + compressed_memory + "\n</short_term_memory>\n"}] + _conversation_history[conversation_index:]
            conversation_index = 0  # Reset index after compression attempt
        except queue.Empty:
            pass

    stm_process.join(timeout=3.0)
    session.close()
    log.info("Storing conversation history before exit...")
    save_conversation_history()
    log.info("Process exited.")


# Run inference with the LLM, handling streaming responses and tool calls.
def _run_inference(prompt: str, input_q: MpQueue, output_q: MpQueue, confirm_q: MpQueue, gui_q: MpQueue, session: requests.Session, stop_event: MpEvent) -> int:
    start_time = time.time()
    tool_counter = 0

    conversation_start_index = len(_conversation_history)
    _conversation_history.append({"role": "user", "content": prompt})

    while True:
        iteration_start_time = time.time()
        messages = build_messages()
        tool_choices = "auto" if tool_counter <= config.MAX_TOOL_CALLS_PER_TURN else "none"

        log.debug(f"Running inference with {len(messages)} messages and tool choice: {tool_choices}")

        payload = {
            "id_slots": 0,
            "messages": messages,
            "temperature": 0.8,
            "top_p": 0.85,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0,
            "max_tokens": config.LLM_MAX_TOKENS,
            "tools": tools.TOOLS,
            "tool_choice": tool_choices,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = session.post(LLM_SERVER_URL, json=payload, stream=True, timeout=config.LLM_REQUEST_TIMEOUT)
        if not response.ok:
            log.error(f"LLM server returned {response.status_code}: {response.text}")
        response.raise_for_status()

        # Extract assistant reply
        reply, reasoning, completion_tokens, context_total_tokens, tool_calls = parse_llm_response(response, input_q, confirm_q)
        if reply == "" and reasoning == "" and completion_tokens == 0 and context_total_tokens == 0 and not tool_calls:
            log.info("Inference interrupted due to new user input.")
            del _conversation_history[conversation_start_index:]
            return 0  # Return empty token count to indicate interruption

        final_time = time.time() - iteration_start_time

        # Check for send in case previous prompt was only partial transcription
        while True:
            try:
                if stop_event.is_set():
                    log.info("Stop event set, exiting inference loop.")
                    del _conversation_history[conversation_start_index:]
                    return 0
                elif tool_counter != 0:
                    log.info("Tool call detected, skipping confirmation and finalizing inference.")
                    break
                elif confirm_q.get(timeout=0.1) == "send":
                    log.info("Received confirm, finalizing inference.")
                    break
            except queue.Empty:
                if input_q.qsize() > 0:
                    log.info("New user input detected, stopping current inference...")
                    del _conversation_history[conversation_start_index:]
                    return 0  # Return empty reply and token count to indicate interruption
                continue

        log.debug(f"Reply: {reply!r}")
        append_message_to_log({"role": "user", "content": prompt})
        append_message_to_log({"role": "assistant", "content": reply})
        output_q.put(reply)

        gui_q.put(ConversationMsg(role="assistant", content=reply, reasoning_content=reasoning, tool_calls=tool_calls))
        gui_q.put(LlmTokenCountMsg(count=completion_tokens))

        log.debug(f"Inference completed in {final_time:.2f}s, waited for confirmation for {time.time() - (final_time + iteration_start_time):.2f}s")
        log.debug(f"Calculated LLM speed for {completion_tokens} tokens: {completion_tokens / final_time:.2f} tokens/s")
        gui_q.put(LlmTokenPerSecondMsg(tokens_per_second=completion_tokens / final_time))

        log.debug(f"Tool calls: {tool_calls}")

        if tool_calls:
            try:
                # Append assistant message WITH tool call
                _conversation_history.append({"role": "assistant", "content": reply, "tool_calls": tool_calls})

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"].get("arguments", "{}")

                    try:
                        arguments = json.loads(raw_args)
                    except json.JSONDecodeError:
                        log.warning("Failed to parse arguments, using empty dict")
                        arguments = {}

                    log.info(f"Function: {function_name}, Arguments: {arguments}")

                    result = tools.dispatch(function_name, arguments)

                    tool_reply = {"role": "tool", "content": f"{result}"}
                    _conversation_history.append(tool_reply)
                    append_message_to_log(tool_reply)
                    gui_q.put(ConversationMsg(role="tool", content=f"{result}"))

                # After processing all tool calls, check for any pending screenshots to attach their metadata to the conversation history
                for path in tools.take_pending_images():
                    img_msg = {"role": "user", "content": [
                        {"type": "text", "text": f"[Screenshot of {config.USER_NAME}'s {_screen_name(path)} monitor, taken {time.strftime('%H:%M:%S')}]"},
                        {"type": "image_ref", "path": path, "keep_turns": SCREENSHOT_KEEP_TURNS},
                    ]}
                    _conversation_history.append(img_msg)
                    append_message_to_log(img_msg)

                tool_counter += 1
                continue
            except Exception as e:
                log.error(f"Tool call failed: {e}")
                continue

        else:
            log.debug(f"Inference and waiting for confirmation/tool calls took {time.time() - start_time:.2f}s")

            gui_q.put(LlmTimeMsg(time=time.time() - start_time))
            _conversation_history.append({"role": "assistant", "content": reply})

            return context_total_tokens



# Parsing the LLM response Stream to extract relevant information such as the assistant's reply, reasoning, token counts, and tool calls.
def parse_llm_response(response, input_q: MpQueue, confirm_q: MpQueue) -> tuple[str, str, int, int, list[dict]]:
    """Parse the LLM response and extract the assistant's reply, reasoning, token counts, and tool calls."""
    reasoning = ""
    completion_tokens = 0
    context_total_tokens = 0
    tool_calls_accumulator = {}
    reply = ""

    for raw_line in response.iter_lines():
        if confirm_q.qsize() == 0 and input_q.qsize() > 0:
            log.info("New user input detected, stopping current inference...")
            response.close()
            return "", "", 0, 0, []  # Return empty reply and token count to indicate interruption
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue

        chunk = json.loads(line[6:])
        if "choices" not in chunk or not chunk["choices"]:
            log.warning(f"Invalid chunk received: {chunk}")
            continue
        delta = chunk["choices"][0]["delta"]

        if token := delta.get("content"):
            reply += token
        if r_token := delta.get("reasoning_content"):  # ik_llama streams this too
            reasoning += r_token
        if usage := chunk.get("usage"):
            completion_tokens = usage.get("completion_tokens", 0)
            context_total_tokens = usage.get("total_tokens", 0)
        if timings := chunk.get("timings"):
            completion_tokens = timings.get("predicted_n", 0)
            context_total_tokens = timings.get("n_past", 0)   # tokens currently in context
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            tool_calls_accumulator[idx]["function"]["arguments"] += tc.get("function", {}).get("arguments", "")
            tool_calls_accumulator[idx]["function"]["name"] += tc.get("function", {}).get("name", "")
            if tc.get("id"):
                tool_calls_accumulator[idx]["id"] = tc["id"]

    tool_calls = list(tool_calls_accumulator.values())
    return reply, reasoning, completion_tokens, context_total_tokens, tool_calls


def save_conversation_history():
    """Save the current conversation history to a file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_conversation_history, f, ensure_ascii=False, indent=2)
        log.info(f"Conversation history saved to {HISTORY_FILE}")
    except Exception as e:
        log.warning(f"Failed to save conversation history: {e}")

def load_conversation_history():
    """Load conversation history from a file if it exists."""
    global _conversation_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                _conversation_history = json.load(f)
            log.info(f"Loaded {len(_conversation_history)} messages from {HISTORY_FILE}")
        except Exception as e:
            log.warning(f"Failed to load conversation history: {e}")
            _conversation_history = []


def append_message_to_log(message: dict):
    """Append a single message to a valid JSON array file."""
    try:
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            # Start fresh
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("[\n" + json.dumps(message, ensure_ascii=False, indent=2) + "\n]")
        else:
            with open(LOG_FILE, "r+", encoding="utf-8") as f:
                # Seek back to just before the closing bracket
                f.seek(0, 2)  # Go to end
                pos = f.tell()
                # Walk back to find the ']'
                while pos > 0:
                    pos -= 1
                    f.seek(pos)
                    if f.read(1) == "]":
                        break
                # Overwrite the ']' with a comma + new entry + closing bracket
                f.seek(pos)
                f.write(",\n" + json.dumps(message, ensure_ascii=False, indent=2) + "\n]")
    except Exception as e:
        log.warning(f"Failed to append message to log: {e}")


def _materialize(msg: dict, turns_after: int) -> dict:
    """Resolve image_ref parts into base64 images, or a placeholder once expired."""
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    parts = []
    for p in content:
        if p.get("type") != "image_ref":
            parts.append(p)
            continue

        # Expired if too many user messages have passed, or the file is gone
        if turns_after > p["keep_turns"] or not os.path.exists(p["path"]):
            parts.append({"type": "text", "text": "[screenshot outdated]"})
        else:
            with open(p["path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    return {**msg, "content": parts}


def build_messages() -> list[dict]:
    """System prompt + history, with image refs resolved. Call on every request."""
    out, turns_after = [], 0

    # Walk newest -> oldest, counting real user prompts (string content) after each message
    for m in reversed(_conversation_history):
        if m["role"] == "user" and isinstance(m.get("content"), str):
            turns_after += 1
        out.append(_materialize(m, turns_after))

    return [{"role": "system", "content": STATIC_SYSTEM_PROMPT}] + out[::-1]


def _strip_images(history: list[dict]) -> list[dict]:
    """Text-only copy of the history, for the STM compressor."""
    res = []
    for m in history:
        c = m.get("content")
        if isinstance(c, list):
            m = {**m, "content": _content_to_text(c)}
        res.append(m)
    return res

def _content_to_text(content) -> str:
    """Flatten list-style content (text + image_ref parts) into a plain string."""
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") in ("image_ref", "image_url"):
                parts.append("[screenshot]")
        return " ".join(parts)
    return content or ""
