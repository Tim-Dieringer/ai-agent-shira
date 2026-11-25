import json
import logging
import multiprocessing as mp
import queue
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent

import requests

import config

log = logging.getLogger("shira.stm")

STM_SERVER_URL = config.STM_SERVER_URL

STATIC_SYSTEM_PROMPT = config.load_prompt("short_term_memory_prompt")

def start_stm_process(stop_event: MpEvent, input_queue: MpQueue, output_queue: MpQueue) -> mp.Process:
    """Spawn and return the short-term-memory (conversation compression) background process."""
    p = mp.Process(
        target=_stm_process_fn,
        args=(stop_event,input_queue, output_queue),
        daemon=True,
        name="stm-process",
    )
    p.start()
    return p


def _stm_process_fn(stop_event: MpEvent, input_queue: MpQueue, output_queue: MpQueue):
    """Background process function for compressing short-term memory."""
    config.setup_logging()
    session = requests.Session()


    while not stop_event.is_set():
        try:
            conversation_history = input_queue.get(timeout=1)
        except queue.Empty:
            continue  # No new conversation history, check if we should stop

        log.debug(f"Received conversation history for compression. Length: {len(conversation_history)} messages.")

        usermessage = json.dumps(conversation_history)  # the slice of history sent by the LLM process


        messages = [{"role": "system", "content": STATIC_SYSTEM_PROMPT}] + [{"role": "user", "content": usermessage}]

        payload = {
                "id_slots": 0,
                "messages": messages,
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.0,
                "repeat_penalty": 1.2,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "max_tokens": config.STM_MAX_TOKENS,
            }
        log.debug(f"Sending conversation history to LLM for compression. Messages sent: {len(messages)}")
        try:
            response = session.post(STM_SERVER_URL, json=payload, timeout=config.STM_REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log.warning(f"Request failed: {e}")
            output_queue.put(None)
            continue
        log.debug(f"Status: {response.status_code}")
        log.debug(f"Body: {response.text[:1000]}")
        if response.status_code != 200:
            output_queue.put(None)
            continue
        data = response.json()

        output = data["choices"][0]["message"]["content"]
        output_queue.put(output)
