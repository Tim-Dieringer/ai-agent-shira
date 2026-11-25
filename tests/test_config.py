import os

import config


def test_load_prompt_replaces_placeholders(monkeypatch, tmp_path):
    (tmp_path / "demo.txt").write_text("{ASSISTANT_NAME} talks to {USER_NAME}.", encoding="utf-8")
    monkeypatch.setattr(config, "PROMPTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "USER_NAME", "Alex")
    monkeypatch.setattr(config, "ASSISTANT_NAME", "Robin")
    assert config.load_prompt("demo") == "Robin talks to Alex."


def test_shipped_prompts_have_no_unresolved_placeholders():
    for name in ("system_prompt", "short_term_memory_prompt"):
        text = config.load_prompt(name)
        assert text.strip()
        assert "{USER_NAME}" not in text
        assert "{ASSISTANT_NAME}" not in text


def test_paths_are_inside_the_project():
    for path in (config.PROMPTS_DIR, config.MEMORY_DIR, config.SCREENSHOT_DIR, config.AUDIO_DIR):
        assert os.path.commonpath([path, config.BASE_DIR]) == config.BASE_DIR
    assert os.path.exists(config.ASR_WARMUP_WAV)


def test_default_urls_point_to_localhost():
    assert config.LLM_SERVER_URL.endswith("/v1/chat/completions")
    assert config.STM_SERVER_URL.endswith("/v1/chat/completions")
    assert config.TTS_SPEECH_URL.endswith("/v1/audio/speech")


def test_asr_warmup_wav_is_mono_16khz():
    """NeMo expects (batch, time) input, so a stereo warm-up file crashes the ASR process on startup."""
    import wave

    import config

    with wave.open(config.ASR_WARMUP_WAV) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
