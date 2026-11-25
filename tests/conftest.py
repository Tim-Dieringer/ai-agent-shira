"""Shared test setup.

The tests only cover pure logic, so they must run on a plain CI machine with
no GPU, no microphone, no display and without the heavy ML dependencies
(torch, NeMo, PySide6). Hardware-related packages that are not installed are
replaced with empty stand-in modules so that `import tools` / `import llm`
still work.
"""
import importlib.util
import sys
import types


def _stub_if_missing(name: str, **attrs) -> None:
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing("mss", mss=None)
