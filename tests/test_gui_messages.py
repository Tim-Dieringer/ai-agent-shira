import dataclasses
import inspect
import os

from gui import gui_messages

BRIDGE_SOURCE = os.path.join(os.path.dirname(gui_messages.__file__), "gui_bridge.py")


def _message_classes():
    return [
        cls for _, cls in inspect.getmembers(gui_messages, inspect.isclass)
        if dataclasses.is_dataclass(cls) and cls.__module__ == gui_messages.__name__
    ]


def test_message_types_are_unique_and_non_empty():
    types_seen = [cls().type for cls in _message_classes()]
    assert all(types_seen)
    assert len(types_seen) == len(set(types_seen))


def test_gui_bridge_handles_every_message_type():
    # gui_bridge imports PySide6, so check its source instead of importing it.
    with open(BRIDGE_SOURCE, encoding="utf-8") as f:
        source = f.read()
    for cls in _message_classes():
        assert f't == "{cls().type}"' in source, f"{cls.__name__} is not dispatched in gui_bridge.py"
