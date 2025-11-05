from pathlib import Path
import types
import pytest

from rascal.utils.cha_files import read_cha_files


class StubReader:
    """Minimal stand-in for pylangacq.Reader; only needs .utterances() in downstream code."""
    def __init__(self, label):
        self._label = label
    def utterances(self):
        return iter([])  # not exercised here


def test_read_cha_files_monkeypatched(monkeypatch, tmp_path):
    # Create a fake directory structure with .cha and non-.cha files
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "one.cha").write_text("*PAR:\thello\n")
    (tmp_path / "a" / "two.cha").write_text("*PAR:\tgoodbye\n")
    (tmp_path / "b" / "ignore.txt").write_text("not a cha")
    (tmp_path / "b" / "three.cha").write_text("*PAR:\thi\n")

    # Monkeypatch pylangacq.read_chat to return a stub without reading/parsing
    import rascal.utils.cha_files as module_under_test
    calls = []
    def fake_read_chat(path_str):
        calls.append(Path(path_str).name)
        return StubReader(Path(path_str).name)
    monkeypatch.setattr(module_under_test, "pylangacq", types.SimpleNamespace(read_chat=fake_read_chat))

    chats = read_cha_files(str(tmp_path), shuffle=False)
    # We should have exactly the three .cha files, keyed by their basenames
    assert set(chats.keys()) == {"one.cha", "two.cha", "three.cha"}
    assert all(isinstance(v, StubReader) for v in chats.values())
    # Ensure we attempted to "read" each file via the monkeypatched function
    assert set(calls) == {"one.cha", "two.cha", "three.cha"}


def test_read_cha_files_shuffle(monkeypatch, tmp_path):
    # Two files; we won't assert order because shuffle=True
    (tmp_path / "x.cha").write_text("*PAR:\tx\n")
    (tmp_path / "y.cha").write_text("*PAR:\ty\n")

    import rascal.utils.cha_files as module_under_test
    monkeypatch.setattr(
        module_under_test,
        "pylangacq",
        types.SimpleNamespace(read_chat=lambda p: StubReader(Path(p).name)),
    )

    chats = read_cha_files(str(tmp_path), shuffle=True)
    assert set(chats.keys()) == {"x.cha", "y.cha"}
