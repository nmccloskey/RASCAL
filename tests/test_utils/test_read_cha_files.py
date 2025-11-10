import random
from pathlib import Path
from rascal.utils.cha_files import read_cha_files


def _make_cha_file(path: Path, text: str):
    """Helper to create a minimal .cha file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_read_cha_files(tmp_path, monkeypatch):
    """Test that read_cha_files reads .cha files correctly and returns dict."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # --- Create fake .cha files ---
    cha1 = _make_cha_file(input_dir / "sample1.cha", "*PAR:\tHello world.\n")
    cha2 = _make_cha_file(input_dir / "nested" / "sample2.cha", "*PAR:\tAnother test.\n")

    # --- Monkeypatch pylangacq.read_chat to avoid heavy dependency ---
    class MockReader:
        def __init__(self, path):
            self.path = path
        def utterances(self):
            return [{"participant": "PAR", "text": "mock"}]

    monkeypatch.setattr(
        "rascal.utils.cha_files.pylangacq.read_chat",
        lambda path: MockReader(path)
    )

    # --- Run function normally ---
    chats = read_cha_files(input_dir)
    assert isinstance(chats, dict)
    assert len(chats) == 2, "Should read both .cha files recursively"
    for name, reader in chats.items():
        assert name.endswith(".cha")
        assert hasattr(reader, "utterances")

    # --- Verify that shuffle changes file order (non-deterministic) ---
    all_names = list(chats.keys())
    random.seed(123)
    chats_shuffled = read_cha_files(input_dir, shuffle=True)
    shuffled_names = list(chats_shuffled.keys())

    # While content may be same, order can differ
    assert set(all_names) == set(shuffled_names), "Shuffle should not change file membership"

    # --- Check behavior with empty directory ---
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result_empty = read_cha_files(empty_dir)
    assert result_empty == {}, "Expected empty dict for empty directory"
