import pandas as pd
from pathlib import Path
from rascal.transcripts.transcription_reliability_selection import select_transcription_reliability_samples


class MockTier:
    """Minimal mock tier with match() and optional partition flag."""
    def __init__(self, label, partition=True):
        self.label = label
        self.partition = partition
        self.name = label

    def match(self, fname):
        # Match if label substring is found in the filename
        return self.label if self.label in fname else None


class MockChat:
    """Fake CHAT object with to_strs() yielding minimal header text."""
    def __init__(self, content):
        self.content = content
    def to_strs(self):
        yield f"@Participants:\tPAR Participant\n@Begin\n{self.content}\n@End"


def _make_chat_map(base_dir: Path, labels: list[str], content="Fake transcript"):
    """Create mock CHAT files and a chats dict for the function input."""
    chats = {}
    for i, label in enumerate(labels, start=1):
        fname = f"{label}_P{i:02d}.cha"
        fpath = base_dir / fname
        fpath.write_text(content)
        chats[str(fpath)] = MockChat(content)
    return chats


def test_select_transcription_reliability_samples(tmp_path):
    """End-to-end test of select_transcription_reliability_samples."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # --- Mock tiers ---
    tiers = {
        "site": MockTier("SiteA", partition=True),
        "group": MockTier("G1", partition=True),
    }

    # --- Create mock CHAT files and chats mapping ---
    labels = ["SiteA_G1", "SiteA_G1", "SiteA_G1"]
    chats = _make_chat_map(input_dir, labels)

    # --- Run the function ---
    select_transcription_reliability_samples(tiers, chats, frac=0.5, output_dir=output_dir)

    # --- Verify output directory structure ---
    base_dir = output_dir / "transcription_reliability_selection" / "SiteA" / "G1"
    assert base_dir.exists(), f"Expected partition directory missing: {base_dir}"

    # --- Verify reliability CHA file(s) ---
    cha_files = list(base_dir.glob("*_reliability.cha"))
    assert cha_files, "No reliability CHA files were written"
    # Ensure headers are correctly written
    cha_text = cha_files[0].read_text()
    assert cha_text.startswith("@Begin")
    assert "@End" in cha_text

    # --- Verify Excel summary ---
    xlsx_files = list(base_dir.glob("*transcription_reliability_samples.xlsx"))
    assert xlsx_files, "Expected Excel summary not found"
    xls = pd.ExcelFile(xlsx_files[0])
    assert {"reliability_selection", "all_transcripts"} <= set(xls.sheet_names)

    df_all = pd.read_excel(xls, sheet_name="all_transcripts")
    df_subset = pd.read_excel(xls, sheet_name="reliability_selection")

    # --- Data validity checks ---
    assert len(df_all) == len(chats)
    assert not df_subset.empty
    assert set(df_all.columns) >= {"file", "site", "group"}

    # Subset rows should all exist in full list
    assert set(df_subset["file"]).issubset(set(df_all["file"]))
