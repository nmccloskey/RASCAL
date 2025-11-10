import pandas as pd
from pathlib import Path
from rascal.coding.coding_files import make_word_count_files


class MockTier:
    """Minimal mock tier that mimics .match() and has no partition requirement."""
    def __init__(self, label):
        self.label = label
    def match(self, fname, return_None=False):
        return self.label if self.label in fname else None


def _make_excel(df, path):
    """Helper: write DataFrame to Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path


def test_make_word_count_files(tmp_path, monkeypatch):
    """End-to-end test of make_word_count_files with a small synthetic dataset."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    # --- Mock tiers ---
    tiers = {"site": MockTier("LabA")}

    # --- Create synthetic CU coding by utterance file ---
    cu_df = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2", "S3"],
        "utterance": ["one two", "three four five", "six", "seven eight nine"],
        "c2_cu": ["ok", "ok", "ok", "ok"],  # required by _prepare_wc_df
        "c1_sv": ["", "", "", ""],
        "c2_sv": ["", "", "", ""],
        "c1_rel": ["", "", "", ""],
        "c2_rel": ["", "", "", ""],
        "c1_comment": ["", "", "", ""],
        "c2_comment": ["", "", "", ""],
    })
    _make_excel(cu_df, input_dir / "LabA_cu_coding_by_utterance.xlsx")

    # --- Monkeypatch get_word_checker to skip nltk dependency ---
    monkeypatch.setattr(
        "rascal.coding.coding_files.get_word_checker",
        lambda: (lambda w: True)  # every token is a valid word
    )

    # --- Run function ---
    make_word_count_files(tiers, frac=0.5, coders=["A", "B", "C"], input_dir=input_dir, output_dir=output_dir)

    # --- Verify outputs ---
    word_count_dir = output_dir / "word_counts" / "LabA"
    assert word_count_dir.exists(), "Expected word_counts/LabA directory not created"

    out_files = list(word_count_dir.glob("*.xlsx"))
    assert len(out_files) == 2, f"Expected 2 Excel outputs (word_counting + reliability), found {len(out_files)}"

    # Identify files
    wc_file = next(f for f in out_files if "word_counting" in f.name)
    rel_file = next(f for f in out_files if "word_count_reliability" in f.name)

    wc_df = pd.read_excel(wc_file)
    rel_df = pd.read_excel(rel_file)

    # --- Structural checks ---
    assert "word_count" in wc_df.columns, "Missing 'word_count' column"
    assert "wc_comment" in wc_df.columns, "Missing 'wc_comment' column"
    assert "sample_id" in rel_df.columns, "Missing 'sample_id' column"
    assert "wc_rel_com" in rel_df.columns, "Missing 'wc_rel_com' column"

    # --- Content checks ---
    # word_count should be > 0 since all words are considered valid
    assert (wc_df["word_count"].astype(str) != "NA").all()
    assert wc_df["word_count"].astype(int).ge(1).all()

    # reliability subset should contain at least one row
    assert len(rel_df) >= 1, "Reliability subset unexpectedly empty"

    # coder assignment
    assert wc_df["c1_id"].notna().all(), "Coders not assigned properly"
