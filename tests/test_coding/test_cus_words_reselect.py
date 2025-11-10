import pandas as pd
from pathlib import Path
from rascal.coding.coding_files import reselect_cu_wc_reliability

class MockTier:
    """Minimal mock tier that provides .match() for label extraction."""
    def __init__(self, label):
        self.label = label
    def match(self, fname, return_None=False):
        return self.label if self.label in fname else None

def _make_excel(df, path):
    """Helper to write a DataFrame to Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path

def test_reselect_cu_reliability(tmp_path):
    """Test reselection for CU reliability files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    tiers = {"group": MockTier("G1")}

    # Original CU coding file
    df_org = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3", "S4", "S5"],
        "utterance": ["a b c", "d e f", "g h i", "j k l", "m n o"],
        "comment": ["", "", "", "", ""],
        "c3_id": ["", "", "", "", ""],
        "c3_comment": ["", "", "", "", ""],
    })
    org_file = _make_excel(df_org, input_dir / "G1_cu_coding.xlsx")

    # Existing reliability file with S1, S2 used
    df_rel = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "utterance": ["a b c", "d e f"],
        "comment": ["", ""],
        "c3_id": ["x", "y"],
        "c3_comment": ["ok", "ok"],
    })
    _make_excel(df_rel, input_dir / "G1_cu_reliability_coding.xlsx")

    # Run reselection
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type="CU", frac=0.4)

    # Validate output file
    out_dir = output_dir / "reselected_cu_coding_reliability"
    out_files = list(out_dir.glob("*.xlsx"))
    assert len(out_files) == 1
    out_df = pd.read_excel(out_files[0])

    # Should contain at least one new sample not in the original reliability
    new_ids = set(out_df["sample_id"])
    assert new_ids.isdisjoint({"S1", "S2"})
    assert all(c in out_df.columns for c in ["utterance", "c3_id", "c3_comment"])


def test_reselect_wc_reliability(tmp_path):
    """Test reselection for WC reliability files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    tiers = {"group": MockTier("G2")}

    # Original WC coding file
    df_org = pd.DataFrame({
        "sample_id": ["A", "B", "C", "D", "E"],
        "utterance": ["one", "two", "three", "four", "five"],
        "comment": ["", "", "", "", ""],
        "word_count": [1, 1, 1, 1, 1],
        "wc_comment": ["", "", "", "", ""],
    })
    org_file = _make_excel(df_org, input_dir / "G2_word_counting.xlsx")

    # Existing reliability file with A, B used
    df_rel = pd.DataFrame({
        "sample_id": ["A", "B"],
        "utterance": ["one", "two"],
        "comment": ["", ""],
        "wc_rel_com": ["good", "ok"],
    })
    _make_excel(df_rel, input_dir / "G2_word_count_reliability.xlsx")

    # Run reselection
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type="WC", frac=0.4)

    # Validate output file
    out_dir = output_dir / "reselected_word_count_reliability"
    out_files = list(out_dir.glob("*.xlsx"))
    assert len(out_files) == 1
    out_df = pd.read_excel(out_files[0])

    # Should exclude A,B and include columns like wc_rel_com
    assert set(out_df["sample_id"]).isdisjoint({"A", "B"})
    assert "wc_rel_com" in out_df.columns
