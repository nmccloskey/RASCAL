import pandas as pd
from pathlib import Path
from rascal.coding.word_count_reliability_evaluation import evaluate_word_count_reliability


class MockTier:
    """Minimal mock tier with .match() and .partition attributes."""
    def __init__(self, label, partition=True):
        self.label = label
        self.partition = partition

    def match(self, fname, return_None=False):
        # Match if the label appears in the filename
        return self.label if self.label in fname else None


def _make_excel(df, path):
    """Helper to write a DataFrame to Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path


def test_evaluate_word_count_reliability(tmp_path):
    """Test evaluate_word_count_reliability end-to-end with synthetic data."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    # --- Mock tier setup ---
    tiers = {"group": MockTier("SiteA")}

    # --- Create synthetic coding file ---
    df_coding = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3"],
        "utterance_id": [1, 2, 3],
        "utterance": ["one", "two", "three"],
        "word_count": [5, 10, 20]
    })
    coding_path = _make_excel(df_coding, input_dir / "SiteA_word_counting.xlsx")

    # --- Create synthetic reliability file ---
    df_rel = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3"],
        "utterance_id": [1, 2, 3],
        "word_count": [6, 9, 18],  # small diffs
        "wc_rel_com": ["ok", "ok", "ok"]
    })
    rel_path = _make_excel(df_rel, input_dir / "SiteA_word_count_reliability.xlsx")

    # --- Run evaluation ---
    evaluate_word_count_reliability(tiers, input_dir, output_dir)

    # --- Validate outputs ---
    word_rel_dir = output_dir / "word_count_reliability" / "SiteA"
    assert word_rel_dir.exists(), "Output directory not created"

    results_files = list(word_rel_dir.glob("*results.xlsx"))
    report_files = list(word_rel_dir.glob("*report.txt"))

    # Expect both results and report
    assert len(results_files) == 1, "No results file created"
    assert len(report_files) == 1, "No report file created"

    # Read merged results and check expected columns
    results_df = pd.read_excel(results_files[0])
    expected_cols = {
        "sample_id", "utterance_id", "word_count_org", "word_count_rel",
        "abs_diff", "perc_diff", "perc_sim", "agmt"
    }
    assert expected_cols.issubset(results_df.columns)

    # Agreement logic check — all small diffs should yield agreement = 1
    assert results_df["agmt"].sum() == len(results_df)

    # Check ICC and summary report text presence
    with open(report_files[0]) as f:
        report_text = f.read()
    assert "ICC" in report_text
    assert "agreement" in report_text.lower()
