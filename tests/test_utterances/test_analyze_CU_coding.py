from pathlib import Path
import os
import pandas as pd
import pytest

# Import target; skip cleanly if package isn't on path
try:
    from rascal.utterances import CU_analyzer as cua
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.CU_analyzer: {e}", allow_module_level=True)


class MockTier:
    """Extract labels from filename split by underscores; choose index per tier."""
    def __init__(self, name, idx, partition=False):
        self.name = name
        self.idx = idx
        self.partition = partition

    def match(self, filename, return_None=False):
        base = Path(filename).stem
        parts = base.split("_")
        try:
            return parts[self.idx]
        except Exception:
            return None if return_None else ""


@pytest.fixture
def tiers():
    # Partitioned site tier (position 0), and a non-partitioned participantID (position 1)
    return {
        "site": MockTier("site", idx=0, partition=True),
        "participantID": MockTier("participantID", idx=1, partition=False),
    }


def _write_coding_file(tmp_path):
    """Create an empty coding file so rglob finds it; content comes from our read_excel stub."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "TU_P01_CUCoding.xlsx").write_bytes(b"")
    return input_dir


def test_analyze_CU_coding_base_columns(tmp_path, monkeypatch, tiers):
    """
    When there are no suffixed c2SV_* columns and CU_paradigms is None,
    the function should:
      - infer [None] and use base columns (c2SV/c2REL),
      - write utterance- and sample-level files,
      - produce summary columns suffixed with 'None' (current behavior).
    """
    input_dir = _write_coding_file(tmp_path)
    output_dir = tmp_path / "out"
    captures = {}

    # Stub read_excel -> only base columns present
    base_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id":    ["S1", "S1", "S1"],
        "c2SV":  [1, 0, 1],
        "c2REL": [1, 0, 1],
        # Droppable columns (ensure harmless)
        "c1ID": ["x", "y", "z"],
        "c1com": ["a", "b", "c"],
        "c2ID": ["d", "e", "f"],
    })

    def fake_read_excel(path, *args, **kwargs):
        return base_df.copy()

    def fake_to_excel(self, path, index=False):
        # Capture the merged summary DF for assertions
        p = os.fspath(path)
        if p.endswith("_CUCoding_BySample.xlsx"):
            captures["summary"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # Run
    cua.analyze_CU_coding(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=None,  # triggers introspection; no suffixed cols => [None]
    )

    # Paths
    outdir = output_dir / "CUCodingAnalysis" / "TU"
    assert outdir.is_dir(), "Expected partitioned output directory."

    by_utt  = outdir / "TU_CUCoding_ByUtterance.xlsx"
    by_samp = outdir / "TU_CUCoding_BySample.xlsx"
    for p in (by_utt, by_samp):
        assert p.exists(), f"Missing expected output: {p}"

    # Check merged summary content captured from to_excel
    assert "summary" in captures, "Summary DataFrame was not written."
    df = captures["summary"]
    # Column names include 'None' suffix (as implemented)
    expected_cols = {
        "sample_id",
        "no_utt_None", "pSV_None", "mSV_None",
        "pREL_None", "mREL_None", "CU_None", "percCU_None",
    }
    assert expected_cols.issubset(df.columns), f"Summary missing expected columns: {expected_cols - set(df.columns)}"

    # Values: 3 utts, 2 positives, 66.667% CU
    row = df.loc[df["sample_id"] == "S1"].iloc[0]
    assert row["no_utt_None"] == 3
    assert row["pSV_None"] == 2
    assert row["pREL_None"] == 2
    assert row["CU_None"] == 2
    assert row["percCU_None"] == pytest.approx(66.667, abs=1e-3)


def test_analyze_CU_coding_multi_paradigms_introspection(tmp_path, monkeypatch, tiers):
    """
    When suffixed columns exist and CU_paradigms is None, the function should
    introspect paradigms (e.g., 'AAE', 'SAE'), compute each, and merge on sample_id.
    """
    input_dir = _write_coding_file(tmp_path)
    output_dir = tmp_path / "out2"
    captures = {}

    # Build a DF with two paradigms that yield different CU% to verify correctness
    multi_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id":    ["S1", "S1", "S1"],
        # SAE -> only first utt is CU => 33.333%
        "c2SV_SAE":  [1, 0, 0],
        "c2REL_SAE": [1, 0, 0],
        # AAE -> first and third utts are CU => 66.667%
        "c2SV_AAE":  [1, 0, 1],
        "c2REL_AAE": [1, 0, 1],
    })

    def fake_read_excel(path, *args, **kwargs):
        return multi_df.copy()

    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_CUCoding_BySample.xlsx"):
            captures["summary"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # Run with CU_paradigms=None -> should discover ['AAE','SAE'] (sorted)
    cua.analyze_CU_coding(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=None,
    )

    outdir = output_dir / "CUCodingAnalysis" / "TU"
    assert outdir.is_dir(), "Expected partitioned output directory."
    assert (outdir / "TU_CUCoding_ByUtterance.xlsx").exists()
    assert (outdir / "TU_CUCoding_BySample.xlsx").exists()

    # Check merged summary includes both paradigms with expected values
    df = captures["summary"]
    needed = {
        "sample_id",
        # AAE columns
        "no_utt_AAE", "pSV_AAE", "pREL_AAE", "CU_AAE", "percCU_AAE",
        # SAE columns
        "no_utt_SAE", "pSV_SAE", "pREL_SAE", "CU_SAE", "percCU_SAE",
    }
    assert needed.issubset(df.columns), f"Summary missing expected columns: {needed - set(df.columns)}"

    row = df.loc[df["sample_id"] == "S1"].iloc[0]
    # AAE
    assert row["no_utt_AAE"] == 3
    assert row["CU_AAE"] == 2
    assert row["percCU_AAE"] == pytest.approx(66.667, abs=1e-3)
    # SAE
    assert row["no_utt_SAE"] == 3
    assert row["CU_SAE"] == 1
    assert row["percCU_SAE"] == pytest.approx(33.333, abs=1e-3)
