from pathlib import Path
import os
import pandas as pd
import pytest

# Import target
try:
    from rascal.coding import cu_analysis as cua
except Exception as e:
    pytest.skip(f"Could not import rascal.coding.cu_analysis: {e}", allow_module_level=True)


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
    (input_dir / "TU_P01_cu_coding.xlsx").write_bytes(b"")
    return input_dir


def test_analyze_cu_coding_base_columns(tmp_path, monkeypatch, tiers):
    """
    When there are no suffixed c2_sv_* columns and cu_paradigms is None,
    the function should:
      - infer [None] and use base columns (c2_sv/c2_rel),
      - write utterance- and sample-level files,
      - produce summary columns suffixed with 'None' (current behavior).
    """
    input_dir = _write_coding_file(tmp_path)
    output_dir = tmp_path / "out"
    captures = {}

    base_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id": ["S1", "S1", "S1"],
        "c2_sv": [1, 0, 1],
        "c2_rel": [1, 0, 1],
        "c1_id": ["x", "y", "z"],
        "c1_comment": ["a", "b", "c"],
        "c2_id": ["d", "e", "f"],
    })

    def fake_read_excel(path, *args, **kwargs):
        return base_df.copy()

    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_cu_coding_by_sample.xlsx"):
            captures["summary"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    cua.analyze_cu_coding(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=None,
    )

    outdir = output_dir / "cu_coding_analysis" / "TU"
    assert outdir.is_dir(), "Expected partitioned output directory."

    by_utt = outdir / "TU_cu_coding_by_utterance.xlsx"
    by_samp = outdir / "TU_cu_coding_by_sample.xlsx"
    for p in (by_utt, by_samp):
        assert p.exists(), f"Missing expected output: {p}"

    assert "summary" in captures, "Summary DataFrame was not written."
    df = captures["summary"]

    expected_cols = {
        "sample_id",
        "no_utt_None", "p_sv_None", "m_sv_None",
        "p_rel_None", "m_rel_None", "cu_None", "perc_cu_None",
    }
    assert expected_cols.issubset(df.columns), f"Summary missing expected columns: {expected_cols - set(df.columns)}"

    row = df.loc[df["sample_id"] == "S1"].iloc[0]
    assert row["no_utt_None"] == 3
    assert row["p_sv_None"] == 2
    assert row["p_rel_None"] == 2
    assert row["cu_None"] == 2
    assert row["perc_cu_None"] == pytest.approx(66.667, abs=1e-3)


def test_analyze_cu_coding_multi_paradigms_introspection(tmp_path, monkeypatch, tiers):
    """
    When suffixed columns exist and cu_paradigms is None, the function should
    introspect paradigms (e.g., 'AAE', 'SAE'), compute each, and merge on sample_id.
    """
    input_dir = _write_coding_file(tmp_path)
    output_dir = tmp_path / "out2"
    captures = {}

    multi_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id": ["S1", "S1", "S1"],
        "c2_sv_SAE": [1, 0, 0],
        "c2_rel_SAE": [1, 0, 0],
        "c2_sv_AAE": [1, 0, 1],
        "c2_rel_AAE": [1, 0, 1],
    })

    def fake_read_excel(path, *args, **kwargs):
        return multi_df.copy()

    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_cu_coding_by_sample.xlsx"):
            captures["summary"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    cua.analyze_cu_coding(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=None,
    )

    outdir = output_dir / "cu_coding_analysis" / "TU"
    assert outdir.is_dir(), "Expected partitioned output directory."
    assert (outdir / "TU_cu_coding_by_utterance.xlsx").exists()
    assert (outdir / "TU_cu_coding_by_sample.xlsx").exists()

    df = captures["summary"]
    needed = {
        "sample_id",
        "no_utt_AAE", "p_sv_AAE", "p_rel_AAE", "cu_AAE", "perc_cu_AAE",
        "no_utt_SAE", "p_sv_SAE", "p_rel_SAE", "cu_SAE", "perc_cu_SAE",
    }
    assert needed.issubset(df.columns), f"Summary missing expected columns: {needed - set(df.columns)}"

    row = df.loc[df["sample_id"] == "S1"].iloc[0]
    assert row["no_utt_AAE"] == 3
    assert row["cu_AAE"] == 2
    assert row["perc_cu_AAE"] == pytest.approx(66.667, abs=1e-3)
    assert row["no_utt_SAE"] == 3
    assert row["cu_SAE"] == 1
    assert row["perc_cu_SAE"] == pytest.approx(33.333, abs=1e-3)
