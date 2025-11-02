from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target; skip cleanly if package isn't on path
try:
    from rascal.coding import make_coding_files as mk
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.make_coding_files: {e}", allow_module_level=True)


# ---- Minimal tier objects so originals and reliabilities match by labels ----
class _SiteTier:
    def match(self, name: str):
        parts = name.split("_")
        return parts[0] if parts else None

class _PidTier:
    def match(self, name: str):
        parts = name.split("_")
        return parts[1] if len(parts) > 1 else None

def _dummy_tiers():
    # Order is stable because we use dict literals
    return {"site": _SiteTier(), "pid": _PidTier()}


# ------------------------------ File helpers ------------------------------- #
def _make_cu_pair(tmp_path):
    """Create matching *CUCoding.xlsx and *CUReliabilityCoding.xlsx placeholders."""
    input_dir = tmp_path / "input_cu"
    input_dir.mkdir(parents=True, exist_ok=True)
    cu = input_dir / "TU_P01_CUCoding.xlsx"
    rel = input_dir / "TU_P01_CUReliabilityCoding.xlsx"
    cu.write_bytes(b"")
    rel.write_bytes(b"")
    return input_dir, cu, rel

def _make_wc_pair(tmp_path):
    """Create matching *WordCounting.xlsx and *WordCountingReliability.xlsx placeholders."""
    input_dir = tmp_path / "input_wc"
    input_dir.mkdir(parents=True, exist_ok=True)
    wc = input_dir / "TU_P01_WordCounting.xlsx"
    rel = input_dir / "TU_P01_WordCountingReliability.xlsx"
    wc.write_bytes(b"")
    rel.write_bytes(b"")
    return input_dir, wc, rel


# ------------------------------- CU tests ---------------------------------- #
def test_reselect_cu_reliability_basic(tmp_path, monkeypatch):
    """
    Selects new IDs from unused samples, writes a single Excel file where:
      - rows are only for the newly chosen sample_id(s),
      - 'c3ID' and 'c3com' exist and are NaN (no coder passed in new API),
      - post-'comment' CU columns (e.g., c3SV/c3REL) exist and are NaN (templated).
    """
    input_dir, cu_file, rel_file = _make_cu_pair(tmp_path)
    output_dir = tmp_path / "out"
    tiers = _dummy_tiers()

    # --- Build synthetic CU (coder2) & REL (used samples) dataframes ---
    # Columns are ordered so that slicing up to 'comment' works.
    df_cu = pd.DataFrame({
        "utterance_id": ["U1","U2","U3","U4","U5","U6"],
        "sample_id":    ["S1","S1","S2","S2","S3","S3"],
        "speaker":      ["PAR"]*6,
        "utterance":    ["a","b","c","d","e","f"],
        "comment":      ["","","","","",""],   # slice anchor
        # coder-2 metadata (intentionally after 'comment' so it's dropped from head)
        "c2ID":         ["2"]*6,
        "c2com":        ["ok"]*6,
        "c2SV":         [1,0,1,1,0,1],
        "c2REL":        [1,0,1,1,0,1],
    })

    # Reliability used S1; S2 and S3 remain available.
    # Include 'comment' so the function can infer post-'comment' template columns.
    df_rel = pd.DataFrame({
        "utterance_id": ["U1","U2"],
        "sample_id":    ["S1","S1"],
        "comment":      ["",""],
        # Template columns after 'comment'
        "c3ID":         ["3","3"],
        "c3com":        ["ok","ok"],
        "c3SV":         [1,0],
        "c3REL":        [1,0],
    })

    # Stub pandas.read_excel
    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("_CUReliabilityCoding.xlsx"):
            return df_rel.copy()
        return df_cu.copy()

    # Capture the written DataFrame
    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_reselected_CUReliabilityCoding.xlsx"):
            captured["df"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    # Stabilize reselection: choose lexicographically first k available IDs
    def fake_sample(population, k):
        return sorted(population)[:k]

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
    monkeypatch.setattr(mk.random, "sample", fake_sample, raising=True)

    # all_sample_ids = {S1,S2,S3}; frac=0.2 -> round(3*0.2)=1 => pick one (S2 via fake_sample)
    mk.reselect_CU_WC_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        rel_type="CU",
        frac=0.2,
        rng_seed=88,
    )

    outdir = output_dir / "reselected_CU_reliability"
    files = list(outdir.glob("*reselected_CUReliabilityCoding.xlsx"))
    assert len(files) == 1, "Expected a single reselected reliability file."
    assert "df" in captured, "Did not capture written DataFrame."

    df = captured["df"]
    # Only rows for S2 should be present (two utterances)
    assert set(df["sample_id"].unique()) == {"S2"}
    assert len(df) == 2

    # 'c3ID' and 'c3com' exist and are NaN in new API
    assert "c3ID" in df.columns and "c3com" in df.columns
    assert df["c3ID"].isna().all()
    assert df["c3com"].isna().all()

    # CU columns (templated post-'comment') exist and are NaN
    cu_cols = [c for c in df.columns if c.startswith("c3SV") or c.startswith("c3REL")]
    assert cu_cols, "Expected CU c3* columns in output"
    for col in cu_cols:
        assert df[col].isna().all(), f"{col} was not NaN"

    # Shared columns retained
    assert set(["utterance_id","sample_id","speaker","utterance","comment"]).issubset(df.columns)


def test_reselect_cu_reliability_no_available(tmp_path, monkeypatch):
    """When all samples were already used, no output file should be created."""
    input_dir, cu_file, rel_file = _make_cu_pair(tmp_path)
    output_dir = tmp_path / "out2"
    tiers = _dummy_tiers()

    df_cu = pd.DataFrame({
        "utterance_id": ["U1","U2","U3","U4"],
        "sample_id":    ["S1","S1","S2","S2"],
        "speaker":      ["PAR"]*4,
        "utterance":    ["a","b","c","d"],
        "comment":      ["","","",""],
        "c2ID":         ["2"]*4,
        "c2com":        ["ok"]*4,
        "c2SV":         [1,0,1,1],
        "c2REL":        [1,0,1,1],
    })
    # Reliability already used S1 and S2 -> no available
    df_rel = pd.DataFrame({
        "utterance_id": ["U1","U2","U3","U4"],
        "sample_id":    ["S1","S1","S2","S2"],
        "comment":      ["","","",""],
    })

    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("_CUReliabilityCoding.xlsx"):
            return df_rel.copy()
        return df_cu.copy()

    def fake_to_excel(self, path, index=False):
        Path(os.fspath(path)).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    mk.reselect_CU_WC_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        rel_type="CU",
        frac=0.5,
        rng_seed=42,
    )
    outdir = output_dir / "reselected_CU_reliability"
    # Directory exists, but no files created
    assert outdir.exists()
    assert not any(outdir.glob("*reselected_CUReliabilityCoding.xlsx"))


# ------------------------------- WC test ----------------------------------- #
def test_reselect_wc_reliability_basic(tmp_path, monkeypatch):
    """
    Selects WC reliability rows using the 'WC' branch. We avoid invoking any
    external word-count helper by setting original 'word_count' to NaN (the code
    falls back to a placeholder), and we assert the templated WC post-'comment'
    column (e.g., 'wc_rel_com') is present and NaN.
    """
    input_dir, wc_file, rel_file = _make_wc_pair(tmp_path)
    output_dir = tmp_path / "out_wc"
    tiers = _dummy_tiers()

    df_wc = pd.DataFrame({
        "utterance_id": ["U1","U2","U3","U4","U5","U6"],
        "sample_id":    ["S1","S1","S2","S2","S3","S3"],
        "speaker":      ["PAR"]*6,
        "utterance":    ["alpha","beta","gamma","delta","epsilon","zeta"],
        "comment":      ["","","","","",""],
        # Keep word_count as NaN so any optional count_words(...) path is skipped
        "word_count":   [np.nan]*6,
    })

    df_wc_rel = pd.DataFrame({
        "utterance_id": ["U1","U2"],
        "sample_id":    ["S1","S1"],
        "comment":      ["",""],
        "wc_rel_com":   ["",""],   # template post-'comment' column
    })

    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("WordCountingReliability.xlsx"):
            return df_wc_rel.copy()
        return df_wc.copy()

    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_reselected_WordCountingReliability.xlsx"):
            captured["df"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    def fake_sample(population, k):
        return sorted(population)[:k]

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
    monkeypatch.setattr(mk.random, "sample", fake_sample, raising=True)

    mk.reselect_CU_WC_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        rel_type="WC",
        frac=0.2,      # all ids {S1,S2,S3} -> round(0.6)=1 -> choose S2 via fake_sample
        rng_seed=7,
    )

    outdir = output_dir / "reselected_WC_reliability"
    files = list(outdir.glob("*_reselected_WordCountingReliability.xlsx"))
    assert len(files) == 1, "Expected a single reselected WC reliability file."
    assert "df" in captured

    df = captured["df"]
    assert set(df["sample_id"].unique()) == {"S2"}
    assert len(df) == 2

    # Templated WC post-'comment' column exists and is NaN
    assert "wc_rel_com" in df.columns
    assert df["wc_rel_com"].isna().all()

    # Head columns kept
    assert set(["utterance_id","sample_id","speaker","utterance","comment"]).issubset(df.columns)
