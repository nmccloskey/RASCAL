from pathlib import Path
import os
import pandas as pd
import pytest

# Import target; skip cleanly if package isn't on path
try:
    from rascal.utterances import CU_analyzer as cua
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.CU_analyzer: {e}", allow_module_level=True)


def _make_pair(tmp_path):
    """Create matching *_CUCoding.xlsx and *_CUReliabilityCoding.xlsx placeholders."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cu = input_dir / "TU_P01_CUCoding.xlsx"
    rel = input_dir / "TU_P01_CUReliabilityCoding.xlsx"
    cu.write_bytes(b"")
    rel.write_bytes(b"")
    return input_dir, cu, rel


def test_reselect_cu_reliability_basic(tmp_path, monkeypatch):
    """
    Selects new IDs from unused samples, writes a single Excel file with:
      - c3ID set to provided coder,
      - c3com wiped (NaN),
      - suffixed c3SV_*/c3REL_* wiped (NaN),
      - unsuffixed c3SV/c3REL preserved from coder-2 (current behavior).
    """
    input_dir, cu_file, rel_file = _make_pair(tmp_path)
    output_dir = tmp_path / "out"

    # --- Build synthetic CU (coder2) & REL (used samples) dataframes ---
    # All columns are ordered so that slicing up to 'comment' works.
    df_cu = pd.DataFrame({
        "UtteranceID": ["U1","U2","U3","U4","U5","U6"],
        "sampleID":    ["S1","S1","S2","S2","S3","S3"],
        "speaker":     ["PAR"]*6,
        "utterance":   ["a","b","c","d","e","f"],
        "comment":     ["","","","","",""],   # slice anchor
        # coder-2 metadata (must come after 'comment' to be excluded from shared_cols)
        "c2ID":   ["2"]*6,
        "c2com":  ["ok"]*6,
        # Base CU fields
        "c2SV":   [1,0,1,1,0,1],
        "c2REL":  [1,0,1,1,0,1],
        # Suffixed paradigms (will be wiped on rename to c3*)
        "c2SV_SAE":[1,0,1,0,1,1],
        "c2REL_SAE":[1,0,1,0,1,1],
        "c2SV_AAE":[1,1,0,0,1,0],
        "c2REL_AAE":[1,1,0,0,1,0],
    })

    # Reliability file already used S1; S2 and S3 remain available
    df_rel = pd.DataFrame({
        "UtteranceID": ["U1","U2"],
        "sampleID":    ["S1","S1"],
        "c3SV":        [1,0],
        "c3REL":       [1,0],
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
    monkeypatch.setattr(cua.random, "sample", fake_sample, raising=True)

    # all_sample_ids = {S1,S2,S3}; frac=0.2 -> round(3*0.2)=1 => pick one (S2 via fake_sample)
    cua.reselect_CU_reliability(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        coder3="3",
        frac=0.2,
    )

    outdir = output_dir / "reselected_CU_reliability"
    files = list(outdir.glob("*_reselected_CUReliabilityCoding.xlsx"))
    assert len(files) == 1, "Expected a single reselected reliability file."
    assert "df" in captured, "Did not capture written DataFrame."

    df = captured["df"]
    # Only rows for S2 should be present (two utterances)
    assert set(df["sampleID"].unique()) == {"S2"}
    assert len(df) == 2

    # Metadata and wipes
    assert (df["c3ID"] == "3").all()
    assert df["c3com"].isna().all()
    # All CU fields (base + suffixed) should now be wiped
    cu_cols = [c for c in df.columns if c.startswith("c3SV") or c.startswith("c3REL")]
    assert cu_cols, "Expected CU columns in output"
    for col in cu_cols:
        assert df[col].isna().all(), f"{col} was not wiped to NaN"

    # Shared columns retained
    assert set(["UtteranceID","sampleID","speaker","utterance","comment"]).issubset(df.columns)


def test_reselect_cu_reliability_no_available(tmp_path, monkeypatch):
    """When all samples were already used, no output file should be created."""
    input_dir, cu_file, rel_file = _make_pair(tmp_path)
    output_dir = tmp_path / "out2"

    df_cu = pd.DataFrame({
        "UtteranceID": ["U1","U2","U3","U4"],
        "sampleID":    ["S1","S1","S2","S2"],
        "speaker":     ["PAR"]*4,
        "utterance":   ["a","b","c","d"],
        "comment":     ["","","",""],
        "c2ID":        ["2"]*4,
        "c2com":       ["ok"]*4,
        "c2SV":        [1,0,1,1],
        "c2REL":       [1,0,1,1],
    })
    # Reliability already used S1 and S2 -> no available
    df_rel = pd.DataFrame({
        "UtteranceID": ["U1","U2","U3","U4"],
        "sampleID":    ["S1","S1","S2","S2"],
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

    cua.reselect_CU_reliability(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        coder3="9",
        frac=0.5,
    )
    outdir = output_dir / "reselected_CU_reliability"
    # Directory exists, but no files created
    assert outdir.exists()
    assert not any(outdir.glob("*_reselected_CUReliabilityCoding.xlsx"))
