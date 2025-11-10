from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target module
try:
    from rascal.coding import corelex as clx
except Exception as e:
    pytest.skip(f"Could not import rascal.coding.corelex: {e}", allow_module_level=True)


def _make_placeholder_files(tmp_path, *, unblind=False, fallback=False):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if unblind:
        p = input_dir / "unblind_utterance_data.xlsx"
        p.write_bytes(b"stub")
        paths.append(p)

    if fallback:
        u = input_dir / "TU_P01_transcript_tables.xlsx"
        u.write_bytes(b"stub")
        paths.append(u)

    return input_dir, paths


@pytest.fixture
def dummy_tiers():
    class DummyTier:
        def __init__(self, name, partition=False):
            self.name = name
            self.partition = partition

    return {"file": DummyTier("file", partition=False)}


def test_run_corelex_unblind_mode(tmp_path, monkeypatch, dummy_tiers):
    input_dir, _ = _make_placeholder_files(tmp_path, unblind=True)
    output_dir = tmp_path / "out"

    unblind_df = pd.DataFrame({
        "sample_id": ["S1"],
        "narrative": ["Sandwich"],
        "utterance": ["put peanut butter on bread then put two together"],
        "speaking_time": [120],
        "c2_cu": [1],
    })

    # Monkeypatch I/O
    def fake_read_excel(path, *a, **k):
        if str(path).endswith("unblind_utterance_data.xlsx"):
            return unblind_df.copy()
        raise AssertionError(f"Unexpected read_excel path in unblind test: {path}")

    captured = {}
    def fake_to_excel(self, path, index=False):
        if str(path).endswith(".xlsx"):
            captured["df"] = self.copy()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    # Stub norms + percentile calculations
    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col: {"control_percentile": 60.0, "pwa_percentile": 80.0})

    clx.run_corelex(dummy_tiers, Path(input_dir), Path(output_dir))

    corelex_dir = output_dir / "core_lex"
    files = list(corelex_dir.glob("core_lex_data_*.xlsx"))
    assert files, "Expected a core_lex_data_<timestamp>.xlsx file."
    assert "df" in captured

    df = captured["df"]
    row = df.iloc[0]
    assert row["sample_id"] == "S1"
    assert row["narrative"] == "Sandwich"
    assert row["num_tokens"] == 9
    assert row["num_core_words"] == 8
    assert row["num_core_word_tokens"] == 9
    assert row["speaking_time"] == 120
    assert pytest.approx(row["core_words_per_min"], 1e-6) == 4.0
    assert row["core_words_control_percentile"] == 60.0
    assert row["core_words_pwa_percentile"] == 80.0
    assert row["cwpm_control_percentile"] == 60.0
    assert row["cwpm_pwa_percentile"] == 80.0
    token_cols = [c for c in df.columns if c.startswith("San_")]
    assert token_cols, "Expected Sandwich token columns in output."
    assert "San_bread" in df.columns
    assert "San_put" in df.columns


def test_run_corelex_transcript_table_mode(tmp_path, monkeypatch, dummy_tiers):
    input_dir, _ = _make_placeholder_files(tmp_path, fallback=True)
    output_dir = tmp_path / "out2"

    utts = pd.DataFrame({
        "sample_id": ["S2", "S2"],
        "narrative": ["Sandwich", "Sandwich"],
        "speaker": ["INV", "PAR"],
        "utterance": ["bread butter", "put"],
        "speaking_time": [np.nan, 60]
    })

    def fake_read_excel(path, *a, **k):
        if str(path).endswith("_transcript_tables.xlsx"):
            return utts.copy()
        raise AssertionError(f"Unexpected read_excel path in fallback test: {path}")

    captured = {}
    def fake_to_excel(self, path, index=False):
        if str(path).endswith(".xlsx"):
            captured["df"] = self.copy()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col: {"control_percentile": 55.0, "pwa_percentile": 65.0})

    # Patch all possible Excel readers inside CoreLex
    monkeypatch.setattr(clx, "_read_excel_safely", lambda path: utts.copy())
    monkeypatch.setattr(clx, "extract_transcript_data", lambda path: utts.copy())

    clx.run_corelex(dummy_tiers, Path(input_dir), Path(output_dir), exclude_participants={"INV"})

    corelex_dir = output_dir / "core_lex"
    files = list(corelex_dir.glob("core_lex_data_*.xlsx"))
    assert files, "Expected a core_lex_data_<timestamp>.xlsx file."
    assert "df" in captured

    df = captured["df"]
    row = df.iloc[0]
    assert row["sample_id"] == "S2"
    assert row["narrative"] == "Sandwich"
    assert row["num_core_words"] == 1
    assert row["num_core_word_tokens"] == 1
    assert row["speaking_time"] == 60
    assert pytest.approx(row["core_words_per_min"], 1e-6) == 1.0
    assert row["core_words_control_percentile"] == 55.0
    assert row["core_words_pwa_percentile"] == 65.0
    assert row["cwpm_control_percentile"] == 55.0
    assert row["cwpm_pwa_percentile"] == 65.0
