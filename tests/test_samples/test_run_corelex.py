from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target module
try:
    from rascal.samples import corelex as clx
except Exception as e:
    pytest.skip(f"Could not import rascal.samples.corelex: {e}", allow_module_level=True)


def _make_placeholder_files(tmp_path, *, unblind=False, fallback=False):
    """
    Create just the minimal placeholder files so rglob finds them.
    We'll return input_dir and a list of paths.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if unblind:
        p = input_dir / "unblindUtteranceData.xlsx"
        p.write_bytes(b"stub")
        paths.append(p)

    if fallback:
        u = input_dir / "TU_P01_Utterances.xlsx"
        t = input_dir / "TU_P01_SpeakingTimes.xlsx"
        u.write_bytes(b"stub")
        t.write_bytes(b"stub")
        paths.extend([u, t])

    return input_dir, paths


def test_run_corelex_unblind_mode(tmp_path, monkeypatch):
    """
    End-to-end check of UNBLIND mode:
      - Reads a single unblindUtteranceData.xlsx
      - Computes token/core-word stats for a Sandwich narrative
      - Writes CoreLex/CoreLexData_<timestamp>.xlsx
    """
    input_dir, _ = _make_placeholder_files(tmp_path, unblind=True, fallback=False)
    output_dir = tmp_path / "out"

    # Unblind utterances: minimal columns needed downstream
    # sample_id, narrative (must be in urls keys), utterance, client_time, c2CU present to pass filter
    unblind_df = pd.DataFrame({
        "sample_id": ["S1"],
        "narrative": ["Sandwich"],
        "utterance": ["put peanut butter on bread then put two together"],
        "client_time": [120],  # seconds -> minutes=2; cwpm = numCoreWords/2
        "c2CU": [1],
    })

    # Monkeypatch IO
    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("unblindUtteranceData.xlsx"):
            return unblind_df.copy()
        raise AssertionError(f"Unexpected read_excel path in unblind test: {p}")

    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        # capture the CoreLex output
        if os.path.basename(p).startswith("CoreLexData_") and p.endswith(".xlsx"):
            captured["df"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # Avoid external data & SciPy percentiles: stub norm preloading + percentile calc
    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col: {"control_percentile": 60.0, "pwa_percentile": 80.0})

    # Run
    clx.run_corelex(str(input_dir), str(output_dir))

    # Check output file existence
    corelex_dir = output_dir / "CoreLex"
    files = list(corelex_dir.glob("CoreLexData_*.xlsx"))
    assert files, "Expected a CoreLexData_<timestamp>.xlsx to be written."
    assert "df" in captured, "Did not capture written CoreLex DataFrame."

    df = captured["df"]
    # One row for S1
    assert list(df["sample_id"]) == ["S1"]
    assert list(df["narrative"]) == ["Sandwich"]

    # Core stats derived from our utterance:
    # tokens: 9 ("put peanut butter on bread then put two together")
    # distinct core lemmas present: put, peanut, butter, on, bread, then, two, together => 8
    # cw tokens count: 9 (both 'put's count)
    row = df.iloc[0]
    assert row["numTokens"] == 9
    assert row["numCoreWords"] == 8
    assert row["numCoreWordTokens"] == 9
    # speaking time propagated
    assert row["speakingTime"] == 120
    # cwpm = 8 / (120/60) = 4.0
    assert row["coreWordsPerMinute"] == pytest.approx(4.0, abs=1e-6)
    # token columns should include some Sandwich-prefixed entries (San_)
    token_cols = [c for c in df.columns if c.startswith("San_")]
    assert token_cols, "Expected Sandwich token columns (San_*) in output."
    assert df.loc[0, "San_bread"] == "bread"
    assert df.loc[0, "San_put"] == "put"
    # stubbed percentiles appear
    assert row["core_words_control_percentile"] == 60.0
    assert row["core_words_pwa_percentile"] == 80.0
    assert row["cwpm_control_percentile"] == 60.0
    assert row["cwpm_pwa_percentile"] == 80.0


def test_run_corelex_fallback_utterances_mode(tmp_path, monkeypatch):
    """
    End-to-end check of FALLBACK mode:
      - Reads *_Utterances.xlsx and *_SpeakingTimes.xlsx
      - Excludes an investigator utterance
      - Merges speaking time and computes stats
      - Writes CoreLex output
    """
    input_dir, _ = _make_placeholder_files(tmp_path, unblind=False, fallback=True)
    output_dir = tmp_path / "out2"

    # Utterances with a mix of speakers; exclude "INV"
    utts = pd.DataFrame({
        "sampleID": ["S2", "S2"],
        "narrative": ["Sandwich", "Sandwich"],
        "speaker": ["INV", "PAR"],
        # If INV utterance were included, it would add more core words.
        "utterance": ["bread butter", "put"],
    })
    times = pd.DataFrame({
        "sampleID": ["S2"],
        "client_time": [60],  # seconds -> minutes=1; cwpm = numCoreWords/1
    })

    # Monkeypatch IO
    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("_Utterances.xlsx"):
            return utts.copy()
        if p.endswith("_SpeakingTimes.xlsx"):
            return times.copy()
        raise AssertionError(f"Unexpected read_excel path in fallback test: {p}")

    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if os.path.basename(p).startswith("CoreLexData_") and p.endswith(".xlsx"):
            captured["df"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # Stub norms + percentiles
    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col: {"control_percentile": 55.0, "pwa_percentile": 65.0})

    # Run with speaker exclusion
    clx.run_corelex(str(input_dir), str(output_dir), exclude_participants={"INV"})

    corelex_dir = output_dir / "CoreLex"
    files = list(corelex_dir.glob("CoreLexData_*.xlsx"))
    assert files, "Expected a CoreLexData_<timestamp>.xlsx to be written."
    assert "df" in captured

    df = captured["df"]
    # One row for sample S2
    assert list(df["sample_id"]) == ["S2"]
    assert list(df["narrative"]) == ["Sandwich"]

    row = df.iloc[0]
    # Only PAR utterance ("put") should be counted -> 1 core lemma, 1 token
    assert row["numCoreWords"] == 1
    assert row["numCoreWordTokens"] == 1
    assert row["speakingTime"] == 60
    # cwpm = 1 / (60/60) = 1.0
    assert row["coreWordsPerMinute"] == pytest.approx(1.0, abs=1e-6)
    # percentiles from stub
    assert row["core_words_control_percentile"] == 55.0
    assert row["core_words_pwa_percentile"] == 65.0
    assert row["cwpm_control_percentile"] == 55.0
    assert row["cwpm_pwa_percentile"] == 65.0
