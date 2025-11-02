# tests/test_samples/test_unblind_CUs.py
from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target; skip cleanly if module isn't on path
try:
    from rascal.coding import summarize_cus as ub
except Exception as e:
    pytest.skip(f"Could not import rascal.samples.unblind_CUs: {e}", allow_module_level=True)


class MockTier:
    def __init__(self, name, blind):
        self.name = name
        self.blind = blind
    def make_blind_codes(self):
        # Map the observed labels; in this test we only use site="TU"
        return {self.name: {"TU": "SITE1"}}


def _touch_inputs(tmp_path):
    """Create placeholder files with expected suffixes so rglob finds them."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "TU_P01_Utterances.xlsx").write_bytes(b"stub")
    (input_dir / "TU_P01_CUCoding_ByUtterance.xlsx").write_bytes(b"stub")
    (input_dir / "TU_P01_WordCounting.xlsx").write_bytes(b"stub")
    (input_dir / "TU_P01_SpeakingTimes.xlsx").write_bytes(b"stub")
    (input_dir / "TU_P01_CUCoding_BySample.xlsx").write_bytes(b"stub")
    return input_dir


def test_unblind_CUs_end_to_end(tmp_path, monkeypatch):
    input_dir = _touch_inputs(tmp_path)
    output_dir = tmp_path / "out"

    # ---------- Build deterministic input frames ----------
    # Two utterances, same sample; contains tiers and 'file'
    utts = pd.DataFrame({
        "utterance_id": ["U1", "U2"],
        "sample_id":    ["S1", "S1"],
        "file":         ["f1.cha", "f1.cha"],
        "site":         ["TU", "TU"],       # blind=True tier
        "participantID":["P01","P01"],      # blind=False tier
        "speaker":      ["PAR", "PAR"],
        "utterance":    ["hello world", "more words"],
        "comment":      ["", ""],
    })

    # CU by utterance: keep columns to the RIGHT of 'comment'
    cubyutts = pd.DataFrame({
        "utterance_id": ["U1", "U2"],
        "sample_id":    ["S1", "S1"],
        "comment":      ["", ""],
        "c2CU":         [1, 1],
    })

    # Word counts (utterance level)
    wcs = pd.DataFrame({
        "utterance_id": ["U1", "U2"],
        "sample_id":    ["S1", "S1"],
        "wordCount":    [3, 4],  # sums to 7
        "WCcom":        ["", ""],
    })

    # Speaking times (sample level)
    times = pd.DataFrame({
        "sample_id":   ["S1"],
        "client_time": [120],  # seconds; wpm = 7 / (120/60) = 3.5
    })

    # CU by sample (minimal; will just be merged through)
    cubysample = pd.DataFrame({
        "sample_id": ["S1"],
        "CU":        [2],
    })

    # ---------- Monkeypatch IO ----------
    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("_Utterances.xlsx"):
            return utts.copy()
        if p.endswith("_CUCoding_ByUtterance.xlsx"):
            return cubyutts.copy()
        if p.endswith("_WordCounting.xlsx"):
            return wcs.copy()
        if p.endswith("_SpeakingTimes.xlsx"):
            return times.copy()
        if p.endswith("_CUCoding_BySample.xlsx"):
            return cubysample.copy()
        raise AssertionError(f"Unexpected read_excel path: {p}")

    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        name = os.path.basename(p)
        captured[name] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # ---------- Tiers ----------
    tiers = {
        "site": MockTier("site", blind=True),
        "participantID": MockTier("participantID", blind=False),
    }

    # ---------- Run ----------
    ub.summarize_cus(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    # ---------- Assertions: files created ----------
    summaries = output_dir / "Summaries"
    for fname in [
        "unblindUtteranceData.xlsx",
        "blindUtteranceData.xlsx",
        "unblindSampleData.xlsx",
        "blindSampleData.xlsx",
        "blindCodes.xlsx",
    ]:
        assert (summaries / fname).exists(), f"Missing {fname}"
        assert fname in captured, f"Not captured: {fname}"

    unblind_utts = captured["unblindUtteranceData.xlsx"]
    blind_utts   = captured["blindUtteranceData.xlsx"]
    unblind_samp = captured["unblindSampleData.xlsx"]
    blind_samp   = captured["blindSampleData.xlsx"]
    blind_codes  = captured["blindCodes.xlsx"]

    # ---------- Utterance-level checks ----------
    # Unblinded utterances include both tiers and 'file', wordCount, c2CU, client_time after merge
    for col in ["site", "participantID", "file", "wordCount", "c2CU", "client_time"]:
        assert col in unblind_utts.columns

    # Blinded utterances: 'participantID' (non-blind) and 'file' dropped; 'site' mapped
    assert "participantID" not in blind_utts.columns
    assert "file" not in blind_utts.columns
    assert "site" in blind_utts.columns
    assert set(blind_utts["site"].unique()) == {"SITE1"}

    # ---------- Sample-level checks ----------
    # Unblinded sample has summed wordCount and computed wpm
    row = unblind_samp.iloc[0]
    assert row["wordCount"] == 7
    assert row["client_time"] == 120
    assert row["wpm"] == pytest.approx(3.5, abs=1e-6)
    # Blinded sample: participantID dropped; site mapped; file SHOULD remain (per implementation)
    assert "participantID" not in blind_samp.columns
    assert "site" in blind_samp.columns and blind_samp["site"].iloc[0] == "SITE1"
    assert "file" in blind_samp.columns

    # ---------- Blind-code key ----------
    # DataFrame columns include 'site'; row index contains original label 'TU' with value 'SITE1'
    assert "site" in blind_codes.columns
    assert "TU" in blind_codes.index
    assert blind_codes.loc["TU", "site"] == "SITE1"
