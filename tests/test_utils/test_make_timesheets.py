from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target
try:
    from rascal.utils import make_timesheets as mts
except Exception as e:
    pytest.skip(f"Could not import rascal.utils.make_timesheets: {e}", allow_module_level=True)


class MockTier:
    def __init__(self, idx=0, partition=True):
        self.idx = idx
        self.partition = partition
    def match(self, filename, return_None=False):
        base = Path(filename).stem
        parts = base.split("_")
        return parts[self.idx] if len(parts) > self.idx else None


def _make_file(tmp_path):
    """Create a dummy utterance file so rglob can find it."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "TU_P01_Utterances.xlsx").write_bytes(b"stub")
    return input_dir


def test_make_timesheets_basic(tmp_path, monkeypatch):
    input_dir = _make_file(tmp_path)
    output_dir = tmp_path / "out"

    # Fake utterance DataFrame
    uttdf = pd.DataFrame({
        "utterance_id": ["U1","U2","U3"],
        "site": ["TU","TU","TU"],
        "participantID": ["P01","P01","P01"],
        "speaker": ["PAR","INV","PAR"],
        "utterance": ["hello","world","foo"],
        "comment": ["","",""],
    })

    captured = {}
    def fake_read_excel(path, *a, **k):
        return uttdf.copy()

    def fake_to_excel(self, path, index=False):
        captured["df"] = self.copy()
        Path(os.fspath(path)).parent.mkdir(parents=True, exist_ok=True)
        Path(os.fspath(path)).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    tiers = {"site": MockTier(idx=0, partition=True),
             "participantID": MockTier(idx=1, partition=False)}

    mts.make_timesheets(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    outdir = output_dir / "TimeSheets" / "TU" / "P01"
    outfile = outdir / "TU_P01_SpeakingTimes.xlsx"
    assert outfile.exists(), "Expected speaking times file not created."
    assert "df" in captured, "Did not capture written DataFrame."

    df = captured["df"]
    # Should no longer have utterance-level columns
    for col in ["utterance_id","speaker","utterance","comment"]:
        assert col not in df.columns
    # Should have new blank time columns
    for col in ["total_time","clinician_time","client_time"]:
        assert col in df.columns
        assert df[col].isna().all()
    # Sorted by tiers (site, participantID)
    assert list(df.columns[:2]) == ["site","participantID"]
