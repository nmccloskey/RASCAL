# tests/test_analyze_CU_reliability.py
from pathlib import Path
import os
import pandas as pd
import pytest

# Try to import the target module; skip cleanly if the package path isn't available.
try:
    from rascal.utterances import CU_analyzer as cua
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.CU_analyzer: {e}", allow_module_level=True)


class MockTier:
    """
    Minimal tier mock that extracts labels from filenames split on underscores.
    Example filenames: TU_P01_CUCoding.xlsx, TU_P01_CUReliabilityCoding.xlsx
    """
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


def _make_input_files(tmp_path):
    """
    Create empty files so rglob finds them. The test stubs read_excel, so content is irrelevant.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    # One coding and one reliability file with matching tier labels
    (input_dir / "TU_P01_CUCoding.xlsx").write_bytes(b"")
    (input_dir / "TU_P01_CUReliabilityCoding.xlsx").write_bytes(b"")
    return input_dir


def _stub_read_excel(monkeypatch):
    """
    Stub pandas.read_excel to return synthetic frames for coding vs reliability.
    Includes both base and suffixed (SAE/AAE) columns so we can reuse it across tests.
    """
    coding_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id":    ["S1", "S1", "S1"],
        # Base columns (used for 0/1-paradigm paths)
        "c2SV":  [1, 0, 1],
        "c2REL": [1, 0, 1],
        # Suffixed columns (used when paradigms >= 2)
        "c2SV_SAE":  [1, 0, 1],
        "c2REL_SAE": [1, 0, 1],
        "c2SV_AAE":  [1, 0, 1],
        "c2REL_AAE": [1, 0, 1],
    })

    reliability_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        # Base columns
        "c3SV":  [1, 0, 0],  # U3 disagrees with coder 2
        "c3REL": [1, 0, 0],
        # Suffixed
        "c3SV_SAE":  [1, 0, 0],
        "c3REL_SAE": [1, 0, 0],
        "c3SV_AAE":  [1, 0, 0],
        "c3REL_AAE": [1, 0, 0],
    })

    def fake_read_excel(path, *args, **kwargs):
        p = os.fspath(path)
        if "CUReliabilityCoding" in p:
            return reliability_df.copy()
        return coding_df.copy()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)


def _stub_to_excel(monkeypatch):
    """Stub DataFrame.to_excel to avoid openpyxl and just drop a tiny file."""
    def fake_to_excel(self, path, index=False):
        path = os.fspath(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)


@pytest.fixture
def tiers():
    # Two tiers: site (partitioned) and participantID (not partitioned)
    return {
        "site": MockTier("site", idx=0, partition=True),
        "participantID": MockTier("participantID", idx=1, partition=False),
    }


def test_analyze_CU_reliability_zero_and_one_paradigm(tmp_path, monkeypatch, tiers):
    """
    0 and 1 paradigm cases should both run the *base* columns and write into
    <out>/CUReliability/<partition_labels>/ with *no* paradigm suffix or subfolder.
    """
    input_dir = _make_input_files(tmp_path)
    output_dir = tmp_path / "out"
    _stub_read_excel(monkeypatch)
    _stub_to_excel(monkeypatch)

    # Case 0 paradigms
    cua.analyze_CU_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=[],
    )
    base_dir = output_dir / "CUReliability" / "TU"  # partition label from filename
    assert base_dir.is_dir()

    # Expected filenames (no paradigm suffix)
    by_utt = base_dir / "TU_CUReliabilityCoding_ByUtterance.xlsx"
    by_samp = base_dir / "TU_CUReliabilityCoding_BySample.xlsx"
    report = base_dir / "TU_CUReliabilityCodingReport.txt"

    for p in (by_utt, by_samp, report):
        assert p.exists(), f"Missing expected output: {p}"

    # Check report has expected lines (one sample, average CU agreement ~66.667)
    txt = report.read_text(encoding="utf-8")
    assert "CU Reliability Coding Report" in txt
    assert "out of 1 total samples" in txt
    assert "Average agreement on CU: 66.667" in txt

    # Case 1 paradigm (should behave the same as 0 paradigms)
    cua.analyze_CU_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=["SAE"],
    )

    # Ensure no per-paradigm subfolder created for the 1-paradigm case
    assert not (output_dir / "CUReliability" / "SAE").exists()


def test_analyze_CU_reliability_two_paradigms(tmp_path, monkeypatch, tiers):
    """
    With 2+ paradigms, the function should use suffixed columns and create
    separate outputs under per-paradigm subfolders.
    """
    input_dir = _make_input_files(tmp_path)
    output_dir = tmp_path / "out2"
    _stub_read_excel(monkeypatch)
    _stub_to_excel(monkeypatch)

    cua.analyze_CU_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=["SAE", "AAE"],
    )

    # Check subfolders and files for both paradigms
    for paradigm in ("SAE", "AAE"):
        pdir = output_dir / "CUReliability" / paradigm / "TU"
        assert pdir.is_dir(), f"Missing paradigm directory: {pdir}"

        by_utt = pdir / f"TU_{paradigm}_CUReliabilityCoding_ByUtterance.xlsx"
        by_samp = pdir / f"TU_{paradigm}_CUReliabilityCoding_BySample.xlsx"
        report = pdir / f"TU_{paradigm}_CUReliabilityCodingReport.txt"

        for p in (by_utt, by_samp, report):
            assert p.exists(), f"Missing expected output: {p}"

        # Spot-check report content
        txt = report.read_text(encoding="utf-8")
        assert "CU Reliability Coding Report" in txt
        assert "Average agreement on CU: 66.667" in txt
