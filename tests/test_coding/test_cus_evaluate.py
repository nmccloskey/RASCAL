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
    """Minimal mock tier that extracts labels from filenames split by underscores."""
    def __init__(self, name, idx, partition=False):
        self.name = name
        self.idx = idx
        self.partition = partition

    def match(self, filename, return_None=False):
        parts = Path(filename).stem.split("_")
        try:
            return parts[self.idx]
        except Exception:
            return None if return_None else ""


def _make_input_files(tmp_path):
    """Create empty input files so rglob finds them (actual content stubbed)."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "TU_P01_cu_coding.xlsx").write_bytes(b"")
    (input_dir / "TU_P01_cu_reliability_coding.xlsx").write_bytes(b"")
    return input_dir


def _stub_read_excel(monkeypatch):
    """Stub pandas.read_excel to return synthetic coding and reliability DataFrames."""
    coding_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id": ["S1", "S1", "S1"],
        "c2_sv": [1, 0, 1],
        "c2_rel": [1, 0, 1],
        "c2_sv_SAE": [1, 0, 1],
        "c2_rel_SAE": [1, 0, 1],
        "c2_sv_AAE": [1, 0, 1],
        "c2_rel_AAE": [1, 0, 1],
    })

    reliability_df = pd.DataFrame({
        "utterance_id": ["U1", "U2", "U3"],
        "sample_id": ["S1", "S1", "S1"],
        "c3_sv": [1, 0, 0],
        "c3_rel": [1, 0, 0],
        "c3_sv_SAE": [1, 0, 0],
        "c3_rel_SAE": [1, 0, 0],
        "c3_sv_AAE": [1, 0, 0],
        "c3_rel_AAE": [1, 0, 0],
    })

    def fake_read_excel(path, *args, **kwargs):
        p = os.fspath(path)
        if "_cu_reliability_coding" in p:
            return reliability_df.copy()
        elif "_cu_coding" in p:
            return coding_df.copy()
        else:
            raise AssertionError(f"Unexpected file requested: {p}")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)


def _stub_to_excel(monkeypatch):
    """Stub DataFrame.to_excel to avoid openpyxl dependency."""
    def fake_to_excel(self, path, index=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)


@pytest.fixture
def tiers():
    return {
        "site": MockTier("site", idx=0, partition=True),
        "participantID": MockTier("participantID", idx=1, partition=False),
    }


def test_evaluate_cu_reliability_zero_and_one_paradigm(tmp_path, monkeypatch, tiers):
    """
    For 0 or 1 paradigms, evaluate_cu_reliability should:
      - use base columns,
      - write output under <out>/cu_reliability/<partition_label>/,
      - not create paradigm subfolders.
    """
    input_dir = _make_input_files(tmp_path)
    output_dir = tmp_path / "out"
    _stub_read_excel(monkeypatch)
    _stub_to_excel(monkeypatch)

    # Case: 0 paradigms
    cua.evaluate_cu_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=[],
    )

    base_dir = output_dir / "cu_reliability" / "TU"
    assert base_dir.is_dir(), "Expected cu_reliability/TU directory."

    by_utt = base_dir / "TU_cu_reliability_coding_by_utterance.xlsx"
    by_samp = base_dir / "TU_cu_reliability_coding_by_sample.xlsx"
    report = base_dir / "TU_cu_reliability_coding_report.txt"

    for p in (by_utt, by_samp, report):
        assert p.exists(), f"Missing expected output: {p}"

    txt = report.read_text(encoding="utf-8")
    assert "CU Reliability Coding Report" in txt
    assert "out of 1 total samples" in txt
    assert "Average agreement on CU: 66.667" in txt

    # Case: 1 paradigm
    cua.evaluate_cu_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=["SAE"],
    )
    # No subfolder expected
    assert not (output_dir / "cu_reliability" / "SAE").exists()


def test_evaluate_cu_reliability_two_paradigms(tmp_path, monkeypatch, tiers):
    """
    For 2+ paradigms, evaluate_cu_reliability should:
      - use suffixed columns,
      - create paradigm subfolders under cu_reliability/,
      - and write expected outputs per paradigm.
    """
    input_dir = _make_input_files(tmp_path)
    output_dir = tmp_path / "out2"
    _stub_read_excel(monkeypatch)
    _stub_to_excel(monkeypatch)

    cua.evaluate_cu_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=["SAE", "AAE"],
    )

    for paradigm in ("SAE", "AAE"):
        pdir = output_dir / "cu_reliability" / paradigm / "TU"
        assert pdir.is_dir(), f"Missing directory for {paradigm}"

        by_utt = pdir / f"TU_{paradigm}_cu_reliability_coding_by_utterance.xlsx"
        by_samp = pdir / f"TU_{paradigm}_cu_reliability_coding_by_sample.xlsx"
        report = pdir / f"TU_{paradigm}_cu_reliability_coding_report.txt"

        for p in (by_utt, by_samp, report):
            assert p.exists(), f"Missing expected output: {p}"

        txt = report.read_text(encoding="utf-8")
        assert "CU Reliability Coding Report" in txt
        assert "Average agreement on CU: 66.667" in txt
