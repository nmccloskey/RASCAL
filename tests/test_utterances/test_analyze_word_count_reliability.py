from pathlib import Path
import os
import pandas as pd
import pytest

# Import target
try:
    from rascal.coding import word_count_reliability_analyzer as wcra
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.word_count_reliability_analyzer: {e}", allow_module_level=True)


class MockTier:
    def __init__(self, idx=0, partition=True):
        self.idx = idx
        self.partition = partition
    def match(self, filename, return_None=False):
        base = Path(filename).stem
        parts = base.split("_")
        return parts[self.idx] if len(parts) > self.idx else None


def _make_files(tmp_path):
    """Create dummy WordCounting + WordCountingReliability files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "TU_P01_WordCounting.xlsx").write_bytes(b"stub")
    (input_dir / "TU_P01_WordCountingReliability.xlsx").write_bytes(b"stub")
    return input_dir


def test_analyze_word_count_reliability(tmp_path, monkeypatch):
    input_dir = _make_files(tmp_path)
    output_dir = tmp_path / "out"

    # Build coding and reliability frames
    WCdf = pd.DataFrame({
        "utterance_id": ["U1","U2","U3"],
        "wordCount": [5, 6, 7],
    })
    # Reliability has slightly different counts
    WCreldf = pd.DataFrame({
        "utterance_id": ["U1","U2","U3"],
        "WCrelCom": ["ok","ok","ok"],
        "wordCount": [5, 7, 6],  # small differences
    })

    captured = {}
    def fake_read_excel(path, *a, **k):
        p = os.fspath(path)
        if p.endswith("Reliability.xlsx"):
            return WCreldf.copy()
        return WCdf.copy()

    def fake_to_excel(self, path, index=False):
        if path.endswith("ReliabilityResults.xlsx"):
            captured["results"] = self.copy()
        Path(os.fspath(path)).parent.mkdir(parents=True, exist_ok=True)
        Path(os.fspath(path)).write_bytes(b"stub")

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    tiers = {"site": MockTier(idx=0, partition=True)}

    wcra.analyze_word_count_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    outdir = output_dir / "WordCountReliability" / "TU"
    results_file = outdir / "TU_WordCountingReliabilityResults.xlsx"
    report_file = outdir / "TU_WordCountReliabilityReport.txt"

    assert results_file.exists()
    assert report_file.exists()
    assert "results" in captured

    # Check merged results
    df = captured["results"]
    assert set(["utterance_id","wordCount_org","wordCount_rel","AbsDiff","PercDiff","PercSim","AG"]).issubset(df.columns)

    # Check report contents
    txt = report_file.read_text(encoding="utf-8")
    assert "Word Count Reliability Report for TU" in txt
    assert "Intraclass Correlation Coefficient" in txt
    # At least one utterance agreed (AG==1)
    assert df["AG"].sum() >= 1
