from pathlib import Path
import os
import pandas as pd
import pytest

try:
    import rascal.transcripts.transcription_reliability_analysis as tra
except Exception as e:
    pytest.skip(f"Could not import transcription_reliability_analysis: {e}", allow_module_level=True)


class MockTier:
    """Minimal tier with a filename-based matcher."""
    def __init__(self, name, partition=False, idx=0):
        self.name = name
        self.partition = partition
        self.idx = idx

    def match(self, filename: str):
        # Example filenames: TU_P01_Sample.cha, TU_P01_SampleReliability.cha
        base = Path(filename).stem.replace("Reliability", "")
        parts = base.split("_")
        try:
            return parts[self.idx]
        except Exception:
            return None


def _make_fake_reader(text: str):
    """Return an object with .utterances() yielding items that have .participant and .tiers."""
    class _Utt:
        def __init__(self, participant, text):
            self.participant = participant
            self.tiers = {participant: text}

    class _Reader:
        def __init__(self, txt):
            self._utts = [_Utt("PAR", txt)]
        def utterances(self):
            return self._utts

    return _Reader(text)


def test_analyze_transcription_reliability_basic(tmp_path, monkeypatch):
    # --- Arrange ----------------------------------------------------------------
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)

    # Create paired original + reliability .cha files (content not used by our stub)
    org = input_dir / "TU_P01_Sample.cha"
    rel = input_dir / "TU_P01_SampleReliability.cha"
    org.write_text("@UTF8\n*PAR: hello world .\n", encoding="utf-8")
    rel.write_text("@UTF8\n*PAR: hello world .\n", encoding="utf-8")

    # Tiers: one partitioning tier ('site'), one non-partition tier ('participantID')
    tiers = {
        "site": MockTier("site", partition=True, idx=0),
        "participantID": MockTier("participantID", partition=False, idx=1),
    }

    # Stub pylangacq.read_chat to avoid real parsing
    def fake_read_chat(path):
        # Make org/rel texts identical so LevenshteinSimilarity should be 1.0
        return _make_fake_reader("hello world.")
    monkeypatch.setattr(tra.pylangacq, "read_chat", fake_read_chat, raising=True)

    # Stub Needleman–Wunsch to avoid Biopython dependency/behavior variance
    def fake_nw(org_text, rel_text):
        class _Align:
            def __getitem__(self, i):
                return org_text if i == 0 else rel_text
        return {
            "NeedlemanWunschScore": float(max(len(org_text), len(rel_text))),
            "NeedlemanWunschNorm": 1.0,
            "alignment": _Align(),
        }
    monkeypatch.setattr(tra, "_needleman_wunsch_global", fake_nw, raising=True)

    # Stub DataFrame.to_excel to avoid openpyxl requirement; create a tiny file so paths exist
    def fake_to_excel(self, path, index=False):
        path = os.fspath(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # --- Act --------------------------------------------------------------------
    results = tra.analyze_transcription_reliability(
        tiers=tiers,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        test=True  # return the grouped DataFrames
    )

    # --- Assert -----------------------------------------------------------------
    # Expect one grouped DataFrame (partitioned by site = 'TU')
    assert isinstance(results, list) and len(results) == 1
    df = results[0]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # one reliability pair

    # Required columns present
    required_cols = {
        "site", "participantID", "OrgFile", "RelFile",
        "LevenshteinDistance", "LevenshteinSimilarity",
        "NeedlemanWunschScore", "NeedlemanWunschNorm",
        "OrgNumTokens", "RelNumTokens", "OrgNumChars", "RelNumChars"
    }
    missing = required_cols.difference(df.columns)
    assert not missing, f"Missing expected columns: {missing}"

    # Values: labels, file names, perfect similarity on identical texts
    row = df.iloc[0]
    assert row["site"] == "TU"
    assert row["participantID"] == "P01"
    assert row["OrgFile"] == org.name
    assert row["RelFile"] == rel.name
    assert row["LevenshteinSimilarity"] == pytest.approx(1.0, abs=1e-12)
    assert row["LevenshteinDistance"] == 0

    # Alignment pretty-print file exists under GlobalAlignments/
    align_glob = list((output_dir / "TranscriptionReliabilityAnalysis").rglob("GlobalAlignments/*Alignment.txt"))
    assert len(align_glob) == 1, "Expected a single alignment output file."

    # Report exists (partitioned by site)
    report_glob = list((output_dir / "TranscriptionReliabilityAnalysis").rglob("*TranscriptionReliabilityReport.txt"))
    assert len(report_glob) == 1, "Expected a single reliability report file."
