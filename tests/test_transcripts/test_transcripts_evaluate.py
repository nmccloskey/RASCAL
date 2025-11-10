import pylangacq
import pandas as pd
from pathlib import Path
from rascal.transcripts.transcription_reliability_evaluation import evaluate_transcription_reliability


class MockTier:
    """Minimal mock tier simulating real RASCAL tier behavior."""
    def __init__(self, label, partition=True):
        self.label = label
        self.partition = partition
        self.name = label

    def match(self, fname):
        # return label if it appears in the filename
        return self.label if self.label in fname else None


def _make_cha_file(path: Path, text: str):
    """Write a minimal pseudo-CHAT .cha file (plain text for test)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_evaluate_transcription_reliability(tmp_path, monkeypatch):
    """End-to-end test for evaluate_transcription_reliability with synthetic .cha files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # --- Mock tiers ---
    tiers = {
        "site": MockTier("SiteA", partition=True),
        "group": MockTier("G1", partition=True),
    }

    # --- Create original and reliability .cha files ---
    original_dir = input_dir / "data"
    reliability_dir = input_dir / "data" / "reliability"
    reliability_dir.mkdir(parents=True, exist_ok=True)

    org_text = "*PAR:\tThis is a test.\n%com:\tOriginal comment.\n"
    rel_text = "*PAR:\tThis is test.\n%com:\tReliability comment.\n"

    org_file = _make_cha_file(original_dir / "SiteA_G1_P01.cha", org_text)
    rel_file = _make_cha_file(reliability_dir / "SiteA_G1_P01_reliability.cha", rel_text)

    # --- Monkeypatch heavy components to make test fast and deterministic ---

    # Avoid using real pylangacq (mock read_chat to return a stub with utterances)
    class MockUtt:
        def __init__(self, text):
            self.participant = "PAR"
            self.tiers = {"PAR": text}

    class MockReader(pylangacq.Reader):
        def __init__(self, text):
            self._text = text
        def utterances(self):
            return [MockUtt(self._text)]

    monkeypatch.setattr(
        "rascal.transcripts.transcription_reliability_evaluation.pylangacq.read_chat",
        lambda path: MockReader(Path(path).read_text())
    )

    # Monkeypatch _convert_cha_names to return the reliability dir as having been processed
    monkeypatch.setattr(
        "rascal.transcripts.transcription_reliability_evaluation._convert_cha_names",
        lambda input_dir: {"renamed": [], "originals": []}
    )

    # --- Run function ---
    results = evaluate_transcription_reliability(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=["INV"],
        test=True,
    )

    # --- Verify outputs ---
    transc_rel_dir = output_dir / "transcription_reliability_evaluation" / "SiteA" / "G1"
    assert transc_rel_dir.exists(), f"Output directory not created: {transc_rel_dir}"

    # Check that results returned (since test=True)
    assert isinstance(results, list) and len(results) == 1
    df = results[0]
    assert not df.empty, "Returned reliability DataFrame is empty"

    # Verify expected metrics
    expected_cols = {
        "original_file",
        "reliability_file",
        "org_num_tokens",
        "rel_num_tokens",
        "levenshtein_similarity",
        "needleman_wunsch_score",
        "needleman_wunsch_norm",
    }
    assert expected_cols.issubset(df.columns), f"Missing expected columns: {expected_cols - set(df.columns)}"

    # Verify Excel and report outputs exist
    xlsx_files = list(transc_rel_dir.rglob("*evaluation.xlsx"))
    report_files = list(transc_rel_dir.rglob("*report.txt"))
    assert xlsx_files, "Expected reliability Excel file not found"
    assert report_files, "Expected reliability report file not found"

    # Basic content checks for report
    report_text = Path(report_files[0]).read_text()
    assert "Levenshtein" in report_text
    assert "Number of samples" in report_text
