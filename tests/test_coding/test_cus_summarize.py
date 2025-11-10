import pandas as pd
import pytest
from pathlib import Path

import rascal.coding.cu_summarization as cu


@pytest.fixture
def mock_tiers(tmp_path):
    """Create minimal mock tier objects with blind and partition attributes."""
    class MockTier:
        def __init__(self, name, blind=True, partition=False):
            self.name = name
            self.blind = blind
            self.partition = partition

        def match(self, _):
            return self.name

        def make_blind_codes(self):
            return {self.name: {"A": "X", "B": "Y"}}

    return {
        "site": MockTier("site", blind=True, partition=True),
        "group": MockTier("group", blind=False, partition=False),
    }


@pytest.fixture
def minimal_datasets(tmp_path):
    """Create minimal Excel datasets expected by summarize_cus."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create fake transcript table
    utt_df = pd.DataFrame({
        "sample_id": [1, 1],
        "utterance_id": [1, 2],
        "utterance": ["Hi", "Bye"],
        "speaker": ["PAR", "PAR"],
        "comment": ["", ""],
        "file": ["file1", "file1"],
        "site": ["A", "A"],
    })
    transcript_path = input_dir / "TU_P01_transcript_tables.xlsx"
    with pd.ExcelWriter(transcript_path) as w:
        utt_df.to_excel(w, index=False)

    # Supporting CU, WC, and sample data
    cu_by_utt = pd.DataFrame({
        "sample_id": [1, 1],
        "utterance_id": [1, 2],
        "c2_comment": ["", ""],
        "c1_CU": [1, 0],
    })
    wc_by_utt = pd.DataFrame({
        "sample_id": [1, 1],
        "utterance_id": [1, 2],
        "word_count": [2, 1],
    })
    cu_by_sample = pd.DataFrame({
        "sample_id": [1],
        "speaking_time": [60],
    })

    for name, df in [
        ("cu_coding_by_utterance", cu_by_utt),
        ("word_counting", wc_by_utt),
        ("cu_coding_by_sample", cu_by_sample),
    ]:
        df.to_excel(input_dir / f"TU_P01_{name}.xlsx", index=False)

    return input_dir, output_dir, transcript_path

def test_summarize_cus_end_to_end(monkeypatch, tmp_path, mock_tiers, minimal_datasets):
    input_dir, output_dir, transcript_path = minimal_datasets

    cu_by_utt = input_dir / "TU_P01_cu_coding_by_utterance.xlsx"
    wc_by_utt = input_dir / "TU_P01_word_counting.xlsx"
    cu_by_sample = input_dir / "TU_P01_cu_coding_by_sample.xlsx"

    def fake_find_files(match_tiers=None, directories=None, search_base="", **_):
        if "cu_coding_by_utterance" in search_base:
            return [cu_by_utt]
        elif "word_counting" in search_base:
            return [wc_by_utt]
        elif "cu_coding_by_sample" in search_base:
            return [cu_by_sample]
        elif "transcript_tables" in search_base:
            return [transcript_path]
        return []

    monkeypatch.setattr(cu, "find_files", fake_find_files)
    monkeypatch.setattr(cu, "extract_transcript_data", lambda p: pd.read_excel(p))

    cu.summarize_cus(mock_tiers, input_dir, output_dir)
