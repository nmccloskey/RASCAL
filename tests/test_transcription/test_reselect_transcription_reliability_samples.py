import pandas as pd
import pytest
from pathlib import Path

from rascal.transcripts.transcription_reliability_selector import (
    reselect_transcription_reliability_samples,
)


@pytest.fixture
def setup_excel(tmp_path):
    """
    Create a fake TranscriptionReliabilitySamples.xlsx file with both
    AllTranscripts and Reliability sheets for testing reselection.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Fake data
    all_data = pd.DataFrame(
        {
            "file": ["f1.cha", "f2.cha", "f3.cha", "f4.cha"],
            "site": ["AC", "AC", "BU", "BU"],
        }
    )
    rel_data = pd.DataFrame(
        {
            "file": ["f1.cha"],  # already used, should be excluded
            "site": ["AC"],
        }
    )

    excel_path = input_dir / "Test_TranscriptionReliabilitySamples.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        all_data.to_excel(writer, sheet_name="AllTranscripts", index=False)
        rel_data.to_excel(writer, sheet_name="Reliability", index=False)

    return input_dir, output_dir, excel_path, all_data, rel_data


def test_reselect_transcription_reliability_samples(setup_excel):
    input_dir, output_dir, excel_path, all_data, rel_data = setup_excel

    # Run reselection with frac=0.5 (should give 2 samples from 4 total)
    reselect_transcription_reliability_samples(input_dir, output_dir, frac=0.5)

    reselect_dir = output_dir / "reselected_TranscriptionReliability"
    assert reselect_dir.exists()

    outpath = reselect_dir / f"reselected_{excel_path.name}"
    assert outpath.exists()

    # Load reselection result
    df_reselected = pd.read_excel(outpath, sheet_name="Reselected")

    # Verify that 'file' column exists
    assert "file" in df_reselected.columns

    # None of the reselected files should be in the original Reliability sheet
    used_files = set(rel_data["file"])
    assert all(f not in used_files for f in df_reselected["file"])

    # Should not exceed total number of candidates
    candidates = set(all_data["file"]) - used_files
    assert set(df_reselected["file"]).issubset(candidates)

    # Sample size should be min(max(1, round(frac * len(all))), len(candidates))
    expected_n = max(1, round(0.5 * len(all_data)))
    expected_n = min(expected_n, len(candidates))
    assert len(df_reselected) == expected_n
