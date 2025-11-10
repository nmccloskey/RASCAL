import pandas as pd
from pathlib import Path
from rascal.transcripts.transcription_reliability_selection import reselect_transcription_reliability_samples


def _make_reliability_excel(path: Path):
    """Create a mock transcription_reliability_samples.xlsx file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_files = [f"file_{i}.cha" for i in range(1, 6)]
    used_files = all_files[:2]  # first two already used

    df_all = pd.DataFrame({"file": all_files, "site": ["A"] * 5, "group": ["G1"] * 5})
    df_rel = pd.DataFrame({"file": used_files, "site": ["A"] * 2, "group": ["G1"] * 2})

    with pd.ExcelWriter(path) as writer:
        df_rel.to_excel(writer, sheet_name="reliability_selection", index=False)
        df_all.to_excel(writer, sheet_name="all_transcripts", index=False)
    return path


def test_reselect_transcription_reliability_samples(tmp_path):
    """Test that reselect_transcription_reliability_samples correctly outputs new Excel with unused files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a mock existing reliability Excel file
    mock_file = input_dir / "mock_transcription_reliability_samples.xlsx"
    _make_reliability_excel(mock_file)

    # Run reselection
    reselect_transcription_reliability_samples(input_dir, output_dir, frac=0.4)

    # Verify output directory and file
    reselect_dir = output_dir / "reselected_transcription_reliability"
    assert reselect_dir.exists(), "Expected reselection directory not created"

    out_files = list(reselect_dir.glob("reselected_*.xlsx"))
    assert out_files, "Expected reselection Excel file not created"

    # Read new reselection file
    new_df = pd.read_excel(out_files[0], sheet_name="reselected_reliability")

    # Check that new_df excludes already used files
    assert not set(new_df["file"]).intersection({"file_1.cha", "file_2.cha"}), \
        "Reselection included already used files"

    # Check that selected files are from remaining candidates
    remaining_candidates = {"file_3.cha", "file_4.cha", "file_5.cha"}
    assert set(new_df["file"]).issubset(remaining_candidates), \
        "Reselection picked unexpected files"

    # Check that at least one file was selected
    assert len(new_df) >= 1, "No files selected in reselection process"

    # Confirm correct sheet and data structure
    xls = pd.ExcelFile(out_files[0])
    assert "reselected_reliability" in xls.sheet_names
    assert {"file", "site", "group"}.issubset(new_df.columns)
