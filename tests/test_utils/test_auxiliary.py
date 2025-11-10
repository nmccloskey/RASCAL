import yaml
import pandas as pd
from pathlib import Path
import pytest
from rascal.utils.auxiliary import (
    project_path,
    as_path,
    find_config_file,
    find_files,
    extract_transcript_data,
)


def test_project_path(tmp_path):
    """Ensure project_path anchors paths to the current working directory."""
    result = project_path("subdir", "file.txt")
    assert isinstance(result, Path)
    assert "subdir" in str(result)
    assert result.is_absolute()


def test_as_path_relative_and_absolute(tmp_path, monkeypatch):
    """Test as_path behavior for relative, absolute, and external paths."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    # Relative path under cwd
    rel_path = as_path("data/file.txt")
    assert rel_path == Path("data/file.txt")
    assert not rel_path.is_absolute()

    # Absolute path within cwd should resolve to relative
    abs_path = as_path(cwd / "data/file.txt")
    assert abs_path == Path("data/file.txt")
    assert not abs_path.is_absolute()

    # Path outside cwd returns absolute
    external = tmp_path.parent / "outside.txt"
    result = as_path(external)
    assert result.is_absolute()
    assert "outside.txt" in str(result)


def test_find_config_file(tmp_path, monkeypatch):
    """Test config file discovery with explicit, default, and fallback logic."""
    # Case 1: explicit user argument
    cfg = tmp_path / "explicit.yaml"
    cfg.write_text("a: 1")
    assert find_config_file(tmp_path, str(cfg)) == cfg.resolve()

    # Case 2: cwd contains config.yaml
    cwd = tmp_path / "cwd_case"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    default_cfg = cwd / "config.yaml"
    default_cfg.write_text("b: 2")
    assert find_config_file(cwd) == default_cfg.resolve()

    # Case 3: fallback to input/config directory
    fallback_dir = cwd / "input" / "config"
    fallback_dir.mkdir(parents=True)
    fb = fallback_dir / "fallback.yaml"
    fb.write_text("c: 3")
    default_cfg.unlink()  # remove config.yaml so fallback is used
    assert find_config_file(cwd) == fb.resolve()

    # Case 4: missing config raises
    fb.unlink()
    with pytest.raises(FileNotFoundError):
        find_config_file(cwd)


def test_find_files_basic(tmp_path):
    """Ensure find_files locates and filters files correctly."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    # Create mock Excel files
    f1 = dir1 / "A_B_data.xlsx"
    f2 = dir2 / "A_B_data.xlsx"
    f3 = dir2 / "A_C_data.xlsx"
    for f in (f1, f2, f3):
        f.write_text("stub")

    matches = find_files(match_tiers=["A", "B"], directories=[dir1, dir2], search_base="data")
    # Deduplication should reduce duplicate names (f1 & f2)
    assert len(matches) == 1
    assert all(f.suffix == ".xlsx" for f in matches)

    # Nonexistent directory
    result = find_files(match_tiers=["X"], directories=[tmp_path / "fake"], search_base="data")
    assert result == []


def test_extract_transcript_data(tmp_path):
    """End-to-end test of extract_transcript_data across all modes."""
    excel_path = tmp_path / "test_transcripts.xlsx"

    # Create dummy sample and utterance DataFrames
    sample_df = pd.DataFrame({"sample_id": ["S1", "S2"], "speaker": ["A", "B"]})
    utt_df = pd.DataFrame({"sample_id": ["S1", "S2"], "utterance": ["Hi", "Bye"]})

    with pd.ExcelWriter(excel_path) as writer:
        sample_df.to_excel(writer, sheet_name="samples", index=False)
        utt_df.to_excel(writer, sheet_name="utterances", index=False)

    # 1. joined type
    joined = extract_transcript_data(excel_path, type="joined")
    assert all(col in joined.columns for col in ["sample_id", "speaker", "utterance"])

    # 2. sample type
    sample = extract_transcript_data(excel_path, type="sample")
    assert list(sample.columns) == ["sample_id", "speaker"]

    # 3. utterance type
    utt = extract_transcript_data(excel_path, type="utterance")
    assert list(utt.columns) == ["sample_id", "utterance"]

    # 4. Invalid type
    with pytest.raises(ValueError):
        extract_transcript_data(excel_path, type="invalid")

    # 5. Missing file
    missing = tmp_path / "no_file.xlsx"
    with pytest.raises(FileNotFoundError):
        extract_transcript_data(missing)
