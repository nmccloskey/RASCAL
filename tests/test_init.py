from __future__ import annotations

import re

import pytest
import yaml

from rascal.cli import main
from rascal.config import ConfigError, init_project, load_project_config


def test_init_lab_full_creates_config_and_canonical_directories(tmp_path):
    result = init_project(tmp_path, profile="lab_full", layout="canonical")

    assert result.config_path == (tmp_path / "config" / "rascal.yaml").resolve()
    assert result.config_path.is_file()
    assert (tmp_path / "config" / "diaad.generated").is_dir()
    assert (tmp_path / "data" / "raw_media").is_dir()
    assert (tmp_path / "data" / "monolog" / "transcript_tables").is_dir()
    assert (tmp_path / "data" / "dialog" / "powers_analysis").is_dir()
    assert (tmp_path / "runs").is_dir()

    loaded = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert loaded["rascal"]["profile"] == "lab_full"
    assert loaded["rascal"]["layout"] == "canonical"


def test_init_canonical_layout_has_no_stage_numbered_directory_names(tmp_path):
    init_project(tmp_path, profile="lab_full", layout="canonical")

    relative_parts = {
        part
        for path in (tmp_path / "data").rglob("*")
        if path.is_dir()
        for part in path.relative_to(tmp_path).parts
    }

    assert not any(re.match(r"^\d{2}_", part) for part in relative_parts)


def test_init_legacy_layout_creates_old_input_and_output_dirs(tmp_path):
    result = init_project(tmp_path, profile="lab_full", layout="legacy")

    assert (tmp_path / "rascal_data" / "input").is_dir()
    assert (tmp_path / "rascal_data" / "output").is_dir()
    assert (tmp_path / "config" / "diaad.generated").is_dir()

    loaded = load_project_config(result.config_path)
    assert loaded.layout == "legacy"
    assert loaded.resolve_path("input_dir") == (tmp_path / "rascal_data" / "input").resolve()


def test_init_refuses_to_overwrite_existing_config_without_force(tmp_path):
    init_project(tmp_path, profile="lab_monolog")

    with pytest.raises(ConfigError, match="already exists"):
        init_project(tmp_path, profile="lab_dialog")


def test_init_force_overwrites_generated_config(tmp_path):
    first = init_project(tmp_path, profile="lab_monolog")
    second = init_project(tmp_path, profile="lab_dialog", force=True)

    loaded = yaml.safe_load(second.config_path.read_text(encoding="utf-8"))

    assert first.config_path == second.config_path
    assert loaded["rascal"]["profile"] == "lab_dialog"


def test_init_does_not_create_private_data_files(tmp_path):
    init_project(tmp_path, profile="lab_full")

    assert not any(path.is_file() for path in (tmp_path / "data").rglob("*"))
    assert not any(path.is_file() for path in (tmp_path / "rascal_data").rglob("*"))


def test_cli_init_creates_project(capsys, tmp_path):
    exit_code = main(["init", "--profile", "lab_full", "--project", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Initialized RASCAL project" in output
    assert (tmp_path / "config" / "rascal.yaml").is_file()
    assert (tmp_path / "data" / "dialog" / "transcript_tables").is_dir()


def test_cli_init_reports_config_error_on_existing_config(capsys, tmp_path):
    init_project(tmp_path, profile="lab_full")

    exit_code = main(["init", "--project", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "already exists" in captured.err

