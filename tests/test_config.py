from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rascal.config import ConfigError, load_project_config


def _write_config(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_missing_config_gives_useful_error(tmp_path):
    with pytest.raises(ConfigError, match="RASCAL config not found"):
        load_project_config(project_root=tmp_path)


def test_project_config_merges_with_packaged_profile(tmp_path):
    config_path = _write_config(
        tmp_path / "config" / "rascal.yaml",
        {
            "rascal": {
                "profile": "lab_monolog",
                "project_root": ".",
                "random_seed": 123,
                "metadata_fields": {"test": ["Screening"]},
            }
        },
    )

    config = load_project_config(config_path)

    assert config.profile_name == "lab_monolog"
    assert config.resolved_profile["random_seed"] == 123
    assert config.resolved_profile["metadata_fields"]["test"] == ["Screening"]
    assert config.resolved_profile["metadata_fields"]["narrative"]
    assert config.resolved_profile["blind_columns"] == ["site", "test"]


def test_explicit_config_path_is_honored(tmp_path):
    config_path = _write_config(
        tmp_path / "custom" / "rascal_custom.yaml",
        {"rascal": {"profile": "lab_dialog", "project_root": "."}},
    )

    config = load_project_config(config_path)

    assert config.config_path == config_path.resolve()
    assert config.profile_name == "lab_dialog"


def test_canonical_paths_resolve_relative_to_project_root(tmp_path):
    config_path = _write_config(
        tmp_path / "config" / "rascal.yaml",
        {"rascal": {"profile": "lab_full", "project_root": "."}},
    )

    config = load_project_config(config_path)

    assert config.project_root == tmp_path.resolve()
    assert config.resolve_path("data_dir") == (tmp_path / "data").resolve()
    assert config.resolve_path("monolog_transcript_tables_dir") == (
        tmp_path / "data" / "monolog" / "transcript_tables"
    ).resolve()
    assert config.resolve_path("diaad_config_dir") == (
        tmp_path / "config" / "diaad.generated"
    ).resolve()


def test_project_path_overrides_are_resolved(tmp_path):
    config_path = _write_config(
        tmp_path / "config" / "rascal.yaml",
        {
            "rascal": {
                "profile": "lab_full",
                "project_root": ".",
                "paths": {"data_dir": "custom_data"},
            }
        },
    )

    config = load_project_config(config_path)

    assert config.resolve_path("data_dir") == (tmp_path / "custom_data").resolve()


def test_legacy_layout_paths_are_available(tmp_path):
    config_path = _write_config(
        tmp_path / "config" / "rascal.yaml",
        {"rascal": {"profile": "lab_full", "layout": "legacy", "project_root": "."}},
    )

    config = load_project_config(config_path)

    assert config.layout == "legacy"
    assert config.resolve_path("input_dir") == (tmp_path / "rascal_data" / "input").resolve()
    assert config.resolve_path("output_dir") == (tmp_path / "rascal_data" / "output").resolve()

