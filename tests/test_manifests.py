from __future__ import annotations

import json
from datetime import datetime

import rascal.manifests as manifests
from rascal.config import init_project, load_project_config
from rascal.runner import run_stage


def _config(tmp_path):
    result = init_project(tmp_path, profile="lab_full")
    return load_project_config(result.config_path)


def test_manifest_has_required_private_data_light_keys(tmp_path):
    result = run_stage(_config(tmp_path), "common", "1", dry_run=True)

    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))

    for key in (
        "schema_version",
        "rascal_version",
        "diaad_version",
        "status",
        "profile",
        "branch",
        "stage_id",
        "stage_type",
        "diaad_commands",
        "generated_config_path",
        "expected_inputs",
        "expected_outputs",
        "random_seed",
        "reliability_fraction",
        "thresholds",
        "preflight",
        "command_results",
    ):
        assert key in manifest
    assert manifest["random_seed"] == 99
    assert manifest["reliability_fraction"] == 0.2


def test_manifest_does_not_embed_transcript_file_contents(tmp_path):
    config = _config(tmp_path)
    transcript = config.resolve_path("auto_transcripts_dir") / "AC001_Pre_BrokenWindow.cha"
    transcript.write_text(
        "@Begin\n*PAR:\tPRIVATE_TRANSCRIPT_CONTENT_SHOULD_NOT_APPEAR .\n@End\n",
        encoding="utf-8",
    )

    result = run_stage(config, "monolog", "4m", dry_run=True)

    manifest_text = result.artifacts.manifest_path.read_text(encoding="utf-8")
    assert "PRIVATE_TRANSCRIPT_CONTENT_SHOULD_NOT_APPEAR" not in manifest_text
    assert "AC001_Pre_BrokenWindow.cha" in manifest_text


def test_missing_diaad_version_is_handled_gracefully(monkeypatch):
    def missing_version(_package_name):
        raise manifests.metadata.PackageNotFoundError

    monkeypatch.setattr(manifests.metadata, "version", missing_version)

    assert manifests.discover_diaad_version() is None


def test_unique_run_directories_get_suffixes(tmp_path):
    config = _config(tmp_path)
    now = datetime(2026, 6, 24, 12, 0, 0)
    first = manifests.create_run_dir(config, "common", "1", now=now)
    second = manifests.create_run_dir(config, "common", "1", now=now)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert second.name.endswith("_02")
