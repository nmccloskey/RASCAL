from __future__ import annotations

import json
from subprocess import CompletedProcess

from rascal.config import init_project, load_project_config
from rascal.runner import run_diaad_passthrough, run_stage


def _config(tmp_path):
    result = init_project(tmp_path, profile="lab_full")
    return load_project_config(result.config_path)


def _successful_runner(calls):
    def fake(command, **_kwargs):
        calls.append(tuple(command))
        return CompletedProcess(command, 0, stdout=f"ran {' '.join(command)}\n", stderr="")

    return fake


def test_dry_run_does_not_call_subprocess(tmp_path):
    calls = []

    result = run_stage(
        _config(tmp_path),
        "common",
        "1",
        dry_run=True,
        subprocess_run=_successful_runner(calls),
    )

    assert result.status == "dry_run"
    assert result.exit_code == 0
    assert calls == []
    assert result.artifacts.manifest_path.is_file()
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["command_results"] == []


def test_successful_fake_diaad_run_writes_manifest_and_command_logs(tmp_path):
    calls = []

    result = run_stage(
        _config(tmp_path),
        "monolog",
        "7m",
        subprocess_run=_successful_runner(calls),
    )

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert calls == [
        ("diaad", "cus", "analyze", "--config", "config/diaad.generated"),
        ("diaad", "templates", "times", "--config", "config/diaad.generated"),
        ("diaad", "words", "files", "--config", "config/diaad.generated"),
    ]
    assert result.artifacts.command_log_path.is_file()
    assert "diaad cus analyze --config config/diaad.generated" in (
        result.artifacts.command_log_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert len(manifest["command_results"]) == 3
    assert result.command_results[0].stdout_path.read_text(encoding="utf-8").startswith("ran diaad")


def test_failed_fake_diaad_run_records_failure_and_stops(tmp_path):
    calls = []

    def fake(command, **_kwargs):
        calls.append(tuple(command))
        return CompletedProcess(
            command,
            9 if command[1:3] == ["templates", "times"] else 0,
            stdout="",
            stderr="template failure\n" if command[1:3] == ["templates", "times"] else "",
        )

    result = run_stage(_config(tmp_path), "monolog", "7m", subprocess_run=fake)

    assert result.status == "failed"
    assert result.exit_code == 9
    assert len(calls) == 2
    assert "templates times" in result.message
    assert result.command_results[-1].stderr_path.read_text(encoding="utf-8") == "template failure\n"
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["command_results"][-1]["returncode"] == 9


def test_preflight_errors_block_execution_but_write_manifest(tmp_path):
    config = _config(tmp_path)
    config.resolve_path("auto_transcripts_dir").rmdir()
    calls = []

    result = run_stage(
        config,
        "monolog",
        "4m",
        subprocess_run=_successful_runner(calls),
    )

    assert result.status == "blocked_preflight"
    assert result.exit_code == 1
    assert calls == []
    assert result.artifacts.manifest_path.is_file()
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_preflight"
    assert manifest["preflight"]["counts"]["error"] >= 1


def test_diaad_passthrough_uses_safe_command_vector():
    calls = []

    result = run_diaad_passthrough(
        ("transcripts", "tabularize"),
        subprocess_run=_successful_runner(calls),
    )

    assert result.returncode == 0
    assert result.command == ("diaad", "transcripts", "tabularize")
    assert calls == [("diaad", "transcripts", "tabularize")]
