from __future__ import annotations

import json
import tomllib
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rascal import __version__
from rascal.cli import build_parser, main, normalize_command, parse_args
from rascal.config import init_project


def test_help_exits_successfully(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])

    assert excinfo.value.code == 0
    assert "Lab-facing workflow wrapper for DIAAD" in capsys.readouterr().out


def test_version_exits_successfully(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])

    assert excinfo.value.code == 0
    assert f"rascal {__version__}" in capsys.readouterr().out


def test_pyproject_console_script_points_to_new_cli():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["rascal"] == "rascal.cli:main"


def test_mvp_subcommands_appear_in_help(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    help_text = capsys.readouterr().out
    for command in (
        "init",
        "check",
        "plan",
        "run",
        "status",
        "next",
        "workflows",
        "diaad",
        "asr",
    ):
        assert command in help_text


def test_workflow_subcommands_appear_in_help(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["workflows", "--help"])

    help_text = capsys.readouterr().out
    assert "list" in help_text
    assert "show" in help_text


def test_main_for_workflows_list_outputs_archive(capsys, monkeypatch, tmp_path):
    archive = tmp_path / "archived_workflows" / "synthetic"
    archive.mkdir(parents=True)
    (archive / "workflow_manifest.yaml").write_text(
        "workflow_id: synthetic_workflow\nname: Synthetic Workflow\nstatus: archived-test\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["workflows", "list", "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["workflows"][0]["workflow_id"] == "synthetic_workflow"


def test_main_for_workflows_show_with_files_reports_references(
    capsys,
    monkeypatch,
    tmp_path,
):
    archive = tmp_path / "archived_workflows" / "synthetic"
    archive.mkdir(parents=True)
    (archive / "workflow_manifest.yaml").write_text(
        """
workflow_id: synthetic_workflow
name: Synthetic Workflow
key_docs:
  - README.md
key_scripts:
  - src/example.py
""".strip(),
        encoding="utf-8",
    )
    (archive / "README.md").write_text("# Synthetic\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "workflows",
            "show",
            "synthetic_workflow",
            "--files",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["workflow_id"] == "synthetic_workflow"
    assert payload["referenced_files"][0] == {
        "kind": "doc",
        "path": "README.md",
        "exists": True,
    }


def test_asr_subcommands_appear_in_help(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["asr", "--help"])

    help_text = capsys.readouterr().out
    assert "split-audio" in help_text
    assert "combine-chat-parts" in help_text


def test_asr_split_audio_options_parse():
    args = parse_args(
        [
            "asr",
            "split-audio",
            "--input",
            "raw",
            "--output",
            "chunks",
            "--max-seconds",
            "45",
        ]
    )

    assert args.command == "asr"
    assert args.asr_command == "split-audio"
    assert args.input == "raw"
    assert args.output == "chunks"
    assert args.max_seconds == 45


def test_asr_combine_chat_options_parse():
    args = parse_args(
        [
            "asr",
            "combine-chat-parts",
            "--input",
            "parts",
            "--output",
            "combined",
        ]
    )

    assert args.command == "asr"
    assert args.asr_command == "combine-chat-parts"
    assert args.input == "parts"
    assert args.output == "combined"


def test_main_for_asr_combine_chat_runs_helper(monkeypatch, capsys, tmp_path):
    calls = []

    class Result:
        base_name = "sample"
        part_paths = (tmp_path / "sample_part1.cha", tmp_path / "sample_part2.cha")
        output_path = tmp_path / "sample_combined.cha"

    def fake_combine(input_dir, output_dir):
        calls.append((input_dir, output_dir))
        return (Result(),)

    monkeypatch.setattr("rascal.cli.combine_chat_parts", fake_combine)

    exit_code = main(
        [
            "asr",
            "combine-chat-parts",
            "--input",
            "parts",
            "--output",
            "combined",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [("parts", "combined")]
    assert "Combined CHAT files: 1" in output
    assert "sample: 2 part(s) -> sample_combined.cha" in output


def test_main_for_asr_split_audio_runs_helper(monkeypatch, capsys, tmp_path):
    calls = []

    class Result:
        input_path = tmp_path / "audio.wav"
        output_paths = (tmp_path / "audio_part1.wav",)

    def fake_split(input_dir, output_dir, *, max_seconds):
        calls.append((input_dir, output_dir, max_seconds))
        return (Result(),)

    monkeypatch.setattr("rascal.cli.split_audio", fake_split)

    exit_code = main(
        [
            "asr",
            "split-audio",
            "--input",
            "raw",
            "--output",
            "chunks",
            "--max-seconds",
            "45",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [("raw", "chunks", 45)]
    assert "Split audio files: 1" in output
    assert "audio.wav: 1 part(s)" in output


def test_plan_branch_and_stage_parse():
    args = parse_args(["plan", "--branch", "monolog", "--stage", "7m"])
    command = normalize_command(args)

    assert command.command == "plan"
    assert command.branch == "monolog"
    assert command.stage == "7m"


def test_old_bare_numeric_alias_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["4"])

    assert excinfo.value.code == 2


def test_old_omnibus_alias_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["10"])

    assert excinfo.value.code == 2


def test_diaad_passthrough_captures_raw_args_after_separator():
    args = parse_args(["diaad", "--", "transcripts", "tabularize"])
    command = normalize_command(args)

    assert command.command == "diaad"
    assert command.diaad_args == ("transcripts", "tabularize")


def test_diaad_passthrough_also_accepts_args_without_separator():
    args = parse_args(["diaad", "transcripts", "tabularize"])
    command = normalize_command(args)

    assert command.diaad_args == ("transcripts", "tabularize")


def test_main_for_plan_prints_stage_plan(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "plan",
            "--branch",
            "dialog",
            "--stage",
            "4d",
            "--config",
            str(result.config_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "RASCAL plan: dialog 4d" in output
    assert "diaad transcripts tabularize --config config/diaad.generated" in output


def test_main_for_plan_format_json_returns_parseable_payload(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "plan",
            "--branch",
            "monolog",
            "--stage",
            "7m",
            "--format",
            "json",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["branch"] == "monolog"
    assert payload["stage_id"] == "7m"
    assert payload["diaad_command_names"] == ["cus analyze", "templates times", "words files"]


def test_main_for_plan_write_config_creates_generated_config(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "plan",
            "--branch",
            "dialog",
            "--stage",
            "4d",
            "--format",
            "json",
            "--write-config",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    generated_dir = result.project_root / "config" / "diaad.generated"
    assert exit_code == 0
    assert payload["generated_config_path"] == "config/diaad.generated"
    assert (generated_dir / "project.yaml").is_file()
    assert (generated_dir / "advanced.yaml").is_file()
    assert (generated_dir / "metadata.yaml").is_file()


def test_main_for_check_defaults_to_common_stage(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "check",
            "--format",
            "json",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["branch"] == "common"
    assert payload["stage_id"] == "1"
    assert payload["counts"]["warning"] >= 1


def test_main_for_check_returns_nonzero_when_required_input_is_missing(
    capsys,
    tmp_path,
):
    result = init_project(tmp_path, profile="lab_full")
    (result.project_root / "data" / "auto_transcripts").rmdir()

    exit_code = main(
        [
            "check",
            "--branch",
            "monolog",
            "--stage",
            "4m",
            "--format",
            "json",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["counts"]["error"] >= 1
    assert any(
        result["code"] == "expected_input_missing"
        for result in payload["results"]
    )


def test_main_for_status_returns_parseable_json(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "status",
            "--format",
            "json",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["profile"] == "lab_full"
    assert payload["branches"]["common"][0]["stage_id"] == "0"


def test_main_for_next_returns_parseable_json(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "next",
            "--format",
            "json",
            "--config",
            str(result.config_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["primary"]["branch"] == "common"
    assert payload["primary"]["stage_id"] == "0"


def test_main_for_run_dry_run_prints_stage_plan(capsys, tmp_path):
    result = init_project(tmp_path, profile="lab_full")

    exit_code = main(
        [
            "run",
            "--branch",
            "dialog",
            "--stage",
            "7d",
            "--dry-run",
            "--config",
            str(result.config_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "RASCAL dry run: no DIAAD commands executed." in output
    assert "diaad powers analyze --config config/diaad.generated" in output


def test_main_for_diaad_runs_passthrough(monkeypatch, capsys):
    calls = []

    def fake(command, **_kwargs):
        calls.append(tuple(command))
        return CompletedProcess(command, 0, stdout="diaad ok\n", stderr="")

    monkeypatch.setattr("rascal.runner.subprocess.run", fake)

    exit_code = main(["diaad", "--", "transcripts", "tabularize"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == "diaad ok\n"
    assert calls == [("diaad", "transcripts", "tabularize")]
