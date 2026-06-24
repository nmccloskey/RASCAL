from __future__ import annotations

import json
import tomllib
from pathlib import Path

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


def test_asr_subcommands_appear_in_help(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["asr", "--help"])

    help_text = capsys.readouterr().out
    assert "split-audio" in help_text
    assert "combine-chat-parts" in help_text


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


def test_main_for_diaad_prints_passthrough_placeholder(capsys):
    exit_code = main(["diaad", "--", "transcripts", "tabularize"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "DIAAD passthrough planned: diaad transcripts tabularize" in output
    assert "later pass" in output
