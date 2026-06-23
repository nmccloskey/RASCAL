from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rascal import __version__
from rascal.cli import build_parser, main, normalize_command, parse_args


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


def test_main_for_plan_prints_placeholder(capsys):
    exit_code = main(["plan", "--branch", "dialog", "--stage", "4d"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "RASCAL command parsed: plan" in output
    assert "Branch: dialog" in output
    assert "Stage: 4d" in output
    assert "later pass" in output


def test_main_for_diaad_prints_passthrough_placeholder(capsys):
    exit_code = main(["diaad", "--", "transcripts", "tabularize"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "DIAAD passthrough planned: transcripts tabularize" in output
    assert "later pass" in output
