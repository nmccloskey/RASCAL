from __future__ import annotations

import pytest

from rascal.diaad_invocation import (
    DiaadInvocationError,
    build_diaad_command,
    build_diaad_commands,
    build_passthrough_command,
)


def test_diaad_command_vector_is_a_list():
    command = build_diaad_command("cus analyze", config_path="config/diaad.generated")

    assert isinstance(command, list)
    assert command == ["diaad", "cus", "analyze", "--config", "config/diaad.generated"]


def test_configured_executable_replaces_diaad():
    command = build_diaad_command(
        "transcripts tabularize",
        executable="diaad-dev",
        config_path="config/diaad.generated",
    )

    assert command[0] == "diaad-dev"
    assert command[1:3] == ["transcripts", "tabularize"]


def test_build_multiple_command_vectors():
    commands = build_diaad_commands(
        ["cus analyze", "templates times", "words files"],
        config_path="config/diaad.generated",
    )

    assert commands == (
        ("diaad", "cus", "analyze", "--config", "config/diaad.generated"),
        ("diaad", "templates", "times", "--config", "config/diaad.generated"),
        ("diaad", "words", "files", "--config", "config/diaad.generated"),
    )


def test_raw_passthrough_preserves_argument_order_after_separator():
    command = build_passthrough_command(["transcripts", "tabularize", "--dry-run-config"])

    assert command == ["diaad", "transcripts", "tabularize", "--dry-run-config"]


def test_no_command_construction_uses_shell_true():
    command = build_diaad_command("powers analyze", config_path="config/diaad.generated")

    assert command == ["diaad", "powers", "analyze", "--config", "config/diaad.generated"]
    assert "shell=True" not in command


def test_empty_diaad_command_is_rejected():
    with pytest.raises(DiaadInvocationError):
        build_diaad_command("", config_path="config/diaad.generated")


def test_empty_passthrough_is_rejected():
    with pytest.raises(DiaadInvocationError):
        build_passthrough_command([])
