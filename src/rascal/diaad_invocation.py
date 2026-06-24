"""DIAAD command-vector construction for RASCAL plans."""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Sequence
from pathlib import Path


class DiaadInvocationError(ValueError):
    """Raised when a DIAAD command cannot be represented safely."""


def command_path(path: str | Path) -> str:
    """Return a stable command-line path string."""

    return Path(path).as_posix()


def build_diaad_command(
    diaad_command: str,
    *,
    executable: str = "diaad",
    config_path: str | Path,
) -> list[str]:
    """Build a DIAAD command vector from a module command string."""

    parts = shlex.split(diaad_command)
    if not parts:
        raise DiaadInvocationError("DIAAD command cannot be empty.")
    return [executable, *parts, "--config", command_path(config_path)]


def build_diaad_commands(
    diaad_commands: Iterable[str],
    *,
    executable: str = "diaad",
    config_path: str | Path,
) -> tuple[tuple[str, ...], ...]:
    """Build command vectors for a sequence of DIAAD commands."""

    return tuple(
        tuple(build_diaad_command(command, executable=executable, config_path=config_path))
        for command in diaad_commands
    )


def build_passthrough_command(
    diaad_args: Sequence[str],
    *,
    executable: str = "diaad",
) -> list[str]:
    """Build a raw DIAAD passthrough command vector."""

    if not diaad_args:
        raise DiaadInvocationError("DIAAD passthrough arguments cannot be empty.")
    return [executable, *diaad_args]


def format_command(command: Sequence[str]) -> str:
    """Format a command vector for display."""

    return " ".join(str(part) for part in command)

