"""Stage execution through the DIAAD CLI boundary."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rascal.config import ProjectConfig
from rascal.diaad_config import GeneratedDiaadConfig, write_diaad_config
from rascal.diaad_invocation import build_passthrough_command, format_command
from rascal.manifests import (
    ManifestArtifacts,
    create_run_dir,
    discover_diaad_version,
    display_path,
    write_manifest,
)
from rascal.planner import Plan, create_stage_plan
from rascal.preflight import PreflightReport, run_preflight

SubprocessRun = Callable[..., Any]


class RunnerError(ValueError):
    """Raised when RASCAL cannot run a stage or passthrough command."""


@dataclass(frozen=True)
class CommandResult:
    """Captured result for one DIAAD command."""

    index: int
    command: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class RunResult:
    """Result of a RASCAL stage run attempt."""

    status: str
    exit_code: int
    message: str
    plan: Plan
    preflight: PreflightReport
    generated_config: GeneratedDiaadConfig
    artifacts: ManifestArtifacts
    command_results: tuple[CommandResult, ...]


@dataclass(frozen=True)
class PassthroughResult:
    """Captured result for a raw DIAAD passthrough command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | None,
    subprocess_run: SubprocessRun | None,
) -> tuple[int, str, str]:
    runner = subprocess_run or subprocess.run
    try:
        completed = runner(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return (
        int(getattr(completed, "returncode", 0)),
        str(getattr(completed, "stdout", "") or ""),
        str(getattr(completed, "stderr", "") or ""),
    )


def _execute_command(
    index: int,
    command: Sequence[str],
    *,
    run_dir: Path,
    cwd: Path,
    subprocess_run: SubprocessRun,
) -> CommandResult:
    returncode, stdout, stderr = _run_subprocess(
        command,
        cwd=cwd,
        subprocess_run=subprocess_run,
    )
    stdout_path = run_dir / f"command_{index:02d}_stdout.txt"
    stderr_path = run_dir / f"command_{index:02d}_stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return CommandResult(
        index=index,
        command=tuple(command),
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def run_stage(
    config: ProjectConfig,
    branch: str,
    stage_id: str,
    *,
    dry_run: bool = False,
    subprocess_run: SubprocessRun | None = None,
) -> RunResult:
    """Run or dry-run one RASCAL stage."""

    plan = create_stage_plan(config, branch, stage_id)
    preflight = run_preflight(config, branch, stage_id)
    generated_config = write_diaad_config(config, plan)
    run_dir = create_run_dir(config, branch, stage_id)
    diaad_version = discover_diaad_version()

    command_results: list[CommandResult] = []
    if preflight.has_errors:
        status = "blocked_preflight"
        exit_code = 1
        message = "Preflight errors blocked DIAAD execution."
    elif dry_run:
        status = "dry_run"
        exit_code = 0
        message = "RASCAL dry run: no DIAAD commands executed."
    elif not plan.diaad_commands:
        status = "no_commands"
        exit_code = 0
        message = "No DIAAD commands are planned for this stage."
    else:
        status = "succeeded"
        exit_code = 0
        message = "RASCAL run completed successfully."
        for index, command in enumerate(plan.diaad_commands, start=1):
            result = _execute_command(
                index,
                command,
                run_dir=run_dir,
                cwd=config.project_root,
                subprocess_run=subprocess_run,
            )
            command_results.append(result)
            if result.returncode != 0:
                status = "failed"
                exit_code = result.returncode
                message = (
                    "DIAAD command failed with exit code "
                    f"{result.returncode}: {format_command(result.command)}"
                )
                break

    artifacts = write_manifest(
        config,
        plan,
        preflight,
        generated_config,
        run_dir,
        status=status,
        dry_run=dry_run,
        command_results=tuple(command_results),
        diaad_version=diaad_version,
    )

    return RunResult(
        status=status,
        exit_code=exit_code,
        message=message,
        plan=plan,
        preflight=preflight,
        generated_config=generated_config,
        artifacts=artifacts,
        command_results=tuple(command_results),
    )


def run_diaad_passthrough(
    args: Sequence[str],
    *,
    executable: str = "diaad",
    cwd: Path | None = None,
    subprocess_run: SubprocessRun | None = None,
) -> PassthroughResult:
    """Run raw DIAAD passthrough args through the same safe subprocess boundary."""

    if not args:
        raise RunnerError("No DIAAD arguments supplied.")
    command = build_passthrough_command(tuple(args), executable=executable)
    returncode, stdout, stderr = _run_subprocess(
        command,
        cwd=cwd,
        subprocess_run=subprocess_run,
    )
    return PassthroughResult(
        command=tuple(command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def render_run_result_text(result: RunResult) -> str:
    """Render a stage run result as concise text."""

    counts = result.preflight.counts
    lines = [
        result.message,
        f"Status: {result.status}",
        f"Run directory: {display_path(result.artifacts.run_dir, result.plan.project_root)}",
        f"Manifest: {display_path(result.artifacts.manifest_path, result.plan.project_root)}",
        f"Preflight: {counts['error']} error(s), {counts['warning']} warning(s), {counts['ok']} ok",
        "",
        "Commands:",
    ]
    if result.plan.diaad_commands:
        lines.extend(
            f"  {index}. {format_command(command)}"
            for index, command in enumerate(result.plan.diaad_commands, start=1)
        )
    else:
        lines.append("  (none)")

    if result.command_results:
        lines.extend(["", "Command results:"])
        for command_result in result.command_results:
            lines.append(
                "  "
                f"{command_result.index}. exit {command_result.returncode} "
                f"stdout={display_path(command_result.stdout_path, result.plan.project_root)} "
                f"stderr={display_path(command_result.stderr_path, result.plan.project_root)}"
            )
    return "\n".join(lines)
