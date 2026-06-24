"""Run manifest writing for the RASCAL DIAAD wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from rascal import __version__
from rascal.config import ProjectConfig
from rascal.diaad_config import GeneratedDiaadConfig
from rascal.diaad_invocation import format_command
from rascal.planner import Plan
from rascal.preflight import PreflightReport


@dataclass(frozen=True)
class ManifestArtifacts:
    """Paths written for one RASCAL run manifest."""

    run_dir: Path
    manifest_path: Path
    command_log_path: Path


def display_path(path: Path, project_root: Path) -> str:
    """Return a project-relative display path when possible."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def create_run_dir(
    config: ProjectConfig,
    branch: str,
    stage_id: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Create and return a unique run directory for a stage attempt."""

    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_branch = branch.replace("/", "_")
    safe_stage = stage_id.replace("/", "_")
    runs_dir = config.resolve_path("runs_dir")
    base_name = f"{timestamp}_{safe_branch}_{safe_stage}"
    run_dir = runs_dir / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def discover_diaad_version() -> str | None:
    """Return an installed DIAAD package version when discoverable."""

    for package_name in ("diaad", "DIAAD"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _command_result_dict(result: Any, project_root: Path) -> dict[str, Any]:
    return {
        "index": result.index,
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout_path": display_path(result.stdout_path, project_root),
        "stderr_path": display_path(result.stderr_path, project_root),
    }


def build_manifest(
    config: ProjectConfig,
    plan: Plan,
    preflight: PreflightReport,
    generated_config: GeneratedDiaadConfig,
    *,
    status: str,
    dry_run: bool,
    command_results: tuple[Any, ...] = (),
    diaad_version: str | None = None,
) -> dict[str, Any]:
    """Build a private-data-light run manifest."""

    profile = config.resolved_profile
    return {
        "schema_version": 1,
        "rascal_version": __version__,
        "diaad_version": diaad_version,
        "status": status,
        "dry_run": dry_run,
        "profile": config.profile_name,
        "layout": config.layout,
        "branch": plan.branch,
        "stage_id": plan.stage_id,
        "stage_name": plan.stage_name,
        "stage_type": plan.stage_type,
        "diaad_command_names": list(plan.diaad_command_names),
        "diaad_commands": [list(command) for command in plan.diaad_commands],
        "generated_config_path": display_path(generated_config.config_dir, config.project_root),
        "expected_inputs": [
            display_path(path, config.project_root) for path in plan.expected_inputs
        ],
        "expected_outputs": [
            display_path(path, config.project_root) for path in plan.expected_outputs
        ],
        "manual_prerequisites": list(plan.manual_prerequisites),
        "random_seed": profile.get("random_seed"),
        "reliability_fraction": profile.get("reliability_fraction"),
        "thresholds": profile.get("thresholds", {}),
        "preflight": preflight.as_dict(),
        "command_results": [
            _command_result_dict(result, config.project_root)
            for result in command_results
        ],
    }


def write_command_log(run_dir: Path, plan: Plan) -> Path:
    """Write the planned DIAAD commands as a plain-text audit file."""

    command_log_path = run_dir / "diaad_commands.txt"
    lines = ["RASCAL DIAAD commands", ""]
    if plan.diaad_commands:
        lines.extend(
            f"{index}. {format_command(command)}"
            for index, command in enumerate(plan.diaad_commands, start=1)
        )
    else:
        lines.append("(none)")
    command_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return command_log_path


def write_manifest(
    config: ProjectConfig,
    plan: Plan,
    preflight: PreflightReport,
    generated_config: GeneratedDiaadConfig,
    run_dir: Path,
    *,
    status: str,
    dry_run: bool,
    command_results: tuple[Any, ...] = (),
    diaad_version: str | None = None,
) -> ManifestArtifacts:
    """Write run manifest and command log files."""

    command_log_path = write_command_log(run_dir, plan)
    manifest_path = run_dir / "rascal_manifest.json"
    manifest = build_manifest(
        config,
        plan,
        preflight,
        generated_config,
        status=status,
        dry_run=dry_run,
        command_results=command_results,
        diaad_version=diaad_version,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return ManifestArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        command_log_path=command_log_path,
    )
