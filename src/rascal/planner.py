"""Stage planning for the RASCAL DIAAD wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rascal.config import ProjectConfig
from rascal.diaad_invocation import build_diaad_commands, format_command
from rascal.stages import Stage, StageError, get_stage


class PlanError(ValueError):
    """Raised when RASCAL cannot produce a stage plan."""


EXPECTED_PATH_KEYS = {
    ("common", "1"): (
        ("auto_transcripts_dir",),
        ("transcription_reliability_dir",),
    ),
    ("common", "2"): (
        ("transcription_reliability_dir",),
        ("transcription_reliability_dir",),
    ),
    ("common", "3"): (
        ("transcription_reliability_dir",),
        ("transcription_reliability_dir",),
    ),
    ("monolog", "4m"): (
        ("auto_transcripts_dir",),
        ("monolog_transcript_tables_dir",),
    ),
    ("monolog", "5m_prepare"): (
        ("monolog_transcript_tables_dir",),
        ("monolog_cu_files_dir",),
    ),
    ("monolog", "5m"): (
        ("monolog_cu_files_dir",),
        ("monolog_cu_files_dir",),
    ),
    ("monolog", "6m"): (
        ("monolog_cu_files_dir",),
        ("monolog_cu_reliability_dir",),
    ),
    ("monolog", "7m"): (
        ("monolog_cu_files_dir",),
        (
            "monolog_cu_analysis_dir",
            "monolog_speaking_times_dir",
            "monolog_word_count_files_dir",
        ),
    ),
    ("monolog", "8m"): (
        ("monolog_word_count_files_dir", "monolog_speaking_times_dir"),
        ("monolog_word_count_files_dir", "monolog_speaking_times_dir"),
    ),
    ("monolog", "9m"): (
        ("monolog_word_count_files_dir",),
        ("monolog_word_reliability_dir",),
    ),
    ("monolog", "10m"): (
        (
            "monolog_cu_analysis_dir",
            "monolog_word_count_files_dir",
            "monolog_speaking_times_dir",
        ),
        ("monolog_corelex_dir", "monolog_final_exports_dir"),
    ),
    ("dialog", "4d"): (
        ("auto_transcripts_dir",),
        ("dialog_transcript_tables_dir",),
    ),
    ("dialog", "5d_prepare"): (
        ("dialog_transcript_tables_dir",),
        ("dialog_powers_files_dir",),
    ),
    ("dialog", "5d"): (
        ("dialog_powers_files_dir",),
        ("dialog_powers_files_dir",),
    ),
    ("dialog", "6d"): (
        ("dialog_powers_files_dir",),
        ("dialog_powers_reliability_dir",),
    ),
    ("dialog", "7d"): (
        ("dialog_powers_files_dir",),
        ("dialog_powers_analysis_dir",),
    ),
}

MANUAL_PREREQUISITES = {
    ("monolog", "6m"): (
        "Completed CU coding and CU reliability workbooks are present and reviewed.",
    ),
    ("monolog", "7m"): (
        "Completed CU coding workbooks are present and reviewed.",
    ),
    ("monolog", "9m"): (
        "Completed word count reliability workbooks are present and reviewed.",
    ),
    ("monolog", "10m"): (
        "Completed word count and speaking-time workbooks are present and reviewed.",
    ),
    ("dialog", "6d"): (
        "Completed POWERS reliability workbooks are present and reviewed.",
    ),
    ("dialog", "7d"): (
        "Completed POWERS coding workbooks are present and reviewed.",
    ),
}


@dataclass(frozen=True)
class Plan:
    """A concrete RASCAL stage plan."""

    profile_name: str
    branch: str
    stage_id: str
    stage_name: str
    stage_type: str
    diaad_command_names: tuple[str, ...]
    diaad_commands: tuple[tuple[str, ...], ...]
    generated_config_path: Path
    expected_inputs: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]
    manual_prerequisites: tuple[str, ...]
    stage: Stage
    project_root: Path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan dictionary."""

        return {
            "profile": self.profile_name,
            "branch": self.branch,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "diaad_command_names": list(self.diaad_command_names),
            "diaad_commands": [list(command) for command in self.diaad_commands],
            "generated_config_path": _display_path(
                self.generated_config_path,
                self.project_root,
            ),
            "expected_inputs": [
                _display_path(path, self.project_root) for path in self.expected_inputs
            ],
            "expected_outputs": [
                _display_path(path, self.project_root) for path in self.expected_outputs
            ],
            "manual_prerequisites": list(self.manual_prerequisites),
        }


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _configured_paths(config: ProjectConfig, path_keys: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in path_keys:
        try:
            path = config.resolve_path(key)
        except Exception as exc:  # noqa: BLE001 - normalized into planner context.
            raise PlanError(f"Stage requires missing configured path key: {key}") from exc
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _legacy_paths(config: ProjectConfig, stage: Stage) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    input_path = config.resolve_path("input_dir")
    output_path = config.resolve_path("output_dir")
    if stage.type == "manual":
        return (input_path,), (input_path,)
    return (input_path,), (output_path,)


def _expected_paths(config: ProjectConfig, stage: Stage) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    if config.layout == "legacy":
        return _legacy_paths(config, stage)

    try:
        input_keys, output_keys = EXPECTED_PATH_KEYS[(stage.branch, stage.stage_id)]
    except KeyError as exc:
        raise PlanError(f"No expected path mapping for {stage.branch}:{stage.stage_id}") from exc
    return _configured_paths(config, input_keys), _configured_paths(config, output_keys)


def _config_path_for_commands(config: ProjectConfig) -> str:
    try:
        return config.resolve_path("diaad_config_dir").relative_to(config.project_root).as_posix()
    except ValueError:
        return config.resolve_path("diaad_config_dir").as_posix()


def create_stage_plan(config: ProjectConfig, branch: str, stage_id: str) -> Plan:
    """Create a RASCAL plan for one branch/stage pair."""

    try:
        stage = get_stage(branch, stage_id)
    except StageError as exc:
        raise PlanError(str(exc)) from exc

    expected_inputs, expected_outputs = _expected_paths(config, stage)
    executable = str(config.resolved_profile.get("diaad_executable", "diaad"))
    command_config_path = _config_path_for_commands(config)
    diaad_commands = (
        build_diaad_commands(
            stage.diaad,
            executable=executable,
            config_path=command_config_path,
        )
        if stage.diaad
        else ()
    )

    return Plan(
        profile_name=config.profile_name,
        branch=branch,
        stage_id=stage.stage_id,
        stage_name=stage.name,
        stage_type=stage.type,
        diaad_command_names=stage.diaad,
        diaad_commands=diaad_commands,
        generated_config_path=config.resolve_path("diaad_config_dir"),
        expected_inputs=expected_inputs,
        expected_outputs=expected_outputs,
        manual_prerequisites=MANUAL_PREREQUISITES.get((branch, stage_id), ()),
        stage=stage,
        project_root=config.project_root,
    )


def render_plan_text(plan: Plan) -> str:
    """Render a stage plan as concise text."""

    lines = [
        f"RASCAL plan: {plan.branch} {plan.stage_id}",
        f"Profile: {plan.profile_name}",
        f"Stage: {plan.stage_name}",
        f"Type: {plan.stage_type}",
        f"Generated DIAAD config: {_display_path(plan.generated_config_path, plan.project_root)}",
        "",
        "Manual prerequisites:",
    ]
    if plan.manual_prerequisites:
        lines.extend(f"  - {item}" for item in plan.manual_prerequisites)
    elif plan.stage_type == "manual":
        lines.append("  - This is a manual stage; no DIAAD command will run.")
    else:
        lines.append("  - None declared.")

    lines.extend(["", "Commands:"])
    if plan.diaad_commands:
        lines.extend(
            f"  {index}. {format_command(command)}"
            for index, command in enumerate(plan.diaad_commands, start=1)
        )
    else:
        lines.append("  (none)")

    lines.extend(["", "Expected inputs:"])
    lines.extend(
        f"  - {_display_path(path, plan.project_root)}" for path in plan.expected_inputs
    )
    lines.extend(["", "Expected outputs:"])
    lines.extend(
        f"  - {_display_path(path, plan.project_root)}" for path in plan.expected_outputs
    )
    return "\n".join(lines)


def render_plan_json(plan: Plan) -> str:
    """Render a stage plan as JSON."""

    return json.dumps(plan.as_dict(), indent=2)

