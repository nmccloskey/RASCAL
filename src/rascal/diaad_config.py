"""Generate DIAAD configuration from a RASCAL stage plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rascal.config import ProjectConfig
from rascal.planner import Plan


class DiaadConfigError(ValueError):
    """Raised when RASCAL cannot generate DIAAD configuration."""


@dataclass(frozen=True)
class GeneratedDiaadConfig:
    """Paths and content written for a generated DIAAD config."""

    config_dir: Path
    project_path: Path
    advanced_path: Path
    metadata_path: Path
    rascal_source_path: Path
    project: dict[str, Any]
    advanced: dict[str, Any]
    metadata: dict[str, Any]
    rascal_source: dict[str, Any]


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _first_path(paths: tuple[Path, ...], label: str) -> Path:
    if not paths:
        raise DiaadConfigError(f"Plan has no expected {label} paths.")
    return paths[0]


def _branch_config(profile: dict[str, Any], branch: str) -> dict[str, Any]:
    branch_data = profile.get(branch)
    return branch_data if isinstance(branch_data, dict) else {}


def metadata_fields_for_plan(config: ProjectConfig, plan: Plan) -> dict[str, Any]:
    """Return metadata fields for the plan's branch/stage."""

    if plan.stage.metadata_fields:
        return dict(plan.stage.metadata_fields)

    profile = config.resolved_profile
    direct_metadata = profile.get("metadata_fields")
    if isinstance(direct_metadata, dict) and config.profile_name.endswith(plan.branch):
        return dict(direct_metadata)

    branch_metadata = _branch_config(profile, plan.branch).get("metadata_fields")
    if isinstance(branch_metadata, dict):
        return dict(branch_metadata)

    return {}


def stimulus_column_for_plan(config: ProjectConfig, plan: Plan) -> str:
    """Return the DIAAD stimulus column for a branch plan."""

    metadata_fields = metadata_fields_for_plan(config, plan)
    if "narrative" in metadata_fields:
        return "narrative"
    if "communication" in metadata_fields:
        return "communication"
    return ""


def build_project_config(config: ProjectConfig, plan: Plan) -> dict[str, Any]:
    """Build DIAAD project.yaml content."""

    profile = config.resolved_profile
    return {
        "input_dir": _relative_to_project(
            _first_path(plan.expected_inputs, "input"),
            config.project_root,
        ),
        "output_dir": _relative_to_project(
            _first_path(plan.expected_outputs, "output"),
            config.project_root,
        ),
        "random_seed": int(profile.get("random_seed", 99)),
        "reliability_fraction": float(profile.get("reliability_fraction", 0.2)),
        "shuffle_samples": bool(profile.get("shuffle_samples", True)),
        "strip_clan": bool(profile.get("strip_clan", True)),
        "prefer_correction": bool(profile.get("prefer_correction", True)),
        "lowercase": bool(profile.get("lowercase", True)),
        "exclude_speakers": list(profile.get("excluded_speakers", [])),
        "auto_tabularize": False,
        "num_bins": int(profile.get("num_bins", 4)),
        "num_coders": int(profile.get("num_coders", 0)),
        "stimulus_column": stimulus_column_for_plan(config, plan),
        "automate_powers": bool(
            _branch_config(profile, "dialog").get("powers_automation", False)
            if plan.branch == "dialog"
            else False
        ),
        "metadata_fields": metadata_fields_for_plan(config, plan),
    }


def build_advanced_config(config: ProjectConfig, plan: Plan) -> dict[str, Any]:
    """Build DIAAD advanced.yaml content."""

    profile = config.resolved_profile
    monolog = _branch_config(profile, "monolog")
    dialog = _branch_config(profile, "dialog")
    return {
        "transcript_table_filename": "transcript_tables.xlsx",
        "sample_id_column": "sample_id",
        "utterance_id_column": "utterance_id",
        "reliability_tag": "_reliability",
        "reliability_dirname": "reliability",
        "cu_paradigms": list(monolog.get("cu_dialects", [])) if plan.branch == "monolog" else [],
        "cu_samples_filename": "cu_coding_by_sample_long.xlsx",
        "cu_utts_filename": "cu_coding_by_utterance.xlsx",
        "word_count_filename": "word_counting.xlsx",
        "word_count_column": "word_count",
        "wc_samples_filename": "word_counting_by_sample.xlsx",
        "speaking_time_filename": "speaking_times.xlsx",
        "speaking_time_column": "speaking_time",
        "powers_coding_filename": "powers_coding.xlsx",
        "powers_reliability_filename": "powers_reliability_coding.xlsx",
        "spacy_model_name": dialog.get("spacy_model", "en_core_web_sm"),
        "dct_coding_filename": "conversation_turns.xlsx",
        "dct_coding_reliability": "conversation_turns_reliability.xlsx",
        "target_vocabulary_resource_path": "",
        "auto_blind": True,
        "blind_columns": list(profile.get("blind_columns", ["site", "test"])),
        "metadata_source": "transcript_tables.xlsx",
        "id_columns": ["sample_id", "utterance_id"],
        "codebook_filename": "",
    }


def build_metadata_config(config: ProjectConfig, plan: Plan) -> dict[str, Any]:
    """Build RASCAL-facing metadata audit content.

    DIAAD consumes metadata through project.yaml. This file is intentionally
    supplemental so a generated config directory remains easy for lab users to
    inspect.
    """

    profile = config.resolved_profile
    return {
        "profile": config.profile_name,
        "branch": plan.branch,
        "stage_id": plan.stage_id,
        "metadata_fields": metadata_fields_for_plan(config, plan),
        "blind_columns": list(profile.get("blind_columns", ["site", "test"])),
        "thresholds": profile.get("thresholds", {}),
        "corelex_stimuli": list(_branch_config(profile, "monolog").get("corelex_stimuli", [])),
    }


def build_rascal_source_config(config: ProjectConfig, plan: Plan) -> dict[str, Any]:
    """Build a compact RASCAL source snapshot for generated DIAAD config."""

    return {
        "rascal_config": _relative_to_project(config.config_path, config.project_root),
        "profile": config.profile_name,
        "layout": config.layout,
        "branch": plan.branch,
        "stage_id": plan.stage_id,
        "stage_name": plan.stage_name,
        "stage_type": plan.stage_type,
        "diaad_commands": [list(command) for command in plan.diaad_commands],
    }


def write_diaad_config(config: ProjectConfig, plan: Plan) -> GeneratedDiaadConfig:
    """Write split DIAAD config files for a stage plan."""

    config_dir = plan.generated_config_path
    config_dir.mkdir(parents=True, exist_ok=True)

    project = build_project_config(config, plan)
    advanced = build_advanced_config(config, plan)
    metadata = build_metadata_config(config, plan)
    rascal_source = build_rascal_source_config(config, plan)

    project_path = config_dir / "project.yaml"
    advanced_path = config_dir / "advanced.yaml"
    metadata_path = config_dir / "metadata.yaml"
    rascal_source_path = config_dir / "rascal_source.yaml"

    for path, data in (
        (project_path, project),
        (advanced_path, advanced),
        (metadata_path, metadata),
        (rascal_source_path, rascal_source),
    ):
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return GeneratedDiaadConfig(
        config_dir=config_dir,
        project_path=project_path,
        advanced_path=advanced_path,
        metadata_path=metadata_path,
        rascal_source_path=rascal_source_path,
        project=project,
        advanced=advanced,
        metadata=metadata,
        rascal_source=rascal_source,
    )

