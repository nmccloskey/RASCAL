"""Stage registry loading for the RASCAL DIAAD wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

STAGE_PACKAGE = "rascal.data.stages"
VALID_STAGE_TYPES = {"manual", "automated", "automated_then_manual", "external_manual"}
REQUIRED_STAGE_FIELDS = {"name", "type"}


class StageError(ValueError):
    """Raised when a RASCAL stage registry cannot be loaded or validated."""


@dataclass(frozen=True)
class Stage:
    """One RASCAL workflow stage."""

    branch: str
    stage_id: str
    name: str
    type: str
    diaad: tuple[str, ...] = ()
    metadata_fields: dict[str, Any] = field(default_factory=dict)
    related_actions: tuple[str, ...] = ()
    thresholds: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_yaml_resource(filename: str) -> dict[str, Any]:
    path = resources.files(STAGE_PACKAGE).joinpath(filename)
    if not path.is_file():
        raise StageError(f"Unknown RASCAL stage registry: {filename.removesuffix('.yaml')}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise StageError(f"Stage resource {filename} must contain a mapping.")
    return loaded


def _coerce_stage(branch: str, stage_id: str, data: dict[str, Any]) -> Stage:
    missing = REQUIRED_STAGE_FIELDS - data.keys()
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise StageError(f"{branch}:{stage_id} missing required fields: {missing_text}")

    stage_type = data["type"]
    if stage_type not in VALID_STAGE_TYPES:
        valid = ", ".join(sorted(VALID_STAGE_TYPES))
        raise StageError(f"{branch}:{stage_id} has invalid type {stage_type!r}; expected {valid}")

    diaad = tuple(data.get("diaad") or ())
    if stage_type == "automated" and not diaad:
        raise StageError(f"{branch}:{stage_id} is automated but has no DIAAD commands.")

    return Stage(
        branch=branch,
        stage_id=str(stage_id),
        name=str(data["name"]),
        type=stage_type,
        diaad=diaad,
        metadata_fields=dict(data.get("metadata_fields") or {}),
        related_actions=tuple(data.get("related_actions") or ()),
        thresholds=dict(data.get("thresholds") or {}),
        raw=dict(data),
    )


def list_stage_branches() -> list[str]:
    """Return packaged stage branch names."""

    return sorted(
        path.name.removesuffix(".yaml")
        for path in resources.files(STAGE_PACKAGE).iterdir()
        if path.name.endswith(".yaml")
    )


def load_branch_stages(branch: str) -> dict[str, Stage]:
    """Load one branch's stages in registry order."""

    registry = _load_yaml_resource(f"{branch}.yaml")
    registry_branch = registry.get("branch")
    if registry_branch != branch:
        raise StageError(
            f"Stage registry {branch}.yaml declares branch {registry_branch!r}."
        )
    stages = registry.get("stages")
    if not isinstance(stages, dict):
        raise StageError(f"Stage registry {branch}.yaml must contain a stages mapping.")
    return {
        str(stage_id): _coerce_stage(branch, str(stage_id), data)
        for stage_id, data in stages.items()
    }


def load_stage_registry() -> dict[str, dict[str, Stage]]:
    """Load all packaged branch stage registries."""

    return {branch: load_branch_stages(branch) for branch in list_stage_branches()}


def get_stage(branch: str, stage_id: str) -> Stage:
    """Return one stage by branch and id."""

    stages = load_branch_stages(branch)
    try:
        return stages[stage_id]
    except KeyError as exc:
        raise StageError(f"Unknown RASCAL stage: {branch}:{stage_id}") from exc


def stage_order(branch: str) -> list[str]:
    """Return stage ids for one branch in registry order."""

    return list(load_branch_stages(branch).keys())


def validate_stage_registry() -> list[str]:
    """Validate all packaged stages and return warning messages.

    Invalid stage definitions raise ``StageError`` during loading. The returned
    list is reserved for non-fatal warnings in later passes.
    """

    load_stage_registry()
    return []

