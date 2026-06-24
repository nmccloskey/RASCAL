"""Workflow status and next-step inference for RASCAL projects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rascal.config import ProjectConfig
from rascal.manifests import display_path
from rascal.planner import Plan, PlanError, create_stage_plan
from rascal.stages import Stage, get_stage, load_branch_stages, stage_order

BRANCHES = ("common", "monolog", "dialog")
COMPLETE_MANIFEST_STATUSES = {"succeeded"}
BLOCKED_MANIFEST_STATUSES = {"failed", "blocked_preflight"}
EFFECTIVE_MANIFEST_STATUSES = COMPLETE_MANIFEST_STATUSES | BLOCKED_MANIFEST_STATUSES | {"no_commands"}


class StatusError(ValueError):
    """Raised when RASCAL cannot infer workflow status."""


@dataclass(frozen=True)
class ManifestRecord:
    """One parsed RASCAL run manifest."""

    path: Path
    data: dict[str, Any]
    mtime_ns: int

    @property
    def branch(self) -> str:
        return str(self.data.get("branch", ""))

    @property
    def stage_id(self) -> str:
        return str(self.data.get("stage_id", ""))

    @property
    def status(self) -> str:
        return str(self.data.get("status", "unknown"))

    @property
    def dry_run(self) -> bool:
        return bool(self.data.get("dry_run", False))


@dataclass(frozen=True)
class StageStatus:
    """Inferred status for one RASCAL workflow stage."""

    branch: str
    stage_id: str
    stage_name: str
    stage_type: str
    state: str
    message: str
    command: str | None = None
    after_manual_command: str | None = None
    diaad_command_names: tuple[str, ...] = ()
    latest_manifest: str | None = None
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable status record."""

        payload: dict[str, Any] = {
            "branch": self.branch,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "state": self.state,
            "message": self.message,
            "diaad_command_names": list(self.diaad_command_names),
            "expected_inputs": list(self.expected_inputs),
            "expected_outputs": list(self.expected_outputs),
            "issues": list(self.issues),
        }
        if self.command:
            payload["command"] = self.command
        if self.after_manual_command:
            payload["after_manual_command"] = self.after_manual_command
        if self.latest_manifest:
            payload["latest_manifest"] = self.latest_manifest
        return payload


@dataclass(frozen=True)
class WorkflowStatusReport:
    """Status report for a whole RASCAL project."""

    profile: str
    branches: dict[str, tuple[StageStatus, ...]]
    manifest_warnings: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """Return counts by stage state."""

        counts = {
            "not_started": 0,
            "ready": 0,
            "manual_pending": 0,
            "complete": 0,
            "blocked": 0,
            "unknown": 0,
        }
        for stages in self.branches.values():
            for status in stages:
                counts[status.state] = counts.get(status.state, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable status report."""

        return {
            "profile": self.profile,
            "counts": self.counts,
            "branches": {
                branch: [stage.as_dict() for stage in stages]
                for branch, stages in self.branches.items()
            },
            "manifest_warnings": list(self.manifest_warnings),
        }


@dataclass(frozen=True)
class NextRecommendation:
    """One branch-level next action recommendation."""

    branch: str
    stage_id: str | None
    stage_name: str | None
    state: str
    action: str
    command: str | None = None
    after_manual_command: str | None = None
    diaad_command_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable recommendation."""

        payload: dict[str, Any] = {
            "branch": self.branch,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "state": self.state,
            "action": self.action,
            "diaad_command_names": list(self.diaad_command_names),
        }
        if self.command:
            payload["command"] = self.command
        if self.after_manual_command:
            payload["after_manual_command"] = self.after_manual_command
        return payload


@dataclass(frozen=True)
class NextReport:
    """Next-step recommendations for a RASCAL project."""

    profile: str
    primary: NextRecommendation
    recommendations: tuple[NextRecommendation, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable next report."""

        return {
            "profile": self.profile,
            "primary": self.primary.as_dict(),
            "recommendations": [
                recommendation.as_dict()
                for recommendation in self.recommendations
            ],
        }


def _path_has_files(path: Path, pattern: str = "*") -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    return any(candidate.is_file() for candidate in path.rglob(pattern))


def _plan_command(branch: str, stage_id: str) -> str:
    return f"rascal run --branch {branch} --stage {stage_id}"


def _load_manifest(path: Path) -> ManifestRecord | str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Could not read manifest {path}: {exc}"
    if not isinstance(data, dict):
        return f"Manifest is not a JSON object: {path}"
    return ManifestRecord(path=path, data=data, mtime_ns=path.stat().st_mtime_ns)


def load_manifest_records(config: ProjectConfig) -> tuple[tuple[ManifestRecord, ...], tuple[str, ...]]:
    """Load RASCAL run manifests from the configured runs directory."""

    runs_dir = config.resolve_path("runs_dir")
    if not runs_dir.exists():
        return (), ()

    records: list[ManifestRecord] = []
    warnings: list[str] = []
    for manifest_path in sorted(runs_dir.rglob("rascal_manifest.json")):
        loaded = _load_manifest(manifest_path)
        if isinstance(loaded, ManifestRecord):
            records.append(loaded)
        else:
            warnings.append(loaded)
    records.sort(key=lambda record: (record.mtime_ns, record.path.as_posix()))
    return tuple(records), tuple(warnings)


def _latest_effective_manifest(
    records: tuple[ManifestRecord, ...],
    branch: str,
    stage_id: str,
) -> ManifestRecord | None:
    matches = [
        record
        for record in records
        if record.branch == branch
        and record.stage_id == stage_id
        and not record.dry_run
        and record.status in EFFECTIVE_MANIFEST_STATUSES
    ]
    return matches[-1] if matches else None


def _latest_manifest(
    records: tuple[ManifestRecord, ...],
    branch: str,
    stage_id: str,
) -> ManifestRecord | None:
    matches = [
        record
        for record in records
        if record.branch == branch and record.stage_id == stage_id
    ]
    return matches[-1] if matches else None


def _manifest_issues(record: ManifestRecord | None) -> tuple[str, ...]:
    if record is None:
        return ()
    preflight = record.data.get("preflight")
    if not isinstance(preflight, dict):
        return ()
    results = preflight.get("results")
    if not isinstance(results, list):
        return ()
    issues: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("severity") in {"warning", "error"}:
            code = str(result.get("code", "unknown"))
            if code not in issues:
                issues.append(code)
    return tuple(issues)


def _common_stage_zero_status(
    config: ProjectConfig,
    stage: Stage,
    records: tuple[ManifestRecord, ...],
) -> StageStatus:
    output_dir = config.resolve_path("auto_transcripts_dir")
    latest = _latest_manifest(records, "common", "0")
    manifest_path = (
        display_path(latest.path, config.project_root)
        if latest is not None
        else None
    )
    if _path_has_files(output_dir, "*.cha"):
        return StageStatus(
            branch="common",
            stage_id="0",
            stage_name=stage.name,
            stage_type=stage.type,
            state="complete",
            message="CHAT transcripts are present for downstream stages.",
            after_manual_command=_plan_command("common", "1"),
            latest_manifest=manifest_path,
            expected_outputs=(display_path(output_dir, config.project_root),),
            issues=_manifest_issues(latest),
        )
    return StageStatus(
        branch="common",
        stage_id="0",
        stage_name=stage.name,
        stage_type=stage.type,
        state="manual_pending",
        message="Prepare ASR/manual transcript outputs; completion is not confirmed.",
        after_manual_command=_plan_command("common", "1"),
        latest_manifest=manifest_path,
        expected_outputs=(display_path(output_dir, config.project_root),),
        issues=_manifest_issues(latest),
    )


def _manual_artifact_complete(config: ProjectConfig, branch: str, stage_id: str) -> bool:
    if branch == "monolog" and stage_id == "8m":
        return (
            _path_has_files(config.resolve_path("monolog_word_count_files_dir"), "word_counting.xlsx")
            and _path_has_files(config.resolve_path("monolog_speaking_times_dir"), "speaking_times.xlsx")
        )
    return False


def _stage_paths(plan: Plan, config: ProjectConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(display_path(path, config.project_root) for path in plan.expected_inputs),
        tuple(display_path(path, config.project_root) for path in plan.expected_outputs),
    )


def _stage_has_output_artifacts(plan: Plan) -> bool:
    return any(_path_has_files(path) for path in plan.expected_outputs)


def _stage_has_input_artifacts(plan: Plan) -> bool:
    return any(_path_has_files(path) for path in plan.expected_inputs)


def _later_successful_manifest(
    records: tuple[ManifestRecord, ...],
    branch: str,
    stage_ids: list[str],
    index: int,
) -> bool:
    later_ids = set(stage_ids[index + 1 :])
    return any(
        record.branch == branch
        and record.stage_id in later_ids
        and not record.dry_run
        and record.status in COMPLETE_MANIFEST_STATUSES
        for record in records
    )


def _next_stage_command(branch: str, stage_ids: list[str], index: int) -> str | None:
    if index + 1 >= len(stage_ids):
        return None
    return _plan_command(branch, stage_ids[index + 1])


def _infer_planned_stage_status(
    config: ProjectConfig,
    branch: str,
    stage_id: str,
    stage_ids: list[str],
    index: int,
    records: tuple[ManifestRecord, ...],
    prior_complete: bool,
) -> StageStatus:
    stage = get_stage(branch, stage_id)
    plan = create_stage_plan(config, branch, stage_id)
    expected_inputs, expected_outputs = _stage_paths(plan, config)
    latest_effective = _latest_effective_manifest(records, branch, stage_id)
    latest_any = _latest_manifest(records, branch, stage_id)
    latest_path = (
        display_path(latest_any.path, config.project_root)
        if latest_any is not None
        else None
    )
    issues = _manifest_issues(latest_any)
    command = _plan_command(branch, stage_id) if plan.diaad_commands else None
    after_manual_command = _next_stage_command(branch, stage_ids, index)

    if latest_effective is not None and latest_effective.status in COMPLETE_MANIFEST_STATUSES:
        return StageStatus(
            branch=branch,
            stage_id=stage_id,
            stage_name=stage.name,
            stage_type=stage.type,
            state="complete",
            message="Latest substantive run manifest succeeded.",
            command=command,
            after_manual_command=after_manual_command if stage.type in {"manual", "external_manual"} else None,
            diaad_command_names=plan.diaad_command_names,
            latest_manifest=latest_path,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            issues=issues,
        )

    if latest_effective is not None and latest_effective.status in BLOCKED_MANIFEST_STATUSES:
        return StageStatus(
            branch=branch,
            stage_id=stage_id,
            stage_name=stage.name,
            stage_type=stage.type,
            state="blocked",
            message=f"Latest substantive run manifest ended as {latest_effective.status}.",
            command=command,
            after_manual_command=after_manual_command if stage.type in {"manual", "external_manual"} else None,
            diaad_command_names=plan.diaad_command_names,
            latest_manifest=latest_path,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            issues=issues,
        )

    if _later_successful_manifest(records, branch, stage_ids, index):
        return StageStatus(
            branch=branch,
            stage_id=stage_id,
            stage_name=stage.name,
            stage_type=stage.type,
            state="complete",
            message="Inferred complete from a later successful stage manifest.",
            command=command,
            after_manual_command=after_manual_command if stage.type in {"manual", "external_manual"} else None,
            diaad_command_names=plan.diaad_command_names,
            latest_manifest=latest_path,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            issues=issues,
        )

    if stage.type in {"manual", "external_manual"}:
        if _manual_artifact_complete(config, branch, stage_id):
            state = "complete"
            message = "Manual completion inferred from expected downstream artifacts."
        elif prior_complete or _stage_has_input_artifacts(plan):
            state = "manual_pending"
            message = "Manual completion is not confirmed."
        else:
            state = "not_started"
            message = "Waiting for prior stages before manual work can be confirmed."
        return StageStatus(
            branch=branch,
            stage_id=stage_id,
            stage_name=stage.name,
            stage_type=stage.type,
            state=state,
            message=message,
            after_manual_command=after_manual_command,
            diaad_command_names=plan.diaad_command_names,
            latest_manifest=latest_path,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            issues=issues,
        )

    if _stage_has_output_artifacts(plan):
        return StageStatus(
            branch=branch,
            stage_id=stage_id,
            stage_name=stage.name,
            stage_type=stage.type,
            state="unknown",
            message="Expected outputs exist, but no successful run manifest was found.",
            command=command,
            diaad_command_names=plan.diaad_command_names,
            latest_manifest=latest_path,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            issues=issues,
        )

    first_branch_stage = index == 0 and branch in {"monolog", "dialog"}
    ready = prior_complete or (first_branch_stage and _stage_has_input_artifacts(plan))
    return StageStatus(
        branch=branch,
        stage_id=stage_id,
        stage_name=stage.name,
        stage_type=stage.type,
        state="ready" if ready else "not_started",
        message="Ready to run." if ready else "Waiting for prerequisite artifacts or stages.",
        command=command,
        diaad_command_names=plan.diaad_command_names,
        latest_manifest=latest_path,
        expected_inputs=expected_inputs,
        expected_outputs=expected_outputs,
        issues=issues,
    )


def build_status_report(config: ProjectConfig) -> WorkflowStatusReport:
    """Build a workflow status report for all active RASCAL branches."""

    records, manifest_warnings = load_manifest_records(config)
    branch_statuses: dict[str, tuple[StageStatus, ...]] = {}
    for branch in BRANCHES:
        stages = load_branch_stages(branch)
        ids = stage_order(branch)
        statuses: list[StageStatus] = []
        prior_complete = True
        for index, stage_id in enumerate(ids):
            if branch == "common" and stage_id == "0":
                status = _common_stage_zero_status(config, stages[stage_id], records)
            else:
                try:
                    status = _infer_planned_stage_status(
                        config,
                        branch,
                        stage_id,
                        ids,
                        index,
                        records,
                        prior_complete,
                    )
                except PlanError as exc:
                    stage = stages[stage_id]
                    status = StageStatus(
                        branch=branch,
                        stage_id=stage_id,
                        stage_name=stage.name,
                        stage_type=stage.type,
                        state="unknown",
                        message=str(exc),
                    )
            statuses.append(status)
            prior_complete = status.state == "complete"
        branch_statuses[branch] = tuple(statuses)
    return WorkflowStatusReport(
        profile=config.profile_name,
        branches=branch_statuses,
        manifest_warnings=manifest_warnings,
    )


def _recommendation_for_stage(status: StageStatus) -> NextRecommendation:
    if status.state == "blocked":
        return NextRecommendation(
            branch=status.branch,
            stage_id=status.stage_id,
            stage_name=status.stage_name,
            state=status.state,
            action=f"Resolve blocked stage {status.branch}:{status.stage_id}.",
            command=status.command,
            diaad_command_names=status.diaad_command_names,
        )
    if status.stage_type in {"manual", "external_manual"} or status.state == "manual_pending":
        return NextRecommendation(
            branch=status.branch,
            stage_id=status.stage_id,
            stage_name=status.stage_name,
            state=status.state,
            action=f"Complete manual work for {status.branch}:{status.stage_id}.",
            after_manual_command=status.after_manual_command,
            diaad_command_names=status.diaad_command_names,
        )
    if status.command:
        return NextRecommendation(
            branch=status.branch,
            stage_id=status.stage_id,
            stage_name=status.stage_name,
            state=status.state,
            action=f"Run {status.branch}:{status.stage_id}.",
            command=status.command,
            diaad_command_names=status.diaad_command_names,
        )
    return NextRecommendation(
        branch=status.branch,
        stage_id=status.stage_id,
        stage_name=status.stage_name,
        state=status.state,
        action=status.message,
        diaad_command_names=status.diaad_command_names,
    )


def _branch_recommendation(branch: str, stages: tuple[StageStatus, ...]) -> NextRecommendation:
    for status in stages:
        if status.state != "complete":
            return _recommendation_for_stage(status)
    return NextRecommendation(
        branch=branch,
        stage_id=None,
        stage_name=None,
        state="complete",
        action=f"All {branch} stages are complete.",
    )


def build_next_report(status_report: WorkflowStatusReport) -> NextReport:
    """Build next-step recommendations from a workflow status report."""

    recommendations = tuple(
        _branch_recommendation(branch, status_report.branches[branch])
        for branch in BRANCHES
    )
    primary = next(
        (recommendation for recommendation in recommendations if recommendation.state != "complete"),
        recommendations[0],
    )
    return NextReport(
        profile=status_report.profile,
        primary=primary,
        recommendations=recommendations,
    )


def render_status_json(report: WorkflowStatusReport) -> str:
    """Render a status report as JSON."""

    return json.dumps(report.as_dict(), indent=2)


def render_status_text(report: WorkflowStatusReport) -> str:
    """Render a status report as concise text."""

    counts = report.counts
    lines = [
        "RASCAL status",
        f"Profile: {report.profile}",
        (
            "Summary: "
            f"{counts['complete']} complete, "
            f"{counts['ready']} ready, "
            f"{counts['manual_pending']} manual pending, "
            f"{counts['blocked']} blocked, "
            f"{counts['unknown']} unknown"
        ),
    ]
    if report.manifest_warnings:
        lines.extend(["", "Manifest warnings:"])
        lines.extend(f"  - {warning}" for warning in report.manifest_warnings)
    for branch in BRANCHES:
        lines.extend(["", f"{branch}:"])
        for status in report.branches[branch]:
            command = f" | {status.command}" if status.command else ""
            lines.append(
                f"  {status.stage_id} {status.stage_name}: {status.state} - {status.message}{command}"
            )
    return "\n".join(lines)


def render_next_json(report: NextReport) -> str:
    """Render next-step recommendations as JSON."""

    return json.dumps(report.as_dict(), indent=2)


def render_next_text(report: NextReport) -> str:
    """Render next-step recommendations as concise text."""

    lines = [
        "RASCAL next",
        f"Profile: {report.profile}",
        f"Primary: {report.primary.branch}:{report.primary.stage_id or 'complete'} - {report.primary.action}",
    ]
    if report.primary.command:
        lines.append(f"Command: {report.primary.command}")
    if report.primary.after_manual_command:
        lines.append(f"After manual completion: {report.primary.after_manual_command}")
    lines.extend(["", "Branch recommendations:"])
    for recommendation in report.recommendations:
        stage = recommendation.stage_id or "complete"
        lines.append(f"  {recommendation.branch}:{stage} - {recommendation.action}")
        if recommendation.command:
            lines.append(f"    command: {recommendation.command}")
        if recommendation.after_manual_command:
            lines.append(f"    after manual completion: {recommendation.after_manual_command}")
        if recommendation.diaad_command_names:
            commands = ", ".join(recommendation.diaad_command_names)
            lines.append(f"    DIAAD: {commands}")
    return "\n".join(lines)
