"""Discovery for archived RASCAL workflow manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ARCHIVE_DIR = Path("archived_workflows")
MANIFEST_NAME = "workflow_manifest.yaml"


class WorkflowError(ValueError):
    """Raised when archived workflow discovery cannot satisfy a request."""


@dataclass(frozen=True)
class WorkflowFileRef:
    """A referenced file in an archived workflow."""

    kind: str
    path: str
    exists: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable file reference."""

        return {
            "kind": self.kind,
            "path": self.path,
            "exists": self.exists,
        }


@dataclass(frozen=True)
class WorkflowRecord:
    """One parsed archived workflow manifest."""

    workflow_id: str
    name: str
    status: str
    intended_reuse: str
    primary_docs: tuple[str, ...]
    manifest_path: Path
    workflow_root: Path
    data: dict[str, Any]

    def as_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "intended_reuse": self.intended_reuse,
            "primary_docs": list(self.primary_docs),
            "manifest_path": self.manifest_path.as_posix(),
        }

    def as_detail_dict(self, *, include_files: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable detailed record."""

        detail = dict(self.data)
        detail.setdefault("workflow_id", self.workflow_id)
        detail.setdefault("name", self.name)
        detail.setdefault("status", self.status)
        detail.setdefault("intended_reuse", self.intended_reuse)
        detail["manifest_path"] = self.manifest_path.as_posix()
        if include_files:
            detail["referenced_files"] = [
                file_ref.as_dict() for file_ref in referenced_files(self)
            ]
        return detail


@dataclass(frozen=True)
class WorkflowDiscovery:
    """Archived workflow discovery result."""

    workflows: tuple[WorkflowRecord, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable discovery result."""

        return {
            "workflows": [workflow.as_summary_dict() for workflow in self.workflows],
            "warnings": list(self.warnings),
        }


def _archive_root(archive_root: str | Path | None = None) -> Path:
    return Path(archive_root or DEFAULT_ARCHIVE_DIR).expanduser().resolve()


def _coerce_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _load_workflow_manifest(manifest_path: Path, archive_root: Path) -> WorkflowRecord:
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise WorkflowError(f"Could not parse {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise WorkflowError(f"Could not read {manifest_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise WorkflowError(f"Workflow manifest must contain a mapping: {manifest_path}")

    workflow_id = loaded.get("workflow_id") or manifest_path.parent.name
    name = loaded.get("name") or workflow_id
    status = loaded.get("status") or "unknown"
    intended_reuse = loaded.get("intended_reuse") or ""
    primary_docs = _coerce_string_list(loaded.get("key_docs"))
    try:
        display_manifest_path = manifest_path.resolve().relative_to(archive_root.parent).as_posix()
    except ValueError:
        display_manifest_path = manifest_path.resolve().as_posix()

    return WorkflowRecord(
        workflow_id=str(workflow_id),
        name=str(name),
        status=str(status),
        intended_reuse=str(intended_reuse),
        primary_docs=primary_docs,
        manifest_path=Path(display_manifest_path),
        workflow_root=manifest_path.parent.resolve(),
        data=dict(loaded),
    )


def discover_workflows(archive_root: str | Path | None = None) -> WorkflowDiscovery:
    """Discover archived workflow manifests."""

    root = _archive_root(archive_root)
    if not root.exists():
        return WorkflowDiscovery(
            workflows=(),
            warnings=(f"Archived workflow directory not found: {root}",),
        )

    workflows: list[WorkflowRecord] = []
    warnings: list[str] = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        try:
            workflows.append(_load_workflow_manifest(manifest_path, root))
        except WorkflowError as exc:
            warnings.append(str(exc))

    workflows.sort(key=lambda workflow: workflow.workflow_id)
    return WorkflowDiscovery(
        workflows=tuple(workflows),
        warnings=tuple(warnings),
    )


def get_workflow(
    workflow_id: str,
    archive_root: str | Path | None = None,
) -> WorkflowRecord:
    """Return one archived workflow by id."""

    discovery = discover_workflows(archive_root)
    for workflow in discovery.workflows:
        if workflow.workflow_id == workflow_id:
            return workflow
    known = ", ".join(workflow.workflow_id for workflow in discovery.workflows)
    suffix = f" Known workflows: {known}" if known else " No workflows were discovered."
    raise WorkflowError(f"Unknown archived workflow: {workflow_id}.{suffix}")


def referenced_files(workflow: WorkflowRecord) -> tuple[WorkflowFileRef, ...]:
    """Return docs and scripts referenced by an archived workflow manifest."""

    refs: list[WorkflowFileRef] = []
    for kind, key in (("doc", "key_docs"), ("script", "key_scripts")):
        for relative_path in _coerce_string_list(workflow.data.get(key)):
            path = workflow.workflow_root / relative_path
            refs.append(
                WorkflowFileRef(
                    kind=kind,
                    path=relative_path,
                    exists=path.exists(),
                )
            )
    return tuple(refs)


def render_workflow_list_text(discovery: WorkflowDiscovery) -> str:
    """Render archived workflow discovery as concise text."""

    lines = ["Archived workflows:"]
    if discovery.workflows:
        for workflow in discovery.workflows:
            docs = ", ".join(workflow.primary_docs) if workflow.primary_docs else "(none)"
            reuse = f" | reuse: {workflow.intended_reuse}" if workflow.intended_reuse else ""
            lines.append(
                f"  - {workflow.workflow_id}: {workflow.name} "
                f"[{workflow.status}] | docs: {docs}{reuse}"
            )
    else:
        lines.append("  (none discovered)")

    if discovery.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in discovery.warnings)
    return "\n".join(lines)


def render_workflow_list_json(discovery: WorkflowDiscovery) -> str:
    """Render archived workflow discovery as JSON."""

    return json.dumps(discovery.as_dict(), indent=2)


def render_workflow_detail_text(workflow: WorkflowRecord, *, include_files: bool = False) -> str:
    """Render one archived workflow as concise text."""

    lines = [
        f"Archived workflow: {workflow.workflow_id}",
        f"Name: {workflow.name}",
        f"Status: {workflow.status}",
    ]
    if workflow.intended_reuse:
        lines.append(f"Intended reuse: {workflow.intended_reuse}")
    if workflow.data.get("purpose"):
        lines.extend(["", "Purpose:", f"  {workflow.data['purpose']}"])
    if workflow.primary_docs:
        lines.extend(["", "Primary docs:"])
        lines.extend(f"  - {path}" for path in workflow.primary_docs)
    scripts = _coerce_string_list(workflow.data.get("key_scripts"))
    if scripts:
        lines.extend(["", "Key scripts:"])
        lines.extend(f"  - {path}" for path in scripts)
    if include_files:
        lines.extend(["", "Referenced files:"])
        refs = referenced_files(workflow)
        if refs:
            lines.extend(
                f"  - {ref.kind}: {ref.path} ({'exists' if ref.exists else 'missing'})"
                for ref in refs
            )
        else:
            lines.append("  (none)")
    return "\n".join(lines)


def render_workflow_detail_json(
    workflow: WorkflowRecord,
    *,
    include_files: bool = False,
) -> str:
    """Render one archived workflow as JSON."""

    return json.dumps(workflow.as_detail_dict(include_files=include_files), indent=2)
