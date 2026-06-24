from __future__ import annotations

import json
from pathlib import Path

import pytest

from rascal.workflows import (
    WorkflowError,
    discover_workflows,
    get_workflow,
    referenced_files,
    render_workflow_detail_json,
    render_workflow_list_json,
)


def _write_manifest(root: Path, folder: str, text: str) -> Path:
    manifest = root / folder / "workflow_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(text, encoding="utf-8")
    return manifest


def test_list_discovers_existing_archived_workflow_manifests():
    discovery = discover_workflows("archived_workflows")
    workflow_ids = {workflow.workflow_id for workflow in discovery.workflows}

    assert "powers_automation_validation_2025" in workflow_ids
    assert "transcript_conversion_2026" in workflow_ids
    assert not discovery.warnings


def test_show_returns_expected_fields_for_synthetic_manifest(tmp_path):
    archive = tmp_path / "archived_workflows"
    _write_manifest(
        archive,
        "synthetic",
        """
workflow_id: synthetic_workflow
name: Synthetic Workflow
status: archived-test
intended_reuse: Test fixture only.
key_docs:
  - README.md
key_scripts:
  - src/example.py
""".strip(),
    )

    workflow = get_workflow("synthetic_workflow", archive)

    assert workflow.name == "Synthetic Workflow"
    assert workflow.status == "archived-test"
    assert workflow.primary_docs == ("README.md",)
    payload = json.loads(render_workflow_detail_json(workflow))
    assert payload["intended_reuse"] == "Test fixture only."


def test_missing_workflow_id_gives_clear_error(tmp_path):
    archive = tmp_path / "archived_workflows"
    _write_manifest(
        archive,
        "synthetic",
        "workflow_id: synthetic_workflow\nname: Synthetic Workflow\n",
    )

    with pytest.raises(WorkflowError, match="Unknown archived workflow: missing"):
        get_workflow("missing", archive)


def test_malformed_manifest_is_reported_without_crashing_list(tmp_path):
    archive = tmp_path / "archived_workflows"
    _write_manifest(
        archive,
        "good",
        "workflow_id: good_workflow\nname: Good Workflow\n",
    )
    _write_manifest(
        archive,
        "bad",
        "workflow_id: [unterminated\n",
    )

    discovery = discover_workflows(archive)

    assert [workflow.workflow_id for workflow in discovery.workflows] == ["good_workflow"]
    assert len(discovery.warnings) == 1
    assert "Could not parse" in discovery.warnings[0]
    payload = json.loads(render_workflow_list_json(discovery))
    assert payload["warnings"]


def test_files_option_reports_docs_and_scripts_as_paths_not_commands(tmp_path):
    archive = tmp_path / "archived_workflows"
    _write_manifest(
        archive,
        "synthetic",
        """
workflow_id: synthetic_workflow
name: Synthetic Workflow
key_docs:
  - README.md
  - docs/missing.md
key_scripts:
  - src/example.py
""".strip(),
    )
    workflow_root = archive / "synthetic"
    (workflow_root / "README.md").write_text("# Synthetic\n", encoding="utf-8")
    (workflow_root / "src").mkdir()
    (workflow_root / "src" / "example.py").write_text("print('not executed')\n", encoding="utf-8")

    workflow = get_workflow("synthetic_workflow", archive)
    refs = referenced_files(workflow)

    assert [(ref.kind, ref.path, ref.exists) for ref in refs] == [
        ("doc", "README.md", True),
        ("doc", "docs/missing.md", False),
        ("script", "src/example.py", True),
    ]
    payload = json.loads(render_workflow_detail_json(workflow, include_files=True))
    assert payload["referenced_files"][2]["path"] == "src/example.py"
    assert "print('not executed')" not in json.dumps(payload)
