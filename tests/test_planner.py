from __future__ import annotations

import pytest

from rascal.config import init_project, load_project_config
from rascal.planner import PlanError, create_stage_plan, render_plan_json, render_plan_text


def _config(tmp_path):
    result = init_project(tmp_path, profile="lab_full")
    return load_project_config(result.config_path)


def test_plans_monolog_stage_4m(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "monolog", "4m")

    assert plan.branch == "monolog"
    assert plan.stage_id == "4m"
    assert plan.diaad_command_names == ("transcripts tabularize",)
    assert plan.diaad_commands == (
        ("diaad", "transcripts", "tabularize", "--config", "config/diaad.generated"),
    )
    assert [path.name for path in plan.expected_outputs] == ["transcript_tables"]


def test_plans_dialog_stage_4d(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "dialog", "4d")

    assert plan.branch == "dialog"
    assert plan.stage_id == "4d"
    assert plan.diaad_commands == (
        ("diaad", "transcripts", "tabularize", "--config", "config/diaad.generated"),
    )
    assert plan.expected_outputs[0].as_posix().endswith("data/dialog/transcript_tables")


def test_plans_monolog_stage_7m_with_three_diaad_commands(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "monolog", "7m")

    assert plan.diaad_command_names == ("cus analyze", "templates times", "words files")
    assert plan.diaad_commands == (
        ("diaad", "cus", "analyze", "--config", "config/diaad.generated"),
        ("diaad", "templates", "times", "--config", "config/diaad.generated"),
        ("diaad", "words", "files", "--config", "config/diaad.generated"),
    )
    assert "Completed CU coding workbooks" in plan.manual_prerequisites[0]
    assert {path.name for path in plan.expected_outputs} == {
        "cu_analysis",
        "speaking_times",
        "word_count_files",
    }


def test_plans_dialog_stage_7d_with_powers_analyze_only(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "dialog", "7d")

    assert plan.diaad_command_names == ("powers analyze",)
    assert plan.diaad_commands == (
        ("diaad", "powers", "analyze", "--config", "config/diaad.generated"),
    )
    assert "powers rates" not in plan.diaad_command_names


def test_manual_stage_planning_does_not_produce_executable_commands(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "monolog", "5m")

    assert plan.stage_type == "manual"
    assert plan.diaad_command_names == ()
    assert plan.diaad_commands == ()
    assert "manual stage" in render_plan_text(plan)


def test_unknown_branch_or_stage_gives_clear_error(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(PlanError, match="Unknown RASCAL stage"):
        create_stage_plan(config, "monolog", "7")
    with pytest.raises(PlanError, match="Unknown RASCAL stage registry"):
        create_stage_plan(config, "unknown", "4m")


def test_plan_text_includes_commands_and_expected_paths(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "monolog", "7m")
    text = render_plan_text(plan)

    assert "RASCAL plan: monolog 7m" in text
    assert "diaad cus analyze --config config/diaad.generated" in text
    assert "data/monolog/cu_files" in text
    assert "data/monolog/word_count_files" in text


def test_plan_json_is_parseable_and_contains_commands(tmp_path):
    plan = create_stage_plan(_config(tmp_path), "dialog", "4d")
    rendered = render_plan_json(plan)

    import json

    payload = json.loads(rendered)
    assert payload["branch"] == "dialog"
    assert payload["stage_id"] == "4d"
    assert payload["diaad_commands"] == [
        ["diaad", "transcripts", "tabularize", "--config", "config/diaad.generated"]
    ]


def test_planned_commands_reference_generated_config_path(tmp_path):
    config = _config(tmp_path)
    plan = create_stage_plan(config, "monolog", "4m")

    assert plan.generated_config_path == config.resolve_path("diaad_config_dir")
    assert plan.diaad_commands == (
        ("diaad", "transcripts", "tabularize", "--config", "config/diaad.generated"),
    )
