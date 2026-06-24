from __future__ import annotations

import json
from subprocess import CompletedProcess

from rascal.config import init_project, load_project_config
from rascal.runner import run_stage
from rascal.status import (
    build_next_report,
    build_status_report,
    render_next_json,
    render_status_json,
)


def _config(tmp_path):
    result = init_project(tmp_path, profile="lab_full")
    return load_project_config(result.config_path)


def _successful_runner(command, **_kwargs):
    return CompletedProcess(command, 0, stdout="", stderr="")


def _failing_runner(command, **_kwargs):
    return CompletedProcess(command, 7, stdout="", stderr="failed\n")


def _stage(report, branch: str, stage_id: str):
    return next(
        status
        for status in report.branches[branch]
        if status.stage_id == stage_id
    )


def _recommendation(report, branch: str):
    return next(
        recommendation
        for recommendation in report.recommendations
        if recommendation.branch == branch
    )


def test_fresh_project_recommends_common_stage_0(tmp_path):
    status_report = build_status_report(_config(tmp_path))
    next_report = build_next_report(status_report)

    assert _stage(status_report, "common", "0").state == "manual_pending"
    assert next_report.primary.branch == "common"
    assert next_report.primary.stage_id == "0"
    assert next_report.primary.after_manual_command == "rascal run --branch common --stage 1"


def test_chat_data_makes_common_stage_1_next(tmp_path):
    config = _config(tmp_path)
    (config.resolve_path("auto_transcripts_dir") / "AC001_Pre_BrokenWindow.cha").write_text(
        "@Begin\n@End\n",
        encoding="utf-8",
    )

    next_report = build_next_report(build_status_report(config))

    assert next_report.primary.branch == "common"
    assert next_report.primary.stage_id == "1"
    assert next_report.primary.command == "rascal run --branch common --stage 1"


def test_monolog_stage_4_manifest_makes_cu_prep_next(tmp_path):
    config = _config(tmp_path)
    run_stage(config, "monolog", "4m", subprocess_run=_successful_runner)

    status_report = build_status_report(config)
    next_report = build_next_report(status_report)
    monolog = _recommendation(next_report, "monolog")

    assert _stage(status_report, "monolog", "4m").state == "complete"
    assert monolog.stage_id == "5m_prepare"
    assert monolog.command == "rascal run --branch monolog --stage 5m_prepare"


def test_dialog_stage_4_manifest_makes_powers_prep_next(tmp_path):
    config = _config(tmp_path)
    run_stage(config, "dialog", "4d", subprocess_run=_successful_runner)

    dialog = _recommendation(build_next_report(build_status_report(config)), "dialog")

    assert dialog.stage_id == "5d_prepare"
    assert dialog.command == "rascal run --branch dialog --stage 5d_prepare"


def test_manual_monolog_stage_5_reports_pending_when_cu_files_exist(tmp_path):
    config = _config(tmp_path)
    (config.resolve_path("monolog_cu_files_dir") / "cu_coding_by_sample_long.xlsx").write_text(
        "placeholder",
        encoding="utf-8",
    )

    status_report = build_status_report(config)

    assert _stage(status_report, "monolog", "5m").state == "manual_pending"
    assert "not confirmed" in _stage(status_report, "monolog", "5m").message


def test_dialog_stage_7_next_command_excludes_powers_rates(tmp_path):
    config = _config(tmp_path)
    run_stage(config, "dialog", "6d", subprocess_run=_successful_runner)

    dialog = _recommendation(build_next_report(build_status_report(config)), "dialog")

    assert dialog.stage_id == "7d"
    assert dialog.command == "rascal run --branch dialog --stage 7d"
    assert dialog.diaad_command_names == ("powers analyze",)
    assert "powers rates" not in render_next_json(build_next_report(build_status_report(config)))


def test_failed_manifest_marks_stage_blocked(tmp_path):
    config = _config(tmp_path)
    run_stage(config, "monolog", "7m", subprocess_run=_failing_runner)

    status_report = build_status_report(config)

    assert _stage(status_report, "monolog", "7m").state == "blocked"
    assert "failed" in _stage(status_report, "monolog", "7m").message


def test_status_json_output_is_parseable(tmp_path):
    payload = json.loads(render_status_json(build_status_report(_config(tmp_path))))

    assert payload["profile"] == "lab_full"
    assert payload["branches"]["common"][0]["stage_id"] == "0"
    assert "counts" in payload
