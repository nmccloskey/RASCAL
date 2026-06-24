from __future__ import annotations

import json

from openpyxl import Workbook

import rascal.preflight as preflight
from rascal.config import init_project, load_project_config
from rascal.preflight import render_preflight_json, run_preflight


def _config(tmp_path):
    result = init_project(tmp_path, profile="lab_full")
    return load_project_config(result.config_path)


def _codes(report, severity: str | None = None) -> set[str]:
    return {
        result.code
        for result in report.results
        if severity is None or result.severity == severity
    }


def _write_transcript_table(path, sample_rows=None, utterance_rows=None, branch="monolog"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rows = sample_rows or [
        {
            "sample_id": "AC001",
            "site": "AC",
            "test": "Pre",
            "study_id": "AC001",
            "narrative": "BrokenWindow",
        }
    ]
    if branch == "dialog":
        sample_rows = sample_rows or [
            {
                "sample_id": "AC001",
                "site": "AC",
                "test": "Pre",
                "study_id": "AC001",
                "communication": "Dialog",
            }
        ]
    utterance_rows = utterance_rows or [
        {
            "sample_id": sample_rows[0]["sample_id"],
            "utterance_id": "1",
            "speaker": "PAR",
            "utterance": "hello",
        }
    ]

    workbook = Workbook()
    samples = workbook.active
    samples.title = "samples"
    sample_headers = list(sample_rows[0])
    samples.append(sample_headers)
    for row in sample_rows:
        samples.append([row.get(header) for header in sample_headers])

    utterances = workbook.create_sheet("utterances")
    utterance_headers = list(utterance_rows[0])
    utterances.append(utterance_headers)
    for row in utterance_rows:
        utterances.append([row.get(header) for header in utterance_headers])

    workbook.save(path)


def test_clean_initialized_project_reports_warnings_without_errors(tmp_path):
    report = run_preflight(_config(tmp_path), "monolog", "4m")

    assert "expected_input_empty" in _codes(report, "warning")
    assert "chat_files_missing" in _codes(report, "warning")
    assert not report.has_errors


def test_duplicate_transcript_tables_are_reported_as_error(tmp_path):
    config = _config(tmp_path)
    input_dir = config.resolve_path("monolog_transcript_tables_dir")
    _write_transcript_table(input_dir / "site_a" / "transcript_tables.xlsx")
    _write_transcript_table(input_dir / "site_b" / "transcript_tables.xlsx")

    report = run_preflight(config, "monolog", "5m_prepare")

    assert "duplicate_expected_file" in _codes(report, "error")
    assert report.has_errors


def test_malformed_chat_filename_is_reported(tmp_path):
    config = _config(tmp_path)
    (config.resolve_path("auto_transcripts_dir") / "badname.cha").write_text(
        "@Begin\n@End\n",
        encoding="utf-8",
    )

    report = run_preflight(config, "monolog", "4m")

    assert "chat_filename_unparsed" in _codes(report, "warning")


def test_duplicate_sample_id_is_reported_from_transcript_table(tmp_path):
    config = _config(tmp_path)
    _write_transcript_table(
        config.resolve_path("monolog_transcript_tables_dir") / "transcript_tables.xlsx",
        sample_rows=[
            {
                "sample_id": "AC001",
                "site": "AC",
                "test": "Pre",
                "study_id": "AC001",
                "narrative": "BrokenWindow",
            },
            {
                "sample_id": "AC001",
                "site": "AC",
                "test": "Post",
                "study_id": "AC001",
                "narrative": "CatRescue",
            },
        ],
    )

    report = run_preflight(config, "monolog", "5m_prepare")

    assert "duplicate_sample_id" in _codes(report, "error")


def test_missing_spacy_model_is_reported_when_powers_automation_is_enabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(preflight, "is_spacy_model_available", lambda _model: False)

    report = run_preflight(_config(tmp_path), "dialog", "5d_prepare")

    assert "spacy_model_missing" in _codes(report, "warning")


def test_thresholds_are_report_only(tmp_path):
    report = run_preflight(_config(tmp_path), "dialog", "4d")

    assert "thresholds_report_only" in _codes(report, "ok")


def test_json_output_is_parseable(tmp_path):
    report = run_preflight(_config(tmp_path), "monolog", "4m")

    payload = json.loads(render_preflight_json(report))

    assert payload["branch"] == "monolog"
    assert payload["stage_id"] == "4m"
    assert payload["counts"]["warning"] >= 1
