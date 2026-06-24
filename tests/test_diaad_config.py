from __future__ import annotations

import yaml

from rascal.config import init_project, load_project_config
from rascal.diaad_config import write_diaad_config
from rascal.planner import create_stage_plan


def _generated(tmp_path, branch: str, stage_id: str, *, layout: str = "canonical"):
    result = init_project(tmp_path, profile="lab_full", layout=layout)
    config = load_project_config(result.config_path)
    plan = create_stage_plan(config, branch, stage_id)
    generated = write_diaad_config(config, plan)
    return config, plan, generated


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_generated_config_directory_contains_expected_split_files(tmp_path):
    _, _, generated = _generated(tmp_path, "monolog", "4m")

    assert generated.project_path.is_file()
    assert generated.advanced_path.is_file()
    assert generated.metadata_path.is_file()
    assert generated.rascal_source_path.is_file()


def test_project_yaml_maps_common_defaults_and_monolog_metadata(tmp_path):
    _, _, generated = _generated(tmp_path, "monolog", "4m")

    project = _read_yaml(generated.project_path)
    metadata = project["metadata_fields"]

    assert project["random_seed"] == 99
    assert project["reliability_fraction"] == 0.2
    assert project["stimulus_column"] == "narrative"
    assert metadata["site"] == ["AC", "BU", "TU"]
    assert metadata["test"] == ["Pre", "Post", "Maint"]
    assert metadata["study_id"] == "(AC|BU|TU)\\d+"
    assert metadata["narrative"] == [
        "CATGrandpa",
        "BrokenWindow",
        "RefusedUmbrella",
        "CatRescue",
        "BirthdayScene",
    ]


def test_project_yaml_maps_dialog_metadata_and_powers_defaults(tmp_path):
    _, _, generated = _generated(tmp_path, "dialog", "4d")

    project = _read_yaml(generated.project_path)
    advanced = _read_yaml(generated.advanced_path)

    assert project["stimulus_column"] == "communication"
    assert project["automate_powers"] is True
    assert project["metadata_fields"]["test"] == ["Pre", "Post", "Maint"]
    assert project["metadata_fields"]["communication"] == ["Dialog"]
    assert advanced["spacy_model_name"] == "en_core_web_trf"


def test_advanced_yaml_maps_blinding_and_identifier_expectations(tmp_path):
    _, _, generated = _generated(tmp_path, "monolog", "4m")

    advanced = _read_yaml(generated.advanced_path)

    assert advanced["auto_blind"] is True
    assert advanced["blind_columns"] == ["site", "test"]
    assert advanced["sample_id_column"] == "sample_id"
    assert advanced["utterance_id_column"] == "utterance_id"
    assert advanced["id_columns"] == ["sample_id", "utterance_id"]


def test_monolog_corelex_stimuli_are_kept_in_audit_metadata(tmp_path):
    _, _, generated = _generated(tmp_path, "monolog", "10m")

    advanced = _read_yaml(generated.advanced_path)
    metadata = _read_yaml(generated.metadata_path)

    assert advanced["target_vocabulary_resource_path"] == ""
    assert metadata["corelex_stimuli"] == [
        "BrokenWindow",
        "RefusedUmbrella",
        "CatRescue",
    ]


def test_stage_specific_paths_differ_for_monolog_and_dialog(tmp_path):
    _, _, monolog = _generated(tmp_path / "monolog_project", "monolog", "4m")
    _, _, dialog = _generated(tmp_path / "dialog_project", "dialog", "4d")

    monolog_project = _read_yaml(monolog.project_path)
    dialog_project = _read_yaml(dialog.project_path)

    assert monolog_project["input_dir"] == "data/auto_transcripts"
    assert monolog_project["output_dir"] == "data/monolog/transcript_tables"
    assert dialog_project["input_dir"] == "data/auto_transcripts"
    assert dialog_project["output_dir"] == "data/dialog/transcript_tables"


def test_legacy_layout_path_generation_uses_legacy_input_and_output(tmp_path):
    _, _, generated = _generated(tmp_path, "monolog", "4m", layout="legacy")

    project = _read_yaml(generated.project_path)

    assert project["input_dir"] == "rascal_data/input"
    assert project["output_dir"] == "rascal_data/output"


def test_rascal_source_snapshot_records_plan_context(tmp_path):
    _, plan, generated = _generated(tmp_path, "dialog", "7d")

    source = _read_yaml(generated.rascal_source_path)

    assert source["profile"] == "lab_full"
    assert source["branch"] == "dialog"
    assert source["stage_id"] == "7d"
    assert source["stage_name"] == plan.stage_name
    assert source["diaad_commands"] == [
        ["diaad", "powers", "analyze", "--config", "config/diaad.generated"]
    ]
