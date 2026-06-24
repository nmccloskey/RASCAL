from __future__ import annotations

import pytest

from rascal.stages import (
    StageError,
    get_stage,
    list_stage_branches,
    load_branch_stages,
    load_stage_registry,
    stage_order,
    validate_stage_registry,
)


def test_all_packaged_stage_files_load():
    assert set(list_stage_branches()) == {"common", "dialog", "monolog"}

    registry = load_stage_registry()

    assert set(registry) == {"common", "dialog", "monolog"}
    assert "1" in registry["common"]
    assert "4m" in registry["monolog"]
    assert "4d" in registry["dialog"]


def test_required_fields_are_valid_for_every_stage():
    assert validate_stage_registry() == []


def test_common_stages_stop_before_transcript_tabulation():
    common_stages = load_branch_stages("common")

    assert "4" not in common_stages
    assert all("transcripts tabularize" not in stage.diaad for stage in common_stages.values())
    assert stage_order("common") == ["0", "1", "2", "3"]


def test_monolog_and_dialog_tabulation_stages_are_branch_specific():
    monolog_stage = get_stage("monolog", "4m")
    dialog_stage = get_stage("dialog", "4d")

    assert monolog_stage.diaad == ("transcripts tabularize",)
    assert dialog_stage.diaad == ("transcripts tabularize",)
    assert "narrative" in monolog_stage.metadata_fields
    assert "communication" not in monolog_stage.metadata_fields
    assert dialog_stage.metadata_fields["communication"] == ["Dialog"]
    assert "narrative" not in dialog_stage.metadata_fields


def test_monolog_stage_10_includes_cu_word_and_vocab_rates():
    stage = get_stage("monolog", "10m")

    assert "cus rates" in stage.diaad
    assert "words rates" in stage.diaad
    assert "vocab analyze" in stage.diaad
    assert "vocab rates" in stage.diaad


def test_dialog_stage_7_excludes_powers_rates():
    stage = get_stage("dialog", "7d")

    assert stage.diaad == ("powers analyze",)
    assert "powers rates" not in stage.diaad


def test_old_numeric_aliases_are_not_registry_stage_ids():
    with pytest.raises(StageError):
        get_stage("common", "4")
    with pytest.raises(StageError):
        get_stage("monolog", "7")
    with pytest.raises(StageError):
        get_stage("monolog", "10")


def test_stage_order_preserves_coding_file_prep_before_manual_coding():
    assert stage_order("monolog")[:3] == ["4m", "5m_prepare", "5m"]
    assert stage_order("dialog")[:3] == ["4d", "5d_prepare", "5d"]

