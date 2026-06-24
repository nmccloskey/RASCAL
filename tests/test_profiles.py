from __future__ import annotations

from rascal.profiles import list_profiles, load_packaged_profile, resolve_profile


def test_packaged_profiles_load():
    assert set(list_profiles()) == {
        "lab_common",
        "lab_dialog",
        "lab_full",
        "lab_monolog",
    }
    assert load_packaged_profile("lab_common")["profile"] == "lab_common"


def test_lab_monolog_inherits_common_defaults():
    profile = resolve_profile("lab_monolog")

    assert profile["profile"] == "lab_monolog"
    assert profile["random_seed"] == 99
    assert profile["reliability_fraction"] == 0.20
    assert profile["blind_columns"] == ["site", "test"]


def test_lab_monolog_has_narrative_metadata():
    profile = resolve_profile("lab_monolog")
    metadata_fields = profile["metadata_fields"]

    assert metadata_fields["test"] == ["Pre", "Post", "Maint"]
    assert metadata_fields["study_id"] == "(AC|BU|TU)\\d+"
    assert "narrative" in metadata_fields
    assert "BirthdayScene" in metadata_fields["narrative"]
    assert profile["monolog"]["cu_dialects"] == ["SAE", "AAE"]


def test_lab_dialog_has_communication_metadata_and_spacy_model():
    profile = resolve_profile("lab_dialog")
    metadata_fields = profile["metadata_fields"]

    assert metadata_fields["communication"] == ["Dialog"]
    assert "narrative" not in metadata_fields
    assert profile["dialog"]["powers_automation"] is True
    assert profile["dialog"]["spacy_model"] == "en_core_web_trf"


def test_thresholds_are_present_and_report_only():
    profile = resolve_profile("lab_common")
    thresholds = profile["thresholds"]

    assert thresholds["report_only"] is True
    assert thresholds["quality_bands"] == {
        "minimal": 0.7,
        "sufficient": 0.8,
        "excellent": 0.9,
    }
    assert thresholds["word_count_icc_2_1"]["target_value"] == 0.9
    assert thresholds["word_count_icc_2_1"]["report_only"] is True


def test_project_overrides_change_values_without_mutating_packaged_defaults():
    overridden = resolve_profile(
        "lab_monolog",
        {
            "random_seed": 123,
            "metadata_fields": {"test": ["Screening"]},
        },
    )
    fresh = resolve_profile("lab_monolog")

    assert overridden["random_seed"] == 123
    assert overridden["metadata_fields"]["test"] == ["Screening"]
    assert fresh["random_seed"] == 99
    assert fresh["metadata_fields"]["test"] == ["Pre", "Post", "Maint"]


def test_lab_full_carries_both_branch_metadata_sets():
    profile = resolve_profile("lab_full")

    assert profile["branches"] == ["monolog", "dialog"]
    assert "narrative" in profile["monolog"]["metadata_fields"]
    assert profile["dialog"]["metadata_fields"]["communication"] == ["Dialog"]

