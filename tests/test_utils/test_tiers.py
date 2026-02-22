from __future__ import annotations

import pytest

from rascal.utils.tiers import Tier, TierManager


def test_default_tiers_when_config_missing_or_invalid():
    tm = TierManager(config={})
    assert tm.get_tier_names() == ["file_name"]

    # default regex: r".*(?=\.cha)" should match everything before ".cha"
    out = tm.match_tiers("AC01_Pre.cha", return_none=True)
    assert out["file_name"] == "AC01_Pre"


def test_tier_match_return_none_vs_default_name_fallback():
    t = Tier(
        name="site",
        order=1,
        kind="values",
        pattern=__import__("re").compile(r"(?:AC|BU)"),
        values=["AC", "BU"],
        regex=None,
    )

    # no match -> default fallback is tier.name
    assert t.match("ZZ99.cha") == "site"
    # no match -> return_none=True returns None
    assert t.match("ZZ99.cha", return_none=True) is None


def test_values_tier_matches_literal_values_and_escapes_regex_metachars():
    # Ensure literal values are escaped (e.g., '.' is treated literally, not "any char")
    config = {
        "tiers": {
            "literal": {
                "values": ["A.C", "B|U"],  # metacharacters that must be escaped
            }
        }
    }
    tm = TierManager(config=config)
    m = tm.match_tiers("XX_A.C_YY.cha", return_none=True)
    assert m["literal"] == "A.C"

    m2 = tm.match_tiers("XX_B|U_YY.cha", return_none=True)
    assert m2["literal"] == "B|U"


def test_values_tier_empty_list_never_matches():
    config = {"tiers": {"site": {"values": []}}}
    tm = TierManager(config=config)

    # no match; by default Tier.match returns tier name (unless return_none=True)
    m = tm.match_tiers("AC01_Pre.cha", return_none=False)
    assert m["site"] == "site"

    m2 = tm.match_tiers("AC01_Pre.cha", return_none=True)
    assert m2["site"] is None


def test_regex_tier_matches():
    config = {"tiers": {"study_id": {"regex": r"(AC|BU|TU)\d+"}}}
    tm = TierManager(config=config)
    m = tm.match_tiers("TU104_BrokenWindow.cha", return_none=True)
    assert m["study_id"] == "TU104"


def test_invalid_tier_spec_type_raises():
    config = {"tiers": {"site": ["AC", "BU"]}}  # must be dict with values/regex
    with pytest.raises(TypeError):
        TierManager(config=config)


def test_tier_must_define_exactly_one_of_values_or_regex():
    with pytest.raises(ValueError):
        TierManager(config={"tiers": {"site": {"values": ["AC"], "regex": r"AC"}}})

    with pytest.raises(ValueError):
        TierManager(config={"tiers": {"site": {"order": 1}}})  # neither values nor regex


def test_invalid_order_type_raises():
    with pytest.raises(TypeError):
        TierManager(config={"tiers": {"site": {"order": "1", "values": ["AC"]}}})


def test_invalid_regex_raises_value_error():
    with pytest.raises(ValueError):
        TierManager(config={"tiers": {"bad": {"regex": 'r"("'}}})  # invalid regex


def test_compute_order_sorts_by_order_then_preserves_unordered_config_order():
    config = {
        "tiers": {
            "tB": {"order": 2, "values": ["B"]},
            "tA": {"order": 1, "values": ["A"]},
            "tC": {"values": ["C"]},  # unordered should come after ordered, preserving appearance
            "tD": {"values": ["D"]},
        }
    }
    tm = TierManager(config=config)
    assert tm.get_tier_names() == ["tA", "tB", "tC", "tD"]


def test_tier_groups_are_read_and_transformed():
    config = {
        "tiers": {
            "site": {"values": ["AC", "BU"]},
            "test": {"values": ["Pre", "Post"]},
        },
        "tier_groups": {
            "blind": ["site"],
            "partition": ["test"],
        },
    }
    tm = TierManager(config=config)
    assert tm.tiers_in_group("blind") == ["site"]
    assert tm.tiers_in_group("partition") == ["test"]
    assert tm.tiers_in_group("missing_group") == []


def test_make_blind_codebook_deterministic_for_values_tiers_only():
    config = {
        "tiers": {
            "site": {"values": ["AC", "BU", "TU"]},
            "study": {"regex": r"(AC|BU|TU)\d+"},  # should be skipped
        },
        "tier_groups": {"blind": ["site", "study"]},
    }
    tm = TierManager(config=config)

    cb1 = tm.make_blind_codebook(seed=123)
    cb2 = tm.make_blind_codebook(seed=123)
    cb3 = tm.make_blind_codebook(seed=124)

    assert set(cb1.keys()) == {"site"}  # regex tier skipped
    assert cb1 == cb2  # deterministic for same seed
    assert cb1["site"] != cb3["site"]  # likely different for different seed (not guaranteed, but extremely likely)

    # mapping is a permutation of 0..n-1
    codes = sorted(cb1["site"].values())
    assert codes == list(range(3))


def test_make_blind_codebook_handles_missing_blind_group_gracefully():
    config = {"tiers": {"site": {"values": ["AC", "BU"]}}}
    tm = TierManager(config=config)
    assert tm.make_blind_codebook(seed=1) == {}
