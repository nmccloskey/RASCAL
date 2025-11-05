import pytest
import re
import logging
from rascal.utils.tier import Tier
from rascal.utils.tier_parsing import read_tiers


def test_multiple_values_literal(monkeypatch):
    config = {
        "site": {"values": ["AC", "BU"], "partition": True, "blind": False}
    }
    tiers = read_tiers(config)

    assert "site" in tiers
    tier = tiers["site"]
    assert isinstance(tier, Tier)
    assert tier.values == ["AC", "BU"]
    assert tier.partition is True
    assert tier.blind is False


def test_single_value_regex_valid():
    config = {"id": {"values": ["[A-Z]{2}\\d{2}"]}}
    tiers = read_tiers(config)

    tier = tiers["id"]
    assert isinstance(tier, Tier)
    assert tier.values == ["[A-Z]{2}\\d{2}"]
    # Verify regex works
    assert re.match(tier.values[0], "TU88")


def test_single_value_regex_invalid(caplog):
    config = {"bad": {"values": ["[A-"]}}  # invalid regex
    with caplog.at_level(logging.ERROR):
        tiers = read_tiers(config)

    # It shouldn't crash, but should log the error
    assert "invalid user regex" in caplog.text
    assert "bad" not in tiers or tiers["bad"].values == []


def test_empty_values_logs_warning(caplog):
    config = {"empty": {"values": []}}
    with caplog.at_level(logging.WARNING):
        tiers = read_tiers(config)

    assert "empty" in tiers
    assert tiers["empty"].values == []
    assert "has no values" in caplog.text


def test_legacy_string_input():
    config = {"participantID": "site##"}
    tiers = read_tiers(config)
    tier = tiers["participantID"]
    assert tier.values == ["site##"]


def test_legacy_list_input():
    config = {"test": ["Pre", "Post"]}
    tiers = read_tiers(config)
    assert tiers["test"].values == ["Pre", "Post"]


def test_invalid_config_type(caplog):
    tiers = read_tiers(["not", "a", "dict"])  # wrong type
    assert tiers == {}
    assert "Invalid tier structure" in caplog.text
