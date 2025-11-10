import re
import types
import pytest
from rascal.utils.tier_parsing import read_tiers, default_tiers


class MockTier:
    """Minimal mock replacement for Tier class with regex validation."""
    def __init__(self, name, values, partition=False, blind=False):
        self.name = name
        self.values = values
        self.partition = partition
        self.blind = blind

    def __repr__(self):
        return f"MockTier({self.name}, {self.values}, partition={self.partition}, blind={self.blind})"


@pytest.fixture(autouse=True)
def patch_tier(monkeypatch):
    """Automatically patch rascal.tier_parsing.Tier for all tests."""
    monkeypatch.setattr("rascal.utils.tier_parsing.Tier", MockTier)


def test_read_tiers_with_valid_config(monkeypatch):
    """Test normal tier parsing with multiple tier types and flags."""
    config = {
        "site": {"values": [r"Site\d+"], "partition": True},
        "group": {"values": ["G1", "G2"], "blind": True},
    }

    tiers = read_tiers(config)
    assert isinstance(tiers, dict)
    assert set(tiers.keys()) == {"site", "group"}

    site_tier = tiers["site"]
    assert site_tier.partition is True
    assert site_tier.blind is False
    assert re.compile(site_tier.values[0])  # Valid regex compiles

    group_tier = tiers["group"]
    assert group_tier.partition is False
    assert group_tier.blind is True
    assert group_tier.values == ["G1", "G2"]


def test_read_tiers_with_invalid_regex(monkeypatch, caplog):
    """Invalid regex should be caught and skipped."""
    config = {"bad": {"values": ["[unclosed"], "partition": False}}
    tiers = read_tiers(config)
    assert "bad" not in tiers  # skipped due to invalid regex
    assert any("invalid regex" in rec.message for rec in caplog.records)


def test_read_tiers_with_legacy_shorthand(monkeypatch):
    """Legacy shorthand list or string should be normalized."""
    config = {
        "legacy_list": ["A", "B", "C"],
        "legacy_str": "pattern",
    }
    tiers = read_tiers(config)
    assert "legacy_list" in tiers
    assert "legacy_str" in tiers
    assert isinstance(tiers["legacy_str"].values, list)
    assert len(tiers["legacy_str"].values) == 1
    assert tiers["legacy_list"].values == ["A", "B", "C"]


def test_read_tiers_with_empty_and_invalid(monkeypatch):
    """Empty or invalid tiers should trigger warning and create empty Tier."""
    config = {
        "empty": {"values": []},
        "invalid_type": 12345,
    }
    tiers = read_tiers(config)
    assert "empty" in tiers
    assert isinstance(tiers["empty"].values, list)
    assert len(tiers["empty"].values) == 0


def test_read_tiers_with_no_config(monkeypatch):
    """None or invalid input should fallback to default_tiers."""
    tiers = read_tiers(None)
    assert "file_name" in tiers
    t = tiers["file_name"]
    assert isinstance(t, MockTier)
    assert re.compile(t.values[0])  # regex compiles
    assert t.partition is False and t.blind is False


def test_read_tiers_with_all_invalid(monkeypatch):
    """If all tiers fail to parse, default_tiers() should still be returned."""
    config = {"bad": {"values": ["["]}}  # invalid regex
    tiers = read_tiers(config)
    assert isinstance(tiers, dict)
    assert any(isinstance(v, MockTier) for v in tiers.values())
