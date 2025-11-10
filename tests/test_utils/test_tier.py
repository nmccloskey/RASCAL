import re
import pytest
from rascal.utils.tier import Tier


def test_tier_single_regex_valid():
    """A single regex string should be treated as a user regex and compile successfully."""
    t = Tier("site", [r"Site\d+"], partition=True, blind=False)
    assert t.is_user_regex
    assert isinstance(t.pattern, re.Pattern)
    assert t.pattern.search("Site12")
    assert t.match("Site12") == "Site12"
    assert t.match("NoMatch", return_None=True) is None
    # default fallback behavior
    assert t.match("NoMatch") == "site"


def test_tier_single_regex_invalid():
    """Invalid regex should raise ValueError."""
    with pytest.raises(ValueError):
        Tier("bad", ["["], partition=False, blind=False)


def test_tier_multiple_literals():
    """Multiple literal values should be escaped and combined correctly."""
    t = Tier("group", ["G1", "G2", "G(3)"], partition=False, blind=True)
    assert not t.is_user_regex
    assert "(?:" in t.search_str
    # Ensure escaped parentheses from literal values
    assert r"\(" in t.search_str
    assert t.pattern.search("G2")
    assert t.match("XG3X", return_None=True) is None


def test_tier_empty_values(monkeypatch, caplog):
    """Empty value list should log a warning and never match."""
    t = Tier("empty", [], partition=False, blind=False)
    assert t._make_search_string([]) == r"(?!x)x"  # matches nothing
    assert not t.pattern.search("anything")
    assert t.match("anything", return_None=True) is None
    assert any("empty values" in rec.message for rec in caplog.records)


def test_make_blind_codes_with_values(monkeypatch):
    """Blinding should return a mapping with shuffled codes."""
    t = Tier("speaker", ["A", "B", "C"], partition=False, blind=True)
    mapping = t.make_blind_codes()
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {"speaker"}
    inner_map = mapping["speaker"]
    assert set(inner_map.keys()) == {"A", "B", "C"}
    # all codes unique
    assert len(set(inner_map.values())) == 3


def test_make_blind_codes_empty(monkeypatch):
    """Blinding with empty values should return empty mapping."""
    t = Tier("empty", [], partition=False, blind=True)
    mapping = t.make_blind_codes()
    assert mapping == {"empty": {}}


def test_match_with_must_match(caplog):
    """must_match=True should log warnings or errors for non-matches."""
    t = Tier("site", [r"Site\d+"], partition=False, blind=False)
    _ = t.match("Site1", must_match=True)
    _ = t.match("Nope", must_match=True)
    assert any("No match" in rec.message for rec in caplog.records)
