import pytest
import re
import logging
from rascal.utils.tier import Tier


# --- Initialization tests ---

def test_user_regex_valid():
    tier = Tier("id", [r"[A-Z]{2}\d{2}"], partition=False, blind=True)
    assert tier.is_user_regex is True
    assert re.match(tier.pattern, "TU88")
    assert tier.match("BU77") == "BU77"


def test_user_regex_invalid():
    with pytest.raises(ValueError) as excinfo:
        Tier("bad", ["[A-"], partition=False, blind=False)
    assert "invalid regex" in str(excinfo.value)


def test_literal_values_regex():
    tier = Tier("site", ["AC", "BU"], partition=True, blind=False)
    assert tier.is_user_regex is False
    assert tier.match("AC") == "AC"
    assert tier.match("BU") == "BU"


def test_empty_values_generates_unmatchable_regex(caplog):
    with caplog.at_level(logging.WARNING):
        tier = Tier("empty", [], partition=False, blind=False)

    assert "(?!x)x" in tier.search_str  # matches nothing
    assert tier.match("anything", return_None=True) is None
    assert "empty values" in caplog.text


# --- _make_search_string tests ---

def test_make_search_string_escapes_special_chars():
    tier = Tier("specials", ["a.c", "d*e"], partition=False, blind=False)
    regex_str = tier.search_str
    assert r"a\.c" in regex_str
    assert r"d\*e" in regex_str


# --- match() tests ---

def test_match_success():
    tier = Tier("digits", [r"\d+"], partition=False, blind=False)
    assert tier.match("abc123") == "123"


def test_match_no_result_return_none(caplog):
    tier = Tier("letters", [r"[A-Z]+"], partition=False, blind=False)
    with caplog.at_level(logging.WARNING):
        result = tier.match("123", return_None=True)
    assert result is None
    assert "No match" in caplog.text


def test_match_no_result_return_name(caplog):
    tier = Tier("letters", [r"[A-Z]+"], partition=False, blind=False)
    with caplog.at_level(logging.ERROR):
        result = tier.match("123", return_None=False)
    assert result == "letters"
    assert "Returning tier name" in caplog.text


# --- make_blind_codes tests ---

def test_make_blind_codes_nonempty():
    tier = Tier("colors", ["red", "blue", "green"], partition=False, blind=True)
    codes = tier.make_blind_codes()
    mapping = codes["colors"]

    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {"red", "blue", "green"}
    # Values should be unique ints
    assert len(set(mapping.values())) == 3


def test_make_blind_codes_empty(caplog):
    tier = Tier("empty", [], partition=False, blind=True)
    with caplog.at_level(logging.WARNING):
        codes = tier.make_blind_codes()
    assert codes == {"empty": {}}
    assert "blind code mapping will be empty" in caplog.text
