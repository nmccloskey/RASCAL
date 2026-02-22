from __future__ import annotations

import random

import pandas as pd
import pytest

import rascal.coding.cu_summarization as cu


@pytest.fixture
def mock_tiers(tmp_path):
    """Create minimal mock tier objects."""
    class MockTier:
        def __init__(self, name, blind=True, partition=False, values=None):
            self.name = name
            self.blind = blind
            self.partition = partition
            # For TierManager-like behavior (blind codebook); keep tiny & explicit
            self.values = values or ["A", "B"]

        def match(self, _text: str):
            # Minimal behavior: just return tier name
            return self.name

    return {
        "site": MockTier("site", blind=True, partition=True, values=["A", "B"]),
        "group": MockTier("group", blind=False, partition=False, values=["A", "B"]),
    }


class MockTierManager:
    """
    Minimal TierManager test double for summarize_cus().

    Only implements what summarize_cus() (and helpers) are likely to call:
      - get_tier_names()
      - match_tiers()
      - tiers_in_group()
      - make_blind_codebook(seed=...)
    """

    def __init__(self, tiers: dict, tier_groups: dict | None = None):
        self.tiers = tiers
        self.tier_groups = tier_groups or {}
        self.order = list(tiers.keys())

    def get_tier_names(self):
        return list(self.order)

    def match_tiers(self, text: str, *, return_none: bool = False, must_match: bool = False):
        # Your MockTier.match only accepts (text); keep it simple.
        out = {}
        for name in self.order:
            tier = self.tiers[name]
            m = tier.match(text)
            out[name] = m if m is not None else (None if return_none else name)
        return out

    def tiers_in_group(self, group: str):
        return list(self.tier_groups.get(group, []))

    def make_blind_codebook(self, *, seed: int):
        """
        Deterministically generate {tier_name: {raw_value: int_code}} for tiers in 'blind' group.
        """
        blind_tiers = self.tiers_in_group("blind")
        if not blind_tiers:
            return {}

        rng = random.Random(seed)
        codebook = {}

        for tier_name in blind_tiers:
            tier = self.tiers.get(tier_name)
            values = getattr(tier, "values", None)
            if not values:
                continue

            codes = list(range(len(values)))
            rng.shuffle(codes)
            codebook[tier_name] = dict(zip(values, codes))

        return codebook


@pytest.fixture
def mock_TM(mock_tiers):
    """
    Default mock TierManager for summarize_cus tests.
    Include a 'blind' group if summarize_cus uses blinding logic.
    """
    tier_groups = {
        # If summarize_cus expects blinded tiers, include them here:
        "blind": ["site"],
        # If summarize_cus expects partition tiers, include if needed:
        "partition": ["site"],
    }
    return MockTierManager(mock_tiers, tier_groups=tier_groups)


@pytest.fixture
def minimal_datasets(tmp_path):
    """Create minimal Excel datasets expected by summarize_cus."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create fake transcript table
    utt_df = pd.DataFrame(
        {
            "sample_id": [1, 1],
            "utterance_id": [1, 2],
            "utterance": ["Hi", "Bye"],
            "speaker": ["PAR", "PAR"],
            "comment": ["", ""],
            "file": ["file1", "file1"],
            "site": ["A", "A"],
        }
    )
    transcript_path = input_dir / "TU_P01_transcript_tables.xlsx"
    with pd.ExcelWriter(transcript_path) as w:
        utt_df.to_excel(w, index=False)

    # Supporting CU, WC, and sample data
    cu_by_utt = pd.DataFrame(
        {
            "sample_id": [1, 1],
            "utterance_id": [1, 2],
            "c2_comment": ["", ""],
            "c1_CU": [1, 0],
        }
    )
    wc_by_utt = pd.DataFrame(
        {
            "sample_id": [1, 1],
            "utterance_id": [1, 2],
            "word_count": [2, 1],
        }
    )
    cu_by_sample = pd.DataFrame(
        {
            "sample_id": [1],
            "speaking_time": [60],
        }
    )

    for name, df in [
        ("cu_coding_by_utterance", cu_by_utt),
        ("word_counting", wc_by_utt),
        ("cu_coding_by_sample", cu_by_sample),
    ]:
        df.to_excel(input_dir / f"TU_P01_{name}.xlsx", index=False)

    return input_dir, output_dir, transcript_path


def test_summarize_cus_end_to_end(monkeypatch, mock_tiers, mock_TM, minimal_datasets):
    input_dir, output_dir, transcript_path = minimal_datasets

    cu_by_utt = input_dir / "TU_P01_cu_coding_by_utterance.xlsx"
    wc_by_utt = input_dir / "TU_P01_word_counting.xlsx"
    cu_by_sample = input_dir / "TU_P01_cu_coding_by_sample.xlsx"

    def fake_find_files(match_tiers=None, directories=None, search_base="", **_):
        if "cu_coding_by_utterance" in search_base:
            return [cu_by_utt]
        elif "word_counting" in search_base:
            return [wc_by_utt]
        elif "cu_coding_by_sample" in search_base:
            return [cu_by_sample]
        elif "transcript_tables" in search_base:
            return [transcript_path]
        return []

    monkeypatch.setattr(cu, "find_files", fake_find_files)
    monkeypatch.setattr(cu, "extract_transcript_data", lambda p: pd.read_excel(p))

    # UPDATED signature: summarize_cus(..., seed, TM)
    cu.summarize_cus(mock_tiers, input_dir, output_dir, seed=1, TM=mock_TM)

    # (Optional) sanity check: ensure something was written
    assert output_dir.exists()
