import pandas as pd
from collections import OrderedDict
from unittest.mock import MagicMock
import pytest
from pathlib import Path

from rascal.utterances.make_utterance_tables import (
    _build_utterance_df,
    _write_utterance_tables,
    prepare_utterance_dfs,
)

def norm(p): return Path(p).as_posix()

# ---------- Fakes (no pylangacq dependency required) ----------

class FakeTier:
    def __init__(self, name, partition, mapping):
        self.name = name
        self.partition = partition
        self._map = mapping  # filename -> label str
    def match(self, filename: str) -> str:
        return self._map.get(filename, "")

class FakeLine:
    def __init__(self, participant, tiers):
        self.participant = participant
        self.tiers = tiers

class FakeChat:
    def __init__(self, lines):
        self._lines = lines
    def utterances(self):
        return iter(self._lines)

@pytest.fixture
def sample_fixtures():
    f1 = "AC_Pre_P001.cha"
    f2 = "BU_Post_P002.cha"
    chats = {
        f1: FakeChat([
            FakeLine("PAR", {"PAR": "hello world", "%com": "c1"}),
            FakeLine("INV", {"INV": "how are you", "%com": "c2"}),
        ]),
        f2: FakeChat([
            FakeLine("PAR", {"PAR": "goodbye", "%com": None}),
        ]),
    }
    return f1, f2, chats

def test_build_df_no_partition(sample_fixtures):
    f1, f2, chats = sample_fixtures
    tiers = OrderedDict([
        ("site", FakeTier("site", False, {f1: "AC", f2: "BU"})),
        ("test", FakeTier("test", False, {f1: "Pre", f2: "Post"})),
    ])
    df, partition_tiers = _build_utterance_df(tiers, chats)
    assert partition_tiers == []
    assert list(df.columns) == ["file", "site", "test", "sample_id", "speaker", "utterance", "comment"]
    assert len(df) == 3
    # Deterministic sample_id by sorted filenames -> f1 is index 0
    assert (df.loc[df["file"] == f1, "sample_id"].unique() == ["S0"]).all()
    assert (df.loc[df["file"] == f2, "sample_id"].unique() == ["S1"]).all()

def test_build_df_with_single_partition(sample_fixtures):
    f1, f2, chats = sample_fixtures
    tiers = OrderedDict([
        ("site", FakeTier("site", True,  {f1: "AC", f2: "BU"})),
        ("test", FakeTier("test", False, {f1: "Pre", f2: "Post"})),
    ])
    df, partition_tiers = _build_utterance_df(tiers, chats)
    assert partition_tiers == ["site"]
    # Ensure labels are there
    assert set(df["site"].unique()) == {"AC", "BU"}

def test_write_tables_no_partition(monkeypatch, tmp_path, sample_fixtures):
    f1, f2, chats = sample_fixtures
    tiers = OrderedDict([
        ("site", FakeTier("site", False, {f1: "AC", f2: "BU"})),
        ("test", FakeTier("test", False, {f1: "Pre", f2: "Post"})),
    ])
    df, pts = _build_utterance_df(tiers, chats)

    # Stub to_excel to avoid openpyxl dependency
    calls = []
    def fake_to_excel(self, path, index=False):
        calls.append(path)
        # write a small CSV next to it for sanity, optional
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    outdir = tmp_path / "out"
    written = _write_utterance_tables(df, str(outdir), pts)
    assert len(written) == 1
    assert written[0].endswith("Utterances.xlsx")

def test_write_tables_multi_partition(monkeypatch, tmp_path, sample_fixtures):
    f1, f2, chats = sample_fixtures
    tiers = OrderedDict([
        ("site", FakeTier("site", True, {f1: "AC", f2: "BU"})),
        ("test", FakeTier("test", True, {f1: "Pre", f2: "Post"})),
    ])
    df, pts = _build_utterance_df(tiers, chats)

    calls = []
    def fake_to_excel(self, path, index=False):
        calls.append(path)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    outdir = tmp_path / "out"
    written = _write_utterance_tables(df, str(outdir), pts)
    assert len(written) == 2
    assert any(norm(p).endswith("AC/Pre/AC_Pre_Utterances.xlsx") for p in written)
    assert any(norm(p).endswith("BU/Post/BU_Post_Utterances.xlsx") for p in written)

def test_prepare_utterance_dfs_end_to_end(monkeypatch, tmp_path, sample_fixtures):
    f1, f2, chats = sample_fixtures
    tiers = OrderedDict([
        ("site", FakeTier("site", True, {f1: "AC", f2: "BU"})),
        ("test", FakeTier("test", False, {f1: "Pre", f2: "Post"})),
    ])
    # Patch to_excel
    calls = []
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda self, p, index=False: calls.append(p), raising=False)

    written = prepare_utterance_dfs(tiers, chats, str(tmp_path / "out"))
    assert len(written) == 2
    assert any(p.endswith("AC_Utterances.xlsx") for p in written)
    assert any(p.endswith("BU_Utterances.xlsx") for p in written)
