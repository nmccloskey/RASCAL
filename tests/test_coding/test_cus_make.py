from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rascal.coding.coding_files import make_cu_coding_files


class FakeTier:
    def __init__(self, name, mapping):
        self.name = name
        self.partition = False
        self._map = mapping
    def match(self, filename, return_None=False):
        if filename in self._map:
            return self._map[filename]
        return None if return_None else ""


def _mk_utt_df():
    """Minimal utterance DataFrame expected downstream."""
    return pd.DataFrame({
        "site": ["AC", "AC", "AC"],
        "test": ["Pre", "Pre", "Pre"],
        "sample_id": ["ACPreS0", "ACPreS0", "ACPreS1"],
        "speaker": ["PAR", "INV", "PAR"],
        "utterance": ["hi", "ok", "bye"],
        "comment": [None, None, None],
    })


@pytest.fixture
def io_tree(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    fname = "AC_Pre_transcript_tables.xlsx"
    fake_utt_path = input_dir / fname
    fake_utt_path.write_text("stub")

    # Patch read_excel to return our synthetic utterance DF
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: _mk_utt_df(), raising=False)
    import rascal.coding.coding_files as mod
    monkeypatch.setattr(mod, "extract_transcript_data", lambda path: _mk_utt_df())

    import rascal.coding.coding_files as mod
    monkeypatch.setattr(mod.random, "sample", lambda seq, k: list(seq)[:k])
    return input_dir, output_dir, fname


def _run_and_collect(io_tree, tiers, coders, cu_paradigms, exclude_participants, monkeypatch):
    input_dir, output_dir, fname = io_tree
    written = []

    def fake_to_excel(self, path, index=False):
        written.append((Path(path), self.copy()))
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    make_cu_coding_files(
        tiers=tiers,
        frac=0.5,
        coders=coders,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cu_paradigms=cu_paradigms,
        exclude_participants=exclude_participants,
    )

    return {p.as_posix(): df for p, df in written}


def test_make_cu_coding_files_zero_paradigms(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree
    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])
    coders = ["A", "B", "C"]
    cu_paradigms = []
    exclude_participants = ["INV"]

    out = _run_and_collect(io_tree, tiers, coders, cu_paradigms, exclude_participants, monkeypatch)

    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_coding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_reliability_coding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    for col in ["c1_id", "c2_id", "c1_sv", "c1_rel", "c1_comment", "c2_sv", "c2_rel", "c2_comment"]:
        assert col in cu_df.columns

    for col in ["c3_id", "c3_sv", "c3_rel", "c3_comment"]:
        assert col in rel_df.columns


def test_make_cu_coding_files_single_paradigm(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree
    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])
    coders = ["A", "B", "C"]
    cu_paradigms = ["SAE"]
    exclude_participants = ["INV"]

    out = _run_and_collect(io_tree, tiers, coders, cu_paradigms, exclude_participants, monkeypatch)

    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_coding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_reliability_coding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    for col in ["c1_id", "c2_id", "c1_sv", "c1_rel", "c1_comment", "c2_sv", "c2_rel", "c2_comment"]:
        assert col in cu_df.columns

    inv_row = cu_df[cu_df["speaker"] == "INV"].iloc[0]
    for c in ["c1_sv", "c1_rel", "c2_sv", "c2_rel"]:
        assert inv_row[c] == "NA"

    assert "c2_sv" not in rel_df.columns
    assert "c2_rel" not in rel_df.columns
    for col in ["c3_id", "c3_sv", "c3_rel", "c3_comment"]:
        assert col in rel_df.columns
    assert set(rel_df["c3_id"].unique()).issubset(set(coders))


def test_make_cu_coding_files_multi_paradigm(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree
    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])
    coders = ["A", "B", "C"]
    cu_paradigms = ["SAE", "AAE"]
    exclude_participants = ["INV"]

    out = _run_and_collect(io_tree, tiers, coders, cu_paradigms, exclude_participants, monkeypatch)

    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_coding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_cu_reliability_coding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    for base in ["c1_sv", "c1_rel", "c2_sv", "c2_rel"]:
        assert base not in cu_df.columns

    for paradigm in cu_paradigms:
        for prefix in ["c1", "c2"]:
            for tag in ["sv", "rel"]:
                assert f"{prefix}_{tag}_{paradigm}" in cu_df.columns

    inv_row = cu_df[cu_df["speaker"] == "INV"].iloc[0]
    for paradigm in cu_paradigms:
        for tag in ["sv", "rel"]:
            for prefix in ["c1", "c2"]:
                assert inv_row[f"{prefix}_{tag}_{paradigm}"] == "NA"

    for paradigm in cu_paradigms:
        for tag in ["sv", "rel"]:
            assert f"c3_{tag}_{paradigm}" in rel_df.columns
            assert f"c2_{tag}_{paradigm}" not in rel_df.columns

    assert "c3_id" in rel_df.columns
    assert set(rel_df["c3_id"].unique()).issubset(set(coders))
