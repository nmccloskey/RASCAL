from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rascal.utterances.make_coding_files import make_CU_coding_files


class FakeTier:
    def __init__(self, name, mapping):
        self.name = name
        self.partition = False  # not used by this function
        self._map = mapping     # filename -> label
    def match(self, filename, return_None=False):
        if filename in self._map:
            return self._map[filename]
        return None if return_None else ""


def _mk_utt_df():
    # Minimal columns this function expects downstream
    # Two sampleIDs, three rows (PAR + INV in S0, PAR in S1)
    return pd.DataFrame({
        "site": ["AC", "AC", "AC"],
        "test": ["Pre", "Pre", "Pre"],
        "sampleID": ["ACPreS0", "ACPreS0", "ACPreS1"],
        "speaker": ["PAR", "INV", "PAR"],
        "utterance": ["hi", "ok", "bye"],
        "comment": [None, None, None],
    })


@pytest.fixture
def io_tree(tmp_path, monkeypatch):
    # Create a fake utterances file under input_dir that Path(...).rglob will find
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    fname = "AC_Pre_Utterances.xlsx"
    fake_utt_path = input_dir / fname
    fake_utt_path.write_text("not really xlsx")  # we monkeypatch read_excel anyway

    # Patch read_excel to return our synthetic utterance DF
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: _mk_utt_df(), raising=False)

    # Make random.sample deterministic: always pick first k
    import rascal.utterances.make_coding_files as mod
    monkeypatch.setattr(mod.random, "sample", lambda seq, k: list(seq)[:k])

    return input_dir, output_dir, fname


def _run_and_collect(io_tree, tiers, coders, CU_paradigms, exclude_participants, monkeypatch):
    input_dir, output_dir, fname = io_tree

    # Intercept DataFrame.to_excel to collect the frames written
    written = []
    def fake_to_excel(self, path, index=False):
        # store a copy to avoid accidental mutation
        written.append((Path(path), self.copy()))
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    make_CU_coding_files(
        tiers=tiers,
        frac=0.5,
        coders=coders,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        CU_paradigms=CU_paradigms,
        exclude_participants=exclude_participants,
    )
    # Return a dict: {"..._CUCoding.xlsx": df, "..._CUReliabilityCoding.xlsx": df}
    out = {}
    for p, df in written:
        out[p.as_posix()] = df
    return out


def test_make_CU_coding_files_zero_paradigms(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree

    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])

    coders = ["A", "B", "C"]
    CU_paradigms = []          # zero paradigms → base columns kept
    exclude_participants = ["INV"]

    out = _run_and_collect(io_tree, tiers, coders, CU_paradigms, exclude_participants, monkeypatch)

    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUCoding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUReliabilityCoding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    # CU file keeps base columns when len(CU_paradigms) == 0
    for col in ["c1ID", "c2ID", "c1SV", "c1REL", "c1com", "c2SV", "c2REL", "c2com"]:
        assert col in cu_df.columns

    # Reliability file must include third-coder columns
    for col in ["c3ID", "c3SV", "c3REL", "c3com"]:
        assert col in rel_df.columns


def test_make_CU_coding_files_single_paradigm(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree

    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])

    coders = ["A", "B", "C"]
    CU_paradigms = ["SAE"]          # single paradigm → keeps base SV/REL columns
    exclude_participants = ["INV"]  # excluded speaker

    out = _run_and_collect(io_tree, tiers, coders, CU_paradigms, exclude_participants, monkeypatch)

    # Find CU and Reliability outputs by filename suffix
    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUCoding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUReliabilityCoding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    # --- CU coding file checks ---
    # Should include base coding columns since len(CU_paradigms) == 1
    for col in ["c1ID", "c2ID", "c1SV", "c1REL", "c1com", "c2SV", "c2REL", "c2com"]:
        assert col in cu_df.columns

    # Excluded participants (INV) get "NA" in coding-value fields
    inv_row = cu_df[cu_df["speaker"] == "INV"].iloc[0]
    assert inv_row["c1SV"] == "NA"
    assert inv_row["c1REL"] == "NA"
    assert inv_row["c2SV"] == "NA"
    assert inv_row["c2REL"] == "NA"

    # --- Reliability coding file checks ---
    # c2* columns should be dropped/renamed to c3*
    assert "c2SV" not in rel_df.columns
    assert "c2REL" not in rel_df.columns
    for col in ["c3ID", "c3SV", "c3REL", "c3com"]:
        assert col in rel_df.columns

    # c3ID should be one of the coders
    assert set(rel_df["c3ID"].unique()).issubset(set(coders))


def test_make_CU_coding_files_multi_paradigm(io_tree, monkeypatch):
    input_dir, output_dir, fname = io_tree

    tiers = OrderedDict([
        ("site", FakeTier("site", {fname: "AC"})),
        ("test", FakeTier("test", {fname: "Pre"})),
    ])

    coders = ["A", "B", "C"]
    CU_paradigms = ["SAE", "AAE"]   # multi-paradigm → suffixed coding columns
    exclude_participants = ["INV"]

    out = _run_and_collect(io_tree, tiers, coders, CU_paradigms, exclude_participants, monkeypatch)

    cu_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUCoding.xlsx")]
    rel_paths = [p for p in out if p.endswith("AC/Pre/AC_Pre_CUReliabilityCoding.xlsx")]
    assert len(cu_paths) == 1 and len(rel_paths) == 1

    cu_df = out[cu_paths[0]]
    rel_df = out[rel_paths[0]]

    # Base columns removed; suffixed columns present
    for base in ["c1SV", "c1REL", "c2SV", "c2REL"]:
        assert base not in cu_df.columns

    for paradigm in CU_paradigms:
        for prefix in ["c1", "c2"]:
            for tag in ["SV", "REL"]:
                assert f"{prefix}{tag}_{paradigm}" in cu_df.columns

    # Excluded participants still get "NA" in the suffixed coding-value fields
    inv_row = cu_df[cu_df["speaker"] == "INV"].iloc[0]
    for paradigm in CU_paradigms:
        assert inv_row[f"c1SV_{paradigm}"] == "NA"
        assert inv_row[f"c2REL_{paradigm}"] == "NA"

    # Reliability file should have c3* suffixed columns present and c2* removed
    for paradigm in CU_paradigms:
        for tag in ["SV", "REL"]:
            assert f"c3{tag}_{paradigm}" in rel_df.columns
            assert f"c2{tag}_{paradigm}" not in rel_df.columns

    # c3ID assigned to one of the coders
    assert "c3ID" in rel_df.columns
    assert set(rel_df["c3ID"].unique()).issubset(set(coders))
