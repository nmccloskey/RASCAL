from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytest

# Import target
try:
    from rascal.utterances import make_coding_files as mcf
except Exception as e:
    pytest.skip(f"Could not import rascal.utterances.make_coding_files: {e}", allow_module_level=True)


class MockTier:
    """Minimal tier: returns constant label from filename split."""
    def __init__(self, idx=0, partition=True):
        self.idx = idx
        self.partition = partition
    def match(self, filename, return_None=False):
        base = Path(filename).stem
        parts = base.split("_")
        return parts[self.idx] if len(parts) > self.idx else None


def _write_cu_file(tmp_path):
    """Create a dummy CU coding utterance-level Excel file."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cu = input_dir / "TU_P01_CUCoding_ByUtterance.xlsx"
    cu.write_bytes(b"stub")  # content stub, replaced by monkeypatch
    return input_dir, cu


def test_make_word_count_files_basic(tmp_path, monkeypatch):
    input_dir, cu_file = _write_cu_file(tmp_path)
    output_dir = tmp_path / "out"

    # Fake CU coding DataFrame
    df_cu = pd.DataFrame({
        "utterance_id": ["U1","U2","U3"],
        "sample_id":    ["S1","S2","S3"],
        "utterance":   ["hello world", "foo bar", "baz"],
        "c2CU":        [1, 1, np.nan],  # one missing -> should produce "NA"
        "c2SV":        [1, 1, 0],
        "c2REL":       [1, 1, 0],
    })

    # Capture outputs
    captured = {}
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        if p.endswith("_WordCounting.xlsx"):
            captured["coding"] = self.copy()
        if p.endswith("_WordCountingReliability.xlsx"):
            captured["reliability"] = self.copy()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"stub")

    # Monkeypatch dependencies
    monkeypatch.setattr(pd, "read_excel", lambda path, *a, **k: df_cu.copy())
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    # count_words should return 5 regardless of input
    monkeypatch.setattr(mcf, "count_words", lambda text, d: 5)
    # assign_CU_coders returns list of assignments [(c1,c2), ...]
    monkeypatch.setattr(mcf, "assign_CU_coders", lambda coders: [(coders[0], coders[1])])
    # segment splits sample_ids evenly across coders
    monkeypatch.setattr(mcf, "segment", lambda ids, n: [ids])

    # Dummy d(word) always True
    monkeypatch.setattr(mcf, "d", lambda word: True)

    tiers = {"site": MockTier(idx=0, partition=True)}
    coders = ["1","2"]

    mcf.make_word_count_files(
        tiers=tiers,
        frac=0.5,
        coders=coders,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    # Check outputs captured
    assert "coding" in captured, "Coding file not written"
    assert "reliability" in captured, "Reliability file not written"

    coding_df = captured["coding"]
    reliability_df = captured["reliability"]

    # Coding file should have wordCount col with ints/NA
    assert "wordCount" in coding_df.columns
    assert set(coding_df["wordCount"].unique()) >= {5, "NA"}

    # Reliability file should have c2ID and WCrelCom
    assert "c2ID" in reliability_df.columns
    assert "WCrelCom" in reliability_df.columns
    # Reliability should only contain a subset of samples (frac=0.5 -> at least 1)
    assert len(set(reliability_df["sample_id"])) >= 1
