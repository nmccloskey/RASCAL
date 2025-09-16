import os
import types
import pandas as pd
import pytest
from pathlib import Path

from rascal.transcription.transcription_reliability_selector import select_transcription_reliability_samples


# ---------- Fakes ----------

class FakeTier:
    def __init__(self, name, partition, mapping):
        self.name = name
        self.partition = partition
        self._map = mapping  # filename -> label
    def match(self, filename: str):
        return self._map.get(filename, None)

class FakeChat:
    def __init__(self, header_lines):
        self._header_lines = header_lines
    def to_strs(self):
        # Simulate pylangacq Reader.to_strs() → iterator of transcript strings
        yield "\n".join(self._header_lines + ["*PAR:\thello", "%com:\tnote"])


@pytest.fixture
def sample_data(tmp_path, monkeypatch):
    f1 = str(tmp_path / "AC_P001.cha")
    f2 = str(tmp_path / "BU_P002.cha")
    chats = {
        f1: FakeChat(["@Begin", "@Participants: PAR Participant"]),
        f2: FakeChat(["@Begin", "@Participants: PAR Participant"]),
    }

    tiers = {
        "site": FakeTier("site", True, {f1: "AC", f2: "BU"}),
    }

    # Force random.sample to pick a deterministic subset (always first element)
    monkeypatch.setattr("rascal.transcription.transcription_reliability_selector.random.sample", lambda seq, k: seq[:k])

    return f1, f2, chats, tiers, tmp_path


def test_select_transcription_reliability_samples_creates_files(sample_data):
    f1, f2, chats, tiers, tmp_path = sample_data

    outdir = tmp_path / "out"
    select_transcription_reliability_samples(tiers, chats, frac=0.5, output_dir=str(outdir))

    # Should have created partition folders under TranscriptionReliability
    rel_dir = outdir / "TranscriptionReliability"
    assert rel_dir.exists()

    # Each partition (AC, BU) should get a subfolder
    ac_dir = rel_dir / "AC"
    bu_dir = rel_dir / "BU"
    assert ac_dir.exists()
    assert bu_dir.exists()

    # Should contain a Reliability .cha file with only header lines
    ac_files = list(ac_dir.glob("*Reliability.cha"))
    bu_files = list(bu_dir.glob("*Reliability.cha"))
    assert len(ac_files) == 1
    assert len(bu_files) == 1

    # Verify headers were written
    content = ac_files[0].read_text().splitlines()
    assert content[0].startswith("@Begin")
    assert content[-1].startswith("@End")

    # Excel files should also be written
    ac_xlsx = ac_dir / "AC_TranscriptionReliabilitySamples.xlsx"
    bu_xlsx = bu_dir / "BU_TranscriptionReliabilitySamples.xlsx"
    assert ac_xlsx.exists()
    assert bu_xlsx.exists()

    # DataFrame content should have columns = file + tier(s)
    df = pd.read_excel(ac_xlsx)
    assert "file" in df.columns
    assert "site" in df.columns
    assert df["site"].iloc[0] == "AC"
