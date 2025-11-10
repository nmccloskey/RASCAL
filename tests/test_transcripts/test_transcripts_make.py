import pandas as pd
from pathlib import Path
from rascal.transcripts.transcript_tables import make_transcript_tables


class MockTier:
    """Minimal mock tier with .match() and optional partition flag."""
    def __init__(self, label, partition=True):
        self.label = label
        self.partition = partition

    def match(self, fname):
        return self.label if self.label in fname else None


class MockUtterance:
    """Fake utterance object mimicking pylangacq's structure."""
    def __init__(self, speaker, text, comment=None):
        self.participant = speaker
        self.tiers = {speaker: text}
        if comment:
            self.tiers["%com"] = comment


class MockChatReader:
    """Fake CHAT reader with utterances()."""
    def __init__(self, utterances):
        self._utterances = utterances

    def utterances(self):
        return self._utterances


def test_make_transcript_tables(tmp_path):
    """End-to-end test of make_transcript_tables with mock CHAT and tiers."""
    # --- Setup directories and mock tiers ---
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    tiers = {"site": MockTier("SiteA"), "group": MockTier("G1")}

    # --- Create mock CHAT input ---
    chat1 = MockChatReader([
        MockUtterance("PAR", "hello world", "greeting"),
        MockUtterance("PAR", "goodbye", "farewell"),
    ])
    chat2 = MockChatReader([
        MockUtterance("PAR", "testing one two", "check"),
    ])

    chats = {
        "SiteA_G1_P01.cha": chat1,
        "SiteA_G1_P02.cha": chat2,
    }

    # --- Run function ---
    make_transcript_tables(tiers, chats, output_dir)

    # --- Verify transcript_tables directory ---
    # The code splits 'SiteA_G1' into ['SiteA', 'G1']
    transcript_dir = output_dir / "transcript_tables" / "SiteA" / "G1"
    assert transcript_dir.exists(), f"Transcript directory not created at {transcript_dir}"

    files = list(transcript_dir.glob("*transcript_tables.xlsx"))
    assert len(files) == 1, f"Expected one transcript table, found {len(files)}"
    excel_path = files[0]

    # --- Validate sheets and structure ---
    xls = pd.ExcelFile(excel_path)
    assert set(xls.sheet_names) == {"samples", "utterances"}

    samples_df = pd.read_excel(xls, sheet_name="samples")
    utterances_df = pd.read_excel(xls, sheet_name="utterances")

    # --- Check sample-level columns ---
    expected_sample_cols = {"sample_id", "file", "site", "group", "speaking_time"}
    assert expected_sample_cols.issubset(samples_df.columns)

    # --- Check utterance-level columns ---
    expected_utt_cols = {"sample_id", "utterance_id", "speaker", "utterance", "comment"}
    assert expected_utt_cols.issubset(utterances_df.columns)

    # --- Validate data consistency ---
    assert set(utterances_df["sample_id"]).issubset(set(samples_df["sample_id"]))
    assert utterances_df["utterance"].notna().all()
    assert any(utterances_df["comment"].notna()), "Comments not populated"

    # Check zero-padded IDs
    assert all(samples_df["sample_id"].str.match(r"^S\d{3}$"))
    assert all(utterances_df["utterance_id"].str.match(r"^U\d{4}$"))
