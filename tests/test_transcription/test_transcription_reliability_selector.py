import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.transcription.transcription_reliability_selector import select_transcription_reliability_samples


class TestSelectTranscriptionReliabilitySamples(unittest.TestCase):
    
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pandas.DataFrame.to_excel")
    @patch("tqdm.tqdm", side_effect=lambda x, *args, **kwargs: x)  # Mock tqdm to pass-through
    def test_select_transcription_reliability_samples(self, mock_tqdm, mock_to_excel, mock_open, mock_makedirs):
        # Mock tiers dictionary.
        tiers = {
            "Tier1": MagicMock(partition=True, match=MagicMock(return_value="A")),
            "Tier2": MagicMock(partition=True, match=MagicMock(return_value="B"))
        }

        # Mock chats dictionary.
        chats = {
            "chat1.cha": MagicMock(to_strs=MagicMock(return_value=["@Begin\nHeader\n@Participants:\n@End"])),
            "chat2.cha": MagicMock(to_strs=MagicMock(return_value=["@Begin\nHeader2\n@Participants:\n@End"]))
        }

        # Call the function.
        select_transcription_reliability_samples(tiers, chats, 0.5, "mock/output_dir")

        # Check that makedirs was called to create output directories.
        self.assertTrue(mock_makedirs.called)

        # Check that to_excel was called to save the DataFrame.
        self.assertTrue(mock_to_excel.called)
        self.assertEqual(mock_to_excel.call_count, 1)

if __name__ == "__main__":
    unittest.main()
