import os
import unittest
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import pandas as pd
from pathlib import Path
from src.utils.read_tiers import read_tiers
from src.samples.unblind_CUs import unblind_CUs
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock, call
from src.utils.read_cha_files import read_cha_files
from src.utterances.make_utterance_tables import prepare_utterance_dfs


class TestPrepareUtteranceDfs(unittest.TestCase):

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_excel")
    def test_unblind_CUs(self, mock_to_excel, mock_makedirs):

        # Test tiers dictionary.
        base_dir = os.path.dirname(__file__)
        input_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step5_input')
        tiers = read_tiers(input_dir=input_dir)

        # Test output data.
        output_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step5_output', 'Summaries')
        test_dfs = [pd.read_excel(file) for file in Path(output_dir).rglob("*Data.xlsx")]

        # Call the function with test=True to yield data.
        dfs = unblind_CUs(tiers, input_dir, "/mock/output_dir", test=True)

        # Ensure matching DataFrames by resetting the index.
        for df, test_df in zip(dfs, test_dfs):
            df = df.reset_index(drop=True)
            df.drop(columns=['site','test'], inplace=True)
            test_df = test_df.reset_index(drop=True)
            test_df.drop(columns=['site','test'], inplace=True)
            test_df['wordCount'] = test_df['wordCount'].astype(float)
            try:
                assert_frame_equal(df, test_df)
            except AssertionError as e:
                self.fail(f"DataFrames are not equal: {e}")

        # Assert makedirs was called to create Utterances directory
        mock_makedirs.assert_called_with(os.path.join("/mock/output_dir", "Summaries"), exist_ok=True)

        # Assert to_excel was called to save the DataFrame
        self.assertTrue(mock_to_excel.called)
        self.assertGreaterEqual(mock_to_excel.call_count, 1)

if __name__ == "__main__":
    unittest.main()
