import os
import unittest
import pandas as pd
from pathlib import Path
from src.utils.read_tiers import read_tiers
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock, call
from src.utterances.make_CU_coding_files import make_CU_coding_files


class TestMakeCUCodingFiles(unittest.TestCase):

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_excel")
    @patch("pandas.read_excel")
    @patch("pathlib.Path.rglob")
    def test_make_CU_coding_files(self, mock_rglob, mock_read_excel, mock_to_excel, mock_makedirs):

        # Test tiers dictionary.
        base_dir = os.path.dirname(__file__)
        input_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step1_input')
        tiers = read_tiers(input_dir=input_dir)

        # Test output data.
        output_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step1_output', 'CUCoding')
        test_CUdfs = [pd.read_excel(str(file)) for file in Path(output_dir).rglob("*CUCoding.xlsx")]

        # Call the function.
        CUdfs = make_CU_coding_files(tiers, 0.2, ["AB", "CD", "EF"],"mock/utterance_dir", "mock/output_dir", test=True)
        
        # Ensure matching DataFrames by resetting the index.
        for df, test_df in zip(CUdfs, test_CUdfs):
            df = df.reset_index(drop=True)
            df.drop(columns=['c1ID','c2ID'], inplace=True)
            test_df = test_df.reset_index(drop=True)
            test_df.drop(columns=['c1ID','c2ID'], inplace=True)
            try:
                assert_frame_equal(df, test_df)
            except AssertionError as e:
                self.fail(f"DataFrames are not equal: {e}")

if __name__ == "__main__":
    unittest.main()
