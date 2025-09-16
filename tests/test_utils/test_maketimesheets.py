import os
import unittest
import pandas as pd
from pathlib import Path
from src.utils.read_tiers import read_tiers
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock, call
from src.utils.make_timesheets import make_timesheets

class TestMakeTimesheets(unittest.TestCase):

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_excel")
    @patch("pandas.read_excel")
    @patch("pathlib.Path.rglob")
    def test_make_timesheets(self, mock_rglob, mock_read_excel, mock_to_excel, mock_makedirs):

        # Read tiers.
        base_dir = os.path.dirname(__file__)
        input_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step3_input')
        tiers = read_tiers(input_dir=input_dir)

        # Test output.
        output_dir = os.path.join(base_dir, '..', '..', 'test_data', 'test_step3_output')
        test_dfs = [pd.read_excel(str(file)) for file in Path(output_dir).rglob("*SpeakingTimes.xlsx")]

        # Call the function.
        dfs = make_timesheets(tiers=tiers, input_dir=input_dir, output_dir="mock/transc_rel_dir", test=True)

        # Ensure matching DataFrames by resetting the index.
        for df, test_df in zip(dfs, test_dfs):
            df = df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            try:
                assert_frame_equal(df, test_df)
            except AssertionError as e:
                self.fail(f"DataFrames are not equal: {e}")

if __name__ == "__main__":
    unittest.main()
