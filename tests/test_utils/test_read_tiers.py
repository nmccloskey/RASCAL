import unittest
from unittest.mock import patch, mock_open
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from src.utils.read_tiers import read_tiers

class TestReadTiers(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="*Tier1:apple,banana,cherry\nTier2:Tier1##\n*Tier3:pear,peach")
    @patch("os.path.join", return_value="tiers.txt")  # Mock path to avoid dependency on file structure
    def test_read_tiers(self, mock_join, mock_open):
        """Test the read_tiers function with mock file data."""
        
        # Execute the function
        tiers = read_tiers(input_dir='input')
        
        # Check that the function returns the correct number of tiers
        self.assertEqual(len(tiers), 3)

        # Verify each Tier object and its properties
        tier1 = tiers["Tier1"]
        self.assertEqual(tier1.name, "Tier1")
        self.assertEqual(tier1.values, ["apple", "banana", "cherry"])
        self.assertTrue(tier1.partition)

        tier2 = tiers["Tier2"]
        self.assertEqual(tier2.name, "Tier2")
        self.assertEqual(tier2.values, ["(apple|banana|cherry)\\d+"])  # Numerical placeholder check
        self.assertFalse(tier2.partition)

        tier3 = tiers["Tier3"]
        self.assertEqual(tier3.name, "Tier3")
        self.assertEqual(tier3.values, ["pear", "peach"])
        self.assertTrue(tier3.partition)

# Run the tests
if __name__ == "__main__":
    unittest.main()
