import unittest
from unittest.mock import patch
import re
from src.utils.tier import Tier


# Assuming Tier class is defined as given
class TestTier(unittest.TestCase):
    
    def setUp(self):
        # Initialize a Tier instance with example values
        self.tier = Tier(name="ExampleTier", values=["apple", "banana", "cherry"], partition=True)
        
    def test_initialization(self):
        """Test the initialization of Tier instance."""
        self.assertEqual(self.tier.name, "ExampleTier")
        self.assertEqual(self.tier.values, ["apple", "banana", "cherry"])
        self.assertTrue(self.tier.partition)
        self.assertIsInstance(self.tier.pattern, re.Pattern)
        
    def test_make_search_string(self):
        """Test the search string creation from values."""
        expected_search_string = "(apple|banana|cherry)"
        self.assertEqual(self.tier._make_search_string(self.tier.values), expected_search_string)
    
    def test_match_found(self):
        """Test matching a value in the text."""
        text = "I have an apple and a banana."
        match = self.tier.match(text)
        self.assertEqual(match, "apple")  # The first match should be returned
    
    def test_match_not_found_return_name(self):
        """
        Test that the match function returns the tier name when no match is found and return_None is False.
        """
        text = "I have a pear and a mango."
        result = self.tier.match(text)
        self.assertEqual(result, "ExampleTier")

    def test_match_not_found_return_none(self):
        """
        Test that the match function returns None when no match is found and return_None is True.
        """
        text = "I have a pear and a mango."
        result = self.tier.match(text, return_None=True)
        self.assertIsNone(result)
        
# Run the tests
if __name__ == "__main__":
    unittest.main()
