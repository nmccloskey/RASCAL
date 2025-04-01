import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from src.utils.read_cha_files import read_cha_files

class TestReadChaFiles(unittest.TestCase):

    @patch('pylangacq.read_chat')
    @patch('pathlib.Path.rglob')
    def test_read_cha_files(self, mock_rglob, mock_read_chat):
        # Mock the rglob method to return a list of mocked paths
        mock_file_1 = MagicMock(spec=Path)
        mock_file_1.name = 'test1.cha'
        mock_file_1.__str__.return_value = 'test1.cha'

        mock_file_2 = MagicMock(spec=Path)
        mock_file_2.name = 'test2.cha'
        mock_file_2.__str__.return_value = 'test2.cha'

        mock_rglob.return_value = [mock_file_1, mock_file_2]

        # Mock the read_chat method to return a mock chat object
        mock_chat_data = MagicMock()
        mock_read_chat.return_value = mock_chat_data

        # Call the function
        result = read_cha_files('mock/input_dir')

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn('test1.cha', result)
        self.assertIn('test2.cha', result)
        self.assertEqual(result['test1.cha'], mock_chat_data)
        self.assertEqual(result['test2.cha'], mock_chat_data)

        # Ensure rglob was called on the correct path
        mock_rglob.assert_called_once_with('*.cha')
        # Ensure read_chat was called twice
        self.assertEqual(mock_read_chat.call_count, 2)

if __name__ == "__main__":
    unittest.main()
