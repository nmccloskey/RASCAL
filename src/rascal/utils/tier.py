import re
import random
import logging

class Tier:
    def __init__(self, name, values, partition):
        """
        Initializes a Tier object.

        Parameters:
        - name (str): The name of the tier.
        - values (list): A list of values used for pattern matching.
        - partition (bool): Whether this tier is used for partitioning.
        """
        self.name = name
        self.values = values
        self.search_str = self._make_search_string(values)
        self.pattern = re.compile(self.search_str)
        self.partition = partition
        logging.info(f"Initialized Tier: {name} with values: {values} and partition: {partition}")

    def _make_search_string(self, values):
        """
        Generates a search string from the provided values.

        Parameters:
        - values (list): A list of values to be used in the search string.

        Returns:
        - str: A regex search string combining the values.
        """
        search_str = '|'.join(values)
        logging.debug(f"Generated search string: {search_str}")
        return f"({search_str})"

    def match(self, text, return_None=False):
        """
        Applies the compiled regex pattern to a given text.

        Parameters:
        - text (str): The text to search against the compiled regex pattern.
        - return_None (bool): If True, return None explicitly if no match is found.

        Returns:
        - str: The matched value if found.
        - None: If no match is found and return_None is True.
        - str: Returns the tier name if no match is found and return_None is False.
        """
        match = self.pattern.search(text)
        if match:
            # logging.info(f"Match found for '{self.name}' in text: {text}")
            return match.group(0)
        elif return_None:
            logging.warning(f"No match found for '{self.name}' in text: {text}, returning None.")
            return None
        else:
            logging.error(f"No match found for '{self.name}' in text: {text}. Returning tier name.")
            return self.name

    def make_blind_codes(self):
        """
        Generates a blinded coding system for the tier values.

        This method assigns a random integer to each value in the tier and returns
        a dictionary mapping the original values to their blinded codes.

        Returns:
        - dict: A dictionary with the tier name as the key and another dictionary as the value.
                The inner dictionary maps each original value to a random blind code.
        """
        logging.info(f"Generating blind codes for tier: {self.name}")
        blind_codes = list(range(len(self.values)))
        random.shuffle(blind_codes)
        blind_code_mapping = {k: v for k, v in zip(self.values, blind_codes)}
        logging.debug(f"Blind code mapping for '{self.name}': {blind_code_mapping}")
        return {self.name: blind_code_mapping}
