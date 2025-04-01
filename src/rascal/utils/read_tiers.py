import os
import re
import logging
from utils.tier import Tier

def read_tiers(input_dir):
    """
    Reads tier information from the tiers.txt file in the specified directory.

    Parameters:
    - input_dir (str): The directory containing the tiers.txt file.

    Returns:
    - dict: A dictionary of Tier objects with tier names as keys.
    """
    
    tiers_path = os.path.join(input_dir, 'tiers.txt')
    logging.info(f"Reading tiers from: {tiers_path}")
    
    try:
        with open(tiers_path) as file:
            tier_lines = file.readlines()
    except FileNotFoundError as e:
        logging.error(f"tiers.txt file not found in directory {input_dir}: {e}")
        return {}
    
    tiers = {}
    for line in tier_lines:
        # Parse the tier name and values.
        tier_name_match = re.match(r'.+(?=\:)', line)
        values_match = re.search(r'(?<=\:).+', line)
        
        if not tier_name_match or not values_match:
            logging.warning(f"Invalid tier format in line: {line.strip()}")
            continue
        
        tier_name = tier_name_match.group(0)
        values = values_match.group(0).split(',')
        logging.info(f"Parsed tier - Name: {tier_name}, Values: {values}")

        # Handle numerical placeholder in values.
        if '##' in values[0]:
            tier_chars = re.sub(r'#', '', values[0])
            try:
                search_str = tiers[tier_chars]._make_search_string(tiers[tier_chars].values) + '\\d+'
                logging.info(f"Generated search string for numerical placeholder: {search_str}")
            except KeyError:
                search_str = tier_chars + '\\d+'
                logging.warning(f"Referenced tier '{tier_chars}' not found. Using default search string: {search_str}")
            tier_obj = Tier(tier_name, [search_str], partition=False)
        else:
            if tier_name.startswith('*'):
                tier_obj = Tier(tier_name[1:], values, partition=True)
                logging.info(f"Tier '{tier_obj.name}' set as partition level.")
            else:
                tier_obj = Tier(tier_name, values, partition=False)

        tiers[tier_obj.name] = tier_obj

    logging.info(f"Finished reading tiers. Total tiers: {len(tiers)}")
    return tiers
