import logging
import re
from utils.tier import Tier

def read_tiers(config_tiers: dict):
    """
    Parses tier definitions from a config dictionary and returns a dict of Tier objects.

    Parameters:
    - config_tiers (dict): Dictionary with tier definitions from config.yaml.

    Returns:
    - dict: Dictionary of Tier objects.
    """
    if not isinstance(config_tiers, dict):
        logging.error("Invalid tier structure in config. Expected a dictionary.")
        return {}

    tiers = {}
    for tier_name, values in config_tiers.items():
        try:
            # Convert string values to list
            if isinstance(values, str):
                values = [values]

            # Handle numerical placeholders (e.g., site##)
            if isinstance(values, list) and any('##' in v for v in values):
                # Strip '##' to get referenced tier
                ref = re.sub(r'#', '', values[0])
                try:
                    # Use existing tier values to generate search string
                    ref_tier = tiers[ref]
                    search_str = ref_tier._make_search_string(ref_tier.values) + r'\d+'
                    logging.info(f"Resolved placeholder in tier '{tier_name}': using search string '{search_str}'")
                except KeyError:
                    search_str = ref + r'\d+'
                    logging.warning(f"Referenced tier '{ref}' not found for placeholder in '{tier_name}'. Using fallback: {search_str}")
                tier_obj = Tier(tier_name, [search_str], partition=False)

            else:
                is_partition = tier_name.startswith('*')
                clean_name = tier_name.lstrip('*')
                tier_obj = Tier(clean_name, values, partition=is_partition)
                if is_partition:
                    logging.info(f"Tier '{clean_name}' marked as partition level.")

            tiers[tier_obj.name] = tier_obj

        except Exception as e:
            logging.error(f"Failed to parse tier '{tier_name}': {e}")

    logging.info(f"Finished parsing tiers from config. Total tiers: {len(tiers)}")
    return tiers
