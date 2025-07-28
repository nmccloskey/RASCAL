import logging
import re
from .tier import Tier

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
    for tier_name, tier_data in config_tiers.items():
        try:
            # If old format (non-nested), convert it
            if isinstance(tier_data, list) or isinstance(tier_data, str):
                values = [tier_data] if isinstance(tier_data, str) else tier_data
                tier_data = {"values": values}

            values = tier_data.get("values", [])
            if isinstance(values, str):
                values = [values]

            partition = tier_data.get("partition", False)
            blind = tier_data.get("blind", False)

            if any('##' in v for v in values):
                ref = re.sub(r'#', '', values[0])
                try:
                    ref_tier = tiers[ref]
                    search_str = ref_tier._make_search_string(ref_tier.values) + r'\d+'
                except KeyError:
                    search_str = ref + r'\d+'
                    logging.warning(f"Referenced tier '{ref}' not found. Using fallback: {search_str}")
                tier_obj = Tier(tier_name, [search_str], partition=partition, blind=blind)
            else:
                tier_obj = Tier(tier_name, values, partition=partition, blind=blind)

            tiers[tier_name] = tier_obj
            if partition:
                logging.info(f"Tier '{tier_name}' marked as partition level.")
            if blind:
                logging.info(f"Tier '{tier_name}' marked as blind column.")

        except Exception as e:
            logging.error(f"Failed to parse tier '{tier_name}': {e}")

    logging.info(f"Finished parsing tiers from config. Total tiers: {len(tiers)}")
    return tiers
