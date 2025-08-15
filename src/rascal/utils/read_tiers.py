import logging
import re
from .tier import Tier

def read_tiers(config_tiers: dict):
    """
    Parses tier definitions from a config dict and returns a dict[str, Tier].

    Rules:
      - If values contain '##', we build a pattern by referencing the prefix tier and appending r'\\d+'.
      - If len(values) > 1: build a regex from literal values (escaped, joined).
      - If len(values) == 1: treat that single string as a user-provided regex and compile it directly.

    Returns:
      dict[str, Tier]
    """
    if not isinstance(config_tiers, dict):
        logging.error("Invalid tier structure in config. Expected a dictionary.")
        return {}

    tiers = {}
    for tier_name, tier_data in config_tiers.items():
        try:
            # Normalize structure from legacy formats
            if isinstance(tier_data, (list, str)):
                values = [tier_data] if isinstance(tier_data, str) else tier_data
                tier_data = {"values": values}

            values = tier_data.get("values", [])
            if isinstance(values, str):
                values = [values]

            # Flags
            partition = bool(tier_data.get("partition", False))
            blind = bool(tier_data.get("blind", False))

            if not values:
                logging.warning(f"Tier '{tier_name}' has no values; it will never match.")
                # Still create the Tier to keep downstream logic predictable
                tier_obj = Tier(tier_name, [], partition=partition, blind=blind)
                tiers[tier_name] = tier_obj
                continue

            # Placeholder behavior: e.g., values like ['site##'] meaning: use previous tier's pattern + \d+
            if any('##' in v for v in values):
                ref_name = re.sub(r'#', '', values[0])
                try:
                    ref_tier = tiers[ref_name]
                    # Build: (ref_pattern) + \d+   — here we reuse the literal-built pattern of the ref tier
                    # If ref_tier is user-regex, we still reuse its compiled pattern string.
                    search_str = ref_tier.search_str + r'\d+'
                    tier_obj = Tier(tier_name, [search_str], partition=partition, blind=blind)
                    logging.info(
                        f"Tier '{tier_name}' constructed from reference '{ref_name}' → {search_str!r}"
                    )
                except KeyError:
                    # Fallback: user forgot to define ref first; make a best-effort pattern
                    search_str = re.escape(ref_name) + r'\d+'
                    logging.warning(
                        f"Referenced tier '{ref_name}' not found before '{tier_name}'. "
                        f"Using fallback regex: {search_str!r}"
                    )
                    tier_obj = Tier(tier_name, [search_str], partition=partition, blind=blind)

            else:
                # Decide behavior based on number of values
                if len(values) == 1:
                    # Single value → user regex
                    user_regex = values[0]
                    try:
                        re.compile(user_regex)
                    except re.error as e:
                        raise ValueError(
                            f"Tier '{tier_name}': invalid user regex {user_regex!r}. Error: {e}"
                        )
                    logging.info(f"Tier '{tier_name}' using user regex: {user_regex!r}")
                    tier_obj = Tier(tier_name, [user_regex], partition=partition, blind=blind)
                else:
                    # Multiple values → build from literals
                    logging.info(
                        f"Tier '{tier_name}' using {len(values)} literal values; "
                        f"regex will match any of them."
                    )
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
