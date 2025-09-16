import logging
import re
from .tier import Tier

def read_tiers(config_tiers: dict):
    """
    Parse tier definitions from a configuration dictionary into Tier objects.

    Behavior:
      - Input must be a dictionary where each key is a tier name and each value
        is either:
          • A dict with keys:
              - "values": list[str] | str (required)
              - "partition": bool (optional, default False)
              - "blind": bool (optional, default False)
          • A legacy list[str] or str (shorthand for {"values": ...}).
      - If "values" contains multiple items, each value is treated literally
        (escaped if necessary). The Tier will match any of them.
      - If "values" contains a single item, it is treated as a user-provided
        regular expression. The function validates that the regex compiles.
      - If "values" is empty or missing, a warning is logged and the Tier is
        created with no values (it will never match).

    Returns:
      dict[str, Tier]
        Mapping from tier name → Tier object with attributes:
          - name: str
          - values: list[str]
          - partition: bool
          - blind: bool

    Logging:
      - Logs warnings for tiers with no values.
      - Logs errors if config_tiers is not a dict or regex compilation fails.
      - Logs info for partition/blind flags and regex behavior.
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
