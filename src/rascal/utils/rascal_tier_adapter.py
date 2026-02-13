from typing import Dict
from rascal.utils.logger import logger


class RascalTier:
    """
    Adapter wrapper around core Tier.

    Adds:
      - partition (bool)
      - blind (bool)

    Keeps original Tier behavior.
    """

    def __init__(self, base_tier, *, partition: bool = False, blind: bool = False):
        self._base = base_tier

        # expected legacy attributes
        self.name = base_tier.name
        self.partition = partition
        self.blind = blind

        # pass-through attributes
        self.pattern = base_tier.pattern
        self.values = getattr(base_tier, "values", [])
        self.regex = getattr(base_tier, "regex", None)
        self.kind = getattr(base_tier, "kind", None)

    # pass-through method
    def match(self, *args, **kwargs):
        return self._base.match(*args, **kwargs)

def adapt_tiers_for_rascal(TM) -> Dict[str, RascalTier]:
    """
    Convert TierManager.tiers into RASCAL-compatible tier dict
    with .partition and .blind attributes.
    """

    partition_set = set(TM.tiers_in_group("partition"))
    blind_set = set(TM.tiers_in_group("blind"))

    adapted = {}

    for name in TM.get_tier_names():
        base = TM.tiers[name]

        partition = name in partition_set
        blind = name in blind_set

        tier = RascalTier(
            base,
            partition=partition,
            blind=blind,
        )

        adapted[name] = tier

        logger.info(
            f"Adapted tier '{name}' "
            f"(partition={partition}, blind={blind})"
        )

    if not adapted:
        logger.warning("No tiers adapted for RASCAL.")

    return adapted
