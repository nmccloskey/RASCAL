from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from rascal.utils.logger import logger


# -------------------------
# Tier
# -------------------------

@dataclass(frozen=True)
class Tier:
    """
    A single filename-parsing tier.

    Config contract example:
      tiers:
        site:
          order: 1          # optional int
          values: [AC, BU]  # OR regex: "(AC|BU)\\d+"
    """
    name: str
    order: Optional[int]
    kind: str                     # "values" | "regex" | "default"
    pattern: re.Pattern
    values: List[str]             # only populated when kind=="values"
    regex: Optional[str] = None   # only populated when kind=="regex"

    def match(
        self,
        text: str,
        *,
        return_none: bool = False,
        must_match: bool = False,
    ) -> Optional[str]:
        m = self.pattern.search(text)
        if m:
            return m.group(0)

        if must_match:
            logger.warning(f"No match for tier '{self.name}' in text: {text!r}")

        return None if return_none else self.name


# -------------------------
# TierManager
# -------------------------

class TierManager:
    """
    Parses tier definitions from config into Tier objects, and provides:
      - ordered tier matching
      - tier group helpers (blind, partition, aggregate, pairwise, ...)
      - deterministic blind codebooks
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        name_transform: Optional[Callable[[str], str]] = None,
    ):
        self._name_transform = name_transform or (lambda s: s)

        self.tiers: Dict[str, Tier] = {}
        self.order: List[str] = []
        self.tier_groups: Dict[str, List[str]] = {}

        self._init_from_config(config)

    # ---- init ----

    def _init_from_config(self, config: Dict[str, Any]) -> None:
        raw_tiers = config.get("tiers", None)

        if not isinstance(raw_tiers, dict) or not raw_tiers:
            logger.warning(
                "Tier config missing/invalid — defaulting to single 'file_name' tier "
                "matching full filename ('.*(?=\\.cha)')."
            )
            self.tiers = self._default_tiers()
            self.order = list(self.tiers.keys())
            self.tier_groups = {}
            return

        self.tiers = self._read_tiers(raw_tiers)
        self.order = self._compute_order(raw_tiers, self.tiers)

        raw_groups = config.get("tier_groups", {})
        self.tier_groups = self._read_groups(raw_groups)

        logger.info(f"Initialized TierManager with {len(self.tiers)} tiers.")
        logger.info(f"Tier order: {self.order}")
        if self.tier_groups:
            logger.info(f"Tier groups: {self.tier_groups}")

    def _default_tiers(self) -> Dict[str, Tier]:
        name = "file_name"
        regex = r".*(?=\.cha)"
        pat = re.compile(regex)
        tier = Tier(
            name=name,
            order=None,
            kind="default",
            pattern=pat,
            values=[],
            regex=regex,
        )
        logger.info(f"Created default tier '{name}' with regex={regex!r}")
        return {name: tier}

    def _read_tiers(self, raw_tiers: Dict[str, Any]) -> Dict[str, Tier]:
        tiers: Dict[str, Tier] = {}

        for raw_name, spec in raw_tiers.items():
            name = self._name_transform(raw_name)

            if not isinstance(spec, dict):
                raise TypeError(
                    f"Tier '{raw_name}' must map to a dict with keys like 'values' or 'regex'. "
                    f"Got: {type(spec).__name__}"
                )

            order = spec.get("order", None)
            if order is not None and not isinstance(order, int):
                raise TypeError(f"Tier '{raw_name}': 'order' must be int or omitted.")

            has_values = "values" in spec
            has_regex = "regex" in spec

            if has_values == has_regex:
                # both True or both False → invalid
                raise ValueError(
                    f"Tier '{raw_name}' must define exactly one of 'values' or 'regex'."
                )

            if has_regex:
                regex = spec["regex"]
                if not isinstance(regex, str) or not regex.strip():
                    raise ValueError(f"Tier '{raw_name}': 'regex' must be a non-empty string.")
                try:
                    pat = re.compile(regex)
                except re.error as e:
                    raise ValueError(f"Tier '{raw_name}': invalid regex {regex!r}: {e}") from e

                tier = Tier(
                    name=name,
                    order=order,
                    kind="regex",
                    pattern=pat,
                    values=[],
                    regex=regex,
                )
                tiers[name] = tier
                logger.info(f"Created tier '{name}' (order={order}) from regex={regex!r}")
                continue

            # values path
            values = spec["values"]
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ValueError(
                    f"Tier '{raw_name}': 'values' must be a list[str]."
                )

            if not values:
                logger.warning(
                    f"Tier '{name}' has empty values list — it will never match anything."
                )
                # matches nothing:
                pat = re.compile(r"(?!x)x")
                search_str = r"(?!x)x"
            else:
                escaped = [re.escape(v) for v in values]
                search_str = "(?:" + "|".join(escaped) + ")"
                pat = re.compile(search_str)

            tier = Tier(
                name=name,
                order=order,
                kind="values",
                pattern=pat,
                values=list(values),
                regex=None,
            )
            tiers[name] = tier
            logger.info(
                f"Created tier '{name}' (order={order}) from {len(values)} literal values; "
                f"regex={search_str!r}"
            )

        if not tiers:
            logger.warning(
                "No tiers constructed (unexpected) — defaulting to 'file_name' tier."
            )
            return self._default_tiers()

        return tiers

    def _compute_order(self, raw_tiers: Dict[str, Any], tiers: Dict[str, Tier]) -> List[str]:
        """
        Order tiers by ascending 'order' when provided; tiers without order come after,
        preserving their appearance order from the config.
        """
        ordered: List[tuple[int, str]] = []
        unordered: List[str] = []

        for raw_name, spec in raw_tiers.items():
            name = self._name_transform(raw_name)
            if name not in tiers:
                continue
            order_val = spec.get("order", None) if isinstance(spec, dict) else None
            if isinstance(order_val, int):
                ordered.append((order_val, name))
            else:
                unordered.append(name)

        ordered_sorted = [name for _, name in sorted(ordered, key=lambda x: x[0])]

        # Avoid duplicates if someone gives same tier both ways (shouldn't happen, but safe)
        seen = set(ordered_sorted)
        unordered_kept = [n for n in unordered if n not in seen]

        # Warn on duplicate order numbers (subtle config bug)
        order_nums = [n for n, _ in ordered]
        if len(order_nums) != len(set(order_nums)):
            logger.warning(
                f"Duplicate tier 'order' values detected: {order_nums}. "
                "Sorting is still deterministic but you should fix the config."
            )

        return ordered_sorted + unordered_kept

    def _read_groups(self, raw_groups: Any) -> Dict[str, List[str]]:
        if raw_groups is None:
            return {}
        if not isinstance(raw_groups, dict):
            raise TypeError("tier_groups must be a dict mapping group_name -> list[tier_name].")

        groups: Dict[str, List[str]] = {}
        for group_name, tier_names in raw_groups.items():
            if not isinstance(group_name, str) or not group_name.strip():
                raise ValueError("tier_groups keys must be non-empty strings.")
            if not isinstance(tier_names, list) or not all(isinstance(x, str) for x in tier_names):
                raise ValueError(
                    f"tier_groups['{group_name}'] must be a list[str] of tier names."
                )
            transformed = [self._name_transform(x) for x in tier_names]
            groups[group_name] = transformed

        # Warn if groups reference unknown tiers
        unknown = sorted({t for names in groups.values() for t in names if t not in self.tiers})
        if unknown:
            logger.warning(f"tier_groups references unknown tiers: {unknown}")

        return groups

    # ---- public API ----

    def get_tier_names(self) -> List[str]:
        return list(self.order)

    def match_tiers(
        self,
        text: str,
        *,
        return_none: bool = False,
        must_match: bool = False,
    ) -> Dict[str, Optional[str]]:
        return {
            tier_name: self.tiers[tier_name].match(
                text, return_none=return_none, must_match=must_match
            )
            for tier_name in self.order
        }

    def tiers_in_group(self, group: str) -> List[str]:
        return list(self.tier_groups.get(group, []))

    def make_blind_codebook(self, *, seed: int) -> Dict[str, Dict[str, int]]:
        """
        Deterministically generate blind codes for tiers in tier_groups['blind'].

        - values tiers: shuffled mapping values -> int codes
        - regex tiers: skipped with a warning (can't enumerate)
        """
        blind_tiers = self.tiers_in_group("blind")
        if not blind_tiers:
            logger.info("No 'blind' tier group defined — no blind codebook created.")
            return {}

        rng = random.Random(seed)
        codebook: Dict[str, Dict[str, int]] = {}

        for tier_name in blind_tiers:
            tier = self.tiers.get(tier_name)
            if tier is None:
                logger.warning(f"Blind tier '{tier_name}' not found — skipping.")
                continue

            if tier.kind != "values" or not tier.values:
                logger.warning(
                    f"No blind codes for tier '{tier_name}' (kind={tier.kind}) — "
                    "regex/default tiers cannot be enumerated."
                )
                continue

            codes = list(range(len(tier.values)))
            rng.shuffle(codes)
            mapping = dict(zip(tier.values, codes))
            codebook[tier_name] = mapping
            logger.info(
                f"Created blind code mapping for tier '{tier_name}' with {len(mapping)} values."
            )

        if not codebook:
            logger.warning("Blind codebook empty after processing — check tier_groups['blind'].")

        return codebook
