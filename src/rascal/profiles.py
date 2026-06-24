"""Profile loading for the RASCAL DIAAD wrapper."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from typing import Any

import yaml

PROFILE_PACKAGE = "rascal.data.profiles"


class ProfileError(ValueError):
    """Raised when a RASCAL profile cannot be loaded or resolved."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_resource(package: str, filename: str) -> dict[str, Any]:
    path = resources.files(package).joinpath(filename)
    if not path.is_file():
        raise ProfileError(f"Unknown RASCAL profile: {filename.removesuffix('.yaml')}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ProfileError(f"Profile resource {filename} must contain a mapping.")
    return loaded


def list_profiles() -> list[str]:
    """Return packaged profile names without file extensions."""

    return sorted(
        path.name.removesuffix(".yaml")
        for path in resources.files(PROFILE_PACKAGE).iterdir()
        if path.name.endswith(".yaml")
    )


def load_packaged_profile(name: str) -> dict[str, Any]:
    """Load one packaged profile without resolving inheritance."""

    return deepcopy(_load_yaml_resource(PROFILE_PACKAGE, f"{name}.yaml"))


def resolve_profile(
    name: str,
    overrides: dict[str, Any] | None = None,
    *,
    _seen: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve a packaged profile and optional project overrides."""

    if name in _seen:
        chain = " -> ".join((*_seen, name))
        raise ProfileError(f"Circular profile inheritance detected: {chain}")

    profile = load_packaged_profile(name)
    parent_name = profile.pop("extends", None)
    if parent_name:
        parent = resolve_profile(parent_name, _seen=(*_seen, name))
        profile = _deep_merge(parent, profile)

    if overrides:
        profile = _deep_merge(profile, overrides)

    return deepcopy(profile)


def merge_profile_overrides(
    profile: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge project overrides into an already resolved profile."""

    if not overrides:
        return deepcopy(profile)
    return _deep_merge(profile, overrides)

