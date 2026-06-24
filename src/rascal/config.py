"""Project configuration and initialization for the RASCAL wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rascal.profiles import ProfileError, resolve_profile

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "rascal.yaml"
CONTROL_KEYS = {"profile", "layout", "project_root", "paths"}

CANONICAL_PATHS = {
    "config_dir": "config",
    "diaad_config_dir": "config/diaad.generated",
    "runs_dir": "runs",
    "data_dir": "data",
    "raw_media_dir": "data/raw_media",
    "asr_chunks_dir": "data/asr_chunks",
    "auto_transcripts_dir": "data/auto_transcripts",
    "manual_edit_round_1_dir": "data/manual_edit_round_1",
    "manual_edit_round_2_dir": "data/manual_edit_round_2",
    "transcription_reliability_dir": "data/transcription_reliability",
    "monolog_transcript_tables_dir": "data/monolog/transcript_tables",
    "monolog_cu_files_dir": "data/monolog/cu_files",
    "monolog_cu_reliability_dir": "data/monolog/cu_reliability",
    "monolog_cu_analysis_dir": "data/monolog/cu_analysis",
    "monolog_word_count_files_dir": "data/monolog/word_count_files",
    "monolog_word_reliability_dir": "data/monolog/word_reliability",
    "monolog_speaking_times_dir": "data/monolog/speaking_times",
    "monolog_corelex_dir": "data/monolog/corelex",
    "monolog_final_exports_dir": "data/monolog/final_exports",
    "dialog_transcript_tables_dir": "data/dialog/transcript_tables",
    "dialog_powers_files_dir": "data/dialog/powers_files",
    "dialog_powers_reliability_dir": "data/dialog/powers_reliability",
    "dialog_powers_analysis_dir": "data/dialog/powers_analysis",
    "dialog_final_exports_dir": "data/dialog/final_exports",
}

LEGACY_PATHS = {
    "config_dir": "config",
    "diaad_config_dir": "config/diaad.generated",
    "runs_dir": "runs",
    "input_dir": "rascal_data/input",
    "output_dir": "rascal_data/output",
}

LAYOUT_PATHS = {
    "canonical": CANONICAL_PATHS,
    "legacy": LEGACY_PATHS,
}


class ConfigError(ValueError):
    """Raised when RASCAL project configuration cannot be loaded."""


@dataclass(frozen=True)
class ProjectConfig:
    """Loaded RASCAL project configuration."""

    config_path: Path
    project_root: Path
    profile_name: str
    layout: str
    paths: dict[str, str]
    raw_config: dict[str, Any]
    resolved_profile: dict[str, Any]

    def resolve_path(self, key: str) -> Path:
        """Resolve a configured project path by key."""

        try:
            raw_path = self.paths[key]
        except KeyError as exc:
            raise ConfigError(f"Unknown configured path key: {key}") from exc
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()


@dataclass(frozen=True)
class InitResult:
    """Summary returned by project initialization."""

    project_root: Path
    config_path: Path
    created_directories: tuple[Path, ...]
    profile_name: str
    layout: str


def default_paths_for_layout(layout: str) -> dict[str, str]:
    """Return the default path mapping for a supported project layout."""

    try:
        return dict(LAYOUT_PATHS[layout])
    except KeyError as exc:
        valid = ", ".join(sorted(LAYOUT_PATHS))
        raise ConfigError(f"Unknown layout {layout!r}; expected one of: {valid}") from exc


def locate_config_path(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Locate a RASCAL project configuration file."""

    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
    else:
        root = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
        resolved = (root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()

    if not resolved.is_file():
        raise ConfigError(f"RASCAL config not found: {resolved}")
    return resolved


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"RASCAL config must contain a mapping: {path}")
    return loaded


def _base_for_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_project_root(config_path: Path, raw_project_root: str | Path | None) -> Path:
    if raw_project_root is None:
        return _base_for_project_root(config_path).resolve()
    root = Path(raw_project_root).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (_base_for_project_root(config_path) / root).resolve()


def _profile_overrides(rascal_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in rascal_config.items()
        if key not in CONTROL_KEYS
    }


def load_project_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> ProjectConfig:
    """Load project config and resolve its packaged profile."""

    path = locate_config_path(config_path, project_root=project_root)
    raw_config = _load_yaml_file(path)
    rascal_config = raw_config.get("rascal")
    if not isinstance(rascal_config, dict):
        raise ConfigError(f"RASCAL config must contain a 'rascal' mapping: {path}")

    profile_name = rascal_config.get("profile", "lab_full")
    if not isinstance(profile_name, str):
        raise ConfigError("'rascal.profile' must be a string.")

    layout = rascal_config.get("layout", "canonical")
    if layout not in LAYOUT_PATHS:
        valid = ", ".join(sorted(LAYOUT_PATHS))
        raise ConfigError(f"'rascal.layout' must be one of: {valid}")

    root = _resolve_project_root(path, rascal_config.get("project_root"))
    paths = default_paths_for_layout(layout)
    project_paths = rascal_config.get("paths") or {}
    if not isinstance(project_paths, dict):
        raise ConfigError("'rascal.paths' must be a mapping when provided.")
    paths.update({str(key): str(value) for key, value in project_paths.items()})

    try:
        resolved_profile = resolve_profile(profile_name, _profile_overrides(rascal_config))
    except ProfileError as exc:
        raise ConfigError(str(exc)) from exc

    return ProjectConfig(
        config_path=path,
        project_root=root,
        profile_name=profile_name,
        layout=layout,
        paths=paths,
        raw_config=raw_config,
        resolved_profile=resolved_profile,
    )


def _directory_paths(project_root: Path, paths: dict[str, str]) -> tuple[Path, ...]:
    directories: list[Path] = []
    for raw_path in paths.values():
        path = Path(raw_path)
        resolved = path if path.is_absolute() else project_root / path
        if resolved not in directories:
            directories.append(resolved)
    return tuple(path.resolve() for path in directories)


def _initial_config(profile: str, layout: str) -> dict[str, Any]:
    return {
        "rascal": {
            "profile": profile,
            "layout": layout,
            "project_root": ".",
            "paths": default_paths_for_layout(layout),
        }
    }


def init_project(
    project: str | Path = ".",
    *,
    profile: str = "lab_full",
    layout: str = "canonical",
    force: bool = False,
) -> InitResult:
    """Initialize a RASCAL project directory."""

    project_root = Path(project).expanduser().resolve()
    try:
        resolve_profile(profile)
    except ProfileError as exc:
        raise ConfigError(str(exc)) from exc

    paths = default_paths_for_layout(layout)
    config_path = project_root / DEFAULT_CONFIG_RELATIVE_PATH
    if config_path.exists() and not force:
        raise ConfigError(
            f"RASCAL config already exists: {config_path}. Use --force to overwrite it."
        )

    created_dirs: list[Path] = []
    for directory in _directory_paths(project_root, paths):
        directory.mkdir(parents=True, exist_ok=True)
        created_dirs.append(directory)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = _initial_config(profile, layout)
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    return InitResult(
        project_root=project_root,
        config_path=config_path.resolve(),
        created_directories=tuple(created_dirs),
        profile_name=profile,
        layout=layout,
    )

