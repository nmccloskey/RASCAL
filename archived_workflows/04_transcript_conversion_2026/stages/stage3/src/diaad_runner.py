"""Run DIAAD CHAT export for Stage 3 transcript tables.

This module runs the DIAAD command::

    diaad transcripts chats

once for each cycle/site transcript-table folder under::

    data/01_reformatted_TTs/<cycle>/<SITE>/

Each output is written to the matching folder under::

    data/02_chat_files/<cycle>/<SITE>/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .tt_processor import CYCLES, SITES
except ImportError:  # pragma: no cover - direct script fallback.
    CYCLES = ("cycle2", "cycle3")
    SITES = ("AC", "BU", "TU")

DIAAD_COMMAND = ("diaad", "transcripts", "chats")
TRANSCRIPT_TABLE_NAME = "transcript_tables.xlsx"


def get_project_root() -> Path:
    """Return the Stage 3 project root, assuming this file lives in ``src/``."""

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
INPUT_ROOT = PROJECT_ROOT / "data" / "01_reformatted_TTs"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "02_chat_files"


@dataclass(frozen=True)
class DiaadRunSpec:
    """A single DIAAD run for one cycle/site partition."""

    cycle: str
    site: str
    input_dir: Path
    output_dir: Path

    @property
    def label(self) -> str:
        """Human-readable run label used in logs."""

        return f"{self.cycle}/{self.site}"

    @property
    def expected_input_file(self) -> Path:
        """Expected transcript-table workbook for this run."""

        return self.input_dir / TRANSCRIPT_TABLE_NAME


def make_logger(name: str = "stage3_diaad_runner") -> logging.Logger:
    """Create a simple console logger if the caller did not provide one."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


def _as_existing_dir(path: Path, label: str) -> Path:
    """Validate that ``path`` exists and is a directory."""

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _build_command(spec: DiaadRunSpec, config_dir: Path) -> list[str]:
    """Build the DIAAD CLI command for one run specification."""

    return [
        *DIAAD_COMMAND,
        "--config",
        str(config_dir),
        "--input-dir",
        str(spec.input_dir),
        "--output-dir",
        str(spec.output_dir),
    ]


def _run_command(
    command: Sequence[str],
    logger: logging.Logger,
    cwd: Path = PROJECT_ROOT,
) -> None:
    """Run one subprocess command and log stdout/stderr."""

    logger.info("Running command: %s", " ".join(command))

    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        logger.error("Could not find executable for command: %s", command[0])
        logger.error("Is DIAAD installed in the active environment and on PATH?")
        raise exc
    except subprocess.CalledProcessError as exc:
        logger.error("DIAAD command failed with return code %s.", exc.returncode)
        logger.error("Command: %s", " ".join(command))
        if exc.stdout:
            logger.error("STDOUT:\n%s", exc.stdout)
        if exc.stderr:
            logger.error("STDERR:\n%s", exc.stderr)
        raise

    if completed.stdout:
        logger.info("STDOUT:\n%s", completed.stdout)
    if completed.stderr:
        logger.warning("STDERR:\n%s", completed.stderr)


def iter_run_specs(
    input_root: Path | str = INPUT_ROOT,
    output_root: Path | str = OUTPUT_ROOT,
    cycles: Iterable[str] = CYCLES,
    sites: Iterable[str] = SITES,
    require_input_files: bool = True,
    logger: logging.Logger | None = None,
) -> list[DiaadRunSpec]:
    """Create DIAAD run specs for existing cycle/site transcript-table folders."""

    logger = logger or make_logger()
    input_root = Path(input_root)
    output_root = Path(output_root)

    specs: list[DiaadRunSpec] = []

    for cycle in cycles:
        for site in sites:
            input_dir = input_root / cycle / site
            output_dir = output_root / cycle / site
            expected_file = input_dir / TRANSCRIPT_TABLE_NAME

            if not input_dir.exists():
                logger.warning("Skipping %s/%s: input directory not found: %s", cycle, site, input_dir)
                continue
            if not input_dir.is_dir():
                logger.warning("Skipping %s/%s: input path is not a directory: %s", cycle, site, input_dir)
                continue
            if require_input_files and not expected_file.exists():
                logger.warning("Skipping %s/%s: missing %s", cycle, site, expected_file)
                continue

            specs.append(
                DiaadRunSpec(
                    cycle=cycle,
                    site=site,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )
            )

    return specs


def run_diaad_specs(
    specs: Sequence[DiaadRunSpec],
    config_dir: Path | str = CONFIG_DIR,
    cwd: Path | str = PROJECT_ROOT,
    logger: logging.Logger | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> dict[str, list[DiaadRunSpec]]:
    """Run DIAAD commands from prepared run specs."""

    logger = logger or make_logger()
    config_dir = Path(config_dir)
    cwd = Path(cwd)

    _as_existing_dir(config_dir, "DIAAD config directory")
    _as_existing_dir(cwd, "Command working directory")

    completed: list[DiaadRunSpec] = []
    failed: list[DiaadRunSpec] = []
    skipped: list[DiaadRunSpec] = []

    logger.info("Preparing to run %s DIAAD command(s).", len(specs))

    for spec in specs:
        logger.info("--- %s ---", spec.label)

        if not spec.expected_input_file.exists():
            logger.warning("Skipping %s: missing %s", spec.label, spec.expected_input_file)
            skipped.append(spec)
            continue

        spec.output_dir.mkdir(parents=True, exist_ok=True)
        command = _build_command(spec, config_dir=config_dir)

        if dry_run:
            logger.info("[DRY RUN] %s", " ".join(command))
            skipped.append(spec)
            continue

        try:
            _run_command(command, logger=logger, cwd=cwd)
            completed.append(spec)
        except (FileNotFoundError, subprocess.CalledProcessError):
            failed.append(spec)
            if not continue_on_error:
                raise

    logger.info("DIAAD CHAT export summary:")
    logger.info("  Completed: %s", len(completed))
    logger.info("  Failed:    %s", len(failed))
    logger.info("  Skipped:   %s", len(skipped))

    return {"completed": completed, "failed": failed, "skipped": skipped}


def run_diaad(
    input_root: Path | str = INPUT_ROOT,
    config_dir: Path | str = CONFIG_DIR,
    output_root: Path | str = OUTPUT_ROOT,
    cwd: Path | str = PROJECT_ROOT,
    continue_on_error: bool = False,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, list[DiaadRunSpec]]:
    """Run DIAAD CHAT export for all reformatted transcript tables."""

    logger = logger or make_logger()
    input_root = _as_existing_dir(Path(input_root), "Reformatted transcript-table root")

    specs = iter_run_specs(
        input_root=input_root,
        output_root=output_root,
        logger=logger,
    )

    if not specs:
        raise RuntimeError(f"No DIAAD run specs were found under {input_root}")

    return run_diaad_specs(
        specs=specs,
        config_dir=config_dir,
        cwd=cwd,
        logger=logger,
        continue_on_error=continue_on_error,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct script execution."""

    parser = argparse.ArgumentParser(
        description="Run DIAAD transcript CHAT export for Stage 3 cycle/site folders."
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--cwd", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> dict[str, list[DiaadRunSpec]]:
    """CLI entry point."""

    args = parse_args()
    return run_diaad(
        input_root=args.input_root,
        output_root=args.output_root,
        config_dir=args.config_dir,
        cwd=args.cwd,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
