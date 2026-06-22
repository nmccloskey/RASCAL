from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .aggregation_utils import (
    AGGREGATED_DIRNAME,
    PRIMARY_FILENAME as AGGREGATED_PRIMARY_FILE,
    REL_EVAL_PRIMARY_FILENAME as AGGREGATED_REL_EVAL_PRIMARY_FILE,
    RELIABILITY_FILENAME as AGGREGATED_RELIABILITY_FILE,
)
from .file_cleaning import REL_EVAL_BASE
from .logging_utils import setup_logger


COMMON_CMDS = ["diaad", "powers"]

# DIAAD action -> output subfolder
FUNCS = {
    "analyze": "coding_analysis",
    "evaluate": "reliability_evaluation",
}

CONFIG_FOLDER = "config"
OUTPUT_FOLDER = "data/02_diaad_output"

CYCLES = [f"cycle{i}" for i in range(2, 5)]
SITES = ["AC", "BU", "TU"]


@dataclass(frozen=True)
class DiaadRunSpec:
    action: str
    input_dir: Path
    output_dir: Path
    set_overrides: tuple[str, ...] = ()
    label: str = ""


class DiaadOutputValidationError(RuntimeError):
    """Raised when DIAAD returns success but expected outputs are missing."""


def _normalize_set_override(override: str) -> str:
    """
    Normalize override strings for DIAAD CLI.

    DIAAD examples have used key=value. The user-facing idea may be described
    as "sample_id_column: expanded_sample_id", so this function accepts either
    style and emits key=value for the subprocess call.
    """
    override = str(override).strip()

    if "=" in override:
        return override

    if ":" in override:
        key, value = override.split(":", 1)
        return f"{key.strip()}={value.strip()}"

    return override


def _extend_set_args(command: list[str], overrides: Sequence[str]) -> None:
    for override in overrides:
        command.extend(["--set", _normalize_set_override(override)])


def _run_command(
    command: list[str],
    logger: logging.Logger,
    cwd: Path | None = None,
) -> None:
    logger.info("Running command: %s", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True,
        )
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


def _latest_diaad_run_dir(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    run_dirs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("diaad_")
    ]

    if not run_dirs:
        return None

    return max(run_dirs, key=lambda path: (path.stat().st_mtime, path.name))


def _run_log_contains_write_failure(run_dir: Path) -> bool:
    log_path = run_dir / "logs" / "run_log.log"

    if not log_path.exists():
        return False

    try:
        return "Failed writing" in log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _validate_diaad_outputs(
    spec: DiaadRunSpec,
    logger: logging.Logger,
) -> None:
    run_dir = _latest_diaad_run_dir(spec.output_dir)

    if run_dir is None:
        raise DiaadOutputValidationError(
            f"{spec.label} did not create a DIAAD run folder under {spec.output_dir}."
        )

    logger.info("Validating DIAAD output folder for %s: %s", spec.label, run_dir)

    if _run_log_contains_write_failure(run_dir):
        raise DiaadOutputValidationError(
            f"{spec.label} completed with a 'Failed writing' entry in "
            f"{run_dir / 'logs' / 'run_log.log'}."
        )

    if spec.action == "evaluate":
        results_path = (
            run_dir
            / "powers_reliability"
            / "powers_reliability_results.xlsx"
        )

        if not results_path.exists():
            raise DiaadOutputValidationError(
                f"{spec.label} is missing expected reliability results: "
                f"{results_path}"
            )

        if results_path.stat().st_size == 0:
            raise DiaadOutputValidationError(
                f"{spec.label} wrote an empty reliability results workbook: "
                f"{results_path}"
            )


def _build_command(
    spec: DiaadRunSpec,
    config_dir: Path,
) -> list[str]:
    command = (
        COMMON_CMDS
        + [spec.action]
        + ["--config", str(config_dir)]
        + ["--input-dir", str(spec.input_dir)]
        + ["--output-dir", str(spec.output_dir)]
    )

    _extend_set_args(command, spec.set_overrides)

    return command


def _existing_partition_dirs(cleaned_root: Path) -> list[tuple[str, str, Path]]:
    """
    Return existing cleaned cycle/site directories.

    This is safer than hard-coding every possible cycle/site combo, while still
    preserving the expected cycle/site order.
    """
    dirs: list[tuple[str, str, Path]] = []

    for cycle in CYCLES:
        for site in SITES:
            candidate = cleaned_root / cycle / site
            if candidate.exists():
                dirs.append((cycle, site, candidate))

    return dirs


def _partition_overrides(cycle: str, action: str) -> tuple[str, ...]:
    """
    Cycle 2 reliability evaluation needs the primary subset file rather than
    the full primary coding file. The cleaned file-renaming convention turns
    powers_coding_for_rel_eval into powers_rel_eval_coding.
    """
    if cycle == "cycle2" and action == "evaluate":
        # return (f"powers_coding_filename=cleaned_*{REL_EVAL_BASE}*.xlsx",)
        return (f"powers_coding_filename={REL_EVAL_BASE}",)
    
    return ()


def build_partition_run_specs(
    cleaned_root: Path | str = "data/01_cleaned_files",
    output_root: Path | str = OUTPUT_FOLDER,
) -> list[DiaadRunSpec]:
    cleaned_root = Path(cleaned_root)
    output_root = Path(output_root)

    specs: list[DiaadRunSpec] = []

    for cycle, site, input_dir in _existing_partition_dirs(cleaned_root):
        for action, output_subdir in FUNCS.items():
            output_dir = output_root / "partitions" / cycle / site / output_subdir

            specs.append(
                DiaadRunSpec(
                    action=action,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    set_overrides=_partition_overrides(cycle, action),
                    label=f"{cycle}/{site}/{action}",
                )
            )

    return specs


def build_aggregate_run_specs(
    cleaned_root: Path | str = "data/01_cleaned_files",
    output_root: Path | str = OUTPUT_FOLDER,
    sample_id_column: str = "expanded_sample_id",
    utterance_id_column: str = "utterance_id",
) -> list[DiaadRunSpec]:
    """
    Build DIAAD specs for pooled aggregate analysis and reliability evaluation.

    Assumes DIAAD will soon support global config keys:
        sample_id_column
        utterance_id_column

    With that update, the aggregate runs can use:
        --set sample_id_column=expanded_sample_id
        --set utterance_id_column=utterance_id

    Aggregate analysis uses the reconciled primary aggregate. Aggregate
    reliability evaluation is run twice for comparison: once with the frozen
    primary aggregate prepared for reliability evaluation, and once with the
    reconciled primary aggregate.

    The aggregate input directory contains separate single-sheet files because
    DIAAD POWERS does not currently accept sheet-specific input specs.
    """
    cleaned_root = Path(cleaned_root)
    output_root = Path(output_root)
    aggregated_dir = cleaned_root / AGGREGATED_DIRNAME

    common_overrides = (
        f"sample_id_column={sample_id_column}",
        f"utterance_id_column={utterance_id_column}",
    )

    return [
        DiaadRunSpec(
            action="analyze",
            input_dir=aggregated_dir,
            output_dir=output_root / "aggregated" / FUNCS["analyze"],
            set_overrides=(
                *common_overrides,
                f"powers_coding_filename={AGGREGATED_PRIMARY_FILE}",
            ),
            label="aggregated/analyze",
        ),
        DiaadRunSpec(
            action="evaluate",
            input_dir=aggregated_dir,
            output_dir=(
                output_root
                / "aggregated"
                / FUNCS["evaluate"]
                / "frozen"
            ),
            set_overrides=(
                *common_overrides,
                f"powers_coding_filename={AGGREGATED_REL_EVAL_PRIMARY_FILE}",
                f"powers_reliability_filename={AGGREGATED_RELIABILITY_FILE}",
            ),
            label="aggregated/evaluate_frozen",
        ),
        DiaadRunSpec(
            action="evaluate",
            input_dir=aggregated_dir,
            output_dir=(
                output_root
                / "aggregated"
                / FUNCS["evaluate"]
                / "reconciled"
            ),
            set_overrides=(
                *common_overrides,
                f"powers_coding_filename={AGGREGATED_PRIMARY_FILE}",
                f"powers_reliability_filename={AGGREGATED_RELIABILITY_FILE}",
            ),
            label="aggregated/evaluate_reconciled",
        ),
    ]


def run_diaad_specs(
    specs: Sequence[DiaadRunSpec],
    config_dir: Path | str = CONFIG_FOLDER,
    logger: logging.Logger | None = None,
    cwd: Path | str | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> dict[str, list[DiaadRunSpec]]:
    logger = logger or setup_logger(name="powers_prep_diaad", log_dir="logs")
    config_dir = Path(config_dir)
    cwd_path = Path(cwd) if cwd is not None else None

    completed: list[DiaadRunSpec] = []
    failed: list[DiaadRunSpec] = []
    skipped: list[DiaadRunSpec] = []

    logger.info("Preparing to run %s DIAAD commands.", len(specs))

    for spec in specs:
        spec.output_dir.mkdir(parents=True, exist_ok=True)

        if not spec.input_dir.exists():
            logger.warning("Skipping %s because input_dir does not exist: %s", spec.label, spec.input_dir)
            skipped.append(spec)
            continue

        command = _build_command(spec, config_dir=config_dir)

        if dry_run:
            logger.info("[DRY RUN] %s", " ".join(command))
            skipped.append(spec)
            continue

        try:
            _run_command(command, logger=logger, cwd=cwd_path)
            _validate_diaad_outputs(spec, logger=logger)
            completed.append(spec)
        except (subprocess.CalledProcessError, DiaadOutputValidationError):
            failed.append(spec)
            if not continue_on_error:
                raise

    logger.info("DIAAD run complete.")
    logger.info("Completed: %s", len(completed))
    logger.info("Failed: %s", len(failed))
    logger.info("Skipped: %s", len(skipped))

    return {
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


def run_diaad(
    input_dir: Path | str = "data/01_cleaned_files",
    config_dir: Path | str = CONFIG_FOLDER,
    output_root: Path | str = OUTPUT_FOLDER,
    run_partitions: bool = True,
    run_aggregates: bool = True,
    sample_id_column_for_aggregates: str = "expanded_sample_id",
    utterance_id_column_for_aggregates: str = "utterance_id",
    log_dir: Path | str = "logs",
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> dict[str, list[DiaadRunSpec]]:
    """
    Run DIAAD POWERS analysis/evaluation over cleaned partition-level files
    and/or aggregate-level files.

    Parameters
    ----------
    input_dir:
        Cleaned files root, usually data/01_cleaned_files.
    config_dir:
        DIAAD config directory.
    output_root:
        Root for DIAAD outputs.
    run_partitions:
        Run DIAAD separately for each cycle/site partition.
    run_aggregates:
        Run DIAAD on pooled aggregate files in input_dir/aggregated.
    sample_id_column_for_aggregates:
        Intended for DIAAD after adding global sample_id_column support.
    utterance_id_column_for_aggregates:
        Intended for DIAAD after adding global utterance_id_column support.
    dry_run:
        Log commands without running them.
    """
    logger = setup_logger(name="powers_prep_diaad", log_dir=log_dir)

    cleaned_root = Path(input_dir)
    output_root = Path(output_root)

    specs: list[DiaadRunSpec] = []

    if run_partitions:
        specs.extend(
            build_partition_run_specs(
                cleaned_root=cleaned_root,
                output_root=output_root,
            )
        )

    if run_aggregates:
        specs.extend(
            build_aggregate_run_specs(
                cleaned_root=cleaned_root,
                output_root=output_root,
                sample_id_column=sample_id_column_for_aggregates,
                utterance_id_column=utterance_id_column_for_aggregates,
            )
        )

    return run_diaad_specs(
        specs=specs,
        config_dir=config_dir,
        logger=logger,
        continue_on_error=continue_on_error,
        dry_run=dry_run,
    )
