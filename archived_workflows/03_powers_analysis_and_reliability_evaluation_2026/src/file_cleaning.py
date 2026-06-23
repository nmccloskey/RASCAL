from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .aggregation_utils import aggregate_cleaned_powers_files
from .cleaning_utils import DataValidationError, process_file
from .revision_utils import RevisionRecorder
from .logging_utils import setup_logger


REL_RENAME_BASE = "powers_coding_for_rel_eval"
REL_EVAL_BASE = "powers_rel_eval_coding"
RESELECT_TAG = "_reselected"
RAW_DIRNAME = "00_raw_files"


def _sort_files(files: list[Path]) -> dict[str, list[Path]]:
    sorted_files: dict[str, list[Path]] = {
        "analysis": [],
        "reliability": [],
    }

    for f in files:
        f = Path(f)

        # Reliability-coding files have c3_* columns.
        # Files named "coding_for_rel_eval" are original/c2 subset files,
        # so they should remain analysis files.
        if "reliability" in f.stem.lower():
            sorted_files["reliability"].append(f)
        else:
            sorted_files["analysis"].append(f)

    return sorted_files


def _read_file(file_path: Path | str, logger: logging.Logger) -> pd.DataFrame:
    file_path = Path(file_path)
    logger.info("Reading raw file: %s", file_path)

    try:
        return pd.read_excel(file_path)
    except Exception:
        logger.exception("Failed to read file: %s", file_path)
        raise


def _write_file(
    df: pd.DataFrame,
    file_path: Path | str,
    logger: logging.Logger,
) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_excel(file_path, index=False)
        logger.info("Wrote processed file: %s", file_path)
    except Exception:
        logger.exception("Failed to write file: %s", file_path)
        raise


def _maybe_rename_file(file_name: str, logger: logging.Logger) -> str:
    new_name = file_name

    if REL_RENAME_BASE in new_name:
        new_name = new_name.replace(REL_RENAME_BASE, REL_EVAL_BASE)

    if RESELECT_TAG in new_name:
        new_name = new_name.replace(RESELECT_TAG, "")

    if new_name != file_name:
        logger.info("Renamed output file: %s -> %s", file_name, new_name)

    return new_name


def prep_out_path(
    file_path: Path | str,
    raw_root: Path | str,
    out_root: Path | str = "data/01_cleaned_files",
    out_tag: str = "cleaned_",
    logger: logging.Logger | None = None,
    create_parent: bool = True,
) -> Path:
    """
    Prepare output subdirs to reflect input.

    Example:
        input:
            data/00_raw_files/cycle2/AC/AC_powers_coding.xlsx

        output:
            data/01_cleaned_files/cycle2/AC/cleaned_AC_powers_coding.xlsx
    """
    logger = logger or logging.getLogger(__name__)

    file_path = Path(file_path)
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    try:
        rel_path = file_path.relative_to(raw_root)
    except ValueError as exc:
        logger.error(
            "Input file is not under raw_root. file=%s raw_root=%s",
            file_path,
            raw_root,
        )
        raise ValueError(
            f"Input file {file_path} is not under raw root {raw_root}."
        ) from exc

    if rel_path.parts and rel_path.parts[0] == RAW_DIRNAME:
        logger.warning(
            "Dropping unexpected raw-directory wrapper from output path: %s",
            rel_path,
        )
        rel_path = Path(*rel_path.parts[1:])

    candidate_name = f"{out_tag}{file_path.name}"
    out_name = _maybe_rename_file(candidate_name, logger)

    out_path = out_root / rel_path.parent / out_name
    if create_parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Prepared output path: %s", out_path)
    return out_path


def collect_files(input_dir: Path | str, logger: logging.Logger) -> list[Path]:
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.rglob("*.xlsx"))

    # Ignore temporary Excel lock files.
    files = [f for f in files if not f.name.startswith("~$")]

    logger.info("Collected %s Excel files from %s.", len(files), input_dir)
    return files


def _has_cycle_dirs(path: Path) -> bool:
    return any(
        child.is_dir() and child.name.lower().startswith("cycle")
        for child in path.iterdir()
    )


def _resolve_input_root(input_dir: Path, logger: logging.Logger) -> Path:
    """
    Resolve the directory whose children are cycle/site folders.

    Some extracted data drops include an extra 00_raw_files wrapper under the
    configured raw root. The cleaner should mirror the logical cycle/site
    layout into 01_cleaned_files, not reproduce that packaging wrapper.
    """
    current = input_dir

    while current.exists() and not _has_cycle_dirs(current):
        nested = current / RAW_DIRNAME
        if not nested.exists() or not nested.is_dir():
            break

        logger.warning(
            "Using nested raw input directory because no cycle folders were "
            "found directly under %s: %s",
            current,
            nested,
        )
        current = nested

    return current


def _timestamped_revisions_path(revisions_dir: Path | str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(revisions_dir) / f"powers_cleaning_revisions_{timestamp}.xlsx"


def clean_files(
    input_dir: Path | str = "data/00_raw_files",
    output_dir: Path | str = "data/01_cleaned_files",
    log_dir: Path | str = "logs",
    revisions_dir: Path | str = "revisions",
    revisions_path: Path | str | None = None,
    continue_on_error: bool = True,
    aggregate: bool = True,
    force_write_cleaned: bool = False,
) -> dict[str, list[Path]]:
    """
    Clean all POWERS coding files for DIAAD analysis.

    Writes a timestamped revisions workbook when validation finds cells that
    need manual correction or review. By default, cleaned files and aggregates
    are written only when there are no revision rows.

    Returns
    -------
    dict
        {
            "written": [Path, ...],
            "failed": [Path, ...],
            "revisions": [Path],
        }
    """
    logger = setup_logger(log_dir=log_dir)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    revisions_dir = Path(revisions_dir)
    raw_root = _resolve_input_root(input_dir, logger)

    if revisions_path is None:
        revisions_path = _timestamped_revisions_path(revisions_dir)

    revisions = RevisionRecorder(revisions_path)

    logger.info("Starting POWERS file cleaning.")
    logger.info("Input directory: %s", input_dir)
    logger.info("Effective raw root: %s", raw_root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Revisions directory: %s", revisions_dir)
    logger.info("Revisions workbook: %s", revisions_path)
    logger.info("Force write cleaned files: %s", force_write_cleaned)

    files = collect_files(raw_root, logger)
    sorted_files = _sort_files(files)

    logger.info("Found %s analysis coding files.", len(sorted_files["analysis"]))
    logger.info("Found %s reliability coding files.", len(sorted_files["reliability"]))

    written: list[Path] = []
    staged: list[tuple[Path, pd.DataFrame]] = []
    failed: list[Path] = []

    for group_name, group_files in sorted_files.items():
        reliability = group_name == "reliability"
        logger.info("Processing group=%s, n_files=%s", group_name, len(group_files))

        for file_path in group_files:
            logger.info("Processing file: %s", file_path)

            try:
                out_file = prep_out_path(
                    file_path=file_path,
                    raw_root=raw_root,
                    out_root=output_dir,
                    logger=logger,
                    create_parent=False,
                )

                df = _read_file(file_path, logger)

                processed_df = process_file(
                    df,
                    reliability=reliability,
                    file_path=file_path,
                    logger=logger,
                    revisions=revisions,
                )

                staged.append((out_file, processed_df))

            except DataValidationError:
                logger.exception("Validation failed for file: %s", file_path)
                failed.append(file_path)

                if not continue_on_error:
                    break

            except Exception:
                logger.exception("Unexpected failure for file: %s", file_path)
                failed.append(file_path)

                if not continue_on_error:
                    break

    revisions_files: list[Path] = []
    has_revisions = bool(revisions.entries)

    if has_revisions:
        revisions_file = revisions.write()
        revisions_files.append(revisions_file)
        logger.info("Wrote revisions workbook: %s", revisions_file)
    else:
        logger.info("No revision rows found; no revisions workbook written.")

    should_write_cleaned = force_write_cleaned or (not has_revisions and not failed)

    if should_write_cleaned:
        for out_file, processed_df in staged:
            _write_file(processed_df, out_file, logger)
            written.append(out_file)
    else:
        logger.warning(
            "Revision rows or failed files were found, so cleaned files and "
            "aggregates were not written. Rerun after correcting source files, or use "
            "force_write_cleaned=True / --force-write-cleaned to override."
        )

    logger.info("Finished POWERS file cleaning.")
    logger.info("Files staged: %s", len(staged))
    logger.info("Files written: %s", len(written))
    logger.info("Files failed: %s", len(failed))
    logger.info("Revision rows written: %s", len(revisions.entries))

    if failed:
        logger.warning("Some files failed validation or processing:")
        for path in failed:
            logger.warning("FAILED: %s", path)
    
    aggregate_files: list[Path] = []
    if aggregate and should_write_cleaned:
        aggregate_result = aggregate_cleaned_powers_files(
            cleaned_root=output_dir,
            logger=logger,
        )
        aggregate_files.extend(
            [
                aggregate_result["metadata_path"],
                aggregate_result["primary_path"],
                aggregate_result["rel_eval_primary_path"],
                aggregate_result["reliability_path"],
            ]
        )
    elif aggregate:
        logger.info("Skipping aggregation because cleaned files were not written.")

    return {
        "written": written,
        "failed": failed,
        "revisions": revisions_files,
        "aggregates": aggregate_files,
    }
