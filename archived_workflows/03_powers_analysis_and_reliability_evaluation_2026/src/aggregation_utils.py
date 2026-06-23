from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


AGGREGATED_DIRNAME = "aggregated"

PRIMARY_FILENAME = "powers_primary_coding_aggregated.xlsx"
REL_EVAL_PRIMARY_FILENAME = "powers_primary_coding_for_rel_eval_aggregated.xlsx"
RELIABILITY_FILENAME = "powers_reliability_coding_aggregated.xlsx"
METADATA_FILENAME = "powers_sample_metadata_aggregated.xlsx"


@dataclass(frozen=True)
class CleanedFileInfo:
    path: Path
    cycle: str
    site: str
    role: str  # "primary", "primary_rel_eval_subset", or "reliability"


def _normalize_cycle(cycle: str) -> str:
    """
    Convert folder names like cycle2, cycle3, cycle4 to C2, C3, C4.
    """
    text = str(cycle).strip().lower()

    if text.startswith("cycle"):
        suffix = text.replace("cycle", "").strip()
        return f"C{suffix}"

    return text.upper()


def _normalize_site(site: str) -> str:
    return str(site).strip().upper()


def make_expanded_sample_id(
    cycle: str,
    site: str,
    sample_id: object,
) -> str:
    """
    Create globally unique sample IDs.

    Example:
        cycle2 + TU + S003 -> C2_TU_S003
    """
    cycle_tag = _normalize_cycle(cycle)
    site_tag = _normalize_site(site)
    sample_text = str(sample_id).strip()

    return f"{cycle_tag}_{site_tag}_{sample_text}"


def infer_cleaned_file_role(path: Path) -> str:
    """
    Infer whether a cleaned file is primary coding, a primary reliability-eval
    subset, or reliability coding.

    Notes
    -----
    The primary rel-eval subset is c2/primary coding, but it is usually a
    duplicate subset of the full primary file, so it should not be pooled into
    the full primary aggregate when full primary files are available.
    """
    stem = path.stem.lower()

    if "reliability" in stem:
        return "reliability"

    if "for_rel_eval" in stem or "rel_eval" in stem:
        return "primary_rel_eval_subset"

    return "primary"


def parse_cleaned_file_info(
    path: Path | str,
    cleaned_root: Path | str,
) -> CleanedFileInfo:
    """
    Extract cycle/site metadata from cleaned file path.

    Expected structure:
        01_cleaned_files/cycle2/AC/cleaned_AC_powers_coding.xlsx
    """
    path = Path(path)
    cleaned_root = Path(cleaned_root)

    rel_path = path.relative_to(cleaned_root)

    if len(rel_path.parts) < 3:
        raise ValueError(
            "Expected cleaned file path to look like "
            f"<cleaned_root>/<cycle>/<site>/<file>. Got: {path}"
        )

    cycle = rel_path.parts[0]
    site = rel_path.parts[1]
    role = infer_cleaned_file_role(path)

    return CleanedFileInfo(
        path=path,
        cycle=cycle,
        site=site,
        role=role,
    )


def _partition_key(info: CleanedFileInfo) -> tuple[str, str]:
    return (_normalize_cycle(info.cycle), _normalize_site(info.site))


def collect_cleaned_files(
    cleaned_root: Path | str,
    logger: logging.Logger,
) -> list[CleanedFileInfo]:
    cleaned_root = Path(cleaned_root)

    if not cleaned_root.exists():
        raise FileNotFoundError(f"Cleaned directory does not exist: {cleaned_root}")

    files = sorted(
        f for f in cleaned_root.rglob("*.xlsx")
        if not f.name.startswith("~$")
    )

    # Avoid accidentally re-ingesting aggregate/revision outputs or stale
    # cleaned files written under an unintended raw-directory wrapper.
    files = [
        f for f in files
        if AGGREGATED_DIRNAME not in f.parts
        and "00_raw_files" not in f.parts
        and "aggregate" not in f.stem.lower()
        and "aggregated" not in f.stem.lower()
        and "revisions" not in f.stem.lower()
    ]

    infos: list[CleanedFileInfo] = []

    for file in files:
        try:
            infos.append(parse_cleaned_file_info(file, cleaned_root))
        except Exception:
            logger.exception("Could not parse cleaned file path: %s", file)
            raise

    logger.info("Collected %s cleaned Excel files.", len(infos))

    if infos:
        role_counts = pd.Series([info.role for info in infos]).value_counts().to_dict()
    else:
        role_counts = {}

    logger.info("Cleaned file role counts: %s", role_counts)

    return infos


def _read_cleaned_file(info: CleanedFileInfo, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Reading cleaned %s file: %s", info.role, info.path)

    df = pd.read_excel(info.path)

    required = {"sample_id", "utterance_id", "speaker"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"{info.path} is missing required columns: {missing}")

    df = df.copy()
    df["cycle"] = _normalize_cycle(info.cycle)
    df["site"] = _normalize_site(info.site)
    df["source_file"] = info.path.name
    df["source_path"] = str(info.path)
    df["file_role"] = info.role

    df["expanded_sample_id"] = [
        make_expanded_sample_id(cycle, site, sample_id)
        for cycle, site, sample_id in zip(df["cycle"], df["site"], df["sample_id"])
    ]

    return df


def _choose_primary_files(
    infos: list[CleanedFileInfo],
    logger: logging.Logger,
) -> list[CleanedFileInfo]:
    """
    Keep canonical primary files.

    By default, this excludes primary_rel_eval_subset files, because they are
    usually duplicate subsets of the full primary coding file.
    """
    primary = [info for info in infos if info.role == "primary"]
    rel_eval_subsets = [info for info in infos if info.role == "primary_rel_eval_subset"]

    logger.info("Canonical primary files: %s", len(primary))
    logger.info("Excluded primary rel-eval subset files: %s", len(rel_eval_subsets))

    if not primary and rel_eval_subsets:
        logger.warning(
            "No full primary files found. Falling back to primary rel-eval subset files."
        )
        return rel_eval_subsets

    return primary


def _choose_rel_eval_primary_files(
    infos: list[CleanedFileInfo],
    logger: logging.Logger,
) -> list[CleanedFileInfo]:
    """
    Build the primary-coding file set used for aggregate reliability evaluation.

    Cycle 2 has frozen primary coding subsets for reliability evaluation. For
    those cycle/site bins, use the frozen rel-eval file instead of the
    reconciled primary coding file. Other bins continue to use their canonical
    primary coding files.
    """
    primary = [info for info in infos if info.role == "primary"]
    rel_eval_subsets = [
        info
        for info in infos
        if info.role == "primary_rel_eval_subset"
        and _normalize_cycle(info.cycle) == "C2"
    ]

    logger.info("Cycle 2 primary rel-eval subset files: %s", len(rel_eval_subsets))

    if not rel_eval_subsets:
        logger.warning(
            "No cycle 2 primary rel-eval subset files found. The reliability "
            "primary aggregate will use canonical primary files only."
        )
        return primary

    subset_counts = Counter(_partition_key(info) for info in rel_eval_subsets)
    duplicate_partitions = {
        key: count
        for key, count in subset_counts.items()
        if count > 1
    }

    if duplicate_partitions:
        logger.warning(
            "Multiple cycle 2 primary rel-eval subset files found for these "
            "cycle/site bins: %s. All will be included.",
            duplicate_partitions,
        )

    subset_partitions = set(subset_counts)
    rel_eval_primary = [
        info
        for info in primary
        if not (
            _normalize_cycle(info.cycle) == "C2"
            and _partition_key(info) in subset_partitions
        )
    ]
    replaced_primary = [
        info
        for info in primary
        if (
            _normalize_cycle(info.cycle) == "C2"
            and _partition_key(info) in subset_partitions
        )
    ]

    logger.info(
        "Replacing %s cycle 2 reconciled primary files with %s frozen rel-eval files.",
        len(replaced_primary),
        len(rel_eval_subsets),
    )

    return sorted(
        [*rel_eval_primary, *rel_eval_subsets],
        key=lambda info: str(info.path),
    )


def _build_sample_metadata(
    primary_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_cols = [
        "expanded_sample_id",
        "sample_id",
        "cycle",
        "site",
        "source_file",
    ]

    if primary_df.empty:
        return pd.DataFrame(
            columns=[
                "expanded_sample_id",
                "sample_id",
                "cycle",
                "site",
                "coding_file",
                "reliability_file",
                "has_reliability",
            ]
        )

    primary_meta = (
        primary_df[metadata_cols]
        .drop_duplicates()
        .rename(columns={"source_file": "coding_file"})
    )

    if reliability_df.empty:
        primary_meta["reliability_file"] = pd.NA
        primary_meta["has_reliability"] = False
        return primary_meta.sort_values(["cycle", "site", "sample_id"])

    reliability_meta = (
        reliability_df[
            [
                "expanded_sample_id",
                "source_file",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"source_file": "reliability_file"})
    )

    reliability_by_sample = (
        reliability_meta
        .groupby("expanded_sample_id", as_index=False)
        .agg({"reliability_file": lambda values: "; ".join(sorted(set(map(str, values))))})
    )

    sample_meta = primary_meta.merge(
        reliability_by_sample,
        on="expanded_sample_id",
        how="left",
    )

    sample_meta["has_reliability"] = sample_meta["reliability_file"].notna()

    return sample_meta.sort_values(["cycle", "site", "sample_id"])


def _check_duplicate_rows(
    df: pd.DataFrame,
    table_name: str,
    logger: logging.Logger,
) -> None:
    """
    Check whether expanded_sample_id + utterance_id uniquely identifies rows.
    """
    if df.empty:
        return

    key_cols = ["expanded_sample_id", "utterance_id"]
    dup_mask = df.duplicated(key_cols, keep=False)

    if dup_mask.any():
        dupes = df.loc[
            dup_mask,
            key_cols + ["cycle", "site", "sample_id", "source_file"],
        ].sort_values(key_cols)

        logger.warning(
            "%s contains %s rows with duplicate expanded_sample_id + utterance_id.",
            table_name,
            len(dupes),
        )
        logger.warning(
            "Duplicate examples from %s:\n%s",
            table_name,
            dupes.head(25).to_string(index=False),
        )


def _check_reliability_matches_primary(
    primary_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    logger: logging.Logger,
    primary_table_name: str = "primary_coding",
) -> None:
    """
    Warn if reliability rows cannot be matched to primary rows.
    """
    if primary_df.empty or reliability_df.empty:
        return

    key_cols = ["expanded_sample_id", "utterance_id"]

    primary_keys = primary_df[key_cols].drop_duplicates()
    reliability_keys = reliability_df[key_cols].drop_duplicates()

    merged = reliability_keys.merge(
        primary_keys,
        on=key_cols,
        how="left",
        indicator=True,
    )

    missing = merged[merged["_merge"].eq("left_only")].drop(columns="_merge")

    if not missing.empty:
        logger.warning(
            "%s reliability rows do not match %s rows by %s.",
            len(missing),
            primary_table_name,
            key_cols,
        )
        logger.warning(
            "Unmatched reliability examples against %s:\n%s",
            primary_table_name,
            missing.head(25).to_string(index=False),
        )


def _write_table(
    df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
        worksheet = writer.sheets["data"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            header = str(column_cells[0].value or "")
            width = min(max(len(header) + 4, 14), 45)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width

    logger.info("Wrote aggregate table: %s rows -> %s", len(df), output_path)
    return output_path


def aggregate_cleaned_powers_files(
    cleaned_root: Path | str = "data/01_cleaned_files",
    aggregated_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """
    Aggregate cleaned POWERS coding files into pooled single-sheet Excel files.

    Outputs four separate files under:
        data/01_cleaned_files/aggregated/

    Files:
        powers_sample_metadata_aggregated.xlsx
        powers_primary_coding_aggregated.xlsx
        powers_primary_coding_for_rel_eval_aggregated.xlsx
        powers_reliability_coding_aggregated.xlsx

    Separate files are used because DIAAD POWERS analysis currently expects
    file-level input, not workbook sheet specifications.
    """
    logger = logger or logging.getLogger(__name__)

    cleaned_root = Path(cleaned_root)

    if aggregated_dir is None:
        aggregated_dir = cleaned_root / AGGREGATED_DIRNAME

    aggregated_dir = Path(aggregated_dir)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    infos = collect_cleaned_files(cleaned_root, logger)

    primary_infos = _choose_primary_files(infos, logger)
    rel_eval_primary_infos = _choose_rel_eval_primary_files(infos, logger)
    reliability_infos = [info for info in infos if info.role == "reliability"]

    primary_frames = [
        _read_cleaned_file(info, logger)
        for info in primary_infos
    ]

    rel_eval_primary_frames = [
        _read_cleaned_file(info, logger)
        for info in rel_eval_primary_infos
    ]

    reliability_frames = [
        _read_cleaned_file(info, logger)
        for info in reliability_infos
    ]

    primary_df = (
        pd.concat(primary_frames, ignore_index=True)
        if primary_frames
        else pd.DataFrame()
    )

    rel_eval_primary_df = (
        pd.concat(rel_eval_primary_frames, ignore_index=True)
        if rel_eval_primary_frames
        else pd.DataFrame()
    )

    reliability_df = (
        pd.concat(reliability_frames, ignore_index=True)
        if reliability_frames
        else pd.DataFrame()
    )

    _check_duplicate_rows(primary_df, "primary_coding", logger)
    _check_duplicate_rows(
        rel_eval_primary_df,
        "primary_coding_for_rel_eval",
        logger,
    )
    _check_duplicate_rows(reliability_df, "reliability_coding", logger)
    _check_reliability_matches_primary(
        primary_df,
        reliability_df,
        logger,
        primary_table_name="primary_coding",
    )
    _check_reliability_matches_primary(
        rel_eval_primary_df,
        reliability_df,
        logger,
        primary_table_name="primary_coding_for_rel_eval",
    )

    sample_metadata = _build_sample_metadata(primary_df, reliability_df)

    metadata_path = _write_table(
        sample_metadata,
        aggregated_dir / METADATA_FILENAME,
        logger,
    )
    primary_path = _write_table(
        primary_df,
        aggregated_dir / PRIMARY_FILENAME,
        logger,
    )
    rel_eval_primary_path = _write_table(
        rel_eval_primary_df,
        aggregated_dir / REL_EVAL_PRIMARY_FILENAME,
        logger,
    )
    reliability_path = _write_table(
        reliability_df,
        aggregated_dir / RELIABILITY_FILENAME,
        logger,
    )

    logger.info("Aggregate POWERS files written under: %s", aggregated_dir)
    logger.info("sample_metadata rows: %s", len(sample_metadata))
    logger.info("primary_coding rows: %s", len(primary_df))
    logger.info("primary_coding_for_rel_eval rows: %s", len(rel_eval_primary_df))
    logger.info("reliability_coding rows: %s", len(reliability_df))

    return {
        "aggregated_dir": aggregated_dir,
        "metadata_path": metadata_path,
        "primary_path": primary_path,
        "rel_eval_primary_path": rel_eval_primary_path,
        "reliability_path": reliability_path,
        "sample_metadata": sample_metadata,
        "primary_coding": primary_df,
        "primary_coding_for_rel_eval": rel_eval_primary_df,
        "reliability_coding": reliability_df,
    }
