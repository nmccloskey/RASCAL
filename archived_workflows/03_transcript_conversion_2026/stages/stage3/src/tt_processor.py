"""Reformat proto transcript tables for DIAAD Stage 3.

This module reads the original proto transcript-table workbooks under::

    data/00_original_proto_TTs/<cycle>/<SITE>_transcript_tables.xlsx

and writes DIAAD-ready transcript-table workbooks under::

    data/01_reformatted_TTs/<cycle>/<SITE>/transcript_tables.xlsx

The output folder layout is intentionally aligned with the next Stage 3 step,
where DIAAD is run once per cycle/site input directory.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Literal, Sequence

import pandas as pd


CYCLES: tuple[str, ...] = ("cycle2", "cycle3")
SITES: tuple[str, ...] = ("AC", "BU", "TU")

SAMPLES_KEEP_COLS = [
    "sample_id",
    "expanded_sample_id",
    "cycle",
    "site",
    "study_id",
    "test",
    "narrative",
]

UTTS_KEEP_COLS = [
    "sample_id",
    "utterance_id",
    "speaker",
    "utterance",
]

SheetKind = Literal["samples", "utterances"]


def get_project_root() -> Path:
    """Return the Stage 3 project root, assuming this file lives in ``src/``."""

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
INPUT_DIR = PROJECT_ROOT / "data" / "00_original_proto_TTs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "01_reformatted_TTs"


class TranscriptTableError(RuntimeError):
    """Raised when a transcript-table workbook fails validation."""


def _log(level: str, message: str) -> None:
    """Print a consistently formatted status message."""

    print(f"[{level}] {message}")


def _require_columns(df: pd.DataFrame, required_cols: Sequence[str], *, context: str) -> None:
    """Raise a clear error if a dataframe is missing required columns."""

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise TranscriptTableError(
            f"{context} is missing required column(s): {', '.join(missing)}"
        )


def _extract_cycle_number(cycle: str) -> int:
    """Extract the numeric cycle value from a label like ``cycle2``."""

    match = re.search(r"\d+", cycle)
    if not match:
        raise TranscriptTableError(f"Could not extract a cycle number from {cycle!r}.")
    return int(match.group())


def _read_tt(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ``samples`` and ``utterances`` sheets from one workbook."""

    if not file_path.exists():
        raise FileNotFoundError(f"Input workbook does not exist: {file_path}")

    _log("INFO", f"Reading transcript tables: {file_path}")
    sheets = pd.read_excel(file_path, sheet_name=None, index_col=False)

    missing_sheets = [name for name in ("samples", "utterances") if name not in sheets]
    if missing_sheets:
        raise TranscriptTableError(
            f"{file_path} is missing required sheet(s): {', '.join(missing_sheets)}"
        )

    return sheets["samples"], sheets["utterances"]


def _write_tt(file_path: Path, sample_df: pd.DataFrame, utt_df: pd.DataFrame) -> None:
    """Write cleaned samples and utterances sheets to one workbook."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        sample_df.to_excel(writer, sheet_name="samples", index=False)
        utt_df.to_excel(writer, sheet_name="utterances", index=False)

    _log("INFO", f"Wrote cleaned transcript tables: {file_path}")


def _fix_sample_col(df: pd.DataFrame, *, file_path: Path, sheet_name: str) -> pd.DataFrame:
    """Replace proto ``sample_id`` with canonical ``sample_no`` -> ``sample_id``."""

    df = df.copy()
    context = f"{file_path} [{sheet_name}]"

    _require_columns(df, ["sample_no"], context=context)

    if "sample_id" in df.columns:
        df = df.drop(columns=["sample_id"])
        _log("INFO", f"Dropped old sample_id column from {context}.")

    return df.rename(columns={"sample_no": "sample_id"})


def _warn_review_rows(df: pd.DataFrame, *, file_path: Path) -> None:
    """Warn if any utterance rows are marked for review."""

    if "review_row" not in df.columns:
        _log("INFO", f"No review_row column found in {file_path} [utterances].")
        return

    review_flags = pd.to_numeric(df["review_row"], errors="coerce").fillna(0)
    n_review = int((review_flags != 0).sum())

    if n_review:
        _log("WARNING", f"{file_path} has {n_review} utterance row(s) marked for review.")
    else:
        _log("INFO", "No utterance rows are marked for review.")


def _remove_flagged_rows(df: pd.DataFrame, *, file_path: Path) -> pd.DataFrame:
    """Remove rows where ``remove_row`` is nonzero, then drop the flag column."""

    df = df.copy()

    if "remove_row" not in df.columns:
        _log("WARNING", f"No remove_row column found in {file_path}; no rows removed.")
        return df

    remove_flags = pd.to_numeric(df["remove_row"], errors="coerce").fillna(0)
    original_len = len(df)
    df = df.loc[remove_flags == 0].copy()
    df = df.drop(columns=["remove_row"])

    _log("INFO", f"Removed {original_len - len(df)} row(s) flagged by remove_row.")
    return df


def _fix_utterance_col(df: pd.DataFrame, *, file_path: Path) -> pd.DataFrame:
    """Rename ``utterance_edited`` to canonical ``utterance``."""

    df = df.copy()

    if "utterance_edited" in df.columns:
        if "utterance" in df.columns:
            df = df.drop(columns=["utterance"])
            _log("INFO", f"Dropped existing utterance column in favor of utterance_edited: {file_path}")
        return df.rename(columns={"utterance_edited": "utterance"})

    if "utterance" in df.columns:
        _log("WARNING", f"Using existing utterance column; utterance_edited not found: {file_path}")
        return df

    raise TranscriptTableError(
        f"{file_path} [utterances] has neither utterance_edited nor utterance."
    )


def _process_samples(df: pd.DataFrame, *, file_path: Path, cycle: str) -> pd.DataFrame:
    """Clean and reorder the samples sheet for one workbook."""

    df = _fix_sample_col(df, file_path=file_path, sheet_name="samples")
    _require_columns(
        df,
        ["site", "study_id", "test", "narrative"],
        context=f"{file_path} [samples]",
    )

    cycle_num = _extract_cycle_number(cycle)
    df["cycle"] = cycle_num
    df["expanded_sample_id"] = df.apply(
        lambda row: f"C{row['cycle']}_{row['site']}_{row['sample_id']}", axis=1
    )

    df = df.sort_values(by=["sample_id"], kind="stable")
    return df[SAMPLES_KEEP_COLS].reset_index(drop=True)


def _process_utterances(df: pd.DataFrame, *, file_path: Path) -> pd.DataFrame:
    """Clean and reorder the utterances sheet for one workbook."""

    df = _fix_sample_col(df, file_path=file_path, sheet_name="utterances")
    _require_columns(df, ["speaker"], context=f"{file_path} [utterances]")

    _warn_review_rows(df, file_path=file_path)
    df = _remove_flagged_rows(df, file_path=file_path)
    df = _fix_utterance_col(df, file_path=file_path)

    df["utterance_id"] = range(1, len(df) + 1)
    return df[UTTS_KEEP_COLS].reset_index(drop=True)


def process_workbook(
    cycle: str,
    site: str,
    input_root: Path | str = INPUT_DIR,
    output_root: Path | str = OUTPUT_DIR,
) -> Path:
    """Process one cycle/site workbook and return the output workbook path."""

    input_root = Path(input_root)
    output_root = Path(output_root)

    in_file = input_root / cycle / f"{site}_transcript_tables.xlsx"
    out_file = output_root / cycle / site / "transcript_tables.xlsx"

    _log("INFO", f"Processing {cycle} / {site}")
    _log("INFO", f"Input:  {in_file}")
    _log("INFO", f"Output: {out_file}")

    sample_df, utt_df = _read_tt(in_file)
    sample_df = _process_samples(sample_df, file_path=in_file, cycle=cycle)
    utt_df = _process_utterances(utt_df, file_path=in_file)
    _write_tt(out_file, sample_df, utt_df)

    _log("INFO", f"Finished {cycle} / {site}: {len(sample_df)} samples, {len(utt_df)} utterances.")
    return out_file


def process_all_workbooks(
    input_root: Path | str = INPUT_DIR,
    output_root: Path | str = OUTPUT_DIR,
    cycles: Iterable[str] = CYCLES,
    sites: Iterable[str] = SITES,
    continue_on_error: bool = False,
) -> list[Path]:
    """Process all configured cycle/site transcript-table workbooks."""

    _log("INFO", "Starting Stage 3 transcript-table processing.")
    outputs: list[Path] = []
    failures: list[tuple[str, str, str]] = []

    for cycle in cycles:
        _log("INFO", f"Processing {cycle}.")
        for site in sites:
            try:
                outputs.append(
                    process_workbook(
                        cycle=cycle,
                        site=site,
                        input_root=input_root,
                        output_root=output_root,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - batch runner reports all failures.
                failures.append((cycle, site, str(exc)))
                _log("ERROR", f"Failed {cycle} / {site}: {exc}")
                if not continue_on_error:
                    raise

    if failures:
        _log("ERROR", f"Completed with {len(failures)} failure(s):")
        for cycle, site, message in failures:
            _log("ERROR", f"  - {cycle} / {site}: {message}")
        raise RuntimeError(f"Transcript-table processing failed for {len(failures)} partition(s).")

    _log("INFO", "All configured transcript tables processed successfully.")
    return outputs


def main() -> None:
    """CLI entry point for direct script execution."""

    process_all_workbooks()


if __name__ == "__main__":
    main()
