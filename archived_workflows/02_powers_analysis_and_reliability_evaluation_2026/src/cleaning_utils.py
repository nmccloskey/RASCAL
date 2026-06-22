from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .revision_utils import RevisionRecorder


META_COLS = [
    "sample_id",
    "utterance_id",
    "speaker",
]

INT_COLS_BASES = [
    "speech_units",
    "content_words",
    "num_nouns",
    "filled_pauses",
    "collab_repair",
]

ANALYSIS_COLS = ["c2_turn_type"] + [f"c2_{col}" for col in INT_COLS_BASES]
REL_EVAL_COLS = ["c3_turn_type"] + [f"c3_{col}" for col in INT_COLS_BASES]
UNTAGGED_COLS = ["turn_type"] + INT_COLS_BASES

TURN_TYPES = {"T", "MT", "ST", "NV"}


@dataclass
class ValidationIssue:
    severity: str
    file: str
    row_number: int | None
    column: str | None
    value: object
    message: str


class DataValidationError(ValueError):
    """Raised when a file contains data that should be manually corrected."""


def _display_file(file_path: Path | str | None) -> str:
    return str(file_path) if file_path is not None else "<dataframe>"


def _excel_row_numbers(index: pd.Index) -> pd.Series:
    """
    Convert pandas zero-based row indices to Excel row numbers.

    Assumes row 1 is the header, so the first data row is Excel row 2.
    """
    return pd.Series(index, index=index) + 2


def _log_issue(
    issue: ValidationIssue,
    logger: logging.Logger,
    revisions: RevisionRecorder | None = None,
    new_value: object = None,
    coded_nonverbal: int | None = None,
) -> None:
    location = f"file={issue.file}"
    if issue.row_number is not None:
        location += f", row={issue.row_number}"
    if issue.column is not None:
        location += f", column={issue.column}"

    msg = f"{location}, value={issue.value!r}: {issue.message}"

    if issue.severity.upper() == "ERROR":
        logger.error(msg)
    else:
        logger.warning(msg)

    if revisions is not None:
        revisions.add(
            file=issue.file,
            row=issue.row_number,
            column=issue.column,
            old_value=issue.value,
            new_value=new_value,
            severity=issue.severity,
            message=issue.message,
            coded_nonverbal=coded_nonverbal,
        )


def _coded_nonverbal(df: pd.DataFrame, idx: object) -> int | None:
    """
    Return 1 when the row's normalized turn_type is NV, 0 otherwise.
    """
    if "turn_type" not in df.columns or idx not in df.index:
        return None

    value = df.at[idx, "turn_type"]

    try:
        if pd.isna(value):
            return 0
    except TypeError:
        pass

    return int(str(value).strip().upper() == "NV")


def _require_columns(
    df: pd.DataFrame,
    keep_cols: list[str],
    file_path: Path | str | None,
    logger: logging.Logger,
    revisions: RevisionRecorder | None = None,
) -> None:
    missing = [col for col in keep_cols if col not in df.columns]

    if missing:
        issue = ValidationIssue(
            severity="ERROR",
            file=_display_file(file_path),
            row_number=None,
            column=None,
            value=missing,
            message="Missing required columns.",
        )
        _log_issue(issue, logger, revisions=revisions)
        raise DataValidationError(
            f"{_display_file(file_path)} is missing required columns: {missing}"
        )


def _filter_cols(
    df: pd.DataFrame,
    keep_cols: list[str],
    file_path: Path | str | None,
    logger: logging.Logger,
    revisions: RevisionRecorder | None = None,
) -> pd.DataFrame:
    _require_columns(df, keep_cols, file_path, logger, revisions=revisions)
    return df.loc[:, keep_cols].copy()


def _rename_cols(df: pd.DataFrame, old_cols: list[str]) -> pd.DataFrame:
    rename_map = dict(zip(old_cols, UNTAGGED_COLS))
    return df.rename(columns=rename_map).copy()


def _ensure_turn_types(
    df: pd.DataFrame,
    file_path: Path | str | None,
    logger: logging.Logger,
    revisions: RevisionRecorder | None = None,
) -> tuple[pd.DataFrame, bool]:
    processed_df = df.copy()

    processed_df["turn_type"] = processed_df["turn_type"].astype("string").str.strip()

    blank_mask = processed_df["turn_type"].isna() | processed_df["turn_type"].eq("")
    nonblank_mask = ~blank_mask

    invalid_mask = nonblank_mask & ~processed_df["turn_type"].isin(TURN_TYPES)
    invalid_rows = processed_df.loc[invalid_mask, "turn_type"]

    if invalid_rows.empty:
        return processed_df, False

    excel_rows = _excel_row_numbers(invalid_rows.index)

    for idx, value in invalid_rows.items():
        issue = ValidationIssue(
            severity="ERROR",
            file=_display_file(file_path),
            row_number=int(excel_rows.loc[idx]),
            column="turn_type",
            value=value,
            message=f"Invalid turn type. Expected blank or one of {sorted(TURN_TYPES)}.",
        )
        _log_issue(
            issue,
            logger,
            revisions=revisions,
            coded_nonverbal=_coded_nonverbal(processed_df, idx),
        )

    return processed_df, True


def _is_nonclient_row(df: pd.DataFrame) -> pd.Series:
    speaker = df["speaker"].astype("string").str.strip().str.upper()
    # return speaker.str.startswith("INV", na=False)
    return speaker.str.startswith(("INV", "PAR1"), na=False)


def _coerce_int_columns(
    df: pd.DataFrame,
    file_path: Path | str | None,
    logger: logging.Logger,
    revisions: RevisionRecorder | None = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Coerce POWERS count columns with context-sensitive blank handling.

    Blank policy:
    - turn_type:
        handled separately; blanks allowed, remain blank

    - collab_repair:
        blanks allowed, remain blank

    - speech_units:
        blanks are interpreted as 0 and recorded as WARNING rows

    - content_words, num_nouns, filled_pauses:
        blanks are allowed for INV* & PAR1 speaker rows
        blanks are interpreted as 0 and recorded as WARNING rows for
        non-INV speaker rows

    Autocorrection:
    - "o" / "O" are interpreted as 0 and recorded as WARNING rows
      in the revisions table.
    - Count-column blanks that are not clinician-only blanks are interpreted
      as 0 and recorded as WARNING rows in the revisions table.
    """
    processed_df = df.copy()
    file_label = _display_file(file_path)
    has_errors = False

    clinician_mask = _is_nonclient_row(processed_df)

    blank_always_allowed_cols = {"collab_repair"}
    blank_zero_all_speakers_cols = {"speech_units"}
    blank_zero_non_inv_cols = {"content_words", "num_nouns", "filled_pauses"}

    for col in INT_COLS_BASES:
        original = processed_df[col]
        normalized = original.copy()

        o_zero_mask = normalized.astype("string").str.strip().str.lower().eq("o")

        if o_zero_mask.any():
            excel_rows = _excel_row_numbers(normalized.index)

            for idx in normalized[o_zero_mask].index:
                issue = ValidationIssue(
                    severity="WARNING",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message='Interpreting "o"/"O" as zero.',
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=0,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            normalized.loc[o_zero_mask] = 0

        blank_mask = normalized.isna() | normalized.astype("string").str.strip().eq("")

        if col in blank_always_allowed_cols:
            allowed_blank_mask = blank_mask
            blank_zero_mask = pd.Series(False, index=processed_df.index)
            disallowed_blank_mask = pd.Series(False, index=processed_df.index)

        elif col in blank_zero_non_inv_cols:
            allowed_blank_mask = blank_mask & clinician_mask
            blank_zero_mask = blank_mask & ~clinician_mask
            disallowed_blank_mask = pd.Series(False, index=processed_df.index)

        elif col in blank_zero_all_speakers_cols:
            allowed_blank_mask = pd.Series(False, index=processed_df.index)
            blank_zero_mask = blank_mask
            disallowed_blank_mask = pd.Series(False, index=processed_df.index)

        else:
            allowed_blank_mask = pd.Series(False, index=processed_df.index)
            blank_zero_mask = pd.Series(False, index=processed_df.index)
            disallowed_blank_mask = blank_mask

        if blank_zero_mask.any():
            excel_rows = _excel_row_numbers(normalized.index)

            for idx in normalized[blank_zero_mask].index:
                issue = ValidationIssue(
                    severity="WARNING",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message="Interpreting blank as zero.",
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=0,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            normalized.loc[blank_zero_mask] = 0

        if disallowed_blank_mask.any():
            excel_rows = _excel_row_numbers(normalized.index)

            for idx in normalized[disallowed_blank_mask].index:
                issue = ValidationIssue(
                    severity="ERROR",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message="Blank value is not allowed for this row/column.",
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=None,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            has_errors = True

        normalized.loc[allowed_blank_mask] = pd.NA

        candidate_mask = ~allowed_blank_mask & ~disallowed_blank_mask
        numeric = pd.to_numeric(normalized.loc[candidate_mask], errors="coerce")

        invalid_numeric_mask = numeric.isna()

        if invalid_numeric_mask.any():
            excel_rows = _excel_row_numbers(numeric.index)

            for idx in numeric[invalid_numeric_mask].index:
                issue = ValidationIssue(
                    severity="ERROR",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message="Could not coerce value to a number.",
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=None,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            has_errors = True

        valid_numeric = numeric.loc[~invalid_numeric_mask]

        non_integer_mask = valid_numeric.mod(1).ne(0)

        if non_integer_mask.any():
            excel_rows = _excel_row_numbers(valid_numeric.index)

            for idx in valid_numeric[non_integer_mask].index:
                issue = ValidationIssue(
                    severity="ERROR",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message="Numeric value is not an integer.",
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=None,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            has_errors = True

        integer_numeric = valid_numeric.loc[~non_integer_mask]

        negative_mask = integer_numeric.lt(0)

        if negative_mask.any():
            excel_rows = _excel_row_numbers(integer_numeric.index)

            for idx in integer_numeric[negative_mask].index:
                issue = ValidationIssue(
                    severity="ERROR",
                    file=file_label,
                    row_number=int(excel_rows.loc[idx]),
                    column=col,
                    value=original.loc[idx],
                    message="Negative counts are not allowed.",
                )
                _log_issue(
                    issue,
                    logger,
                    revisions=revisions,
                    new_value=None,
                    coded_nonverbal=_coded_nonverbal(processed_df, idx),
                )

            has_errors = True

        valid_values = integer_numeric.loc[~negative_mask]

        processed_df[col] = pd.Series(pd.NA, index=processed_df.index, dtype="Int64")
        processed_df.loc[valid_values.index, col] = valid_values.astype("Int64")

    return processed_df, has_errors


def process_file(
    df: pd.DataFrame,
    reliability: bool = False,
    file_path: Path | str | None = None,
    logger: logging.Logger | None = None,
    revisions: RevisionRecorder | None = None,
) -> pd.DataFrame:
    """
    Prepare one POWERS coding file for DIAAD analysis.
    """
    logger = logger or logging.getLogger(__name__)

    old_cols = REL_EVAL_COLS if reliability else ANALYSIS_COLS
    keep_cols = META_COLS + old_cols

    logger.info(
        "Processing file=%s as %s.",
        _display_file(file_path),
        "reliability" if reliability else "analysis",
    )

    filtered_df = _filter_cols(
        df,
        keep_cols,
        file_path=file_path,
        logger=logger,
        revisions=revisions,
    )

    renamed_df = _rename_cols(filtered_df, old_cols)

    tt_checked_df, has_turn_type_errors = _ensure_turn_types(
        renamed_df,
        file_path=file_path,
        logger=logger,
        revisions=revisions,
    )

    int_checked_df, has_int_errors = _coerce_int_columns(
        tt_checked_df,
        file_path=file_path,
        logger=logger,
        revisions=revisions,
    )

    if has_turn_type_errors or has_int_errors:
        raise DataValidationError(
            f"{_display_file(file_path)} contains validation errors. "
            "See revisions workbook for all detected issues."
        )

    return int_checked_df
