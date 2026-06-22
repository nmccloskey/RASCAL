"""
env: c2c3tt

Convert old CU-coding sheets into DIAAD-style transcript-table utterance sheets.

Expected project structure
--------------------------
OneDrive - Temple University/SLABlab/Projects/c2c3_transcript_conversion/
│   proto_utterances.py
│   sample_sheets.py
│
├───.vscode
│       settings.json
│
├───original_cu_files_copied_2603111
│   ├───c2_coding_files
│   │       AC_c2_utterances.xlsx
│   │       BU_c2_utterances.xlsx
│   │       TU_c2_utterances.xlsx
│   │
│   ├───c3_coding_files
│   │       AC_c3_utterances.xlsx
│   │       BU_c3_utterances.xlsx
│   │       TU_c3_utterances.xlsx
│   │
│   └───metadata
│           c2_metadata.xlsx
│           c3_metadata.xlsx
│
└───proto_transcript_tables
    ├───cycle2
    │       AC_transcript_tables.xlsx
    │       BU_transcript_tables.xlsx
    │       TU_transcript_tables.xlsx
    │
    └───cycle3
            AC_transcript_tables.xlsx
            BU_transcript_tables.xlsx
            TU_transcript_tables.xlsx

            
Stuff to handle:
            
nonverbals - need * or ( plus action word like point/gesture/laugh - not just everything in () or **
*pointing to fish bowl* -> &=pointing:to:fish:bowl
*sighs* -> &=sighs
(yawns) -> &=yawns
(gesturing falling down) -> &=ges:falling:down
(gesturing, imitating angry mother) -> &=ges:imitating:angry:mother
*laughs* (pointing to books falling) -> &=laughs &=pointing:to:books:falling
(indicating done with task) -> &=indicating:done:with:task
[writes something] -> &=writes:something

corrections:
ricycle {tricycle} -> ricycle [: tricycle] [*]

regularize unintelligibles:
xx/XX/xxx -> XXX

case regularize disfluencies:
if tok.startswith('&-'):
    tok = tok.lower()

handle IPA:
/sə/ /əf/ -> [= /sə/] [= /əf/]

phonological fragments - if lowercase (so not 'B-S-E'):
h-home -> &+h home
n-n-not -> &+n &+n not
pr-presents -> &+pr presents

"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from helpers import (
    require_columns,
    first_pass_transform_utterance,
    detect_clinician,
    flag_for_removal,
    strip_excel_illegal_chars,
    convert_numbers_to_words,
)


def _prep_single_utterance_sheet(
    utt_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an utterances sheet plus a review log.

    Returns
    -------
    utterances_df : pd.DataFrame
        Excel-safe utterance sheet for transcript-table workbooks.
    review_df : pd.DataFrame
        Sidecar review log for rows where illegal Excel characters were removed.
    """
    utterance_rows: list[dict] = []
    review_rows: list[dict] = []

    for _, row in utt_df.iterrows():
        raw_utt = row["utterance"]

        speaker = "INV" if detect_clinician(raw_utt) else "PAR0"
        remove = flag_for_removal(raw_utt)

        transform_result = first_pass_transform_utterance(raw_utt)

        text = transform_result.transformed
        flags = list(transform_result.flags)
        notes = list(transform_result.notes)

        # Apply num2words only for rows not tagged for auto-removal
        if remove == 0:
            text, step_flags, step_notes = convert_numbers_to_words(text)
            flags.extend(step_flags)
            notes.extend(step_notes)

        auto_utt_pre_excel = text
        auto_utt_excel, n_illegal = strip_excel_illegal_chars(auto_utt_pre_excel)

        comment_parts: list[str] = []

        if flags:
            comment_parts.append(
                f"{', '.join(sorted(set(flags)))}"
            )

        if n_illegal > 0:
            comment_parts.append(
                f"Removed {n_illegal} illegal Excel control character(s); check against raw text."
            )

            review_rows.append(
                {
                    "sample_no": row["sample_no"],
                    "raw_utterance": raw_utt,
                    "utterance_auto_before_excel_cleaning": auto_utt_pre_excel,
                    "utterance_auto_written_to_excel": auto_utt_excel,
                    "illegal_char_count": n_illegal,
                    "transform_flags": "; ".join(flags) if flags else None,
                    "transform_notes": " | ".join(notes) if notes else None,
                }
            )

        comment = " | ".join(comment_parts) if comment_parts else None

        utterance_rows.append(
            {
                "sample_no": row["sample_no"],
                "speaker": speaker,
                "utterance_raw": raw_utt,
                "utterance_edited": auto_utt_excel,
                "remove_row": remove,
                "review_row": 0,
                "comment_reviewer": "",               
                "comment_auto": comment,
            }
        )

    utterances_df = pd.DataFrame(utterance_rows)
    review_df = pd.DataFrame(review_rows)
    return utterances_df, review_df


def prep_utterance_sheets(utt_dir: Path, out_dir: Path, cycle: str) -> None:
    """
    Add utterances sheets to existing transcript-table workbooks.

    For each *_utterances.xlsx file:
    - read old utterance data
    - infer speaker
    - flag probable metadata/timing rows for removal
    - create a conservative first-pass transformed utterance
    - strip Excel-illegal control chars from the transformed utterance
    - write utterances sheet into the corresponding transcript-table workbook
    - write a sidecar TSV review log for rows altered during Excel sanitization
    """
    for utt_file in sorted(utt_dir.rglob("*_utterances.xlsx")):
        site = utt_file.stem[:2]

        utt_df = pd.read_excel(utt_file)
        require_columns(utt_df, ["sample_no", "utterance"], utt_file)

        utterances_df, review_df = _prep_single_utterance_sheet(utt_df)

        tt_file = out_dir / f"cycle{cycle}" / f"{site}_transcript_tables.xlsx"
        if not tt_file.exists():
            raise FileNotFoundError(
                f"Transcript table workbook does not exist yet: {tt_file}"
            )

        with pd.ExcelWriter(
            tt_file,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            utterances_df.to_excel(writer, sheet_name="utterances", index=False)

        print(f"Wrote utterances sheet in {tt_file}")

        if not review_df.empty:
            review_file = out_dir / f"cycle{cycle}" / f"{site}_utterance_review.tsv"
            review_df.to_csv(review_file, sep="\t", index=False, encoding="utf-8")
            print(f"Wrote review log: {review_file}")
