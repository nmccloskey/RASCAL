# Stage 1 Technical README

> NB: This is a no-data workflow archive and does not include private clinical
> data, identifiable transcripts, or lab-internal spreadsheets.

## Purpose

This folder contains the Stage 1 utilities for converting legacy CU coding
metadata and utterance workbooks into reviewable proto-transcript tables for
Stage 2.

The Stage 1 pipeline has two programmatic steps:

1. Convert metadata workbooks into site-partitioned transcript-table `samples`
   sheets.
2. Add `utterances` sheets with conservative first-pass CHAT-oriented cleanup,
   speaker labels, removal flags, and review comments.

The orchestrator is `src/main.py`. The helper modules are:

- `src/sample_sheets.py`
- `src/proto_utterances.py`
- `src/helpers.py`

## Source-Documented Layout

The original script docstrings describe this active-project structure:

```text
c2c3_transcript_conversion/
    sample_sheets.py
    proto_utterances.py
    original_cu_files_copied_2603111/
        c2_coding_files/
            AC_c2_utterances.xlsx
            BU_c2_utterances.xlsx
            TU_c2_utterances.xlsx
        c3_coding_files/
            AC_c3_utterances.xlsx
            BU_c3_utterances.xlsx
            TU_c3_utterances.xlsx
        metadata/
            c2_metadata.xlsx
            c3_metadata.xlsx
    proto_transcript_tables/
        cycle2/
            AC_transcript_tables.xlsx
            BU_transcript_tables.xlsx
            TU_transcript_tables.xlsx
        cycle3/
            AC_transcript_tables.xlsx
            BU_transcript_tables.xlsx
            TU_transcript_tables.xlsx
```

The archived `src/main.py` points to the later active input folder name:

```text
stage1_in/
    cu_files_modified_260330/
        c2_coding_files/
        c3_coding_files/
        metadata/
            c2_metadata.xlsx
            c3_metadata.xlsx
stage1_out/
    proto_transcript_tables_<YYMMDD_HHMM>/
```

Because `main.py` uses `Path.cwd().parent`, the archived script should be run
from `stages/stage1/src` if reusing the paths as written, or the path constants
should be adjusted before reuse.

## Run Command

From `stages/stage1/src`:

```bash
python main.py
```

The script creates a timestamped output directory:

```text
stage1_out/proto_transcript_tables_<YYMMDD_HHMM>/
```

## Inputs and Outputs

### Step 1: `sample_sheets.py`

Inputs:

```text
stage1_in/cu_files_modified_260330/metadata/c2_metadata.xlsx
stage1_in/cu_files_modified_260330/metadata/c3_metadata.xlsx
```

Required metadata columns:

```text
site, sample_no, participant_no, test
```

The implementation also maps `test_id` values to timepoint labels:

```text
1 -> Pre
2 -> Post
3 -> Maint
```

Output:

```text
stage1_out/proto_transcript_tables_<timestamp>/cycle2/<SITE>_transcript_tables.xlsx
stage1_out/proto_transcript_tables_<timestamp>/cycle3/<SITE>_transcript_tables.xlsx
```

Processing behavior:

- Groups metadata by site.
- Builds `sample_id` as site plus a zero-padded three-digit sample number, such
  as `TU005`.
- Builds `study_id` as site plus a zero-padded two-digit participant number,
  such as `TU05`.
- Removes unnamed spreadsheet columns.
- Writes one workbook per cycle/site with a `samples` sheet.

### Step 2: `proto_utterances.py`

Inputs:

```text
stage1_in/cu_files_modified_260330/c2_coding_files/*_utterances.xlsx
stage1_in/cu_files_modified_260330/c3_coding_files/*_utterances.xlsx
```

Required utterance columns:

```text
sample_no, utterance
```

Output:

```text
stage1_out/proto_transcript_tables_<timestamp>/cycle2/<SITE>_transcript_tables.xlsx
stage1_out/proto_transcript_tables_<timestamp>/cycle3/<SITE>_transcript_tables.xlsx
```

The module appends or replaces an `utterances` sheet in each matching workbook.
Rows with Excel-illegal control characters removed may also produce:

```text
stage1_out/proto_transcript_tables_<timestamp>/cycle<cycle>/<SITE>_utterance_review.tsv
```

The `utterances` sheet contains:

```text
sample_no
speaker
utterance_raw
utterance_edited
remove_row
review_row
comment_reviewer
comment_auto
```

Processing behavior:

- Uses `INV` when the raw utterance appears to begin with a clinician label,
  otherwise uses `PAR0`.
- Sets `remove_row` to `1` for likely timing, metadata, or non-utterance rows.
- Preserves the original text in `utterance_raw`.
- Writes the transformed first-pass text to `utterance_edited`.
- Leaves `review_row` initialized to `0` for Stage 2 reviewers to update.
- Leaves `comment_reviewer` blank.
- Writes automatic transformation flags into `comment_auto`.

## Text Transformation Helpers

Most text cleanup logic lives in `src/helpers.py`.

The primary wrapper is `first_pass_transform_utterance()`, which applies these
steps in order:

1. `convert_nonverbals()`
2. `convert_corrections()`
3. `regularize_unintelligibles()`
4. `convert_ipa_slashes()`
5. `convert_phonological_fragments()`
6. `autotag_disfluencies()`
7. `regularize_disfluency_case()`
8. `ensure_terminal_punct()`

After that, `proto_utterances.py` applies `convert_numbers_to_words()` only to
rows not marked for removal.

Transformation examples documented in the source include:

```text
*pointing to fish bowl* -> &=pointing:to:fish:bowl
(gesturing falling down) -> &=ges:falling:down
ricycle {tricycle} -> ricycle [: tricycle] [*]
xx / XXX-style tokens -> XXX
/s.../ -> [= /s.../]
h-home -> &+h home
n-n-not -> &+n &+n not
2 -> two
50% -> fifty percent
```

## Error Handling and Validation

`require_columns()` raises a `ValueError` when an expected workbook column is
missing.

`proto_utterances.py` raises a `FileNotFoundError` when an utterance workbook
does not have a corresponding transcript-table workbook produced by
`sample_sheets.py`.

The scripts remove Excel-illegal control characters before writing workbooks.
When this occurs, they write a sidecar TSV log with the raw utterance,
pre-cleaned transformed utterance, Excel-safe transformed utterance, and
transformation notes.

## Important Maintenance Notes

- The active path constants are hard-coded in `src/main.py`.
- `sample_sheets.py` checks for a `test` column but maps from `test_id`; if the
  input schema changes, review that required-column check before rerunning.
- Site and cycle coverage is inferred from available input workbooks rather than
  a centralized cycle/site constant.
- The first-pass transformations are intentionally conservative and should not
  be treated as a replacement for Stage 2 review.

## Minimal Expected End State

After a successful Stage 1 run, the output directory should contain one workbook
per cycle/site:

```text
cycle2/AC_transcript_tables.xlsx
cycle2/BU_transcript_tables.xlsx
cycle2/TU_transcript_tables.xlsx
cycle3/AC_transcript_tables.xlsx
cycle3/BU_transcript_tables.xlsx
cycle3/TU_transcript_tables.xlsx
```

Each workbook should include both `samples` and `utterances` sheets and should
be ready for Stage 2 manual review.
