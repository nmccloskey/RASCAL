# Monologic Narrative Transcript Conversion Workflow

> NB: This is a no-data workflow archive and does not include private clinical
> data, identifiable transcripts, or lab-internal spreadsheets.

## Procedural Overview

This archived workflow documents a four-stage, semi-automated protocol for
converting monologic narrative transcript materials from study years 2022 (cycle 2) & 2023 (cycle 3) into
CHAT-formatted files suitable for CLAN validation. For cycles 1 & 4, an ASR-based protocol was applied
and these transcripts were already in CHAT format.

See [DIAAD](https://github.com/nmccloskey/DIAAD/tree/main/docs) documentation for full details about transcript tables.

The workflow was designed as a reproducible protocol artifact rather than a
fully automated data product. Automation was used where it reduced repetitive
formatting work, but human review remained required because the source
transcripts contained inconsistent formatting and transcript-level ambiguities.

The four stages are:

1. Stage 1: collect data and metadata, create proto-transcript tables, and apply
   conservative first-pass auto-annotations.
2. Stage 2: manually review the proto-transcript tables and correct utterance
   text, speaker labels, removal flags, and CHAT annotations.
3. Stage 3: reformat reviewed transcript tables, assign utterance IDs, run
   DIAAD to generate CHAT files, and sort the output files.
4. Stage 4: manually open the generated CHAT files in CLAN and validate them
   with Esc + L.

## Stage Summary

### Stage 1: Proto-Transcript Table Preparation

Stage 1 reads legacy CU coding metadata and utterance workbooks. It writes one
proto-transcript-table workbook per cycle and site, with a `samples` sheet and
an `utterances` sheet.

The automation formats identifiers, maps test timepoints, infers basic speaker
labels, flags likely non-utterance rows, preserves raw utterance text, and
creates an editable `utterance_edited` column with conservative CHAT-oriented
transformations.

The archived source is in:

```text
stages/stage1/src/
```

The active command, when paths are arranged as expected by the archived script,
is:

```bash
cd stages/stage1/src
python main.py
```

Detailed Stage 1 documentation:

```text
stages/stage1/README_procedural.txt
stages/stage1/README_technical.md
```

### Stage 2: Manual Transcript-Table Review

Stage 2 is manual. Reviewers work in the `utterances` sheet of each
proto-transcript-table workbook and edit `utterance_edited` while using
`utterance_raw` as reference.

Reviewers confirm whether each row is a real utterance, correct speaker labels,
split rows when multiple speakers appear, apply CHAT annotations, spell out
numbers, remove trailing hyphens, ensure terminal punctuation, and flag
uncertain rows in `review_row` with notes in `comment_reviewer`.

Detailed Stage 2 documentation:

```text
stages/stage2/README.md
```

### Stage 3: CHAT File Generation and Sorting

Stage 3 takes the manually reviewed Stage 2 transcript tables and prepares
final-review CHAT files. It first reformats the reviewed workbooks into a
DIAAD-ready structure, then runs DIAAD's transcript-to-CHAT command, then sorts
and renames the generated `.cha` files into review folders.

The main Stage 3 command is:

```bash
cd stages/stage3
python -m src.main
```

The Stage 3 pipeline writes sorted CHAT files under:

```text
data/03_sorted_files/<cycle>/<SITE>/<timepoint>/<study_id>/<file>.cha
```

Detailed Stage 3 documentation:

```text
stages/stage3/README_procedural.txt
stages/stage3/README_technical.md
```

### Stage 4: Final CLAN Validation

Stage 4 is manual. Reviewers copy or download the sorted Stage 3 `.cha` files,
open each file in CLAN, run Esc + L, correct any highlighted CHAT errors, and
repeat validation until CLAN reports no errors. Validated files are saved and
uploaded into the matching validated-file folder structure.

Detailed Stage 4 documentation:

```text
stages/stage4/README.md
```

## Workflow Handoff Points

Stage 1 produces proto-transcript-table workbooks for Stage 2 manual review.

Stage 2 produces reviewed proto-transcript-table workbooks for Stage 3
conversion.

Stage 3 produces sorted, unvalidated CHAT files for Stage 4 review.

Stage 4 produces CLAN-validated CHAT files.

## Archive Scope

This archive preserves workflow logic and protocol documentation. It does not
include private source data, identifiable transcript content, lab-internal
working spreadsheets, Teams folders, or generated clinical outputs.

The source code should be read as a reproducibility record for the archived
workflow. Before rerunning against new data, review hard-coded paths, expected
workbook schemas, DIAAD command availability, and CLAN validation requirements.
