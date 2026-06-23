# Stage 4 CHAT Validation Protocol

> NB: This README archives the Stage 4 protocol content converted from
> `Stage4_Transcript_Protocol.docx`.

## Purpose

This document describes Stage 4 of 4 in the conversion of Cycle 2 and Cycle 3
monologic narratives into CHAT-formatted transcripts.

At this stage, data exist as unvalidated CHAT files from Stage 3. The goal is
to manually check each file and make sure the Esc + L command in CLAN returns
no errors.

## Pipeline Overview

1. Stage 1: Collect data and metadata; clean and auto-annotate.
2. Stage 2: Manual review of annotations; ensure CHAT format.
3. Stage 3: Assign utterance IDs; auto-generate CHAT files.
4. **Stage 4: Final CHAT review and validation with Esc + L.**

## Notes on Stage 3 Automation

The proto-transcript tables were cleaned and reformatted, and the utterances
edited in Stage 2 were written into separate CHAT files.

- Automation serves as a first pass for efficiency.
- Human review is required and expected.
- Both original and transformed text are preserved for transparency.

## Setup

The active Stage 4 folder was located on Teams in:

```text
Data and Data Analysis/revisions/c2c3_monolog_transcript_conversion/
```

Stage 3 final output is a set of sorted CHAT files copied to:

```text
stage4/data/00_sorted_files
```

The files are organized first by cycle, then by site, then by timepoint, and
finally by study ID or participant. The intended structure is:

```text
stage4/data/00_sorted_files/cycle/site/timepoint/study_id/file.cha
```

For example:

```text
stage4/data/00_sorted_files/cycle2/TU/03_Maintenance/TU58/TU58_Maint_CATGrandpa.cha
```

Within `stage4/data/00_validated_files` are the same subfolders, but with no
`.cha` files before validation/upload.

## Procedure

1. Navigate to `stage4/data/00_sorted_files/` and then to a specific participant
   folder, such as `TU75`.
2. Download the folder as a ZIP file from Teams.
3. Extract the files. Working from the Downloads folder is acceptable.
4. Open each file in CLAN.
5. Press Esc + L.
6. If CLAN indicates an error, correct it. CLAN will highlight the problem
   area.
7. Repeat Esc + L and correction until CLAN returns a success/no-errors message.
8. Save the file.
9. Navigate to `stage4/data/00_validated_files` through the same subfolders to
   the participant folder that was downloaded.
10. Upload all validated `.cha` files.

Example:

```text
Downloaded and validated:
stage4/data/00_sorted_files/cycle2/TU/01_PreTx/TU75

Upload to:
stage4/data/00_validated_files/cycle2/TU/01_PreTx/TU75
```
