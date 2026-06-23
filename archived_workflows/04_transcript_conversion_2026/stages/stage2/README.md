# Stage 2 Transcript Review Protocol

> NB: This README archives the Stage 2 protocol content converted from
> `Stage2_Transcript_Protocol.docx`.

## Purpose

This document describes Stage 2 of 4 in the conversion of Cycle 2 and Cycle 3
monologic narratives into CHAT-formatted transcripts.

At this stage, data exist as proto-transcript tables derived from CU coding
files. The goal is to review automated annotations and ensure all utterances
conform to CHAT standards.

## Pipeline Overview

1. Stage 1: Collect data and metadata; clean and auto-annotate.
2. **Stage 2: Manual review of annotations; ensure CHAT format.**
3. Stage 3: Assign utterance IDs; auto-generate CHAT files.
4. Stage 4: Final CHAT review and validation with Esc + L.

## Notes on Stage 1 Automation

Due to inconsistent formatting in the source transcripts, full automation was
not feasible.

- Automated annotations serve as a first pass for efficiency.
- Human review is required and expected.
- Both original and transformed text are preserved for transparency.

## Setup

There is one Excel file per site by cycle, for six total files.

Each file contains:

- `samples` sheet: metadata. Do not edit.
- `utterances` sheet: all transcript lines. This is the review workspace.

## Utterances Sheet Fields

| Column | Description |
| --- | --- |
| `sample_no` | Connects to sample metadata. |
| `speaker` | `PAR0` for participant or `INV` for clinician. |
| `utterance_raw` | Original text for reference, not editing. |
| `utterance_edited` | Preprocessed version of `utterance_raw` for manual editing. |
| `remove_row` | `0` = keep, `1` = remove. |
| `review_row` | `0` = reviewer is confident, `1` = question or revisit needed. |
| `comment_reviewer` | Reviewer notes. |
| `comment_auto` | List of auto-transformations. |

The `utterance_edited` column contains the automated first pass and should be
corrected as needed. The `utterance_raw` column is included to help check the
auto-transformations.

## Review Protocol

For each row, check that the text in `utterance_edited` meets the four criteria
below.

## 1. Utterance Status: Keep vs Remove

Not all rows are true utterances. Non-utterance rows should be easy to spot.

Examples of lines to mark for removal:

- timestamps, such as `1:33:19 to 1:34:09`;
- initials, such as `JD`;
- empty cells or stray punctuation marks.

Check the `remove_row` column:

- `0`: valid utterance.
- `1`: not an utterance, to be removed in Stage 3.

Speaker label does not affect removal.

## 2. Speaker Attribution

Each utterance must have the correct speaker:

- `PAR0`: participant.
- `INV`: clinician.

Common clinician indicators include `*CL:`, `Cl:`, `*Clinician`, or similar
tags. These tags should be removed in the `utterance_edited` column.

Example:

```text
*Clinician: Mhm -> mhm .
```

If multiple speakers appear in one line, split into separate rows.

Unsegmented:

| sample_no | speaker | utterance_edited |
| --- | --- | --- |
| 25 | PAR0 | and then this one (CL: this one?) yes that one . |

Properly segmented:

| sample_no | speaker | utterance_edited |
| --- | --- | --- |
| 25 | PAR0 | and then this one (CL: this one?) |
| 25 | INV | this one? |
| 25 | PAR0 | yes that one . |

When re-segmenting, which should be relatively rare:

- duplicate `sample_no` across new rows;
- set `remove_row = 0` for all valid utterances;
- leave other columns blank unless needed.

## 3. CHAT Annotation

Apply CHAT-style tags where appropriate.

| Tag | Meaning | Example |
| --- | --- | --- |
| `&-` | Non-word filler | `I &-um &-uh I went home .` |
| `&+` | Fragment | `&+s &+st stuck at home .` |
| `&=` | Nonverbal | `&=laughs that was funny &=points:to:picture .` |
| `[: target] [*]` | Phonemic paraphasia | `the gat [: cat] [*] ran .` |

Multi-word nonverbal descriptions use colons to link words, such as
`&=points:to:picture`. A `ges` tag like `&=ges` or `&=ges:cold` explicitly
indicates a gesture and is often included but not required.

Non-word fillers include `um`, `uh`, `er`, `eh`, and `erm`. The tokens `hmm`,
`mm`, `oh`, and `ah` are considered interjections, meaning true words, and need
no annotation. If incorrectly auto-annotated, remove the tags.

Example:

```text
&-ah I see -> ah I see .
```

## 4. CHAT Formatting

Ensure all utterances follow CHAT formatting rules.

- Terminal punctuation: all utterances must end with `.`, `!`, or `?`.
- Numbers: spell out numbers, such as `two` instead of `2`.
- Hyphens: no trailing hyphens.
- Stage directions: move to the comment column, such as `(exhausted)`.

The auto-expander may incorrectly write out numeric sequences. For example,
`911` may become `nine hundred eleven` instead of `nine one one`.

For hyphens, remove the hyphen from full words, such as `dog-` to `dog`. Convert
fragments or disfluencies accordingly, such as `eh-` to `&-eh`.

## Flagging Uncertain Cases

If unsure:

- set `review_row = 1`;
- describe the issue in `comment_reviewer`;
- continue annotating and flagging rows as needed.

Flagged items will be reviewed separately. Email the lab to resolve
uncertainties.

## Stage 2 Transcript Review Checklist

Before you begin:

- work only in the `utterances` sheet;
- do not edit the `samples` sheet;
- edit text in the `utterance_edited` column.

For each row:

- decide whether this is a real utterance;
- if no, set `remove_row = 1`;
- if yes, ensure `remove_row = 0`;
- fix incorrect speaker labels;
- split rows if multiple speakers appear;
- add `&-` for fillers and `&+` for fragments;
- add `&=` for nonverbal actions;
- add `[: target] [*]` for recognizable phonemic paraphasias;
- end every utterance with `.`, `!`, or `?`;
- spell out numbers;
- remove trailing hyphens;
- move stage directions to `comment_reviewer`.

For questions or issues:

- set `review_row = 1`;
- write the question in `comment_reviewer`;
- continue reviewing and email the lab with inquiries.

Final pass:

- all `utterance_edited` entries end with punctuation;
- formatting is consistent;
- speaker labels are correct;
- all rows have correct `remove_row` and `review_row` values.
