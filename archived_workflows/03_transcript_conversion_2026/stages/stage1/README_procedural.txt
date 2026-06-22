C2-3 Transcript Conversion Project Stage 1 Procedural README
================================

Purpose of Stage 1
------------------

Stage 1 creates the first reviewable proto-transcript tables for the Cycle 2
and Cycle 3 monologic narrative transcript conversion workflow.

In plain terms, this stage takes legacy CU coding files and matching metadata,
combines them into DIAAD-style transcript-table workbooks, and applies a
conservative first pass of CHAT-oriented cleanup to each utterance. The result
is not considered final. It is a starting point for human review in Stage 2.

This stage sits at the beginning of the larger workflow:

1. Stage 1: Collect data and metadata; clean and auto-annotate.
2. Stage 2: Manually review annotations and make sure the transcript content is
   ready for CHAT conversion.
3. Stage 3: Assign utterance IDs and auto-generate CHAT files.
4. Stage 4: Open the final CHAT files in CLAN and validate them, including
   Esc + L checking.

Why Stage 1 Is Only Semi-Automated
----------------------------------

The source transcripts used inconsistent formatting, so this stage was designed
to be conservative. The scripts do not try to fully solve CHAT conversion.
Instead, they preserve the original text, create an editable transformed
version, and add flags/comments where automatic transformations were applied.

Human review is expected in Stage 2. Stage 1 is meant to reduce repetitive work
and make likely issues visible.

What Goes Into Stage 1
----------------------

The original active workflow expected one metadata workbook per cycle and three
utterance workbooks per cycle, one for each study site.

The input layout documented in the stage source was:

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

In the archived `main.py`, the active input root was later named:

    stage1_in/cu_files_modified_260330

Those private data folders are not included in this no-data workflow archive.

What the Stage 1 Scripts Do
---------------------------

Stage 1 has two automated preparation steps.

First, the metadata processor reads the cycle metadata workbook and partitions
it by site. For each site, it writes a transcript-table workbook with a
`samples` sheet. It formats sample IDs, participant/study IDs, and test
timepoint labels in the style expected by later DIAAD processing.

Second, the utterance processor reads the legacy utterance files and appends an
`utterances` sheet to each site/cycle workbook. For each row, it:

- infers whether the speaker is the participant (`PAR0`) or clinician (`INV`);
- flags probable timing or metadata rows for removal;
- preserves the original text in `utterance_raw`;
- writes an editable first-pass version to `utterance_edited`;
- adds automatic comments when transformations were applied;
- creates sidecar review logs when Excel-illegal control characters were
  removed.

The automatic text transformations include conservative handling of likely
nonverbal actions, phonemic corrections, unintelligible tokens, disfluencies,
IPA-like slash-delimited forms, phonological fragments, terminal punctuation,
and simple numbers.

What Comes Out of Stage 1
-------------------------

The main output is one proto-transcript-table workbook for each cycle and site.
The source docstrings describe this output structure:

    proto_transcript_tables/
        cycle2/
            AC_transcript_tables.xlsx
            BU_transcript_tables.xlsx
            TU_transcript_tables.xlsx
        cycle3/
            AC_transcript_tables.xlsx
            BU_transcript_tables.xlsx
            TU_transcript_tables.xlsx

Each workbook contains:

- `samples`: metadata and sample-level identifiers;
- `utterances`: raw and editable utterance text, speaker labels, review flags,
  and reviewer/comment columns.

Rows with Excel control-character cleanup may also produce sidecar TSV review
logs, such as:

    cycle2/AC_utterance_review.tsv

How This Supports Stage 2
-------------------------

Stage 2 reviewers should work in the `utterances` sheet, using
`utterance_raw` as reference and correcting `utterance_edited` as needed.
The automatic columns are intended as guidance, not as final judgments.

As a general rule, corrections to transcript content should be made in Stage 2,
not by editing Stage 1 source outputs after they have moved downstream.

Practical Notes for Reviewers
-----------------------------

Automatic annotations should be checked closely. They are intentionally
conservative, but they can still be incomplete or wrong.

Rows marked for removal should be reviewed, especially if a timing or metadata
pattern appears inside a valid utterance. Speaker labels should also be checked,
because automatic clinician detection is based on recognizable text prefixes
such as `CL:` or `Clinician:`.

The presence of an automatic comment does not mean the row is wrong. It means
the row was changed by the first-pass script and deserves attention during
manual review.
