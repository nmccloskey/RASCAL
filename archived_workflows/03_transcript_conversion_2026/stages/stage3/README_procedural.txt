C2-3 Transcript Conversion Project Stage 3 Procedural README
================================

Purpose of Stage 3
------------------

Stage 3 converts the manually reviewed transcript tables from Stage 2 into CHAT files that can be opened and checked in CLAN. In plain terms, this stage takes the reviewed spreadsheet version of each transcript, assigns final utterance IDs, uses DIAAD to auto-generate .cha files, and then places those files into a cleaner folder structure for final review.

This stage sits in the middle of the larger workflow:

1. Stage 1: Collect data and metadata; clean and auto-annotate.
2. Stage 2: Manually review annotations and make sure the transcript content is ready for CHAT conversion.
3. Stage 3: Assign utterance IDs and auto-generate CHAT files.
4. Stage 4: Open the final CHAT files in CLAN and validate them, including Esc + L checking.

Stage 3 is not meant to replace final human review. It is a preparation step. Its job is to reduce the amount of repetitive manual formatting work needed before final CLAN validation.

What goes into Stage 3
----------------------

The input is a set of six reviewed transcript-table Excel files from Stage 2. These represent two treatment cycles and three study sites:

- cycle 2: AC, BU, TU
- cycle 3: AC, BU, TU

At the start of Stage 3, those files are expected in the folder:

    data/00_original_proto_TTs

For example:

    data/00_original_proto_TTs/cycle2/AC_transcript_tables.xlsx
    data/00_original_proto_TTs/cycle3/TU_transcript_tables.xlsx

Each workbook contains a samples sheet and an utterances sheet. The samples sheet contains file-level information, such as site, study ID, test timepoint, and narrative. The utterances sheet contains the reviewed utterance text and speaker information.

What the Stage 3 scripts do
---------------------------

There are three automated steps.

First, the transcript-table processor reformats the Stage 2 workbooks into the simpler structure DIAAD expects. It keeps the important metadata, adds new metadata like cycle & expanded sample identifiers, removes rows that were marked for removal, uses the edited utterance text, and assigns utterance IDs in order. This produces a new transcript_tables.xlsx file for each cycle and site.

Second, the DIAAD runner calls DIAAD's CHAT-generation command on each of those six reformatted transcript-table folders. This is the step that actually produces the .cha files.

Third, the CHAT sorter renames and organizes the generated .cha files. DIAAD's generated filenames include more metadata than we need in the file name. This is a harmless artifact of having added fields like 'cycle' and 'expanded_sample_id' (DIAAD just combines all non-sample_id metadata fields in the .cha file name). The sorter trims those names down to the participant ID, test timepoint, and narrative. For example, a file like:

    C2_TU_5_2_TU_TU58_Maint_CATGrandpa.cha

is renamed as:

    TU58_Maint_CATGrandpa.cha

The sorter then places files into timepoint and participant folders, such as:

    data/03_sorted_files/cycle2/TU/03_Maintenance/TU58/

This makes the final review folders easier to browse and reduces the chance of reviewers opening the wrong file.

What comes out of Stage 3
-------------------------

The main output is a set of sorted CHAT files under:

    data/03_sorted_files

The files are organized first by cycle, then by site, then by timepoint, and finally by participant/study ID. The intended structure is:

    data/03_sorted_files/cycle/site/timepoint/study_id/file.cha

For example:

    data/03_sorted_files/cycle2/TU/03_Maintenance/TU58/TU58_Maint_CATGrandpa.cha

These are the files that should be carried forward into Stage 4.

How this supports Stage 4
-------------------------

Stage 4 is the final CHAT review and validation stage. Reviewers should open the Stage 3 output files in CLAN and run the usual CHAT-format checks, including Esc + L. Stage 4 remains necessary because automatic conversion can create files that are structurally close but not guaranteed to be perfectly valid for every downstream CHAT/CLAN use case.

Stage 3 therefore handles the mechanical work, while Stage 4 confirms that the resulting files are truly ready for analysis, archiving, or additional coding.

Practical notes for reviewers
-----------------------------

The sorted file names intentionally keep only the information we typically keep: participant/study ID, test timepoint, and narrative. The cycle and site are still preserved in the folder path. This means that the full context of a file is represented by both its folder location and its filename.

If an expected file is missing from the sorted output, check whether it was present in the upstream data inputs. If a filename looks strange or was not sorted, it may mean that DIAAD generated a filename that did not contain the expected study ID, timepoint, and narrative pattern.

As a general rule, Stage 3 should be rerun only after the Stage 2 transcript tables have been corrected. Manual edits should generally happen upstream in the reviewed transcript tables, not in the intermediate auto-generated outputs, unless a specific exception has been documented.
