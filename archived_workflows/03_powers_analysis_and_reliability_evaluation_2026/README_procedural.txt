May 14, 2026

-----
TL;DR
-----

NB: This is a no-data workflow archive and does not include private clinical data, identifiable transcripts, or lab-internal spreadsheets. Furthermore, it was preserved for reproducibility and documentation and is not guaranteed to generalize outside the original data structure.

Timestamped powers_cleaning_revisions_YYYYMMDD_HHMMSS.xlsx files under revisions/ contain auto-tabulated warnings/errors from processing raw manual POWERS coding with the below blank-handling policy.

This is part of the data prep for POWERS coding analysis and reliability evaluation.

Manual coders should check the revisions table and make any necessary changes in the raw data files.


-----------------------------------------------
POWERS Coding Preparation Mini-Program Overview
-----------------------------------------------

Expected project layout

project_root/
├── config/
├── data/
│   ├── 00_raw_files/
│   │   ├── cycle2/
│   │   │   ├── AC/
│   │   │   ├── BU/
│   │   │   └── TU/
│   │   ├── cycle3/
│   │   │   ├── AC/
│   │   │   ├── BU/
│   │   │   └── TU/
│   │   └── cycle4/
│   │       ├── AC/
│   │       ├── BU/
│   │       └── TU/
│   ├── 01_cleaned_files/
│   └── 02_diaad_output/
├── logs/
└── src/
    ├── aggregation_utils.py
    ├── cleaning_utils.py
    ├── diaad_running.py
    ├── file_cleaning.py
    ├── logging_utils.py
    ├── main.py
    ├── revision_utils.py
    └── __init__.py

Purpose

This mini-program (in src) helps prepare manually coded POWERS data for DIAAD analysis. It is not intended to replace manual review. Instead, it supports a semi-automated workflow: the scripts clean and check the coding files, identify cells that may need correction, allow coders to revise the files, and then help run DIAAD only after the data are ready.

Why this is needed

The POWERS coding files come from multiple study cycles and sites. Each cycle/site folder may contain primary coding files and reliability coding files. These files need to be standardized before DIAAD can analyze them. For example, coder-specific columns may need to be renamed, numeric columns may need to be checked, and sample identifiers may need to be made unique across the full dataset.

The scripts help catch common problems before analysis. For example, they can flag non-numeric entries in count columns, unexpected turn-type labels, or likely zero values entered as blanks or as the letter "O".

Overall workflow

The intended workflow is:

1. Place raw POWERS coding files in the raw data folder.
2. Run the cleaning script with the rdt conda environment.
3. Review the timestamped revisions workbook created by the script, if one is created.
4. Manually correct the source coding files as needed.
5. Re-run the cleaning script.
6. Once a cleaning run produces no revision rows, cleaned files and aggregate files are written and the DIAAD workflow can be run.

This means that cleaning and DIAAD analysis are intentionally separated - important because the first cleaning pass may reveal issues that require human judgment or manual correction before analysis should proceed.

What the cleaning step does:

The cleaning step reads the raw POWERS Excel files and creates standardized cleaned files. It checks that the expected columns are present, renames the coding columns into the format DIAAD expects, and checks whether values are appropriate for analysis.

Some blanks are acceptable and should remain blank. For example, blank values in turn type and collaborative repair are allowed. Blank values in content words, nouns, and filled pauses are also allowed for non-client rows, because these are not coded for clinicians or non-study participants. For count data that should be present, blanks are interpreted as zero and logged as warnings.

    Explicit blank/autocorrection policy:

    - turn_type & collab_repair:
        blanks allowed, remain blank
        (handled in DIAAD analysis)

    - speech_units:
        blanks are interpreted as 0 for all rows and logged as warnings

    - content_words, num_nouns, filled_pauses:
        blanks are allowed for INV* & PAR1 speaker rows
        blanks are interpreted as 0 for non-INV/PAR1 rows and logged as warnings

    Autocorrection:
    - "o" / "O" are interpreted as 0 and recorded as WARNING rows in the revisions table.
    - Count-column blanks inferred as 0 are recorded as WARNING rows in the revisions table.

    Within each file, validation reports all checkable issues it can find before marking that file as failed. For example, errors in one count column do not prevent later checks of content_words, num_nouns, or filled_pauses. Missing required columns remain fail-fast because later validation depends on those columns existing.

By default, the script writes cleaned versions of the files to the cleaned data folder only when no revision rows are found. If revision rows are found, it writes only the timestamped revisions workbook under revisions/ and does not refresh cleaned files or aggregates. Use --force-write-cleaned only when you intentionally want cleaned files despite revision rows.

What the revisions workbook does:

The revisions workbook is meant to make the manual correction process easier. It records the file, row, coded_nonverbal, column, old value, severity, and a message explaining the issue. The coded_nonverbal field is 1 when that row's turn_type is NV and 0 otherwise, which helps sort zero-inferred warnings by nonverbal versus verbal turns. On 5/14/26, we manually added columns: comments & transcript_edit_reqd. The latter binary field indicates if 1, that the revision affects not just POWERS coding, but transcription as well. If 0, the revision is a self-contained issue requiring no changes in the 'speaker' or 'utterance' columns. Transcript edits should be conducted according to the protocol on Teams under Data and Data Analysis/revisions/coding_revision_protocol.docx.

Typical commands:

conda run -n rdt python -m src.main clean
conda run -n rdt python -m src.main clean --no-aggregate
conda run -n rdt python -m src.main clean --force-write-cleaned
conda run -n rdt python -m src.main run-diaad --dry-run

Rows marked as errors generally need manual correction before analysis. Rows marked as warnings may indicate that the script made an automatic correction, such as interpreting “O” as zero, but these may still be worth reviewing.

What the aggregation step does:

After cleaned files are created, the scripts can also create pooled aggregate files. These files combine data across cycles and sites. The aggregate files add metadata such as cycle, site, and source file. They also create an expanded sample identifier, such as C2_TU_S003, so that samples are uniquely identified across the full dataset rather than only within one cycle/site folder.

The aggregate outputs include one reconciled primary coding file for analysis, one frozen/reliability-evaluation primary coding file that swaps in the cycle 2 frozen coding files where available, one reliability coding file, and one sample metadata file.

The aggregate files are useful for global reliability evaluation and downstream statistical analysis.

What the DIAAD-running step does:

After revision and re-cleaning are complete, a separate DIAAD-running command can call DIAAD on the cleaned files. This can be done for the separate cycle/site partitions and/or for the pooled aggregate files. For aggregate reliability evaluation, DIAAD is called once with the frozen primary aggregate and once with the reconciled primary aggregate so the results can be compared.

This DIAAD-running step is kept separate from cleaning so that analysis is not accidentally run on files that still need manual correction.

Record-keeping value:

The mini-program supports record-keeping in several ways. It preserves the raw files, writes cleaned versions separately, records validation issues in a revisions workbook, writes timestamped logs, and organizes DIAAD outputs in a separate output folder. This makes it easier to document what was checked, what needed revision, and what files were ultimately used for analysis.

In short, these scripts automate the repetitive and error-prone parts of file preparation while preserving the human review needed for responsible manual coding workflows.


----------------
Input File Notes
----------------

The input tree is below. For cycle 2, POWERS coding was 'frozen' for reliability evaluation before coders reconciled their annotations. For POWERS coding analysis, the *powers_coding.xlsx version is used; for reliability evaluation, the *powers_coding_for_rel_eval*.xlsx version. The aggregate workflow preserves the same distinction by writing a separate frozen primary aggregate for reliability evaluation.

For cycles 3 & 4, we have just the full coding datasets plus reliability subsets.

All files containing "reliability" are the reliability subsets.


00_raw_files
│  
├───cycle2
│   ├───AC
│   │       AC_powers_coding.xlsx
│   │       AC_powers_coding_for_rel_eval_260407.xlsx
│   │       AC_powers_reliability_coding.xlsx
│   │
│   ├───BU
│   │       BU_powers_coding.xlsx
│   │       BU_powers_coding_for_rel_eval_260402.xlsx
│   │       BU_powers_reliability_coding.xlsx
│   │
│   └───TU
│           TU_powers_coding.xlsx
│           TU_powers_coding_for_rel_eval_260406.xlsx
│           TU_powers_reliability_coding.xlsx
│
├───cycle3
│   ├───AC
│   │       AC_powers_coding_2023.xlsx
│   │       AC_powers_reselected_reliability_coding.xlsx
│   │
│   ├───BU
│   │       BU_powers_coding_2023.xlsx
│   │       BU_powers_reselected_reliability_coding.xlsx
│   │
│   └───TU
│           TU_powers_coding_2023.xlsx
│           TU_powers_reselected_reliability_coding.xlsx
│
└───cycle4
    ├───AC
    │       AC_powers_coding_2024.xlsx
    │       AC_powers_reselected_reliability_coding.xlsx
    │
    ├───BU
    │       BU_powers_coding_2024.xlsx
    │       BU_powers_reselected_reliability_coding.xlsx
    │
    └───TU
            TU_powers_coding_2024.xlsx
            TU_powers_reselected_reliability_coding.xlsx
