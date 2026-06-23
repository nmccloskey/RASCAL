# POWERS Coding Preparation Mini-Program

This mini-program prepares manually coded POWERS Excel files for DIAAD analysis. It is designed for a semi-automated workflow in which raw manual coding files are cleaned, validation issues are reviewed and corrected by human coders, cleaned files are regenerated, and DIAAD is run only after the data are ready.

> NB: This is a no-data workflow archive and does not include private clinical data, identifiable transcripts, or lab-internal spreadsheets.

## Purpose

The scripts support three related tasks:

1. **Clean partition-level POWERS coding files**
   - Reads raw Excel files from `data/00_raw_files/`
   - Keeps and standardizes the columns DIAAD needs
   - Renames coder-specific columns such as `c2_speech_units` or `c3_speech_units` to untagged names such as `speech_units`
   - Applies context-sensitive data validation and type coercion
   - Writes cleaned Excel files to `data/01_cleaned_files/` only when no revision rows are found, unless explicitly forced

2. **Create a revisions workbook**
   - Records validation errors and automatic corrections in a structured `.xlsx` file
   - Identifies the file, row, whether that row was coded nonverbal, column, old value, suggested/new value when applicable, severity, and message
   - Helps coders manually correct source files before re-running cleaning
   - Writes timestamped revisions workbooks to `revisions/`

3. **Aggregate cleaned data for pooled analysis**
   - Adds metadata from folder structure, such as cycle and site
   - Creates globally unique sample IDs such as `C2_TU_S003`
   - Writes separate aggregate Excel files for sample metadata, reconciled primary coding, frozen reliability-evaluation primary coding, and reliability coding
   - Keeps aggregate files separate because DIAAD POWERS analysis currently expects file-level inputs rather than sheet-specific workbook inputs

DIAAD execution is intentionally separated from cleaning because the workflow includes an intermediate manual revision step.

---

## Expected project layout

```text
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
```

The project root also contains `revisions/`, where timestamped cleaning revision workbooks are written when the cleaner finds rows needing review.

---

## Raw input file conventions

The cleaner expects Excel files organized by cycle and site. For example:

```text
data/00_raw_files/cycle2/AC/AC_powers_coding.xlsx
data/00_raw_files/cycle2/AC/AC_powers_coding_for_rel_eval_260407.xlsx
data/00_raw_files/cycle2/AC/AC_powers_reliability_coding.xlsx
```

Primary coding files are treated as analysis files. Reliability files are identified by `reliability` in the filename. Primary reliability-evaluation subset files, such as `*_coding_for_rel_eval_*`, are treated as primary/c2 subset files and are not treated as reliability/c3 files.

---

## Key scripts

### `main.py`

Provides the command-line entry point for the mini-program. It should dispatch to either cleaning or DIAAD execution, rather than running both automatically.

Recommended workflow:

```bash
conda run -n rdt python -m src.main clean
# Review and manually correct issues listed in revisions workbook
conda run -n rdt python -m src.main clean
conda run -n rdt python -m src.main run-diaad
```

### `file_cleaning.py`

Coordinates the cleaning process:

- Collects raw Excel files
- Sorts files into primary and reliability groups
- Reads raw files
- Calls `process_file()` from `cleaning_utils.py`
- Stages cleaned files in memory
- Writes a timestamped revisions workbook when revision rows exist
- Writes cleaned files only when the revisions workbook would be empty, unless `force_write_cleaned=True` or `--force-write-cleaned` is used
- Optionally runs aggregation after cleaned files are written

### `cleaning_utils.py`

Handles column selection, renaming, validation, and type coercion.

Important validation behavior:

- `turn_type`
  - Valid nonblank values: `T`, `MT`, `ST`, `NV`
  - Blank values are allowed and remain blank

- `collab_repair`
  - Blank values are allowed and remain blank

- `speech_units`
  - Blank values are interpreted as `0` in all rows
  - This inference is logged as a warning

- `content_words`, `num_nouns`, `filled_pauses`
  - Blank values are allowed for rows where `speaker` starts with `INV` or `PAR1`
  - Blank values are interpreted as `0` for non-`INV`/`PAR1` rows
  - This inference is logged as a warning

- Count columns
  - Integer-like values are coerced to integers
  - `o` and `O` are interpreted as `0` and logged as warnings
  - Count-column blanks inferred as `0` are logged as warnings
  - Non-numeric values, non-integer numeric values, and negative values are logged as errors

Within each file, validation reports all checkable issues it can find before marking that file as failed. For example, errors in one count column do not prevent later checks of `content_words`, `num_nouns`, or `filled_pauses`. Missing required columns remain fail-fast because later validation depends on those columns existing.

### `revision_utils.py`

Collects validation issues during cleaning and writes a structured revisions workbook. The revisions workbook is meant to help manual coders find and correct problematic cells.

Default output when revision rows exist:

```text
revisions/powers_cleaning_revisions_YYYYMMDD_HHMMSS.xlsx
```

Suggested columns:

```text
date | file | row | coded_nonverbal | column | old_value | new_value | severity | message
```

`coded_nonverbal` is a binary 0/1 helper column derived from that row's cleaned `turn_type`; it is `1` when `turn_type` is `NV` and `0` otherwise. File-level issues that do not refer to a specific row leave this field blank.

### `aggregation_utils.py`

Aggregates cleaned partition-level files into pooled files under:

```text
data/01_cleaned_files/aggregated/
```

Default aggregate outputs:

```text
powers_sample_metadata_aggregated.xlsx
powers_primary_coding_aggregated.xlsx
powers_primary_coding_for_rel_eval_aggregated.xlsx
powers_reliability_coding_aggregated.xlsx
```

The canonical primary aggregate uses the reconciled `*_powers_coding.xlsx`
versions for analysis. The reliability-evaluation primary aggregate uses the
cycle 2 frozen `*_powers_rel_eval_coding*.xlsx` versions where available, while
keeping the canonical primary files for other cycle/site bins.

The aggregate files include metadata derived from folder structure:

- `cycle`
- `site`
- `source_file`
- `source_path`
- `file_role`
- `expanded_sample_id`

The `expanded_sample_id` is designed to be unique across the full dataset. For example:

```text
cycle2 + TU + S003 -> C2_TU_S003
```

### `diaad_running.py`

Builds and runs DIAAD subprocess calls on cleaned files.

It supports:

- Partition-level DIAAD runs for each cycle × site folder
- Aggregate-level DIAAD runs using pooled files in `data/01_cleaned_files/aggregated/`
- Aggregate-level reliability evaluation against both frozen and reconciled primary aggregates for comparison
- Dry runs
- Logging of subprocess commands, standard output, and errors

Aggregate-level DIAAD calls anticipate global DIAAD config settings such as:

```yaml
sample_id_column: sample_id
utterance_id_column: utterance_id
```

For aggregate runs, these can be overridden as:

```bash
--set sample_id_column=expanded_sample_id
--set utterance_id_column=utterance_id
```

---

## Example CLI calls

### Clean raw files

```bash
conda run -n rdt python -m src.main clean
```

This reads from:

```text
data/00_raw_files/
```

and writes cleaned files to this folder only if no revision rows are found:

```text
data/01_cleaned_files/
```

It also writes logs to:

```text
logs/
```

and, when needed, a timestamped revisions workbook to:

```text
revisions/powers_cleaning_revisions_YYYYMMDD_HHMMSS.xlsx
```

If the revisions workbook has rows, cleaned files and aggregates are not written by default. This keeps the workflow centered on manual review before analysis-ready outputs are refreshed.

### Force cleaned output despite revisions

```bash
conda run -n rdt python -m src.main clean --force-write-cleaned
```

Use this only when you intentionally want cleaned files for inspection even though the revisions workbook contains rows.

### Clean without aggregation

```bash
conda run -n rdt python -m src.main clean --no-aggregate
```

Use this when you want to focus only on partition-level file cleaning and revision checks.

### Preview DIAAD commands without running them

```bash
conda run -n rdt python -m src.main run-diaad --dry-run
```

This is recommended before the first real DIAAD run.

### Run DIAAD on all cleaned partition and aggregate files

```bash
conda run -n rdt python -m src.main run-diaad
```

If neither `--partitions` nor `--aggregates` is specified, the dispatcher can default to running both.

### Run DIAAD only on aggregate files

```bash
conda run -n rdt python -m src.main run-diaad --aggregates
```

### Run DIAAD only on cycle/site partition files

```bash
conda run -n rdt python -m src.main run-diaad --partitions
```

### Continue after DIAAD errors

```bash
conda run -n rdt python -m src.main run-diaad --continue-on-error
```

For DIAAD execution, failing fast is usually safer. Use `--continue-on-error` only when you intentionally want to collect all failed DIAAD calls in one pass.

---

## Recommended semi-automated workflow

### 1. Place raw files

Put all raw POWERS coding files under:

```text
data/00_raw_files/
```

using the cycle/site folder structure.

### 2. Run cleaning

```bash
conda run -n rdt python -m src.main clean
```

### 3. Review the revisions workbook

Open:

```text
revisions/powers_cleaning_revisions_YYYYMMDD_HHMMSS.xlsx
```

Use this file to identify source files and rows that need manual correction.

### 4. Correct the source files

Make manual corrections in the appropriate raw coding files or agreed-upon revision source files.

### 5. Re-run cleaning

```bash
conda run -n rdt python -m src.main clean
```

Repeat the revise/re-clean loop until the cleaning run finds no revision rows. On that run, cleaned files and aggregates are written.

### 6. Dry-run DIAAD

```bash
conda run -n rdt python -m src.main run-diaad --dry-run
```

Check the commands and paths.

### 7. Run DIAAD

```bash
conda run -n rdt python -m src.main run-diaad
```

DIAAD outputs are written to:

```text
data/02_diaad_output/
```

---

## Notes on aggregate analysis

The aggregate files support global reliability evaluation and streamlined downstream statistical processing. The key field is:

```text
expanded_sample_id
```

This field should be used instead of `sample_id` for pooled analysis, because original sample IDs may only be unique within cycle × site partitions.

A typical reliability pairing key is:

```text
expanded_sample_id + utterance_id
```

Aggregate DIAAD reliability outputs are separated for comparison:

```text
data/02_diaad_output/aggregated/reliability_evaluation/frozen/
data/02_diaad_output/aggregated/reliability_evaluation/reconciled/
```

---

## Logs

Logs are written to:

```text
logs/
```

Log files include timestamped messages about:

- Files collected
- Files processed
- Validation warnings
- Validation errors
- Files written
- Aggregate files written
- DIAAD subprocess commands
- DIAAD stdout/stderr

The revisions workbook is better for manual correction. The log is better for technical debugging and record-keeping.

---

## Design principle

This mini-program should not hide the manual nature of the workflow. It automates repetitive preparation, validation, aggregation, and command execution, but it preserves a deliberate human review step before DIAAD analysis.

## Requirements/Setup

The DIAAD program is required, and a conda environment is recommended:

```bash
conda create -n diaad python=3.12
conda activate diaad
python -m pip install diaad
```
