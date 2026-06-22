# DIAAD Stage 3 Technical README

> NB: This is a no-data workflow archive and does not include private clinical data, identifiable transcripts, or lab-internal spreadsheets.

## Purpose

This folder contains the Stage 3 conversion utilities for taking manually reviewed proto-transcript tables from Stage 2 and preparing final-review CHAT files for Stage 4.

The Stage 3 pipeline has three programmatic steps:

1. Reformat Stage 2 proto-transcript tables into DIAAD-ready transcript tables.
2. Run DIAAD's transcript-to-CHAT export command on each cycle/site partition.
3. Rename and sort the generated `.cha` files into final review folders.

The orchestrator is `src/main.py`. The helper modules are:

- `src/tt_processor.py`
- `src/diaad_runner.py`
- `src/cha_sorter.py`

The expected project layout is:

```text
stage3_project/
├── config/
│   ├── advanced.yaml
│   └── project.yaml
├── data/
│   ├── 00_original_proto_TTs/
│   ├── 01_reformatted_TTs/
│   ├── 02_chat_files/
│   └── 03_sorted_files/
└── src/
    ├── main.py
    ├── tt_processor.py
    ├── diaad_runner.py
    └── cha_sorter.py
```

Run from the project root:

```bash
python -m src.main
```

## Inputs and outputs

### Step 1: `tt_processor.py`

Input root:

```text
data/00_original_proto_TTs/<cycle>/<SITE>_transcript_tables.xlsx
```

Expected files:

```text
data/00_original_proto_TTs/cycle2/AC_transcript_tables.xlsx
data/00_original_proto_TTs/cycle2/BU_transcript_tables.xlsx
data/00_original_proto_TTs/cycle2/TU_transcript_tables.xlsx
data/00_original_proto_TTs/cycle3/AC_transcript_tables.xlsx
data/00_original_proto_TTs/cycle3/BU_transcript_tables.xlsx
data/00_original_proto_TTs/cycle3/TU_transcript_tables.xlsx
```

Output root:

```text
data/01_reformatted_TTs/<cycle>/<SITE>/transcript_tables.xlsx
```

The module exports `process_all_workbooks()` and `process_workbook()`.

Processing behavior:

- Reads the `samples` and `utterances` sheets.
- In both sheets, drops old `sample_id` when present and renames `sample_no` to canonical `sample_id`.
- In `samples`, adds numeric `cycle` and `expanded_sample_id`.
- Keeps and orders sample columns as:

```text
sample_id, expanded_sample_id, cycle, site, study_id, test, narrative
```

- In `utterances`, warns if `review_row` contains nonzero flags.
- Removes rows where `remove_row` is nonzero, when that column exists.
- Renames `utterance_edited` to `utterance`, preferring the edited column over any existing `utterance` column.
- Creates sequential `utterance_id` values from 1 to n after row removal.
- Keeps and orders utterance columns as:

```text
sample_id, utterance_id, speaker, utterance
```

### Step 2: `diaad_runner.py`

Input root:

```text
data/01_reformatted_TTs/<cycle>/<SITE>/transcript_tables.xlsx
```

Output root:

```text
data/02_chat_files/<cycle>/<SITE>/
```

For each existing cycle/site partition, the runner executes:

```bash
diaad transcripts chats --config config --input-dir <input_dir> --output-dir <output_dir>
```

The module exports `run_diaad()`, `iter_run_specs()`, and `run_diaad_specs()`.

`DiaadRunSpec` stores one cycle/site command specification, including the input folder, output folder, and expected `transcript_tables.xlsx` path.

Important assumptions:

- DIAAD is installed in the active environment.
- The `diaad` command is available on `PATH`.
- The `config` directory contains valid DIAAD config files.
- Each input folder contains exactly the transcript table workbook DIAAD expects.

### Step 3: `cha_sorter.py`

Input root:

```text
data/02_chat_files/<cycle>/<SITE>/
```

Output root:

```text
data/03_sorted_files/<cycle>/<SITE>/<timepoint_folder>/<study_id>/
```

The module exports `sort_all_chat_files()`, `sort_chat_files_for_partition()`, and `sort_chat_file()`.

The sorter searches recursively by default for `.cha` files under each cycle/site output folder. It extracts three compact naming components from DIAAD's metadata-rich filenames:

- Study ID: `AC\d+`, `BU\d+`, or `TU\d+`
- Timepoint: `Pre`, `Post`, or `Maint`
- Narrative: the final filename component before `.cha`

Example:

```text
C2_TU_5_2_TU_TU58_Maint_CATGrandpa.cha
```

becomes:

```text
TU58_Maint_CATGrandpa.cha
```

The timepoint folder mapping is:

```python
{
    "Pre": "01_PreTx",
    "Post": "02_PostTx",
    "Maint": "03_Maintenance",
}
```

So the example above would be copied to something like:

```text
data/03_sorted_files/cycle2/TU/03_Maintenance/TU58/TU58_Maint_CATGrandpa.cha
```

By default, files are copied, not moved. Existing sorted files are skipped unless overwrite is enabled.

## Orchestrator: `main.py`

`main.py` imports the three helper modules and runs them in order:

1. `tt_processor.process_all_workbooks()`
2. `diaad_runner.run_diaad()`
3. `cha_sorter.sort_all_chat_files()`

Run the full pipeline:

```bash
python -m src.main
```

Useful options:

```bash
python -m src.main --dry-run-diaad
python -m src.main --skip-tt
python -m src.main --skip-diaad
python -m src.main --skip-sort
python -m src.main --continue-on-error
python -m src.main --move-chat-files
python -m src.main --overwrite-sorted
python -m src.main --nonrecursive-sort
```

Recommended development runs:

```bash
# Confirm DIAAD commands before executing them.
python -m src.main --skip-tt --dry-run-diaad --skip-sort

# Re-run only the sorting step after DIAAD has already produced CHAT files.
python -m src.main --skip-tt --skip-diaad --overwrite-sorted
```

## Error handling and validation

The scripts intentionally fail loudly when required structural inputs are missing. This is preferable to silently creating malformed CHAT files.

`tt_processor.py` raises errors for missing workbooks, missing sheets, or required columns. It logs warnings for noncritical issues such as missing `review_row` or missing `remove_row`.

`diaad_runner.py` validates that the config directory and working directory exist. It logs stdout and stderr from DIAAD. It raises subprocess errors unless `continue_on_error` is enabled.

`cha_sorter.py` skips unparsable filenames and existing destinations by default. It reports counts of sorted, skipped, and failed files.

## Important maintenance notes

- Keep `CYCLES` and `SITES` centralized in `tt_processor.py` unless this project expands beyond `cycle2`, `cycle3`, and the three sites `AC`, `BU`, `TU`.
- Keep the project-root logic based on `Path(__file__).resolve().parents[1]` as long as scripts live directly under `src/`.
- If DIAAD changes its transcript command again, update `DIAAD_COMMAND` in `diaad_runner.py`.
- If DIAAD changes the generated CHAT filename format, update `CHAT_NAME_PATTERN` in `cha_sorter.py`.
- If Stage 2 changes its workbook schema, update `SAMPLES_KEEP_COLS`, `UTTS_KEEP_COLS`, and the required-column checks in `tt_processor.py`.

## Minimal expected end state

After a successful run, Stage 3 should produce final-review CHAT files under:

```text
data/03_sorted_files/cycle2/AC/
data/03_sorted_files/cycle2/BU/
data/03_sorted_files/cycle2/TU/
data/03_sorted_files/cycle3/AC/
data/03_sorted_files/cycle3/BU/
data/03_sorted_files/cycle3/TU/
```

Within each site folder, files should be organized by timepoint and participant/study ID, for example:

```text
data/03_sorted_files/cycle2/TU/03_Maintenance/TU58/TU58_Maint_CATGrandpa.cha
```
