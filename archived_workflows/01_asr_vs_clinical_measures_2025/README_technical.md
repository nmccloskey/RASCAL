# ASR vs Clinical Measures Technical README

> NB: This is a no-data workflow archive and does not include private clinical
> data, identifiable transcripts, or lab-internal spreadsheets. Furthermore,
> it was preserved for reproducibility and documentation and is
> not guaranteed to generalize outside the original data structure.

## Purpose

This folder contains utility scripts and analysis code for a 2025 workflow that
prepared CHAT files for transcription reliability analysis and compared
participant-level ASR/transcription reliability with clinical measures.

The source files are:

- `src/normalize_narrative_names.py`
- `src/cex_to_cha.py`
- `src/prep_cha_files.py`
- `src/WABvsLS.py`

The configuration file is:

- `config/wab_auto_config.yaml`

## Source Layout

```text
01_asr_vs_clinical_measures_2025/
    config/
        wab_auto_config.yaml
    src/
        cex_to_cha.py
        normalize_narrative_names.py
        prep_cha_files.py
        WABvsLS.py
```

## CHAT Preparation Utilities

### `normalize_narrative_names.py`

This script recursively scans a root directory for `.cha` files whose filenames
contain `CATPicDesc`, case-insensitively, and renames that substring to
`CATGrandpa`.

Example:

```text
AC01Pre_CATPicDesc.cha -> AC01Pre_CATGrandpa.cha
```

CLI usage:

```bash
python src/normalize_narrative_names.py <root_dir>
python src/normalize_narrative_names.py <root_dir> --apply
```

The default is dry-run mode. Existing target filenames are skipped.

### `cex_to_cha.py`

This script recursively scans for `.chstr.cex` files and writes their
debulletized contents to paired `.cha` files.

Pairing rule:

```text
BU75PostTxBrokenWindow.chstr.cex -> BU75PostTxBrokenWindow.cha
```

CLI usage:

```bash
python src/cex_to_cha.py <root_dir>
python src/cex_to_cha.py <root_dir> --apply
python src/cex_to_cha.py <root_dir> --apply --no-backup
```

The default is dry-run mode. When applying changes, existing `.cha` files are
backed up to `.cha.bak` unless `--no-backup` is passed. If `pylangacq` is
available, the script attempts a parse sanity check but still copies raw text
when parsing fails.

### `prep_cha_files.py`

This script recursively finds `.cha` files and renames eligible files to include
`_Reliability` before the extension.

Example:

```text
AC01Pre_CATGrandpa.cha -> AC01Pre_CATGrandpa_Reliability.cha
```

Files containing `INV:` are skipped, because investigator speech indicates they
are not ready for this reliability-file naming step.

The function is:

```python
rename_chat_reliability_files(root_dir, dry_run=True)
```

The `__main__` block contains an active-workflow path:

```python
rename_chat_reliability_files("transcriptions/final", dry_run=False)
```

Review and edit that path before reuse.

## Clinical Measures Analysis

### `WABvsLS.py`

This script merges participant-level transcription reliability with clinical
measures, then plots relationships between mean Levenshtein similarity and
selected clinical scores.

Referenced inputs:

```text
../July2025_PrepSpreadsheets/MASTERDiscourseInter_DATA_2025-02-19_1542.csv
rascal_d_output_250910_0901/TranscriptionReliabilityAnalysis/TranscriptionReliabilityAnalysis.xlsx
```

The reliability workbook is grouped by `study_id` to compute:

```text
mean_ls = mean LevenshteinSimilarity
num_stim = count of OrgFile
```

The grouped reliability data are merged with these clinical fields:

```text
study_id
wabaq
wabseverity
wabaphasiasyndrome
namtotscore1
```

Generated plots:

```text
wab_vs_accuracy.png
cat_naming_vs_accuracy.png
```

The plotting helper adds a scatterplot, regression line when enough unique
points are available, and a text box containing the regression equation, R
squared, p value, and standard error.

## RASCAL/DIAAD Configuration

`config/wab_auto_config.yaml` records the reliability-analysis configuration
used by the archived workflow.

Key settings include:

- `input_dir: data/input`
- `output_dir: data/output`
- `reliability_fraction: 0.2`
- coders `1`, `2`, and `3`
- CU paradigms `SAE` and `AAE`
- participant/site tiers for `AC`, `BU`, and `TU`
- narratives `CATGrandpa`, `BrokenWindow`, `RefusedUmbrella`, `CatRescue`, and
  `BirthdayScene`
- text normalization options including `strip_clan`, `prefer_correction`, and
  `lowercase`

## Dependencies

The scripts use:

- Python standard library modules: `argparse`, `logging`, `os`, `pathlib`,
  `re`, and `shutil`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- optionally `pylangacq`

## Maintenance Notes

- Several paths are hard-coded from the active 2025 workflow and should be
  reviewed before reuse.
- `normalize_narrative_names.py` and `cex_to_cha.py` default to dry-run mode.
- `prep_cha_files.py` has a `dry_run` option in the function, but its
  `__main__` block calls the function with `dry_run=False`.
- `WABvsLS.py` is a script-style analysis file, not a parameterized command-line
  tool.
