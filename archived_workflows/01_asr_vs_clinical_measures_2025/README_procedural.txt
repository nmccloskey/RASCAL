ASR vs Clinical Measures Workflow Procedural README
================================

Purpose
-------

This archived workflow documents a 2025 analysis connecting automatic speech
recognition/transcription reliability outputs with participant-level clinical
measures.

In plain terms, the workflow prepared CHAT files for ASR-only vs twice-edited 
transcription reliability evaluation with a DIAAD precursor, and then
compared participant-level transcription accuracy with WAB-AQ and CAT Naming
scores.

This is a no-data archive. It preserves the protocol logic and utility scripts,
but does not include private transcripts, clinical spreadsheets, or generated
analysis outputs.

Workflow Overview
-----------------

The workflow has two broad phases.

First, CHAT files were cleaned and normalized before reliability analysis. This
included harmonizing narrative names, replacing CHAT files with debulletized
CLAN `.chstr.cex` output where needed, and preparing reliability transcript
files by adding a `_Reliability` suffix when files did not contain investigator
speech.

Second, transcription reliability output was merged with clinical measures.
The analysis aggregated Levenshtein similarity by participant and plotted the
relationship between mean transcription similarity and selected clinical scores.

Manual/Review Points
--------------------

The preparation utilities are intentionally conservative. Most scripts default
to dry-run behavior or include checks that skip potentially problematic files.

Reviewers should check proposed filename changes before applying them, confirm
that `.chstr.cex` files are the intended debulletized replacements for paired
`.cha` files, and inspect any files skipped because they include `INV:` lines.

Expected Inputs
---------------

The archived code references the following kinds of inputs:

- CHAT transcript files with `.cha` extensions;
- paired debulletized CLAN export files ending in `.chstr.cex`;
- RASCAL/DIAAD transcription reliability output workbook;
- clinical measures CSV containing fields such as `study_id`, `wabaq`,
  `wabseverity`, `wabaphasiasyndrome`, and `namtotscore1`.

Expected Outputs
----------------

Depending on which scripts were run, outputs included:

- normalized `.cha` filenames using `CATGrandpa` rather than `CATPicDesc`;
- `.cha` files replaced by paired debulletized `.chstr.cex` contents, with
  backups when enabled;
- reliability-ready `.cha` files with `_Reliability` in the filename;
- figures such as `wab_vs_accuracy.png` and `cat_naming_vs_accuracy.png`;
- participant-level and descriptive summary dataframes produced during the
  analysis.

Practical Notes
---------------

This archive should be treated as a record of the analysis workflow, not as a
turnkey package. Paths in the analysis script are relative to the original
working directory and should be reviewed before reuse.

Before rerunning the workflow, confirm that clinical score variables, transcript
folder names, and RASCAL/DIAAD output paths match the current project layout.
