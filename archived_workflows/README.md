# Archived Workflows

This directory preserves selected RASCAL workflows that are useful as
reproducibility records, protocol references, or examples of clinical-language
data-processing design. These are no-data archives: private clinical data,
identifiable transcripts, lab-internal spreadsheets, Teams folders, and
generated analysis outputs are intentionally excluded.

The archived code should be read as preserved project logic, not as a promise
that every workflow is turnkey in the current package. Start with each
`workflow_manifest.yaml`, then use the workflow README files for procedural and
technical details.

## Archive Contents

| Workflow | Status | Intended reuse | Primary docs |
| --- | --- | --- | --- |
| [ASR vs Clinical Measures 2025](01_asr_vs_clinical_measures_2025/) | Archived/project-specific | Documentation and adaptable reference for CHAT reliability preparation and participant-level clinical-measure analysis | [manifest](01_asr_vs_clinical_measures_2025/workflow_manifest.yaml), [technical](01_asr_vs_clinical_measures_2025/README_technical.md), [procedural](01_asr_vs_clinical_measures_2025/README_procedural.txt) |
| [POWERS Automation Validation 2025](02_powers_automation_validation_2025/) | Archived/deprecated precursor | Reference for validation sampling, manual-vs-automatic comparison, and agreement interpretation | [manifest](02_powers_automation_validation_2025/workflow_manifest.yaml), [README](02_powers_automation_validation_2025/README.md) |
| [POWERS Analysis and Reliability Evaluation 2026](03_powers_analysis_and_reliability_evaluation_2026/) | Archived/lab-specific | Adaptable pattern for revise-before-analysis cleaning, aggregation, and DIAAD reliability execution | [manifest](03_powers_analysis_and_reliability_evaluation_2026/workflow_manifest.yaml), [technical](03_powers_analysis_and_reliability_evaluation_2026/README_technical.md), [procedural](03_powers_analysis_and_reliability_evaluation_2026/README_procedural.txt) |
| [Monologic Narrative Transcript Conversion 2026](04_transcript_conversion_2026/) | Archived/lab-specific | Protocol reference for staged spreadsheet-to-CHAT conversion with manual review and CLAN validation | [manifest](04_transcript_conversion_2026/workflow_manifest.yaml), [README](04_transcript_conversion_2026/README.md) |

## Design Themes Incorporated

- Reproducible clinical-language workflows with documented inputs, outputs, and handoff points.
- Human-in-the-loop validation before analysis-ready files are accepted.
- Reliability-oriented data processing for transcription, coding, and automation checks.
- Staged conversion between spreadsheets, transcript tables, DIAAD outputs, CHAT files, and final validation artifacts.
- Conservative automation with dry runs, revision workbooks, review flags, and explicit manual checkpoints.
- Privacy-conscious archive boundaries that preserve workflow logic without including sensitive data.
- Separation of reusable RASCAL/DIAAD package concepts from project-specific preserved scripts.
- Clear maintenance labels that distinguish active package code from preserved research artifacts.

## Status Labels

| Label | Meaning in this archive |
| --- | --- |
| Archived/project-specific | Preserved for reproducibility; useful as a reference but tied to original paths, schemas, or analyses. |
| Archived/lab-specific | Preserved workflow logic may be adaptable, but assumes local lab conventions, file structures, or manual review protocols. |
| Archived/deprecated precursor | Documents behavior from an older RASCAL/DIAAD precursor system and should not be treated as current package API. |

## Notes for Reuse

Before rerunning or adapting any archived workflow, review hard-coded paths,
expected workbook schemas, command-line assumptions, dependency versions, and
manual review requirements. The manifests intentionally summarize the workflow
surface area; the detailed READMEs describe the procedural context and
maintenance cautions.

For privacy and reproducibility, keep new examples synthetic unless the data
can be shared under the repository's privacy expectations.
