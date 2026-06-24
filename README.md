# RASCAL- Resources for Analyzing Speech in Clinical Aphasiology Labs

RASCAL is a lab-facing command-line wrapper for
[DIAAD](https://github.com/nmccloskey/DIAAD). It packages the BU-TU lab's discourse analysis
workflow as profiles, stage registries, generated DIAAD config, preflight
checks, run manifests, status/next-step guidance, archived workflow discovery,
and a few ASR helpers.

## Install

For local development:

```powershell
conda create -n rascal python=3.12 -y
conda activate rascal
python -m pip install -e ".[dev]"
```

For audio splitting helpers, install the optional ASR extra:

```powershell
python -m pip install -e ".[asr]"
```

The package requires Python `>=3.12,<3.13` and DIAAD `>=0.3.1,<0.4.0`.

## Requirement Locks

Pinned requirement files live in `reqs/`. They mirror the current dependency
splits:

- `base`: RASCAL plus DIAAD
- `dev`: base plus RASCAL test tools
- `asr`: base plus audio helper dependencies
- `nlp`: RASCAL plus DIAAD's NLP extra
- `full`: dev, ASR, and NLP together

The `.in` files install RASCAL in editable mode and resolve DIAAD from the
package metadata in `pyproject.toml`.

Regenerate a split with:

```powershell
conda run -n rascal python -m piptools compile --no-build-isolation --output-file=reqs\base.txt reqs\base.in
```

## Quick Start

Create a wrapper project:

```powershell
rascal init --profile lab_full --project path\to\project
```

Inspect a stage:

```powershell
rascal plan --branch monolog --stage 4m --config path\to\project\config\rascal.yaml
```

Run preflight checks:

```powershell
rascal check --branch monolog --stage 4m --config path\to\project\config\rascal.yaml
```

Write generated DIAAD config without running DIAAD:

```powershell
rascal plan --branch monolog --stage 4m --write-config --config path\to\project\config\rascal.yaml
```

Dry-run a stage and write a lightweight manifest:

```powershell
rascal run --branch monolog --stage 4m --dry-run --config path\to\project\config\rascal.yaml
```

Check workflow progress:

```powershell
rascal status --config path\to\project\config\rascal.yaml
rascal next --config path\to\project\config\rascal.yaml
```

## Project Layout

`rascal init` creates a canonical layout:

```text
config/
  rascal.yaml
  diaad.generated/
data/
  raw_media/
  asr_chunks/
  auto_transcripts/
  manual_edit_round_1/
  manual_edit_round_2/
  transcription_reliability/
  monolog/
    transcript_tables/
    cu_files/
    cu_reliability/
    cu_analysis/
    word_count_files/
    word_reliability/
    speaking_times/
    corelex/
    final_exports/
  dialog/
    transcript_tables/
    powers_files/
    powers_reliability/
    powers_analysis/
    final_exports/
runs/
```

A legacy layout preserving `rascal_data/input` and `rascal_data/output` is also
available:

```powershell
rascal init --layout legacy --project path\to\project
```

## Workflow Stages

Common stages:

| Stage | Type | DIAAD command |
| --- | --- | --- |
| `0` | external/manual | Stage 0 ASR and manual transcript preparation |
| `1` | automated | `transcripts select` |
| `2` | manual | Transcribe reliability samples |
| `3` | automated | `transcripts evaluate` |

Monolog stages:

| Stage | Type | DIAAD command(s) |
| --- | --- | --- |
| `4m` | automated | `transcripts tabularize` |
| `5m_prepare` | automated | `cus files` |
| `5m` | manual | CU coding and review |
| `6m` | automated | `cus evaluate` |
| `7m` | automated | `cus analyze`, `templates times`, `words files` |
| `8m` | manual | Word counts and speaking times |
| `9m` | automated | `words evaluate` |
| `10m` | automated | `words analyze`, `words rates`, `cus rates`, `vocab analyze`, `vocab rates` |

Dialog stages:

| Stage | Type | DIAAD command(s) |
| --- | --- | --- |
| `4d` | automated | `transcripts tabularize` |
| `5d_prepare` | automated | `powers files` |
| `5d` | manual | POWERS review and coding |
| `6d` | automated | `powers evaluate` |
| `7d` | automated | `powers analyze` |

Dialog `powers rates` is intentionally excluded from the MVP until speaking-time
handling is settled.

## Generated DIAAD Config

RASCAL writes split DIAAD config under `config/diaad.generated/`:

```text
project.yaml
advanced.yaml
metadata.yaml
rascal_source.yaml
```

`project.yaml` and `advanced.yaml` are DIAAD-facing. `metadata.yaml` and
`rascal_source.yaml` are RASCAL-facing audit files for inspection and review.

## Run Manifests

`rascal run` writes an audit directory under `runs/`:

```text
runs/YYYYMMDD_HHMMSS_<branch>_<stage>/
  rascal_manifest.json
  diaad_commands.txt
  command_01_stdout.txt
  command_01_stderr.txt
```

Manifests include command vectors, config paths, preflight summaries, expected
inputs/outputs, random seed, reliability fraction, and threshold settings. They
avoid embedding transcript or workbook contents.

## ASR Helpers

Stage 0 helper commands live inside `src/rascal`:

```powershell
rascal asr split-audio --input data\raw_media --output data\asr_chunks --max-seconds 60
rascal asr combine-chat-parts --input data\asr_chunks --output data\auto_transcripts
```

`combine-chat-parts` merges numbered `.cha` parts, removes duplicate `@End` and
`@Media` lines, and converts standalone digits to words only on CHAT speaker
tiers. `split-audio` processes `.wav` files only and requires the optional
`pydub` dependency.

## Archived Workflows

Archived workflows are discoverable as protocol references:

```powershell
rascal workflows list
rascal workflows show transcript_conversion_2026 --files
```

RASCAL does not execute archived scripts or copy archived code into new
projects.

## Raw DIAAD Passthrough

For low-level inspection or escape hatches:

```powershell
rascal diaad -- transcripts tabularize --help
```

RASCAL passes arguments through as command vectors and does not construct shell
strings.

## Testing

Use the tracked Windows helper:

```powershell
.\scripts\run_tests.ps1 tests
```

Focused tests:

```powershell
.\scripts\run_tests.ps1 tests\test_cli.py
```

Current MVP validation:

```text
125 passed
```

## Notes For Developers

- Keep implementation logic in `src/rascal`.
- Keep archived workflows read-only and discovery-only.
- Keep manual stages conservative: RASCAL should report manual work as pending
  unless completion can be inferred from reliable downstream evidence.
- Keep private clinical data outside the repository and outside generated test
  fixtures.
- See [DIAAD lineage](https://github.com/nmccloskey/DIAAD/blob/main/docs/lineage.md) for repo history.