# RASCAL - Resources for Analyzing Speech in Clinical Aphasiology Labs

RASCAL is a tool designed to facilitate the analysis of speech in clinical aphasiology research. It processes CHAT-formatted (.cha) transcriptions, organizes data into structured tiers, and automates key analytical steps in transcription reliability, CU coding, word counting, and core lexicon analysis.

---

## Analysis Pipeline

### **BU-TU Semi-Automated Monologic Narrative Analysis Overview**

- **Stage 0 (ASR + Manual):** Complete transcription for all samples.
- **Stage 1 (RASCAL):**
   - **Input:** Transcriptions (`.cha`)
   - **Output:** Transcription reliability samples
- **Stage 2 (Manual):** Transcribe reliability samples
- **Stage 3 (RASCAL):**
   - **Input:** Original & reliability transcriptions
   - **Output:** Transcription reliability reports, reselected reliability samples
- **Stage 4 (RASCAL):**
   - **Input:** Transcriptions (`.cha`)
   - **Output:** Utterance tables, CU coding files, timesheets
- **Stage 5 (Manual):** CU coding and reliability checks
- **Stage 6 (RASCAL):**
   - **Input:** Manually completed CU coding files
   - **Output:** CU reliability analysis, reselected CU reliability samples
- **Stage 7 (RASCAL):**
   - **Input:** CU coding
   - **Output:** CU coding summaries, word count files
- **Stage 8 (Manual):** Word counting and reliability checks
- **Stage 9 (RASCAL):**
    - **Input:** Manually completed word count files
    - **Output:** Word count reliability analysis, reselected WC reliability samples
- **Stage 10 (RASCAL):**
    - **Input:** CU and word count data
    - **Output:** Blind & unblind summaries, CoreLex analysis
---

## Web App

You can use RASCAL in your browser — no installation required:

👉 [Launch the RASCAL Web App](https://rascal.streamlit.app/)

---

## Installation

We recommend installing RASCAL into a dedicated virtual environment using Anaconda:

### 1. Create and activate your environment:

```bash
conda create --name rascal_env python=3.9
conda activate rascal_env
```

### 2. Install RASCAL directly from PyPI:
```bash
pip install rascal-speech
```

### ...or from GitHub:
```bash
pip install git+https://github.com/nmccloskey/rascal.git@main
```

---

## Setup

To prepare for running RASCAL, complete the following steps:

### 1. Create your working directory:

We recommend creating a fresh project directory where you'll run your analysis.

Example structure:

```plaintext
your_project/
├── config.yaml           # Configuration file (see below)
└── data/
    └── input/            # Place your CHAT (.cha) files and/or Excel data here
                          # (RASCAL will make an output directory)
```

### 2. Provide a `config.yaml` file

This file specifies the directories, coders, reliability settings, and tier structure.

You can download the example config file from the repo or create your own like this:

```yaml
input_dir: data/input
output_dir: data/output
reliability_fraction: 0.2
coders:
- '1'
- '2'
- '3'
CU_paradigms:
- SAE
- AAE
exclude_participants:
- INV
strip_clan: true
prefer_correction: true
lowercase: true
tiers:
  site:
    values:
    - AC
    - BU
    - TU
    partition: true
    blind: true
  test:
    values:
    - Pre
    - Post
    - Maint
    blind: true
  study_id:
    values: (AC|BU|TU)\d+
  narrative:
    values:
    - CATGrandpa
    - BrokenWindow
    - RefusedUmbrella
    - CatRescue
    - BirthdayScene
```

### Explanation:

- General

  - `reliability_fraction` - the proportion of data to subset for reliability (default 20%).

  - `coders` - alphanumeric coder identifiers (2 required for function **g** and 3 for **c**, see below).

  - `CU_paradigms` - allows users to accommodate multiple dialects if desired. If at least two paradigms are entered, parallel coding columns will be prepared and processed in all CU functions.

  - `exclude_participants` - speakers appearing in .cha files to exclude from transcription reliability and CU coding (neutral utterances).

- Transcription Reliability

  - `strip_clan` - removes CLAN markup but preserve speech-like content, including filled pauses (e.g., '&um' -> 'um') and partial words.

  - `prefer_correction` - toggles policy for accepted corrections '[: x] [*]': True keeps x, False keeps original.

  - `lowercase` - toggles case regularization.

**Specifying tiers:**
The tier system facilitates tabularization by associating a unit of analysis with its possible values and extracting this information from the file name of individual transcripts.

- **Multiple values**: enter as a comma- or newline-separated list. These are treated as **literal choices** and combined into a regex internally. See below examples.
  - *narrative*: `BrokenWindow, RefusedUmbrella, CatRescue`
  - *test*: `PreTx, PostTx`
  
- **Single value**: treated as a **regular expression** and validated immediately. Examples include:
  - Digits only: `\\d+`
  - Lab site + digits: `(AC|BU|TU)\\d+`
  - Three uppercase letters + three digits: `[A-Z]{3}\\d{3}`

- **Tier attributes**
  - **Partition**: creates separate coding files and **separate reliability** subsets by that tier. In this example, separate CU coding files will be generated for each site (AC, BU, TU), but not for each narrative or test value.
  - **Blind**: generates blind codes for CU summaries (function **j** below).

***Example: Tier-Based Tabularization from Filenames (according to the above config).***

Source files:
- `TU88PreTxBrokenWindow.cha`
- `BU77Maintenance_CatRescue.cha`

Tabularization:

| Site | Test  | ParticipantID | Narrative     |
|------|-------|---------------|---------------|
| TU   | Pre   | TU88          | BrokenWindow  |
| BU   | Maint | BU77          | CatRescue     |
---

## Running the Program

Once installed, RASCAL can be run from any directory using the command-line interface:

```bash
rascal <step or function>
```

For example, to run the CU coding analysis function:

```bash
rascal f
```

### Pipeline Commands

| Command | Function (name)                                   | Input                                        | Output (described)                                    |
|---------|---------------------------------------------------|----------------------------------------------|-------------------------------------------------------|
| a       | Select transcription reliability samples (*select_transcription_reliability_samples*) | Raw `.cha` files                             | Reliability sample list + paired reliability `.cha` files |
| b       | Analyze transcription reliability (*analyze_transcription_reliability*) | Reliability `.cha` pairs                     | Agreement metrics + alignment text reports             |
| c       | Reselect transcription reliability (*reselect_transcription_reliability_samples*) | Original + reliability transcription files   | New reliability subsets                               |
| d       | Prepare utterance tables (*prepare_utterance_dfs*) | Raw `.cha` files                             | Utterance spreadsheets                                |
| e       | Make CU coding files (*make_CU_coding_files*)     | Utterance tables (from **d**)                | CU coding spreadsheets                                |
| f       | Make timesheets (*make_timesheets*)               | Utterance tables (from **d**)                | Speaking time entry sheets                            |
| g       | Analyze CU reliability (*analyze_CU_reliability*) | Manually completed CU coding (from **e**)    | Reliability summary tables + reports                   |
| h       | Reselect CU reliability (*reselect_CU_reliability*) | Manually completed CU coding (from **e**)    | New reliability subsets                               |
| i       | Analyze CU coding (*analyze_CU_coding*)           | Manually completed CU coding (from **e**)    | Sample- and utterance-level CU summaries              |
| j       | Make word count files (*make_word_count_files*)   | CU coding tables (from **i**)                | Word count spreadsheets                               |
| k       | Analyze word count reliability (*analyze_word_count_reliability*) | Manually completed word counts (from **j**) | Reliability summaries + agreement reports              |
| l       | Reselect WC reliability (*reselect_WC_reliability*) | Manually completed word counts (from **j**) | New reliability subsets                               |
| m       | Unblind samples (*unblind_CUs*)                   | CU and WC coding results                     | Blind + unblind utterance and sample summaries         |
| n       | Run CoreLex analysis (*run_corelex*)              | CU and WC sample summaries                   | CoreLex coverage and percentile metrics                |

---

## 📊 RASCAL Workflow

Below is the current RASCAL pipeline, represented as a flow chart:

![RASCAL Flowchart](images/RASCAL_flowchart.png)

Black arrows indicate the central analysis pipeline. Red arrows represent the path required if CU reliability coding fails to meet agreement threshold and needs redone. Blue arrows show the alternate inputs to CoreLex analysis: function **b** output is required, and **h** output is optional.

## Notes on Input Transcriptions

- `.cha` files must be formatted correctly according to CHAT conventions.
- Ensure filenames match tier values as specified in `config.yaml`.
- RASCAL searches tier values using exact spelling and capitalization.

## 🧪 Testing

This project uses [pytest](https://docs.pytest.org/) for its testing suite.  
All tests are located under the `tests/` directory, organized by module/function.

### Running Tests
To run the full suite:

```bash
pytest
```
Run with verbose output:
```bash
pytest -v
```
Run a specific test file:
```bash
pytest tests/test_samples/test_run_corelex.py
```

### Notes
- Tests stub out heavy dependencies (e.g., `openpyxl`, external web requests) to keep them fast and reproducible.
- Many tests use temporary directories (`tmp_path`) to simulate file I/O without affecting your real data.

## Status and Contact

I warmly welcome feedback, feature suggestions, or bug reports. Feel free to reach out by:

- Submitting an issue through the GitHub Issues tab

- Emailing me directly at: nsm [at] temple.edu

Thanks for your interest and collaboration!

## Citation

If using RASCAL in your research, please cite:

> McCloskey, N., et al. (2025, April). *The RASCAL pipeline: User-friendly and time-saving computational resources for coding and analyzing language samples*. Poster presented at the Aphasia Access Leadership Summit, Pittsburgh, PA.

## Acknowledgments

RASCAL builds on and integrates functionality from two excellent open-source tools which I highly recommend to researchers and clinicians working with language data:

- [**batchalign2**](https://github.com/TalkBank/batchalign2) – Developed by the TalkBank team, batchalign provides a robust backend for automatic speech recognition. RASCAL is designed to function downstream of this system, leveraging its debulletized `.cha` files as input. This integration allows researchers to significantly expedite batch transcription, which without an ASR springboard might bottleneck discourse analysis.

- [**coreLexicon**](https://github.com/rbcavanaugh/coreLexicon) – A web-based interface for Core Lexicon analysis developed by Rob Cavanaugh, et al. RASCAL implements its own Core Lexicon analysis that has high reliability with this web app: ICC(2) values (two-way random, absolute agreement) on primary metrics are 0.9627 for accuracy (number of core words) and 0.9689 for efficiency (core words per minute) - measured on 402 narratives (Brokem Window, Cat Rescue, and Refused Umbrella) in our study. RASCAL does not use the webapp but accesses the normative data associated with this repository (using Google sheet IDs) to calculate percentiles.
