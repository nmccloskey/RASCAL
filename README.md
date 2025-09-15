# RASCAL - Resources for Analyzing Speech in Clinical Aphasiology Labs

RASCAL is a tool designed to facilitate the analysis of speech in clinical aphasiology research. It processes CHAT-formatted (.cha) transcriptions, organizes data into structured tiers, and automates key analytical steps in transcription reliability, CU coding, word counting, and core lexicon analysis.

---

## Analysis Pipeline

### **BU-TU Semi-Automated Monologic Narrative Analysis Overview**

1. **Step 0 (Manual):** Complete transcription for all samples.
2. **Step 1 (RASCAL):**
   - **Input:** Transcriptions (`.cha`)
   - **Output:** Transcription reliability files, utterance files, CU coding and reliability files
3. **Step 2 (Manual):** CU coding and reliability checks
4. **Step 3 (RASCAL):**
   - **Input:** Original & reliability transcriptions, CU coding & reliability files
   - **Output:** Reliability reports, coding summaries, word count & reliability files, speaking time file
5. **Step 4 (Manual):** Finalize word counts and record speaking times
6. **Step 5 (RASCAL):**
   - **Input:** Utterance file, utterance-level CU summary, speaking times, word counts & reliability
   - **Output:** Blind & unblind, utterance- & sample-level CU coding summaries, word count reliability, core lexicon analysis
---

## Try the Web App

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

### 2. Install RASCAL from GitHub:
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

Explanation:

- `partition: true` indicates partition tiers — these group files for separate coding and output. In this example, separate CU coding files will be generated for each `site` (AC, BU, TU), but not for each `narrative` or `test` value.

- `blind: true` indicates fields to remove for blinding purposes in CU summaries.

- `"study_id": (AC|BU|TU)\d+` enables pattern matching where the study (participant) ID is derived from the site name followed by digits (e.g., AC01, TU23).

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

| Command | Step (Python function)                       | Input                                  | Output (described)                                  |
|---------|----------------------------------------------|----------------------------------------|-----------------------------------------------------|
| a       | Select transcription reliability samples (*select_transcription_reliability_samples*) | Raw `.cha` files                       | Sample list + paired reliability `.cha` files        |
| b       | Prepare utterance tables (*prepare_utterance_dfs*) | Raw `.cha` files                       | Utterance spreadsheets                              |
| c       | Create CU coding files (*make_CU_coding_files*) | Utterance tables (from **b**)          | CU coding + reliability coding spreadsheets         |
| d       | Analyze transcription reliability (*analyze_transcription_reliability*) | Reliability `.cha` pairs               | Agreement metrics + alignment text reports           |
| e       | Analyze CU reliability (*analyze_CU_reliability*) | Manually completed CU coding (from **c**) | Reliability summary tables + reports                 |
| f       | Analyze CU coding (*analyze_CU_coding*)      | Manually completed CU coding (from **c**) | Sample-level CU summaries                           |
| g       | Create word count files (*make_word_count_files*) | CU coding tables (from **f**)          | Word count + reliability spreadsheets               |
| h       | Make timesheets (*make_timesheets*)          | Utterance tables (from **b**)          | Speaking time entry sheets                          |
| i       | Analyze word count reliability (*analyze_word_count_reliability*) | Manually completed word counts (from **g**) | Reliability summaries + agreement reports            |
| j       | Unblind samples (*unblind_CUs*)              | Utterance tables (**b**), CU coding (**c**), timesheets (**h**), word counts (**i**) | Blind + unblind utterance and sample summaries       |
| k       | Run CoreLex analysis (*run_corelex*)         | Sample summaries (from **j**)          | CoreLex coverage and percentile metrics             |
| l       | Reselect CU reliability (*reselect_CU_reliability*) | Manually completed CU coding (from **c**) | New reliability subsets                             |

### Step mapping:
| Step | Letters | Description                                |
|------|----------|--------------------------------------------|
| 1    | a, b, c  | Read CHA, select reliability, prepare utterances |
| 3    | d–h      | Analyze transcription & CU, word counts, timesheets |
| 5    | i, j, k  | Word count reliability, unblind CUs, CoreLex |

---

### 📊 RASCAL Workflow Overview

```mermaid
flowchart TD
    A[a: Select transcription reliability samples] --> B[b: Prepare utterance tables]
    B --> C[c: Create CU coding files]

    C --> D[d: Analyze transcription reliability]
    C --> E[e: Analyze CU reliability]
    C --> F[f: Analyze CU coding]

    F --> G[g: Create word count files]
    B --> H[h: Make timesheets]

    G --> I[i: Analyze word count reliability]

    B & C & H & I --> J[j: Unblind samples]

    J --> K[k: Run CoreLex analysis]

    C --> L[l: Reselect CU reliability]
```

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
