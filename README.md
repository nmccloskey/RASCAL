# RASCAL - Resources for Analyzing Speech in Clinical Aphasiology Labs

RASCAL is a tool designed to facilitate the analysis of speech in clinical aphasiology research. It processes CHAT-formatted (.cha) transcriptions, organizes data into structured tiers, and automates key analytical steps in transcription reliability, CU coding, word counting, and core lexicon analysis.

---

## Analysis Pipeline

### **BU-TU Conversation Treatment Monologic Narrative Analysis Protocol**

1. **Step 0 (Manual):** Complete transcription for all samples.
2. **Step 1 (RASCAL):**
   - **Input:** Transcriptions (`.cha`), Tiers (`.txt`)
   - **Output:** Transcription reliability files (`.xlsx`, `.cha`), CU coding files
3. **Step 2 (Manual):** CU coding and reliability checks
4. **Step 3 (RASCAL):**
   - **Input:** Transcriptions, CU coding files
   - **Output:** Reliability reports, coding summaries, word count templates
5. **Step 4 (Manual):** Finalize word counts and speaking times
6. **Step 5 (RASCAL):**
   - **Input:** Word counts, speaking times, participant data
   - **Output:** Blind/unblind CU coding summaries, word count reliability, core lexicon analysis

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
└── rascal_data/
    └── input/            # Place your CHAT (.cha) files and/or Excel data here
                          # (RASCAL will make an output directory)
```

### 2. Provide a `config.yaml` file

This file specifies the directories, coders, reliability settings, and tier structure.

You can download the example config file from the repo or create your own like this:

```yaml
input_dir: rascal_data/input
output_dir: rascal_data/output
reliability_fraction: 0.2
coders:
- '1'
- '2'
- '3'
tiers:
  '*site':
  - AC
  - BU
  - TU
  test:
  - Pre
  - Post
  - Maint
  participantID: site##
  narrative:
  - CATGrandpa
  - BrokenWindow
  - RefusedUmbrella
  - CatRescue
  - BirthdayScene
```

Explanation:

- Keys prefixed with `*` and surrounded by `""` (e.g., `"*site"`) define partition tiers — these group files for separate coding and output.

- In this example, separate CU coding files will be generated for each `site` (AC, BU, TU), but not for each `narrative` or `test` value.

- `"participantID": "site##"` enables pattern matching where the participant ID is derived from the site name followed by digits (e.g., AC01, BU23).

---

## Running the Program

Once installed, RASCAL can be run from any directory using the command-line interface:

```bash
rascal <step>
```

For example, to run the CU analysis step:

```bash
rascal f
```

| Command | Function                                     | Input                                                                                      | Output                                                                                   |
| ------- | -------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| a       | select_transcription_reliability_samples()   | .cha files                                                                                 | _TranscriptionReliabilitySamples.xlsx, _Reliability.cha files                           |
| b       | prepare_utterance_dfs()                      | .cha files                                                                                 | _Utterances.xlsx                                                                         |
| c       | make_CU_coding_files()                       | b output                                                                                   | _CU_Coding.xlsx, _CUReliabilityCoding.xlsx                                               |
| d       | analyze_transcription_reliability()          | .cha file pairs                                                                            | _TranscriptionReliabilityAnalysis.xlsx, .txt alignment files                             |
| e       | analyze_CU_reliability()                     | Manually completed c output                                                                | _CUReliabilityCoding_BySample.xlsx, _CUReliabilityCodingReport.txt                      |
| f       | analyze_CU_coding()                          | Manually completed c output                                                                | _CUCoding_BySample.xlsx, _CUCoding_ByUtterance.xlsx                                     |
| g       | make_word_count_files()                      | f output                                                                                   | _WordCounting.xlsx, _WordCountingReliability.xlsx                                       |
| h       | make_timesheets()                            | b output                                                                                   | _SpeakingTimes.xlsx                                                                      |
| i       | analyze_word_count_reliability()             | Manually completed g output                                                                | _WordCountingReliabilityResults.xlsx, _WordCountingReliabilityReport.txt                |
| j       | unblind_CUs()                                | b output, manually completed c, h & i outputs, (optional) ParticipantData.xlsx             | Blind/unblind sample data files                                                          |
| k       | run_corelex()                                | j output                                                                                   | CoreLexDataYYMMDD.xlsx                                                                   |

---

## Notes

- `.cha` files must be formatted correctly according to CHAT conventions.
- Ensure filenames match tier values as specified in `config.yaml`.
- RASCAL searches values using exact spelling and capitalization.

## Status and Contact

I warmly welcome feedback, feature suggestions, or bug reports. Feel free to reach out by:

- Submitting an issue through the GitHub Issues tab

- Emailing me directly at: nsm [at] temple.edu

Thanks for your interest and collaboration!

## Citation

If using RASCAL in your research, please cite:

> McCloskey, N., et al. (2025, April). *The RASCAL pipeline: User-friendly and time-saving computational resources for coding and analyzing language samples*. Poster presented at the Aphasia Access Leadership Summit, Pittsburgh, PA.

A copy of the poster will be available through Aphasia Access shared resources.

## Acknowledgments

RASCAL builds on and integrates functionality from two excellent open-source tools which I highly recommend to researchers and clinicians working with language data:

- [**batchalign2**](https://github.com/TalkBank/batchalign2) – Developed by the TalkBank team, batchalign provides a robust backend for automatic speech recognition. RASCAL is designed to function downstream of this system, leveraging its debulletized `.cha` files as input. This integration allows researchers to significantly expedite batch transcription, which without an ASR springboard might bottleneck discourse analysis.

- [**coreLexicon**](https://github.com/rbcavanaugh/coreLexicon) – A web-based interface for Core Lexicon analysis developed by Rob Cavanaugh. RASCAL interfaces with this tool via Selenium to enable batch submission of samples to an otherwise single-sample system. The underlying functionality and analysis methodology are essential to the Core Lexicon scoring components in RASCAL.
