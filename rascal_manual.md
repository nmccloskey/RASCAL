::: only-pdf
---
title: "RASCAL Instruction Manual"
subtitle: "Resources for Analyzing Speech in Clinical Aphasiology Labs"
author: "Nick McCloskey"
date: "Version 1.0.1"
geometry: margin=3cm
titlepage: true
titlepage-color: "FFFFFF"
titlepage-text-color: "000000"
titlepage-rule-height: 1
toc: false
---

\pagebreak

\tableofcontents

\pagebreak
:::

## Section 0: About RASCAL  
**Version:** manual written for version 1.0.0; will be edited as needed.  
**Author:** Nick McCloskey, M.S. | Temple University | Speech, Language & Brain Lab  
**Repository:** [https://github.com/nmccloskey/rascal](https://github.com/nmccloskey/rascal)    
**Report Issues:** [https://github.com/nmccloskey/rascal/issues](https://github.com/nmccloskey/rascal/issues)    
**PyPI Distribution:**  [https://pypi.org/project/rascal-speech/](https://pypi.org/project/rascal-speech/)  
**Web App:** [https://rascal.streamlit.app/](https://rascal.streamlit.app/)  
**Zenodo record:** [https://zenodo.org/records/17624074](https://zenodo.org/records/17624074)  
**Zenodo DOI:** 10.5281/zenodo.17624073  
**License:** MIT  

---

### 0.1 Program Overview

**RASCAL** is an open-source system that streamlines the preparation and analysis of discourse data from people with aphasia.  
It supports both command-line and web-based operation, enabling efficient, reproducible, and scalable processing of hundreds or thousands of CHAT-formatted transcripts.

The program automates or facilitates:

- **Transcription reliability** selection and evaluation  
- **Complete Utterance (CU)** coding summaries and reliability calculations  
- **Batched Core Lexicon analysis** with detailed output  
- **Blinded sample management** for clinical coding teams  
- **Integrated data organization** for tiered multi-speaker transcripts  

RASCAL’s modular design allows clinicians and researchers to select only the components that support their workflow while ensuring full compatibility across pipeline stages.

---

### 0.2 System Requirements

| Component | Recommended Specification |
|------------|---------------------------|
| **Python** | 3.12 |
| **Operating System** | Windows 10+, macOS 13+, Linux (Ubuntu 20.04+) |
| **RAM** | >= 8 GB recommended for large batches |
| **Dependencies** | Automatically installed via `pip` (see `requirements.txt`) |
| **Other Programs** | CLAN (for transcription), Excel (for manual coding files) |

---

::: only-pdf
\pagebreak
:::
## Section 1: Installation and Access

RASCAL can be used in **two ways**:

1. **Web App** – no installation required; runs entirely in your browser.  
2. **Local Installation (CLI)** – for advanced users or batch processing.

---

### 1.1  Using the Web App

1. Visit [**https://rascal.streamlit.app/**](https://rascal.streamlit.app/)  
2. Upload your `config.yaml` and input files when prompted.  
3. Upload inputs (`.cha` and/or `.xlsx` files).  
4. Download the zipped output files when complete.  

> **Tip:** The web app saves temporary files in session memory only; export your outputs before closing the tab.

---

### 1.2  Installing RASCAL Locally (Command Line Interface)

#### Step 1  – Create a Virtual Environment

```bash
conda create --name rascal python=3.12
conda activate rascal
```

(If you prefer `venv`: `python -m venv rascal && source rascal/bin/activate`)

#### Step 2 – Install RASCAL

Choose one of the following methods:

```bash
# From PyPI (recommended)
pip install rascal-speech

# From GitHub (latest development version)
pip install git+https://github.com/nmccloskey/rascal.git@main
```

#### Step 3 – Verify Installation

```bash
rascal --help
```

If the command runs and lists available stages, installation is complete.

---

### 1.3  Project Directory Setup

Create a working folder for each analysis project:

```plaintext
your_project/
  config.yaml           # Configuration file (edit as needed)
  rascal_data/
    input/            # Place CHAT (.cha) or Excel input files here
                          # RASCAL automatically creates 'output/' on run
```

#### notes
> - Ensure that any directories synchronized to cloud storage (e.g., OneDrive, Google Drive) are HIPAA-compliant if they contain real clinical data.  
> - Use consistent participant encoding and file naming across projects to enable cross-run merging.  

---

### 1.4  Updating RASCAL

```bash
pip install --upgrade rascal-speech
```

---

### 1.5  Uninstalling

```bash
pip uninstall rascal-speech
conda remove --name rascal --all   # optional, removes environment
```

---

## Section 2: Incorporation of ASR Tools

RASCAL is designed to integrate seamlessly with the **Batchalign** and **Whisper AI** automatic speech recognition (ASR) systems to accelerate the transcription phase preceding linguistic analysis.  

The recommended workflow begins by using **Batchalign** with **Whisper AI**, an open-source ASR system developed by OpenAI that performs high-accuracy offline transcription. Both tools can be installed locally, allowing all ASR to occur **entirely offline**, which ensures compatibility with privacy and clinical data protection standards.  

Batchalign generates CHAT-formatted (`.cha`) transcripts that can be imported directly into RASCAL for further processing. Once transcribed, these files can also be coded or annotated in **CLAN** (MacWhinney & Fromm, 2022) and subsequently analyzed through RASCAL’s modular pipeline (see section 4 for the CHAT coding supported by RASCAL).

While Batchalign’s authors note that its performance may degrade for speech characterized by severe impairments, our experience indicates that the **Batchalign + Whisper** combination provides sufficiently accurate outputs across a wide range of aphasia severities to substantially expedite transcription. 

---

::: only-pdf
\pagebreak
:::
## Section 3: Privacy and Data Security

RASCAL adheres to ethical and technical standards that prioritize **participant confidentiality** and **data integrity**.

### 3.1  Web App Operation

The RASCAL web application runs on **Streamlit Community Cloud**, which processes user data in a **secure, ephemeral environment**:  
- Uploaded files are stored **temporarily** in the session memory only.  
- No data are written to or retained on any persistent server.  
- Files are deleted automatically when the browser tab or session ends.  

This ensures that all participant or client information remains confidential and is not accessible after a session concludes.

> **Best Practice:**  
> We strongly recommend **blinding or pseudonymizing** participant identifiers (e.g., replacing “John_P01” with “TU01P01”) prior to upload. This maintains full compliance with privacy standards in clinical and research settings.

Note that Streamlit flattens directories upon file upload. This matters in particular for the transcription reliability functionality (only CLI version supports the `/reliability` directory).

### 3.2  Local Use

When installed and run locally (via the CLI), all data remain within the user’s computer environment. For the CoreLex functionality, normative data for percentile calculation (Cavanaugh et al., 2021) is accessed online using Google Sheet IDs, but otherwise the CLI version runs offline. The program never collects or transmits any user information. RASCAL therefore supports secure use under HIPAA-compliant privacy frameworks.

---

## Section 4: Configuration

### 4.1  Purpose

RASCAL requires a **configuration file** (e.g., `config.yaml`) to define how the program interprets, organizes, and processes your data.  
This file specifies the following:

1. Input and output directories (automatic in web app mode)  
2. Reliability subset fraction (default: 0.2)  
3. Coder identifiers (for dividing and blinding samples)  
4. CU coding paradigms (to handle dialectal variants if desired)  
5. Speakers in transcripts to exclude from analyses  
6. Tier definitions for identifying (nested) units of analysis in filenames  
7. Optional preprocessing settings for transcription reliability

RASCAL reads this configuration at runtime for both the **CLI** and **web app**, ensuring a reproducible workflow across users and datasets.

---

### 4.2  Example Configuration File

You can download an editable example (`example_config.yaml`) from the GitHub repository or create your own using the format below:

```yaml
input_dir: rascal_data/input
output_dir: rascal_data/output
random_seed: 8
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

---

### 4.3  Explanation of Parameters

#### General Settings

| Key | Description |
|-----|--------------|
| **input_dir / output_dir** | Paths for input and output data. The web app presets these automatically (the latter is zipped for download). |
| **random_seed** | Ensures deterministic selections for replicability |
| **reliability_fraction** | Fraction of samples randomly selected for reliability coding (default = 0.2). |
| **coders** | List of alphanumeric identifiers (e.g., 1–3). At least 2 are required for word count and 3 for CU coding. RASCAL automatically distributes samples evenly and prevents overlap between coders. |
| **CU_paradigms** | Defines CU coding systems (e.g., SAE = Standard American English, AAE = African American English). Multiple paradigms generate parallel coding columns. |
| **exclude_participants** | Excludes specified speakers (e.g., “INV”) from transcription reliability and CU coding. |

#### Transcription Reliability Settings

| Key | Description |
|-----|--------------|
| **strip_clan** | Removes CLAN markup while preserving speech-like elements, including filled pauses (`&um → um`) and partial words (`&+wor → wor`). |
| **prefer_correction** | Determines how corrections of the form `[: correction] [*]` are handled: `true` keeps the correction, `false` keeps the original. |
| **lowercase** | Converts all text to lowercase for normalization. |

---

### 4.4  Tiers: Structuring Units of Analysis

Tiers define the **metadata dimensions** (e.g., site, test phase, participant ID, narrative) that RASCAL extracts from transcript filenames.  
Each tier has:  
- A set of **values** (either listed explicitly or expressed as a regular expression)  
- Optional **partition** and **blind** flags that control analysis grouping and CU summary anonymization.

#### Tier Attributes

| Attribute | Description |
|------------|--------------|
| **values** | List of literal identifiers or a single regular expression defining possible values in filenames. |
| **partition** | When `true`, RASCAL creates **separate coding files and reliability subsets** for each value of that tier (e.g., site). |
| **blind** | When `true`, generates blinding codes for CU coding summaries (function **10a**). |

#### Example: File Naming and Tier Extraction

Suppose transcript filenames follow this pattern:

```
studyID_test_narrative.cha
```

Example files:  
- `TU88PreTxBrokenWindow.cha`
- `BU77Maintenance_CatRescue.cha`

Based on the above configuration (section 4.2), RASCAL tabularizes as follows:

| site | test  | study_id | narrative     |
|------|-------|---------------|---------------|
| TU   | Pre   | TU88          | BrokenWindow  |
| BU   | Maint | BU77          | CatRescue     |

Each row represents one transcript, and each tier becomes a column in sample-level transcript table (see below).  

See [Python - re](https://docs.python.org/3/library/re.html) for more information on regular expressions (Python Software Foundation, n.d.)

---

### 4.5  Dialectal CU Paradigms

Including multiple CU paradigms (e.g., `SAE`, `AAE`) allows RASCAL to automatically produce **parallel CU coding columns**.  
This is useful when coding across dialectal or sociolectal variations while maintaining consistent reliability analysis.

---

### 4.6  Building the Config File Online

Users may create a custom configuration file directly from the **RASCAL web application** using its interactive Config Builder:
1. Launch the web app: [https://rascal.streamlit.app/](https://rascal.streamlit.app/)  
2. Select “Config Builder.”  
3. Define tiers, coders, and reliability settings using dropdowns and checkboxes.  
4. Click “Download config.yaml” to save it for future sessions.  

The downloaded file can later be uploaded for use in both the CLI and web app interfaces.

---

### 4.7  Recommendations

- Keep the configuration file alongside your project data (e.g., in `your_project/`).
- Use descriptive, consistent naming for sites, tests, and narratives.
- Validate regular expressions before running large analyses.
- Always back up configuration files used in published analyses to preserve reproducibility.

---

::: only-pdf
\pagebreak
:::
## Section 5: Operation

> This section is organized by **functionality**. For each function you’ll find: purpose (in the RASCAL pipeline and in general), inputs, outputs, CLI commands (succinct & expanded), Streamlit selection, and any associated **manual** procedures. See the `README.md` on GitHub for a tabular summary of RASCAL functions.

### 5.0 Command & App Basics

- **CLI pattern:**  
  ```bash
  rascal [-h] [--config CONFIG] command [command ...]
  ```  
  - *Succinct* commands: numeric/letter codes (e.g., `4b`).  
  - *Expanded* commands: quoted phrases (e.g., `"cus make"`).  
  - *Omnibus* commands: whole stage (e.g., `4` equivalent to `4a,4b`).  

- **Web app:** Streamlit Community Cloud. Use **Part 3: Select functions to run** to choose the same functions listed below.

---

### 5.1 Function 1a — Select Transcription Reliability Samples
*Pipeline context:* Stage 1 (after semi-automated & edited CHAT transcripts are ready).  
*General purpose:* Randomly subset transcripts for *reliability* transcription from scratch; produce a complete sample list and templates.

*Inputs*

- CHAT transcripts (`.cha`) in `input_dir` (recursive search).
- `config.yaml` (tiers, `reliability_fraction`, exclusions).

*Outputs*

 - Excel with *reliability subset* and *full sample list* (tier-labeled).
 - Template `.cha` files for reliability transcription (blank from scratch).

*CLI*

 - Succinct: `rascal 1a`
 - Expanded: `rascal "transcripts select"`

*Streamlit*

 - “*1a. Select transcription reliability samples*”

*Manual step associated*

 - *Stage 2:* Reliability transcripts are created *manually from scratch* (not edits of auto-transcripts).

---

### 5.2 Function 3a — Evaluate Transcription Reliability
*Pipeline context:* Stage 3.  
*General purpose:* Compare original vs. reliability transcripts to compute agreement metrics and alignment text reports.

*Inputs*

 - `config.yaml` (text normalization settings: `strip_clan`, `prefer_correction`, `lowercase`).
 - Paired `.cha` files (original & reliability) from Stage 2.  

In both the CLI and webapp versions, RASCAL function *3a* matches original with reliability transcripts based on common tiers plus a `reliability` tag in the file name, e.g., `TU88_PreTxBrokenWindow.cha` & `TU88PreTxBrokenWindow_reliability.cha`.  

Function *1a* generates empty `.cha` file templates with the `reliabiilty` tag for the randomly selected samples. In the CLI version, reliability samples can be collected into a `/reliability` subdirectory in the input folder. The tier values still must match the originals, but this provides an alternative to tagging filenames.

*Outputs*

 - Agreement metrics (per sample + summary).
 - Alignment / difference reports (textual).

*CLI*

 - Succinct: `rascal 3a`
 - Expanded: `rascal "transcripts evaluate"`

*Streamlit*

 - “*3a. Evaluate transcription reliability*”

*Manual step associated*

 - None (evaluation is automated); if thresholds not met, see *3b*.

---

### 5.3 Function 3b — Reselect Transcription Reliability Samples
*Pipeline context:* Stage 3 (fallback).  
*General purpose:* When thresholds aren’t met, reselect a new reliability subset using 1a’s tables.

If transcription reliability reselection must recur, just append subsequent selections to the original reliability selection.

*Inputs*

- Reliability tables generated by *1a*.
- Optional thresholds/flags in config.

*Outputs*

- New reliability subset table(s).

*CLI*

- Succinct: `rascal 3b`
- Expanded: `rascal "transcripts reselect"`

*Streamlit*

- “*3b. Reselect transcription reliability samples*”

*Manual step associated*

- Proceed again to *Stage 2* reliability transcription for the reselection.

---

### 5.4 Function 4a — Make Transcript Tables
*Pipeline context:* Stage 4.  
*General purpose:* Tabularize CHAT transcripts into *sample-level* and *utterance-level* Excel sheets with unique IDs (e.g., `S008`, `U0246`).

This encoded tabularization: 
- establishes unique, human-readable identifiers that satisfy database logic
- facilitates data management across RASCAL inputs and outputs, including joins between tables
- promotes transparency and consistency in text processing
- minimizes potential bias during manual coding

If not provided, these tables are automatically generated from `.cha` inputs for functions `4b` & `10b`.

*Inputs*

- `.cha` files.
- `config.yaml` (tiers/regex to parse filenames; text normalization options).

*Outputs*

- `transcript_tables.xlsx` with:  
  - *samples* sheet for transcript metadata: filename + tier values + speaking_time column
  - *utterances* sheet for transcript content: speaker + (CHAT-coded) utterances + comments (from `%com` lines)

*CLI*

- Succinct: `rascal 4a`
- Expanded: `rascal "transcripts make"`

*Streamlit*

- “*4a. Make transcript tables*”

*Manual step associated*

- Optional: Fill *speaking_time* (seconds) in the *samples* sheet if you intend to include speaking rate later (e.g., for CoreLex).

---

### 5.5 Function 4b — Make CU Coding & Reliability Files
*Pipeline context:* Stage 4.  
*General purpose:* Blind samples, *assign coders*, and generate *CU primary & reliability coding templates*.

*Inputs*

- Transcript tables from *4a*.
- `config.yaml` with tiers and *stimulus tier* (`narrative`/`scene`/`story`/`stimulus`), coder list, partition/blind flags, optional multiple *CU_paradigms*.

*Outputs*

- *CU coding spreadsheets* (primary + reliability), with parallel columns per CU paradigm when configured.

*CLI*

- Succinct: `rascal 4b`
- Expanded: `rascal "cus make"`
- Omnibus to run 4a+4b: `rascal 4` or `rascal 4a,4b`

*Streamlit*

- “*4b. Make CU coding & reliability files*”

*Manual step associated*

- *Stage 5:* Coders *manually complete CU coding* (primary by C1, adjudication by C2, blind reliability by C3).

---

### 5.6 Function 6a — Analyze CU Reliability
*Pipeline context:* Stage 6.  
*General purpose:* Compute reliability between *C2 and C3* (post-adjudication), at utterance and sample levels; summarize sufficiency.

*Inputs*

- *Manually completed CU coding* files from *4b*.
- `config.yaml` (coder IDs, reliability thresholds/flags).

*Outputs*

- Reliability detail tables (utterance + sample).
- Summary text (e.g., % of samples meeting >=80% agreement; lab threshold 80%).

*CLI*

- Succinct: `rascal 6a`
- Expanded: `rascal "cus evaluate"`

*Streamlit*

- “*6a. Analyze CU reliability*”

*Manual step associated*

- If thresholds not met, use *6b* to reselect samples for another reliability pass.

---

### 5.7 Function 6b — Reselect CU Reliability Samples
*Pipeline context:* Stage 6 (fallback).  
*General purpose:* Reselect a new blind reliability subset for CU coding.

*Inputs*

- Original + reliability CU coding files (from *4b*).

*Outputs*

- New reliability subset for CU.

*CLI*

- Succinct: `rascal 6b`
- Expanded: `rascal "cus reselect"`

*Streamlit*

- “*6b. Reselect CU reliability samples*”

*Manual step associated*

- Run *Stage 5* manual CU reliability again with the reselection.

---

### 5.8 Function 7a — Analyze CU Coding
*Pipeline context:* Stage 7.  
*General purpose:* Aggregate CU coding to compute counts/percentages of *+SV (grammatical), +REL (relevant), +CU (complete)* per sample.

*Inputs*

- Manually completed CU coding files (from *4b / Stage 5*).

*Outputs*

- Utterance-level and sample-level CU summaries (numbers and percentages).

*CLI*

- Succinct: `rascal 7a`
- Expanded: `rascal "cus analyze"`

*Streamlit*

- “*7a. Analyze CU coding*”

*Manual step associated*

- None.

---

### 5.9 Function 7b — Make Word Count Files
*Pipeline context:* Stage 7.  
*General purpose:* Generate *word count (WC)* coding templates (primary + reliability), downstream of CU so neutral utterances are removed.

*Inputs*

- CU summary tables from *7a* (to know which utterances to include).

*Outputs*

- WC spreadsheets (primary + reliability), with coder assignments.

*CLI*

- Succinct: `rascal 7b`
- Expanded: `rascal "words make"`

*Streamlit*

- “*7b. Make word count files*”

*Manual step associated*

- *Stage 8:* Coders *manually count words* per protocol (exclude repetitions, neologisms, exam prompts, etc.; cf. Nicholas & Brookshire, 1993; Forbes et al., 2012).

---

### 5.10 Function 9a — Evaluate Word Count Reliability
*Pipeline context:* Stage 9.  
*General purpose:* Compute WC reliability as *percent difference* and *ICC(2,1)* (two-way random, single measurement, absolute agreement).

*Inputs*

- Manually completed WC files from *7b*.

*Outputs*

- Reliability summaries (per sample + overall), including ICC values.

*CLI*

- Succinct: `rascal 9a`
- Expanded: `rascal "words evaluate"`

*Streamlit*

- “*9a. Evaluate word count reliability*”

*Manual step associated*

- None; if thresholds not met, see *9b*.

---

### 5.11 Function 9b — Reselect Word Count Reliability Samples
*Pipeline context:* Stage 9 (fallback).  
*General purpose:* Reselect new WC reliability subset when agreement is insufficient.

*Inputs*

- Completed WC files (from *7b*).

*Outputs*

- New reliability subset for WC.

*CLI*

- Succinct: `rascal 9b`
- Expanded: `rascal "words reselect"`

*Streamlit*

- “*9b. Reselect word count reliability samples*”

*Manual step associated*

- Re-run *Stage 8* manual WC on the reselection.

---

### 5.12 Function 10a — Summarize CU Samples (Blind & Unblind)
*Pipeline context:* Stage 10.  
*General purpose:* Produce *blinded and unblinded* CU/WC summaries with blinding keys; incorporate speaking rate if available.

*Inputs*

- Outputs from *4a* (Samples sheet for speaking_time), *7a*, and *7b*.
- `config.yaml` (tiers; which can be selectively blinded in config).

*Outputs*

- Blinded & unblinded *utterance- and sample-level* summaries.
- *Blinding keys* mapping blind codes to tier values.

*CLI*

- Succinct: `rascal 10a`
- Expanded: `rascal "cus summarize"`

*Streamlit*

- “*10a. Summarize CU samples*”

*Manual step associated*

- Optional: ensure *speaking_time* is filled before running for rate metrics.

---

### 5.13 Function 10b — Run CoreLex Analysis
*Pipeline context:* Stage 10.  
*General purpose:* Compute CoreLex coverage and percentiles; optionally incorporate speaking rate (from *4a Samples*).

*Inputs*

1. main path: CU summaries (from *10a*) & transcript tables with speaking times (from *4a*).  
2. minimal path: analyze CoreLex directly on `cha` inputs, or run *4a* and optionally complete the speaking_time column.

*Outputs*

- CoreLex analysis tables (coverage, percentile metrics).

*CLI*

- Succinct: `rascal 10b`
- Expanded: `rascal "corelex analyze"`

*Streamlit*

- “*10b. Run CoreLex analysis*”

*Manual step associated*

- None; ensure prerequisite summaries exist.

---

### 5.14  Example Omnibus & Combined Runs

```bash
# Prepare transcript tables + CU coding files in one go
rascal 4            # runs 4a,4b

# Full CU analysis after manual CU coding is finished
rascal 6a,7a,10a

# Minimal batched CoreLex on input .cha files
rascal 10b
```

---

### 5.15  Notes & Best Practices
- Always keep `config.yaml` versioned alongside your project outputs for reproducibility.  
- For privacy, use blinded identifiers in filenames before upload (web app).  
- If your lab requires thresholds, document them (e.g., >=0.80 agreement for CU; ICC(2,1) >= 0.90 for WC).  
- Use **partition** tiers to silo outputs by site or cohort when needed.  
- The **Streamlit multiselect** mirrors all CLI functions one-for-one.
- Program log messages and in/output files are documented in the automatically generated `logs` directory in the output

---

::: only-pdf
\pagebreak
:::
## Section 6: Example Data

RASCAL is distributed with a folder of **example data** to support hands-on exploration and verification of the pipeline. All included materials are synthetic and non-identifiable. Users can safely test every functionality without access to clinical data.

### 6.1  Folder Structure
```
rascal_files/
  example_data/
    toy_data/
      toy_narratives/
      toy_cu_codes/
      toy_transcript_tables/
      toy_transcription_reliability/
    function_1a/
      rascal_output_YYYYMM_DD_HHMM/
    ...
```

- **toy_data/** provides minimal inputs for trial runs.
- **function_xx/** folders each contain a timestamped output directory from a complete RASCAL operation (e.g., `rascal_output_YYYYMMDD_HHMM/`).  
  These runs illustrate expected outputs, internal metadata, and file naming conventions without repeating inputs unnecessarily.

### 6.2  Usage Notes
- Example data can be used as templates for preparing new projects.
- Metadata files within each output directory specify the exact input paths and parameters used.
- Users handling real participant data must ensure all paths and directories are **HIPAA-compliant**; avoid auto-synchronized cloud folders (e.g., OneDrive, Google Drive) unless institutional data-security approval has been obtained.

---

::: only-pdf
\pagebreak
:::
## Section 7: References

- Cavanaugh, R., Dalton, S. G., & Richardson, J. (2021). coreLexicon: *An open-source web-app for scoring core lexicon analysis*. R package version 0.0.1.0000.
  - https://github.com/aphasia-apps/coreLexicon
- Liu, H., MacWhinney, B., Fromm, D., & Lanzi, A. (2023). *Automation of Language Sample Analysis.* *Journal of Speech, Language, and Hearing Research*, 66(7), 2421–2433.
  - https://pubmed.ncbi.nlm.nih.gov/37348510/
- MacWhinney, B., & Fromm, D. (2022). *CLAN (Computerized Language ANalysis) Manual.* Carnegie Mellon University.
  - https://talkbank.org/0info/manuals/CLAN.pdf
- Python Software Foundation. (n.d.). *re - Regular expression operations.*
  - https://docs.python.org/3/library/re.html
