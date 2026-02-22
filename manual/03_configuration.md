# Setting up the config file

## 1. Purpose

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

## 2. Example Configuration File

You can download an editable example (`example_config.yaml`) from the GitHub repository or create your own using the format below:

```yaml
input_dir: rascal_data/input
output_dir: rascal_data/output
random_seed: 99
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

## 3. Explanation of Parameters

### 3.1 General Settings

| Key | Description |
|-----|--------------|
| **input_dir / output_dir** | Paths for input and output data. The web app presets these automatically (the latter is zipped for download). |
| **random_seed** | Ensures deterministic selections for replicability |
| **reliability_fraction** | Fraction of samples randomly selected for reliability coding (default = 0.2). |
| **coders** | List of alphanumeric identifiers (e.g., 1–3). At least 2 are required for word count and 3 for CU coding. RASCAL automatically distributes samples evenly and prevents overlap between coders. |
| **CU_paradigms** | Defines CU coding systems (e.g., SAE = Standard American English, AAE = African American English). Multiple paradigms generate parallel coding columns. |
| **exclude_participants** | Excludes specified speakers (e.g., “INV”) from transcription reliability and CU coding. |

### 3.2 Transcription Reliability Settings

| Key | Description |
|-----|--------------|
| **strip_clan** | Removes CLAN markup while preserving speech-like elements, including filled pauses (`&um → um`) and partial words (`&+wor → wor`). |
| **prefer_correction** | Determines how corrections of the form `[: correction] [*]` are handled: `true` keeps the correction, `false` keeps the original. |
| **lowercase** | Converts all text to lowercase for normalization. |

---

## 4. Tiers: Structuring Units of Analysis

Tiers define the **metadata dimensions** (e.g., site, test phase, participant ID, narrative) that RASCAL extracts from transcript filenames.  
Each tier has:  
- A set of **values** (either listed explicitly or expressed as a regular expression)  
- Optional **partition** and **blind** flags that control analysis grouping and CU summary anonymization.

### 4.1 Tier Attributes

| Attribute | Description |
|------------|--------------|
| **values** | List of literal identifiers or a single regular expression defining possible values in filenames. |
| **partition** | When `true`, RASCAL creates **separate coding files and reliability subsets** for each value of that tier (e.g., site). |
| **blind** | When `true`, generates blinding codes for CU coding summaries (function **10a**). |

### 4.2 Example: File Naming and Tier Extraction

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

## 5. Dialectal CU Paradigms

Including multiple CU paradigms (e.g., `SAE`, `AAE`) allows RASCAL to automatically produce **parallel CU coding columns**.  
This is useful when coding across dialectal or sociolectal variations while maintaining consistent reliability analysis.

---

## 6. Building the Config File Online

Users may create a custom configuration file directly from the **RASCAL web application** using its interactive Config Builder:
1. Launch the web app: [https://rascal.streamlit.app/](https://rascal.streamlit.app/)  
2. Select “Config Builder.”  
3. Define tiers, coders, and reliability settings using dropdowns and checkboxes.  
4. Click “Download config.yaml” to save it for future sessions.  

The downloaded file can later be uploaded for use in both the CLI and web app interfaces.

---

## 7. Recommendations

- Keep the configuration file alongside your project data (e.g., in `your_project/`).
- Use descriptive, consistent naming for sites, tests, and narratives.
- Validate regular expressions before running large analyses.
- Always back up configuration files used in published analyses to preserve reproducibility.

---
