# Using program functionalities

## 1. Operation

### 1.1 Command & App Basics

- **CLI pattern:**  
  ```bash
  rascal [-h] [--config CONFIG] command [command ...]
  ```  
  - *Succinct* commands: numeric/letter codes (e.g., `4b`).  
  - *Expanded* commands: quoted phrases (e.g., `"cus make"`).  
  - *Omnibus* commands: whole stage (e.g., `4` equivalent to `4a,4b`).  

- **Web app:** Streamlit Community Cloud. Use **Part 3: Select functions to run** to choose the same functions listed below.

---

### 1.3 RASCAL Pipeline Commands

| Stage (succinct command) | Expanded command | Description | Input | Output | Function name |
|---------------------------|------------------|--------------|--------|---------|----------------|
| 1a | transcripts select | Select transcription reliability samples | Raw `.cha` files | Reliability & full sample lists + template `.cha` files | `select_transcription_reliability_samples` |
| 3a | transcripts evaluate | Evaluate transcription reliability | Reliability `.cha` pairs | Agreement metrics + alignment text reports | `evaluate_transcription_reliability` |
| 3b | transcripts reselect | Reselect transcription reliability samples | Original + reliability transcription tables (from **1a**) | New reliability subset(s) | `reselect_transcription_reliability_samples` |
| 4a | transcripts make | Prepare transcript tables | Raw `.cha` files | Sample & utterance-level spreadsheets | `make_transcript_tables` |
| 4b | cus make | Make CU coding & reliability files | Utterance tables (from **4a**) | CU coding + reliability spreadsheets | `make_cu_coding_files` |
| 6a | cus evaluate | Analyze CU reliability | Manually completed CU coding (from **4b**) | Reliability summary tables + reports | `evaluate_cu_reliability` |
| 6b | cus reselect | Reselect CU reliability samples | Manually completed CU coding (from **4b**) | New reliability subset(s) | `reselect_cu_wc_reliability` |
| 7a | cus analyze | Analyze CU coding | Manually completed CU coding (from **4b**) | Sample- and utterance-level CU analyses | `analyze_cu_coding` |
| 7b | words make | Make word count & reliability files | CU coding tables (from **7a**) | Word count + reliability spreadsheets | `make_word_count_files` |
| 9a | words evaluate | Evaluate word count reliability | Manually completed word counts (from **7b**) | Reliability summaries + agreement reports | `evaluate_word_count_reliability` |
| 9b | words reselect | Reselect word count reliability samples | Manually completed word counts (from **7b**) | New reliability subset(s) | `reselect_cu_wc_reliability` |
| 10a | cus summarize | Summarize CU coding & word counts | CU and WC coding results | Blind + unblind utterance and sample summaries + blind codes | `summarize_cus` |
| 10b | corelex analyze | Run CoreLex analysis | CU and WC sample summaries | CoreLex coverage and percentile metrics | `run_corelex` |

---

### 1.4 Command Mappings
| Omnibus command | Succinct command | Expanded command |
|--|--|--|
| 1 | 1a | transcripts select |
| 4 | 4a, 4b | utterances make, cus make |
| 7 | 7a, 7b | cus analyze, words make |
| 10 | 10a, 10b | cus summarize, corelex analyze

---


### 1.5 Example Omnibus & Combined Runs

```bash
# Prepare transcript tables + CU coding files in one go
rascal 4            # runs 4a,4b

# Full CU analysis after manual CU coding is finished
rascal 6a,7a,10a

# Minimal batched CoreLex on input .cha files
rascal 10b
```

---

### 1.6  Notes & Best Practices
- Always keep `config.yaml` versioned alongside your project outputs for reproducibility.  
- For privacy, use blinded identifiers in filenames before upload (web app).  
- If your lab requires thresholds, document them (e.g., >=0.80 agreement for CU; ICC(2,1) >= 0.90 for WC).  
- Use **partition** tiers to silo outputs by site or cohort when needed.  
- The **Streamlit multiselect** mirrors all CLI functions one-for-one.
- Program log messages and in/output files are documented in the automatically generated `logs` directory in the output

---

## 2: Example Data

RASCAL is distributed with a folder of **example data** to support hands-on exploration and verification of the pipeline. All included materials are synthetic and non-identifiable. Users can safely test every functionality without access to clinical data.

### 2.1  Folder Structure
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

## 3: Usage Notes
- Example data can be used as templates for preparing new projects.
- Metadata files within each output directory specify the exact input paths and parameters used.
- Users handling real participant data must ensure all paths and directories are **HIPAA-compliant**; avoid auto-synchronized cloud folders (e.g., OneDrive, Google Drive) unless institutional data-security approval has been obtained.

---

> The following sections are organized by **functionality**. For each function you’ll find: purpose (in the RASCAL pipeline and in general), inputs, outputs, CLI commands (succinct & expanded), Streamlit selection, and any associated **manual** procedures. See the `README.md` on GitHub for a tabular summary of RASCAL functions.

---