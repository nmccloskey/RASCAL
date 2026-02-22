## Operation

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

## 6: Example Data

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