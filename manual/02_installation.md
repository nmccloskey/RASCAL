## 1: Installation and Access

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