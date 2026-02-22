## 0: About RASCAL  
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


## 2: Incorporation of ASR Tools

RASCAL is designed to integrate seamlessly with the **Batchalign** and **Whisper AI** automatic speech recognition (ASR) systems to accelerate the transcription phase preceding linguistic analysis.  

The recommended workflow begins by using **Batchalign** with **Whisper AI**, an open-source ASR system developed by OpenAI that performs high-accuracy offline transcription. Both tools can be installed locally, allowing all ASR to occur **entirely offline**, which ensures compatibility with privacy and clinical data protection standards.  

Batchalign generates CHAT-formatted (`.cha`) transcripts that can be imported directly into RASCAL for further processing. Once transcribed, these files can also be coded or annotated in **CLAN** (MacWhinney & Fromm, 2022) and subsequently analyzed through RASCAL’s modular pipeline (see section 4 for the CHAT coding supported by RASCAL).

While Batchalign’s authors note that its performance may degrade for speech characterized by severe impairments, our experience indicates that the **Batchalign + Whisper** combination provides sufficiently accurate outputs across a wide range of aphasia severities to substantially expedite transcription. 

---

## 3: Privacy and Data Security

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