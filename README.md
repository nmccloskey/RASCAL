# RASCAL (Resources for Analyzing Speech in Clinical Aphasiology Labs)

## February 2025

RASCAL is a tool designed to facilitate the analysis of speech in clinical aphasiology research. It processes CHAT-formatted (.cha) transcriptions, organizes data into structured tiers, and automates key analytical steps in transcription reliability, CU coding, word counting, and core lexicon analysis.

---

## Installation

### Recommended: Anaconda Navigator Command Line

1. Create a virtual environment:
   ```bash
   conda create --name rascal_env python=3.9
   ```
2. Activate the environment:
   ```bash
   conda activate rascal_env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Setup

### Required Files

1. **Tiers Specification (**``**)**

   - Specifies tier labels and values for organizing .cha files.
   - Example content:
     ```
     *site:AC,BU,TU
     test:PreTx,PostTx,Maint
     participantID:site##
     narrative:CATGrandpa,BrokenWindow,RefusedUmbrella,CatRescue,BirthdayScene
     ```
   - `*` indicates the partition tier that groups samples.
   - Placeholder `##` in `participantID` enables pattern matching for participant codes.

2. **Configuration File (**``**)**

   - Specifies input/output directories, reliability fraction, and coder initials.
   - Example:
     ```yaml
     input_dir: "input"
     output_dir: "output"
     reliability_fraction: 0.2
     coders: FK,SV,NM
     ```

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

## Individual Functionalities

Each function is executed via:

```bash
python src/main.py <command>
```

| Command | Function                                     | Input                                                     | Output                                                                       |
| ------- | -------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `ab`    | `select_transcription_reliability_samples()` | `.cha` files                                              | `_TranscriptionReliabilitySamples.xlsx`, `_Reliability.cha` files            |
| `ac`    | `prepare_utterance_dfs()`                    | `.cha` files                                              | `_Utterances.xlsx`                                                           |
| `d`     | `make_CU_coding_files()`                     | c output                                                  | `_CU_Coding.xlsx`, `_CUReliabilityCoding.xlsx`                               |
| `e`     | `analyze_transcription_reliability()`        | `.cha` file pairs                                         | `_TranscriptionReliabilityAnalysis.xlsx`, `.txt` alignment files             |
| `f`     | `analyze_CU_reliability()`                   | Manually completed d output                               | `_CUReliabilityCoding_BySample.xlsx`, `_CUReliabilityCodingReport.txt`       |
| `g`     | `analyze_CU_coding()`                        | Manually completed d output                               | `_CUCoding_BySample.xlsx`, `_CUCoding_ByUtterance.xlsx`                      |
| `h`     | `make_word_count_files()`                    | g output                                                  | `_WordCounting.xlsx`, `_WordCountingReliability.xlsx`                        |
| `i`     | `make_timesheets()`                          | c output                                                  | `_SpeakingTimes.xlsx`                                                        |
| `j`     | `analyze_word_count_reliability()`           | h output                                                  | `_WordCountingReliabilityResults.xlsx`, `_WordCountingReliabilityReport.txt` |
| `k`     | `unblind_CUs()`                              | c output, d output, h & i outputs, `ParticipantData.xlsx` | Blind/unblind sample data files                                              |
| `l`     | `run_corelex()`                              | k output                                                  | `CoreLexDataYYMMDD.xlsx`                                                     |

---

## Notes

- `.cha` files must be formatted correctly according to CHAT conventions.
- Ensure filenames match tier values as specified in `tiers.txt`.
- RASCAL searches values using exact spelling and capitalization.
- No spaces allowed in `tiers.txt` (use ` ` for new lines only).

<!-- ---

## Citation

If using RASCAL in your research, please cite:

> XXX (Year). Title of the related study. Journal/Conference.

--- -->

<!-- For further details, contact: **[Your Lab Contact Information]**
 -->
