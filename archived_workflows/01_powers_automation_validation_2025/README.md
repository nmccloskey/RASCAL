# POWERS automated measure validation notes

> The below represents an archive; code is not expected to run as presented. The current version of DIAAD would enable such a workflow, although not as directly/explicitly. When the deprecated system is referenced, this indicates one of DIAAD's precursors (see [lineage](https://github.com/nmccloskey/DIAAD/blob/main/docs/lineage.md) for full details).

> NB: This is a no-data workflow archive and does not include private clinical data, identifiable transcripts, or lab-internal spreadsheets.

## Profile of Word Errors and Retrieval in Speech (POWERS) coding - specific measures

The POWERS coding system addresses the need to assess language abilities (particularly lexical retrieval) in conversation for people with aphasia. This deprecated system facilitated quantification of the following subset of POWERS variables for both the clinician and client (see the [POWERS](https://doi.org/10.3233/ACS-2013-20107) manual for full details): 

Using regex and spaCy (`en_core_web_trf`):

   - **filled pauses** - disfluencies like "um", "uh", "er", etc.
   - **speech units** - these more or less map onto non-punctuation tokens excluding filled pauses
   - **content words** - nouns (including proper nouns), non-auxiliary verbs, adjectives, *-ly*-terminal adverbs, and numerals
   - **nouns** - a subset of content words

The below were coded manually, but were not amenable to automation:

   - **number of turns** - a verbal contribution to the conversation with three types:
      - *substantial turn* - contains at least one content word
      - *minimal turn* - hands the turn back to the other conversation partner
      - *subminimal turn (a nonce, non-canonical term)* - not classifiable as either type above (a bespoke laboratory category)
   - **collaborative repair** - sequences of turns devoted to overcoming communicative error/difficulty

## Typical Workflow

1. **Tabularize utterances (if needed)**: read `.cha` files and tabularize utterances, assigning samples unique identifiers at the utterance and transcript levels.

2. **Prepare POWERS coding files**: create full dataset plus reliability coding workbooks, with 4 select metrics automated.

3. **Human coding**: coders complete POWERS annotations in the generated spreadsheets.

4. **Analyze**: aggregate and reports POWERS metrics at the turn, speaker, and dialog levels.

5. **Reliability evaluation**: match reliability files and runs ICC2/Cohen's kappa evaluation.

6. **Reliability subset (optional)**: rselects reliability coding subset if ICC2 measures fail to meet threshold (0.7 a typical minimum).

---

## Automation Validation

The deprecated system included CLI utilities to validate automatic POWERS coding against manual coding.

This workflow had two main steps:

### 1. Select Validation Samples
Use (stratified) random sampling to create a balanced subset of samples for manual validation.

**Arguments:**

- `--stratify`: Optional fields to group by (comma, space, or repeated flags) in random sample selection.
   
   Example: `--stratify site,test` or `--stratify site --stratify test`.

- `--strata`: Number of samples to draw per stratum (default: 5).

- `--seed`: Random number generator seed for reproducibility (default: 42).

**Output:**

- An Excel file `POWERS_validation_selection_<timestamp>.xlsx` containing the selected samples.

- The `stratum_no` column facilitates "chunking" the reliability subset. For example:

   - Code through stratum numbers 1 & 2
   - Evaluate reliability
   - Work through further strata if agreement is poor

- If POWERS coding tables exist in the input folder, labeled versions with `stratum_no` will also be written.


### 2. Validate Automation

Merge the automatic and manual coding files for side-by-side comparison and reliability checks.

**Requirements:**

- Place your coding files in two subdirectories under the input folder:

   - `Auto/` containing automatically generated coding files

   - `Manual/` containing manually coded files

**Arguments:**

- `--selection`: Path to the selection Excel file from the previous step. Required if `stratum_no` is not already in the Manual coding files.

- `--numbers`: Optional comma- or space-separated list of stratum numbers to include (e.g., `--numbers 1,2`).

**Output:**

- An Excel file POWERS_Coding_Auto_vs_Manual.xlsx inside a new AutomationValidation/ folder.
This file contains paired automatic and manual codes, restricted to the requested strata if specified.


**Typical Workflow**

1. Run `powers select` to generate a stratified subset of samples.

2. Manually code samples marked with `stratum_no`.

3. After manual coding, run `powers validate` to merge auto vs manual annotations.

4. Use the merged file to compute inter-coder reliability or other evaluation metrics.

---

## Validation Results

### Sample selection

Sample pool: 181 clinician-client dyadic dialogic samples conducted before testing battery at the Pre- and Post-Tx testing timepoints, 
collected across 3 sites and over two study cycles (years).

Random sample selection with a 20% reliability target was conducted with stratification by:
- cycle, representing aphasia severity category (as measured by WAB)
   - cycle 3: severe profiles
   - cycle 4: mild profiles
- site, i.e., treatment/testing location
- test, i.e., pre- or post-tx

We manually coded 36/181 dialog samples (19.9% coverage) for automation validation.

### Agreement with Automation

ICC2 measures were calculated on utterance-level values across automated & manual codes. Reliability was calculated on samples pooled across site and test, but the cycle stratification remains, as aphasia profile severely impacts accuracy.

|metric|ICC2-severe|ICC2-mild|
|------|----|---------|
|speech units| 0.9981 | 0.9998 |
|content words| 0.7645 | 0.9078 |
|no. nouns| 0.5087 | 0.8018 |
|filled pauses| 0.9786 | 0.9873 |

>**Conclusion:** Counting speech units & filled pauses is easy to automate, and reliability is high across both severe and mild profiles. Content words agreement was good (~0.91) for mild profiles and acceptable (~0.76) for the severe cohort. Noun counts, however, while fair for Cycle 4 (~0.80) were not reliable for Cycle 3 (~0.51). Thus, the automation on this latter metric is not likely to be accurate for sample from people with severe aphasia. It should be noted, however, that the lab had to develop its own coding rules for counting content words and nouns, as the POWERS manual did not specify its paradigm in sufficient detail. These figures depend on the manual coding protocol, and do not reflect the accuracy of the underlying spaCy model.
