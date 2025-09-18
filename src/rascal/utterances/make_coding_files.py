import os
import re
import random
import logging
import itertools
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
import nltk
# Download only if missing
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

from nltk.corpus import words
valid_words = set(words.words())
d = lambda word: word in valid_words

stim_cols = ["narrative", "scene", "story", "stimulus",
             "Narrative", "Scene", "Story", "Stimulus",]

def segment(x, n):
    """
    Segment a list x into n batches of roughly equal length.
    
    Parameters:
    - x (list): List to be segmented.
    - n (int): Number of segments to create.
    
    Returns:
    - list of lists: Segmented batches of roughly equal length.
    """
    segments = []
    # seg_len = math.ceil(len(x) / n)
    seg_len = int(round(len(x) / n))
    for i in range(0, len(x), seg_len):
        segments.append(x[i:i + seg_len])
    # Correct for small trailing segment.
    if len(segments) > n:
        last = segments.pop(-1)
        segments[-1] = segments[-1] + last
    return segments

def assign_CU_coders(coders):
    """
    Assign each coder to each role (coder 1, coder 2, coder 3) in different segments.
    
    Parameters:
    - coders (list): List of coder names.
    
    Returns:
    - list of tuples: Each tuple contains an assignment of coders.
    """
    random.shuffle(coders)
    perms = list(itertools.permutations(coders))
    assignments = [perms[0]]
    for p in perms[1:]:
        newp = True
        for ass in assignments:
            if any(np.array(p) == np.array(ass)):
                newp = False
        if newp:
            assignments.append(p)
    random.shuffle(assignments)
    return assignments

def make_CU_coding_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    CU_paradigms,
    exclude_participants,
):
    """
    Build and write Complete Utterance (CU) coding workbooks and reliability
    workbooks from previously generated utterance tables.

    This function scans both `input_dir` and `output_dir` for files named
    `*Utterances.xlsx`, loads each into memory, and produces two Excel files
    per input (under `{output_dir}/CUCoding/<labels>/`):

      1) `<labels>_CUCoding.xlsx` – primary coding workbook
      2) `<labels>_CUReliabilityCoding.xlsx` – third-coder reliability workbook

    Where `<labels>` is derived from the provided `tiers` by calling
    `t.match(file.name, return_None=True)` for each tier and joining all
    non-None results with underscores (e.g., `AC_Pre`).

    Parameters
    ----------
    tiers : Mapping[str, Tier]
        A dict-like of tier objects used to extract label text from the
        utterance filename. Each tier must implement:
          - .name : str
          - .match(filename: str, return_None: bool = False) -> Optional[str]
        Only the returned *strings* are used here to construct label folders
        and filenames; `.partition` is not used by this function.

    frac : float
        Fraction (0–1) of samples, **within each coder segment**, to include
        in the reliability subset. The actual number per segment is
        `max(1, round(frac * len(segment)))`.

    coders : list[str]
        A list of coder identifiers. If fewer than three are provided, the
        function logs a warning and falls back to `['1', '2', '3']`.
        Internally, coders are assigned to segments such that each segment
        gets two primary coders (`c1ID`, `c2ID`) and a third reliability coder
        (`c3ID` in the reliability file).

    input_dir : str or Path
        Root directory to search (recursively) for `*Utterances.xlsx`.

    output_dir : str or Path
        Root directory under which `CUCoding/` will be created and outputs
        written.

    CU_paradigms : list[str]
        If length >= 2, the primary coding workbook will drop the base columns
        (`c1SV`, `c1REL`, `c2SV`, `c2REL`) and instead create suffixed variants
        per paradigm (e.g., `c1SV_SAE`, `c1SV_AAE`, ...). The reliability file
        mirrors this structure and uses `c3` prefixes for the third coder.
        If length == 1, base columns are kept (no suffixed variants).

    exclude_participants : list[str]
        Speaker codes (e.g., `['INV']`) for which coding *values* should be
        prefilled with `"NA"` (e.g., `c1SV`, `c2REL`, etc.). ID columns
        (`c1ID`, `c2ID`, and `c3ID`) are still assigned to maintain workflow
        consistency, but content fields remain `"NA"` for excluded speakers.

    Behavior
    --------
    - For each input workbook, the function constructs a CU coding DataFrame by
      dropping bookkeeping columns (e.g., 'file' and any tier columns that are
      not stimulus labels), adding coder ID/comment and coding-value columns,
      and pre-filling content fields with either `np.nan` (normal) or `"NA"`
      (if the row's `speaker` is in `exclude_participants`).
    - Samples are segmented (roughly evenly) across the provided coders; within
      each segment, two primary coder IDs are assigned (`c1ID`, `c2ID`).
    - A reliability subset is sampled from each segment according to `frac`.
      For those rows, a **reliability** DataFrame is built by removing the
      second coder’s columns and introducing third-coder columns (`c3*`).
    - Two Excel files are written per input, under:
        {output_dir}/CUCoding/<label1>/<label2>/.../<labels>_CUCoding.xlsx
        {output_dir}/CUCoding/<label1>/<label2>/.../<labels>_CUReliabilityCoding.xlsx

    Returns
    -------
    None
        Outputs are written to disk.

    Notes
    -----
    - This function reads any `*Utterances.xlsx` found in **either**
      `input_dir` or `output_dir`. This lets you pipe data from earlier steps
      without moving files around.
    - Column expectations for the input utterance table include at least:
      `['sample_id', 'speaker']` and any tier columns used for labeling.
    - Randomness is used for selecting reliability subsets; seed externally or
      monkeypatch `random.sample` for deterministic tests.
    """

    if len(coders) < 3:
        logging.warning(f"Coders entered: {coders} do not meet minimum of 3. Using default 1, 2, 3.")
        coders = ['1', '2', '3']

    base_cols = ['c1ID', 'c1SV', 'c1REL', 'c1com', 'c2ID', 'c2SV', 'c2REL', 'c2com']
    CU_coding_dir = os.path.join(output_dir, 'CUCoding')
    logging.info(f"Writing CU coding files to {CU_coding_dir}")
    utterance_files = list(Path(input_dir).rglob("*Utterances.xlsx")) + list(Path(output_dir).rglob("*Utterances.xlsx"))

    for file in tqdm(utterance_files, desc="Generating CU coding files"):
        logging.info(f"Processing file: {file}")
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]

        assignments = assign_CU_coders(coders)

        try:
            uttdf = pd.read_excel(str(file))
            logging.info(f"Successfully read file: {file}")
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")
            continue

        CUdf = uttdf.drop(columns=[col for col in ['file'] + [t for t in tiers if t not in stim_cols] if col in uttdf.columns]).copy()

        # Set up base coding columns
        for col in base_cols:
            CUdf[col] = CUdf.apply(lambda row: 'NA' if row['speaker'] in exclude_participants else np.nan, axis=1)

        # Dynamically add multiple paradigms if length >= 2
        if len(CU_paradigms) >= 2:

            for prefix in ['c1', 'c2']:
                for tag in ['SV', 'REL']:
                    base_col = f'{prefix}{tag}'
                    CUdf.drop(columns=[base_col], inplace=True, errors='ignore')  # remove original

                    for paradigm in CU_paradigms:
                        new_col = f"{prefix}{tag}_{paradigm}"
                        CUdf[new_col] = CUdf.apply(lambda row: 'NA' if row['speaker'] in exclude_participants else np.nan, axis=1)

        unique_sample_ids = list(CUdf['sample_id'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))
        rel_subsets = []

        for seg, ass in zip(segments, assignments):
            CUdf.loc[CUdf['sample_id'].isin(seg), 'c1ID'] = ass[0]
            CUdf.loc[CUdf['sample_id'].isin(seg), 'c2ID'] = ass[1]

            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = CUdf[CUdf['sample_id'].isin(rel_samples)].copy()

            relsegdf.drop(columns=['c1ID', 'c1com'], inplace=True, errors='ignore')

            if len(CU_paradigms) >= 2:
                # Multi-paradigm: rename c2*_{paradigm} -> c3*_{paradigm}, then drop remaining c1*_{paradigm}
                for tag in ['SV', 'REL']:
                    for paradigm in CU_paradigms:
                        old = f'c2{tag}_{paradigm}'
                        new = f'c3{tag}_{paradigm}'
                        if old in relsegdf.columns:
                            relsegdf.rename(columns={old: new}, inplace=True)
                # Optional comment column for coder 3
                if 'c2com' in relsegdf.columns:
                    relsegdf.rename(columns={'c2com': 'c3com'}, inplace=True)
                # Remove c2ID; c3ID is set explicitly below
                relsegdf.drop(columns=['c2ID'], inplace=True, errors='ignore')
                # Drop c1* coding-value columns (we don’t need them in reliability)
                for tag in ['SV', 'REL']:
                    for paradigm in CU_paradigms:
                        relsegdf.drop(columns=[f'c1{tag}_{paradigm}'], inplace=True, errors='ignore')

            else:
                # Single or zero paradigm: use base columns
                # Rename c2* -> c3* before dropping any remaining c2*
                renames = {'c2SV': 'c3SV', 'c2REL': 'c3REL', 'c2com': 'c3com'}
                to_rename = {k: v for k, v in renames.items() if k in relsegdf.columns}
                if to_rename:
                    relsegdf.rename(columns=to_rename, inplace=True)
                # Remove c2ID; c3ID is set explicitly below
                relsegdf.drop(columns=['c2ID'], inplace=True, errors='ignore')
                # Drop c1* value columns
                relsegdf.drop(columns=['c1SV', 'c1REL'], inplace=True, errors='ignore')

                # Ensure expected c3 columns exist
                for col in ['c3SV', 'c3REL', 'c3com']:
                    if col not in relsegdf.columns:
                        relsegdf[col] = np.nan

            try:
                idx_c3id = relsegdf.columns.tolist().index('c3com')
            except:
                tdx_c3id = len(relsegdf)
            relsegdf.insert(idx_c3id, 'c3ID', ass[2])
            rel_subsets.append(relsegdf)

        reldf = pd.concat(rel_subsets)
        logging.info(f"Selected {len(set(reldf['sample_id']))} samples for reliability from {len(set(CUdf['sample_id']))} total samples.")

        lab_str = '_'.join(labels) + '_' if labels else ''

        cu_filename = os.path.join(CU_coding_dir, *labels, lab_str + 'CUCoding.xlsx')
        rel_filename = os.path.join(CU_coding_dir, *labels, lab_str + 'CUReliabilityCoding.xlsx')

        try:
            os.makedirs(os.path.dirname(cu_filename), exist_ok=True)
            CUdf.to_excel(cu_filename, index=False)
            logging.info(f"Successfully wrote CU coding file: {cu_filename}")
        except Exception as e:
            logging.error(f"Failed to write CU coding file {cu_filename}: {e}")

        try:
            reldf.to_excel(rel_filename, index=False)
            logging.info(f"Successfully wrote CU reliability coding file: {rel_filename}")
        except Exception as e:
            logging.error(f"Failed to write CU reliability coding file {rel_filename}: {e}")


def count_words(text, d):
    """
    Prepares a transcription text string for counting words.
    
    Parameters:
        text (str): Input transcription text.
        d (function): A function or callable to check if a word exists in the dictionary.
        
    Returns:
        int: Count of valid words.
    """
    # Normalize text
    text = text.lower().strip()
    
    # Handle specific contractions and patterns
    text = re.sub(r"(?<=(he|it))'s got", ' has got', text)
    text = ' '.join([contractions.fix(w) for w in text.split()])
    text = text.replace(u'\xa0', '')
    text = re.sub(r'(^|\b)(u|e)+?(h|m|r)+?(\b|$)', '', text)
    text = re.sub(r'(^|\b|\b.)x+(\b|$)', '', text)
    
    # Remove annotations and special markers
    text = re.sub(r'\[.+?\]', '', text)
    text = re.sub(r'\*.+?\*', '', text)
    
    # Convert numbers to words
    text = re.sub(r'\d+', lambda x: n2w.num2words(int(x.group(0))), text)
    
    # Remove non-word characters and clean up spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bcl\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    # Tokenize and validate words
    tokens = [word for word in text.split() if d(word)]
    return len(tokens)

def make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    """
    Generate utterance-level word count coding files and reliability subsets
    from existing CU coding outputs.

    Workflow
    --------
    1. Locate all "*CUCoding_ByUtterance.xlsx" files under both `input_dir`
       and `output_dir`.
    2. For each file:
       - Extract partition labels from filename using `tiers`.
       - Read CU coding DataFrame.
       - Drop CU-specific columns (c2SV*, c2REL*, c2CU*, c2com*, AG*).
       - Add empty coder-1 assignment column ('c1ID') and comment ('wc_com').
       - Compute 'wordCount' for each utterance using `count_words(utterance, d)`
         if c2CU is not NaN, otherwise assign "NA".
       - Assign coders to samples via `assign_CU_coders(coders)` and
         distribute sample_ids across coders using `segment(...)`.
       - Select a fraction (`frac`) of samples for reliability per coder pair.
         For these, rename 'c1ID'→'c2ID' and 'wc_com'→'wc_rel_com',
         and assign the second coder ID.

    Outputs
    -------
    Under "<output_dir>/WordCounts[/<partition_labels...>]":
      - "<labels>_WordCounting.xlsx"
        Full utterance-level coding frame with 'wordCount', 'c1ID', 'wc_com'.
      - "<labels>_WordCountingReliability.xlsx"
        Subset of samples (≈ frac of total) for reliability, with 'c2ID' and 'wc_rel_com'.

    Parameters
    ----------
    tiers : dict[str, Any]
        Tier objects with `.match(filename, ...)` and `.partition` attributes.
        Used to derive subdirectories and labels for outputs.
    frac : float
        Fraction of unique sample_ids to include in the reliability subset
        (minimum 1 per coder assignment).
    coders : list[str]
        Coder IDs; first two are used for assignments.
    input_dir : str | os.PathLike
        Directory containing CU coding utterance-level Excel files.
    output_dir : str | os.PathLike
        Directory to save word count outputs.

    Returns
    -------
    None
        Saves Excel files to disk; does not return.
    """
    
    # Make word count coding file path.
    word_count_dir = os.path.join(output_dir, 'WordCounts')
    logging.info(f"Writing word count files to {word_count_dir}")

    # Convert utterance-level CU coding files to word counting files.
    CU_files = list(Path(input_dir).rglob("*CUCoding_ByUtterance.xlsx")) + list(Path(output_dir).rglob("*CUCoding_ByUtterance.xlsx"))
    for file in tqdm(CU_files, desc="Generating word count coding files"):

        logging.info(f"Processing file: {file}")
        
        # Extract partition tier info from file name.
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]
        logging.debug(f"Extracted labels: {labels}")

        # Read and copy CU df.
        try:
            CUdf = pd.read_excel(str(file))
            logging.info(f"Successfully read file: {file}")
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")
            continue
        WCdf = CUdf.copy()

        # Add counter and word count comment column.
        empty_col = [np.nan for _ in range(len(WCdf))]
        WCdf = WCdf.assign(**{'c1ID': empty_col})
        # Set to string type explicitly to avoid warning in the .isin part.
        WCdf['c1ID'] = WCdf['c1ID'].astype('string')
        WCdf = WCdf.assign(**{'WCcom': empty_col})

        # Add word count column and pull neutrality from CU2.
        WCdf['word_count'] = WCdf.apply(lambda row: count_words(row['utterance'], d) if not np.isnan(row['c2CU']) else 'NA', axis=1)

        # Winnow columns.
        drop_cols = [c for c in WCdf.columns if c.startswith(('c2SV', 'c2REL', 'c2CU', 'c2com', 'AG'))]
        WCdf = WCdf.drop(columns=drop_cols)
        logging.debug("Dropped CU-specific columns.")

        # Only first two coders used in these assignments.
        assignments = assign_CU_coders(coders)

        # Select samples for reliability.
        unique_sample_ids = list(WCdf['sample_id'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))

        # Assign coders and prep reliability file.
        rel_subsets = []
        for seg, ass in zip(segments, assignments):
            WCdf.loc[WCdf['sample_id'].isin(seg), 'c1ID'] = ass[0]
            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = WCdf[WCdf['sample_id'].isin(rel_samples)].copy()
            relsegdf.rename(columns={'c1ID': 'c2ID', 'wc_com': 'wc_rel_com'}, inplace=True)
            relsegdf['c2ID'] = ass[1]
            rel_subsets.append(relsegdf)

        WCreldf = pd.concat(rel_subsets)
        logging.info(f"Selected {len(set(WCreldf['sample_id']))} samples for reliability from {len(set(WCdf['sample_id']))} total samples.")

        lab_str = '_'.join(labels) + '_' if labels else ''

        # Save word count coding file.
        filename = os.path.join(word_count_dir, *labels, lab_str + 'WordCounting.xlsx')
        logging.info(f"Writing word counting file: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            WCdf.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write word count coding file {filename}: {e}")

        # Word count reliability coding file.
        filename = os.path.join(word_count_dir, *labels, lab_str + 'WordCountingReliability.xlsx')
        logging.info(f"Writing word count reliability coding file: {filename}")
        try:
            WCreldf.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write word count reliability coding file {filename}: {e}")

def reselect_CU_WC_reliability(
    tiers,
    input_dir,
    output_dir,
    rel_type: str = "CU",   # "CU" or "WC"
    frac: float = 0.2,
    rng_seed: int = 88,
):
    """
    Reselect a new set of reliability samples for either CU or Word Counting (WC),
    *excluding* samples already present across any existing reliability files
    that match the same tier labels.

    For each original coder-2 coding file (CU or WC) found under `input_dir`,
    this function gathers **all** paired reliability sheets that match by tier
    labels (so multiple failed attempts are supported), computes the set
    difference between all coder-2 samples and samples already used in any
    reliability sheet, randomly selects a fraction of the total unique samples,
    and writes a fresh reliability workbook restricted to those new sample_ids.

    Parameters
    ----------
    tiers : dict[str, Tier]
        Mapping of tier-name -> Tier object. Each Tier must expose a method usable
        for filename labeling. This function will try (in order): `t.match(name)`,
        `t.extract(name)`, `t.parse(name)`, or a `t.regex` attribute. If none work,
        `None` is used for that tier label.

    input_dir : str | os.PathLike
        Root directory searched (recursively) for coder-2 coding files and their
        paired reliability files.

    output_dir : str | os.PathLike
        Base directory where outputs are written under:
          "<output_dir>/reselected_<rel_type>_reliability/"

    rel_type : {"CU","WC"}, default "CU"
        Which reliability type to process.

        - "CU": looks for "*CUCoding.xlsx" and paired "*CUReliabilityCoding.xlsx"
        - "WC": looks for "*WordCounting.xlsx" and paired "*WordCountingReliability.xlsx"

    coder3 : str, default "3"
        Value written to the reliability coder ID column (e.g., 'c3ID').

    frac : float in (0,1], default 0.2
        Target fraction of **all unique samples** (from the coder-2 file) to select.
        The number selected is `max(1, round(len(all_sample_ids) * frac))`. If fewer
        than this number of **unused** samples are available, all available unused
        samples are selected (with a warning).

    rng_seed : int, default 88
        RNG seed for reproducible selection within a run.

    Behavior & Output
    -----------------
    - Tier-label matching allows multiple prior reliability attempts to be excluded
      (all matched reliability files contribute to the "used sample_ids" set).
    - Selection excludes any `sample_id` already present across *any* matched reliability file.
    - Selection is random with a fixed seed per function call.
    - The shape (columns) of the new reliability sheet is derived from the first matched
      reliability file when present. If multiple matched reliability files have different
      columns, the **intersection** after 'comment' is used (a warning is logged).
    - Carry-forward of columns:
        * We copy "header" columns from the original coder-2 table up to and including
          the column named 'comment' (if present); otherwise we copy the full table then
          trim to the reliability template if available.
    - CU-specific niceties:
        * If columns like 'c2SV', 'c2REL', 'c2SV_*', 'c2REL_*' exist, they are renamed
          to 'c3SV', 'c3REL', 'c3SV_*', 'c3REL_*', respectively.
        * All **suffixed** c3 CU columns ('c3SV_*', 'c3REL_*') are wiped to NaN so the
          reliability coder must re-enter them. Base 'c3SV'/'c3REL' are preserved.
        * 'c3ID' is set to `coder3`, and 'c3com' (if present) is wiped to NaN.

    Files Written
    -------------
    One Excel file per coder-2 table:
      "<output_dir>/reselected_<rel_type>_reliability/<base>_reselected_<RelName>.xlsx"

      where:
        - For CU: RelName = "CUReliabilityCoding"
        - For WC: RelName = "WordCountingReliability"

    Returns
    -------
    None

    Notes
    -----
    - If *no* matched reliability file is found for a given coder-2 file, the file is
      **skipped** (logged warning). This avoids guessing the reliability schema.
    - This function requires a 'sample_id' column to exist in both the coder-2 file
      and the reliability file(s). Missing 'sample_id' will be logged and that file
      will be skipped.
    - If no unused samples remain, the file is skipped (with warning).
    """
    rel_type = (rel_type or "CU").upper().strip()
    if rel_type not in {"CU", "WC"}:
        logging.error(f"Invalid rel_type '{rel_type}'. Must be 'CU' or 'WC'.")
        return

    random.seed(rng_seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / f"reselected_{rel_type}_reliability"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified directory: {out_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {out_dir}: {e}")
        return

    # --- discover original coder-2 files and their reliability mates ---
    if rel_type == "CU":
        coding_glob = "*CUCoding.xlsx"
        rel_glob = "*CUReliabilityCoding.xlsx"
        out_rel_name = "CUReliabilityCoding"
    else:  # "WC"
        coding_glob = "*WordCounting.xlsx"
        rel_glob = "*WordCountingReliability.xlsx"
        out_rel_name = "WordCountingReliability"

    coding_files = list(input_dir.rglob(coding_glob))
    rel_files = list(input_dir.rglob(rel_glob))

    if not coding_files:
        logging.warning(f"No original {rel_type} files found under {input_dir} (pattern: {coding_glob}).")
        return

    # --- helpers ---
    def _label_one(tier_obj, fname: str):
        try:
            if hasattr(tier_obj, "match"):
                return tier_obj.match(fname)
        except Exception:
            pass
        # Fallback: no label for this tier
        return None

    def _labels_for(path: Path):
        if not tiers:
            # If tiers are not provided, fallback to just the stem so pairs can still match
            return (path.stem,)
        labels = []
        for t in tiers.values():
            try:
                labels.append(_label_one(t, path.name))
            except Exception:
                labels.append(None)
        return tuple(labels)

    # Build map from original -> list of matching reliability files (by tier labels)
    org_rel_matches: dict[Path, list[Path]] = {}
    rel_labels_cache = {p: _labels_for(p) for p in rel_files}

    for org in coding_files:
        org_labels = _labels_for(org)
        matches = [p for p, labs in rel_labels_cache.items() if labs == org_labels]
        if not matches:
            logging.warning(f"[{rel_type}] No reliability files found for {org.name}. Skipping.")
        org_rel_matches[org] = matches

    # --- process each original coding file ---
    desc = f"Reselecting {rel_type} reliability samples"
    for org_file in tqdm(coding_files, desc=desc):
        rel_mates = org_rel_matches.get(org_file, [])
        if not rel_mates:
            continue  # already warned

        try:
            df_org = pd.read_excel(org_file)
        except Exception as e:
            logging.error(f"Failed reading original coding file {org_file}: {e}")
            continue

        # Collect reliability DataFrames & validate 'sample_id'
        rel_dfs = []
        for rf in rel_mates:
            try:
                rel_df = pd.read_excel(rf)
                rel_dfs.append(rel_df)
            except Exception as e:
                logging.warning(f"Failed reading reliability file {rf}: {e}")

        if not rel_dfs:
            logging.warning(f"[{rel_type}] All matched reliability files failed to read for {org_file.name}. Skipping.")
            continue

        # Confirm 'sample_id' columns exist
        if "sample_id" not in df_org.columns:
            logging.warning(f"[{rel_type}] 'sample_id' missing in {org_file.name}. Skipping.")
            continue
        missing_sid = [rf for rf, rdf in zip(rel_mates, rel_dfs) if "sample_id" not in rdf.columns]
        if missing_sid:
            logging.warning(f"[{rel_type}] 'sample_id' missing in reliability file(s): {[p.name for p in missing_sid]}. Skipping {org_file.name}.")
            continue

        # Compute used sample_ids across all matched reliability files
        used_sample_ids: set = set()
        for rdf in rel_dfs:
            used_sample_ids.update(set(rdf["sample_id"].dropna().astype(str).unique()))

        all_sample_ids = set(df_org["sample_id"].dropna().astype(str).unique())
        available_ids = list(all_sample_ids - used_sample_ids)

        if len(available_ids) == 0:
            logging.warning(f"[{rel_type}] No available samples to reselect for {org_file.name}. Skipping.")
            continue

        num_to_select = max(1, round(len(all_sample_ids) * float(frac)))
        if len(available_ids) < num_to_select:
            logging.warning(
                f"[{rel_type}] Not enough unused samples in {org_file.name}. "
                f"Selecting {len(available_ids)} instead of target {num_to_select}."
            )
            num_to_select = len(available_ids)

        # Sample deterministically (within run) from available ids
        random.seed(rng_seed)  # reset per file to make selection reproducible if code reruns
        reselected_ids = set(random.sample(available_ids, k=num_to_select))

        # --- Build new reliability frame ---
        # Determine the template (columns) from reliability files.
        # Use columns up to 'comment' + any columns after; if multiple have different
        # post-'comment' sets, take intersection to reduce mismatch risk.
        def _cols_to_comment(df):
            if "comment" in df.columns:
                idx = df.columns.get_loc("comment")
                return list(df.columns[: idx + 1])
            return list(df.columns)

        # "Head" columns (up through 'comment') will come from original coding table, not from reliability.
        head_cols = _cols_to_comment(df_org)

        # Post-comment columns from reliability (template)
        post_sets = []
        # for rdf in rel_dfs:
        if "comment" in rdf.columns:
            start = rdf.columns.get_loc("comment") + 1
            post_cols = rdf.columns[start:]
        else:
            logging.error(f"No 'comment' column found in {rel_mates[0]}.")
            post_sets.append(set(rdf.columns))  # fallback

        # if not post_sets:
        #     logging.warning(f"[{rel_type}] Could not determine reliability template for {org_file.name}. Skipping.")
        #     continue

        # post_cols = list(set.intersection(*post_sets)) if len(post_sets) > 1 else list(post_sets[0])

        # # Warn if templates differ
        # if len(post_sets) > 1 and any(post_cols != list(s) for s in post_sets):
        #     logging.warning(
        #         f"[{rel_type}] Matched reliability files for {org_file.name} "
        #         f"have differing columns; using intersection of post-'comment' columns."
        #     )

        # Subset original coding rows for the chosen sample_ids
        sub = df_org[df_org["sample_id"].astype(str).isin(reselected_ids)].copy()
        if sub.empty:
            logging.warning(f"[{rel_type}] Resolved 0 rows after filtering by reselected sample_ids for {org_file.name}. Skipping.")
            continue

        # Keep up through 'comment' if present
        if "comment" in sub.columns:
            end_idx = sub.columns.get_loc("comment")
            sub = sub.iloc[:, : end_idx + 1]
        # else: keep all (we'll align columns later)

        # Add post-'comment' reliability columns (NaN by default)
        for col in post_cols:
            if col not in sub.columns:
                sub[col] = np.nan

        # CU-specific adjustments
        if rel_type == "CU":
            # # Rename coder-2 CU columns to coder-3 equivalents where applicable
            # rename_map = {}
            # for col in list(sub.columns):
            #     if col == "c2SV":
            #         rename_map[col] = "c3SV"
            #     elif col == "c2REL":
            #         rename_map[col] = "c3REL"
            #     elif col.startswith("c2SV_"):
            #         rename_map[col] = "c3SV_" + col.split("c2SV_", 1)[1]
            #     elif col.startswith("c2REL_"):
            #         rename_map[col] = "c3REL_" + col.split("c2REL_", 1)[1]
            # if rename_map:
            #     sub.rename(columns=rename_map, inplace=True)

            # Ensure c3ID/c3com exist, then set/wipe
            if "c3ID" not in sub.columns:
                sub["c3ID"] = np.nan
            if "c3com" not in sub.columns:
                sub["c3com"] = np.nan
            # sub["c3ID"] = coder3
            # sub["c3com"] = np.nan

            # # Wipe suffixed c3 CU columns
            # for col in sub.columns:
            #     if col.startswith("c3SV_") or col.startswith("c3REL_"):
            #         sub[col] = np.nan

        else:  # "WC"
            # Ensure c3ID exists if present in template; set it to coder3
            if "c2ID" not in sub.columns:
                sub["c2ID"] = np.nan

                
            # If a 'c3com' exists in the template, wipe it
            if "wc_rel_com" not in sub.columns:
                sub["wc_rel_com"] = np.nan
            # Add word count column and pull neutrality from original.
            org_sub = df_org[df_org['sample_id'].isin(reselected_ids)].copy()
            sub['word_count'] = org_sub.apply(lambda row: count_words(row['utterance'], d) if not np.isnan(row['word_count']) else 'NA', axis=1)

        # Order columns: head-cols first (as they exist), then post-cols in template order
        ordered_cols = [c for c in head_cols if c in sub.columns] + [c for c in post_cols if c in sub.columns and c not in head_cols]
        sub = sub.loc[:, ordered_cols]

        # --- Write output file ---
        stem = org_file.stem
        if rel_type == "CU" and stem.endswith("CUCoding"):
            base = stem[: -len("CUCoding")].rstrip("_")
        elif rel_type == "WC" and stem.endswith("WordCounting"):
            base = stem[: -len("WordCounting")].rstrip("_")
        else:
            base = stem

        out_path = out_dir / f"{base}_reselected_{out_rel_name}.xlsx".lstrip("_")
        try:
            sub.to_excel(out_path, index=False)
            logging.info(f"[{rel_type}] Saved reselected reliability file: {out_path}")
        except Exception as e:
            logging.error(f"[{rel_type}] Failed writing reselected file {out_path}: {e}")
