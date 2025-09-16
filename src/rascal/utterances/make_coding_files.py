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
        logging.debug("Dropped 'file', 'test', and 'participantID' columns.")

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

            relsegdf['c3ID'] = ass[2]
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
       - Add empty coder-1 assignment column ('c1ID') and reliability comment ('WCcom').
       - Compute 'wordCount' for each utterance using `count_words(utterance, d)`
         if c2CU is not NaN, otherwise assign "NA".
       - Assign coders to samples via `assign_CU_coders(coders)` and
         distribute sample_ids across coders using `segment(...)`.
       - Select a fraction (`frac`) of samples for reliability per coder pair.
         For these, rename 'c1ID'→'c2ID' and 'WCcom'→'WCrelCom',
         and assign the second coder ID.

    Outputs
    -------
    Under "<output_dir>/WordCounts[/<partition_labels...>]":
      - "<labels>_WordCounting.xlsx"
        Full utterance-level coding frame with 'wordCount', 'c1ID', 'WCcom'.
      - "<labels>_WordCountingReliability.xlsx"
        Subset of samples (≈ frac of total) for reliability, with 'c2ID' and 'WCrelCom'.

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
        WCdf['wordCount'] = WCdf.apply(lambda row: count_words(row['utterance'], d) if not np.isnan(row['c2CU']) else 'NA', axis=1)

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
            relsegdf.rename(columns={'c1ID': 'c2ID', 'WCcom': 'WCrelCom'}, inplace=True)
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
