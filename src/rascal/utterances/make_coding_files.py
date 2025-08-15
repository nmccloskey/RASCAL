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

def make_CU_coding_files(tiers, frac, coders, input_dir, output_dir, CU_paradigms, exclude_participants, test=False):
    """
    Generate CU coding and CU reliability coding files from utterance DataFrames.
    """

    if len(coders) < 3:
        logging.warning(f"Coders entered: {coders} do not meet minimum of 3. Using default 1, 2, 3.")
        coders = ['1', '2', '3']

    base_cols = ['c1ID', 'c1SV', 'c1REL', 'c1com', 'c2ID', 'c2SV', 'c2REL', 'c2com']
    CU_coding_dir = os.path.join(output_dir, 'CUCoding')
    logging.info(f"Writing CU coding files to {CU_coding_dir}")
    utterance_files = list(Path(input_dir).rglob("*_Utterances.xlsx")) + list(Path(output_dir).rglob("*_Utterances.xlsx"))
    results = []

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

        # UPDATE Just select columns that are desired as opposed to dropping 
        CUdf = uttdf.drop(columns=[col for col in ['file', 'test', 'participantID'] if col in uttdf.columns]).copy()
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

        unique_sample_ids = list(CUdf['sampleID'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))
        rel_subsets = []

        for seg, ass in zip(segments, assignments):
            CUdf.loc[CUdf['sampleID'].isin(seg), 'c1ID'] = ass[0]
            CUdf.loc[CUdf['sampleID'].isin(seg), 'c2ID'] = ass[1]

            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = CUdf[CUdf['sampleID'].isin(rel_samples)].copy()

            for col in ['c1ID', 'c1com', 'c2ID', 'c2com']:
                relsegdf.drop(columns=[col], inplace=True, errors='ignore')

            # Drop all SV/REL columns, whether original or suffixed
            if len(CU_paradigms) >= 2:
                for prefix in ['c1', 'c2']:
                    for tag in ['SV', 'REL']:
                        for paradigm in CU_paradigms:
                            relsegdf.drop(columns=[f"{prefix}{tag}_{paradigm}"], inplace=True, errors='ignore')
            else:
                for col in ['c1SV', 'c1REL', 'c2SV', 'c2REL']:
                    relsegdf.drop(columns=[col], inplace=True, errors='ignore')

            # Rename c2 -> c3
            relsegdf.rename(columns={
                'c2ID': 'c3ID',
                'c2com': 'c3com'
            }, inplace=True)

            if len(CU_paradigms) >= 2:
                for tag in ['SV', 'REL']:
                    for paradigm in CU_paradigms:
                        relsegdf.rename(columns={f"c2{tag}_{paradigm}": f"c3{tag}_{paradigm}"}, inplace=True)
                        relsegdf[f"c3{tag}_{paradigm}"] = relsegdf.get(f"c3{tag}_{paradigm}", np.nan)
            else:
                relsegdf.rename(columns={
                    'c2SV': 'c3SV',
                    'c2REL': 'c3REL'
                }, inplace=True)

            relsegdf['c3ID'] = ass[2]
            rel_subsets.append(relsegdf)

        reldf = pd.concat(rel_subsets)
        logging.info(f"Selected {len(set(reldf['sampleID']))} samples for reliability from {len(set(CUdf['sampleID']))} total samples.")

        cu_filename = os.path.join(CU_coding_dir, *labels, '_'.join(labels) + '_CUCoding.xlsx')
        rel_filename = os.path.join(CU_coding_dir, *labels, '_'.join(labels) + '_CUReliabilityCoding.xlsx')

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

        if test:
            results.append(CUdf)

    if test:
        return results


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
    
    # Handle parentheses
    open_par_count = text.count('(')
    closed_par_count = text.count(')')
    if open_par_count == closed_par_count:
        for _ in range(open_par_count):
            text = re.sub(r'\([^\(]*?\)', '', text)
    else:
        print('Mismatched parentheses detected')
    
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

def make_word_count_files(tiers, frac, coders, input_dir, output_dir, test=False):
    """
    Generate word count coding and reliability files from CU coding DataFrames.

    Parameters:
    - tiers (dict): Dictionary of tier objects used for partitioning.
    - frac (float): Fraction of samples to be selected for reliability.
    - CU_coding_dir (str): Directory containing the CU coding Excel files.
    - input_dir (str): Directory where the CU coding files could be saved.
    - output_dir (str): Directory where the CU coding files could be saved.

    Returns:
    - None. Saves word count coding and reliability coding files to output directory.
    """
    
    # Make word count coding file path.
    word_count_dir = os.path.join(output_dir, 'WordCounts')
    logging.info(f"Writing word count files to {word_count_dir}")

    # Store results for test.
    results = []

    # Convert utterance-level CU coding files to word counting files.
    CU_files = list(Path(input_dir).rglob("*_CUCoding_ByUtterance.xlsx")) + list(Path(output_dir).rglob("*_CUCoding_ByUtterance.xlsx"))
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
        unique_sample_ids = list(WCdf['sampleID'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))

        # Assign coders and prep reliability file.
        rel_subsets = []
        for seg, ass in zip(segments, assignments):
            WCdf.loc[WCdf['sampleID'].isin(seg), 'c1ID'] = ass[0]
            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = WCdf[WCdf['sampleID'].isin(rel_samples)].copy()
            relsegdf.rename(columns={'c1ID': 'c2ID', 'WCcom': 'WCrelCom'}, inplace=True)
            relsegdf['c2ID'] = ass[1]
            rel_subsets.append(relsegdf)

        WCreldf = pd.concat(rel_subsets)
        logging.info(f"Selected {len(set(WCreldf['sampleID']))} samples for reliability from {len(set(WCdf['sampleID']))} total samples.")

        # Save word count coding file.
        filename = os.path.join(word_count_dir, *labels, '_'.join(labels) + '_WordCounting.xlsx')
        logging.info(f"Writing word counting file: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            WCdf.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write word count coding file {filename}: {e}")

        # Word count reliability coding file.
        filename = os.path.join(word_count_dir, *labels, '_'.join(labels) + '_WordCountingReliability.xlsx')
        logging.info(f"Writing word count reliability coding file: {filename}")
        try:
            WCreldf.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write word count reliability coding file {filename}: {e}")
        
        if test:
            results.append(WCdf)
        
    if test:
        return results
