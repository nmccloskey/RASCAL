import os
import re
import random
import enchant
import logging
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path


def count_words(text, d):
    """Prepare a transcription text string for counting words."""
    # Capital letters shouldn't count as different.
    text = text.lower().strip('\n').strip(' ')
    # Accommodate "he/she/it's got" so that "have" is counted.
    text = re.sub(r"(?<=(he|it))'s got",' has got',text)
    # Expand contracted forms.
    text = ' '.join([contractions.fix(w) for w in text.split(' ')])
    # Remove weird space encoding.
    text = text.replace(u'\xa0','')
    # Remove um, uh, eh, er.
    text = re.sub(r'(^|\b)(u|e)+?(h|m|r)+?(\b|$)','',text)
    # Remove unintelligibles.
    text = re.sub(r'(^|\b|\b.)x+(\b|$)','',text)
    # # Replace double parentheses.
    # Count open and closed parentheses.
    open_par_count = text.count('(')
    closed_par_count = text.count(')')
    # If equal numbers,
    if open_par_count == closed_par_count:
        # Iteratively remove content to handle nesting (()).
        for _ in range(open_par_count):
            text = re.sub(r'\([^\(]*?\)','',text)
    else:
        # Direct user to sample that requires editing.
        print('has mismatched number of ()')
    text = re.sub(r'\[.+?\]','',text)
    # Remove everything between asterisks (gesturing, etc.).
    text = re.sub(r'\*.+?\*','',text)
    # Also, spell out all numbers.
    text = re.sub(r'\d+',lambda x: n2w.num2words(int(x.group(0))),text)
    # And non-word characters.
    text = re.sub(r'[^\w\s]',' ',text)
    # Remove clinician utterance marker.
    text = re.sub(r'(\bcl\b)','',text)
    # Remove extra spaces.
    text = re.sub(r'\s{2,}',' ',text)
    # Remove tabs.
    text = re.sub(r'\t','',text)
    # Strip whitespace.
    text = text.strip(' ')
    # Tokenize and count.
    words = [w for w in text.split(' ') if w and d.check(w)]
    return len(words)


def make_word_count_files(tiers, frac, input_dir, output_dir, test=False):
    """
    Generate word count coding and reliability files from CU coding DataFrames.

    Parameters:
    - tiers (dict): Dictionary of tier objects used for partitioning.
    - frac (float): Fraction of samples to be selected for reliability.
    - input_dir (str): Directory possibly containing the CU coding Excel files.
    - output_dir (str): Directory where the CU coding files should be saved.

    Returns:
    - None. Saves word count coding and reliability coding files to output directory.
    """

    # Initialize dictionary of English words.
    d = enchant.Dict('en_US')

    # Make word count coding file path.
    word_count_dir = os.path.join(output_dir, 'WordCounts')
    logging.info(f"Writing word count files to {word_count_dir}")

    # Store results for test.
    results = []

    # Convert utterance-level CU coding files to word counting files.
    cu_utterance_files = list(Path(input_dir).rglob("*_CUCoding_ByUtterance.xlsx")) + list(Path(output_dir).rglob("*_CUCoding_ByUtterance.xlsx"))
    for file in tqdm(cu_utterance_files, desc="Generating word count coding files"):
        logging.info(f"Processing file: {file}")
        
        # Extract partition tier info from file name.
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]
        logging.debug(f"Extracted labels: {labels}")

        # Read CUs and pull neutrality from CU2.
        try:
            CUdf = pd.read_excel(str(file))
            logging.info(f"Successfully read file: {file}")
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")
            continue
        
        CUdf['wordCount'] = CUdf.apply(lambda row: count_words(row['utterance'], d) if not np.isnan(row['CU2']) else 'NA')

        WCdf = CUdf.drop(columns=['c1SV', 'c1REL', 'c1CU', 'c2SV', 'c2REL', 'c2CU'])
        logging.debug("Dropped CU-specific columns.")

        # Select samples for reliability.
        samples = list(set(WCdf['sampleID']))
        rel_samples = random.sample(samples, k=max(1, round(len(samples) * frac)))
        WCreldf = WCdf[WCdf['sampleID'].isin(rel_samples)]
        logging.info(f"Selected {len(rel_samples)} samples for reliability from {len(samples)} total samples.")

        # Word count coding file.
        empty_col = [np.nan for _ in range(len(WCdf))]
        CUdf = CUdf.assign(**{'WCcomment': empty_col})
        filename = os.path.join(word_count_dir, *labels, '_'.join(labels) + '_WordCounting.xlsx')
        logging.info(f"Writing word counting file: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            WCdf.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write word count coding file {filename}: {e}")

        # Word count reliability coding file.
        empty_col = [np.nan for _ in range(len(WCreldf))]
        WCreldf = WCreldf.assign(**{'WCRelcomment': empty_col})
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
