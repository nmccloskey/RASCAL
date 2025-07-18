import os
import re
import pylangacq
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Align import PairwiseAligner
import logging


def percent_difference(value1, value2):
    """
    Calculate percent difference between two values.
    """
    return (abs(value1 - value2) / ((value1 + value2) / 2)) * 100

def extract_cha_text(chat_data):
    """
    Extract text from CHAT data, excluding clinician ('INV') utterances.
    """
    text = ''
    for line in chat_data.utterances():
        if line.participant != 'INV':
            utterance = line.tiers.get(line.participant, "")
            # Remove space before utterance delimiter.
            utterance = re.sub(r'\s(?=[.!?])','',utterance)
            text += utterance + ' '
    # Remove non-word characters, regularize whitespace, and convert to lowercase.
    text = re.sub(r'[^\w\s\n.!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Helper function to wrap lines at approximately 80 characters or based on delimiters
def wrap_text(text, width=80):
    """
    Wrap text to a specified width or based on utterance delimiters for better readability.
    """
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        # Add the word to the current line if it doesn't exceed the width limit
        if len(current_line) + len(word) + 1 <= width:
            current_line += ' ' + word
        else:
            # If the width limit is exceeded, append the current line and start a new one
            lines.append(current_line)
            current_line = word

    # Append the last line if there is any content left
    if current_line:
        lines.append(current_line)

    return lines


def analyze_transcription_reliability(tiers, input_dir, output_dir, test=False):
    """
    Analyze transcription reliability by comparing original and reliability CHAT files.

    Parameters:
    - tiers (dict): Dictionary of tier information for partitioning.
    - input_dir (str): Directory containing input CHAT files.
    - output_dir (str): Directory where analysis results will be saved.
    - test (bool): If True, run in test mode.
    """
    dfcols = list(tiers.keys()) + ['OrgFile', 'RelFile',
                                   'OrgNumTokens', 'RelNumTokens', 'PercDiffNumTokens',
                                   'OrgNumChars', 'RelNumChars', 'PercDiffNumChars',
                                   'LevenshteinDistance','LevenshteinSimilarity',
                                   'NeedlemanWunschDistance', 'NeedlemanWunschScore']
    transc_rel_df = pd.DataFrame(columns=dfcols)

    # Make output folder.
    transc_rel_dir = os.path.join(output_dir, 'TranscriptionReliabilityAnalysis')
    try:
        os.makedirs(transc_rel_dir, exist_ok=True)
        logging.info(f"Created directory: {transc_rel_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {transc_rel_dir}: {e}")
        return

    # Output subfolders based on partition tiers.
    partition_tiers = [t.name for t in tiers.values() if t.partition]

    # Read all .cha files in input directory.
    cha_files = [cha for cha in Path(input_dir).rglob("*.cha")]
    logging.info(f"Found {len(cha_files)} .cha files in the input directory.")
    rel_chats = [cha for cha in cha_files if 'Reliability' in cha.name]
    org_chats = [cha for cha in cha_files if 'Reliability' not in cha.name]
    
    logging.info("Analyzing transcription reliability...")
    for rel_cha in tqdm(rel_chats, desc="Analyzing reliability transcripts"):
        rel_labels = [t.match(rel_cha.name) for t in tiers.values()]
        logging.debug(f"Reliability file labels: {rel_labels}")

        for org_cha in org_chats:
            org_labels = [t.match(org_cha.name) for t in tiers.values()]
            # logging.info(f"Original file labels: {org_labels}")

            if rel_labels == org_labels:
                try:
                    org_chat_data = pylangacq.read_chat(str(org_cha))
                    rel_chat_data = pylangacq.read_chat(str(rel_cha))
                    logging.info(f"Matching original file: {org_cha.name} with reliability file: {rel_cha.name}")
                except Exception as e:
                    logging.error(f"Failed to read CHAT files {org_cha} or {rel_cha}: {e}")
                    continue

                try:
                    # Extract text from both samples.
                    org_text = extract_cha_text(org_chat_data)
                    rel_text = extract_cha_text(rel_chat_data)

                    # Simple analysis.
                    org_num_tokens = len(org_text.split(' '))
                    rel_num_tokens = len(rel_text.split(' '))
                    pdiff_num_tokens = percent_difference(org_num_tokens, rel_num_tokens)
                    org_num_chars = len(org_text)
                    rel_num_chars = len(rel_text)
                    pdiff_num_chars = percent_difference(org_num_chars, rel_num_chars)
                    
                    # Levenshtein algorithm.
                    Ldist = distance(org_text, rel_text)
                    max_len = max(len(org_text), len(rel_text))
                    Lscore = 1 - (Ldist / max_len)

                    # Initialize the Needleman-Wunsch algorithm aligner
                    aligner = PairwiseAligner()
                    aligner.mode = 'global'
                    alignments = aligner.align(org_text, rel_text)
                    best_alignment = alignments[0]
                    best_score = best_alignment.score
                    longer_length = max(len(org_text), len(rel_text))
                    normalized_score = best_score / longer_length

                    row = rel_labels + [org_cha.name, rel_cha.name,
                                        org_num_tokens, rel_num_tokens, pdiff_num_tokens,
                                        org_num_chars, rel_num_chars, pdiff_num_chars,
                                        Ldist, Lscore, best_score, normalized_score]
                    transc_rel_df.loc[len(transc_rel_df)] = row
                    logging.debug(f"Appended row to DataFrame: {row}")

                    # Prepare the alignment output as formatted text
                    alignment_str = f"Global alignment score: {best_score}\n"
                    alignment_str += f"Normalized score (by length): {normalized_score}\n\n"

                    # Prepare the alignment text with match indicators
                    seq1 = best_alignment[0]
                    seq2 = best_alignment[1]

                    # Wrap each line to 80 characters
                    seq1_lines = wrap_text(seq1)
                    seq2_lines = wrap_text(seq2)

                    for s1, s2 in zip(seq1_lines, seq2_lines):
                        alignment_str += f"Sequence 1: {s1}\n"
                        alignment_line = ''.join(['|' if a == b else ' ' for a, b in zip(s1, s2)])
                        alignment_str += f"Alignment : {alignment_line}\n"
                        alignment_str += f"Sequence 2: {s2}\n\n"

                    # Extract partition tier info from file name.
                    partition_labels = [t.match(rel_cha.name) for t in tiers.values() if t.partition]
                    text_filename = f"{''.join(rel_labels)}_TranscriptionReliabilityAlignment.txt"
                    text_file_path = os.path.join(transc_rel_dir, *partition_labels, text_filename)
                    try:
                        os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
                        with open(text_file_path, 'w') as file:
                            file.write(alignment_str)
                        # logging.info(f"Saved alignment text to: {text_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to write alignment file {text_file_path}: {e}")
                
                except Exception as e:
                    logging.error(f"Failed to analyze transcription reliability for {org_cha} and {rel_cha}: {e}.")
    
    # Store results for testing.
    results = []
    
    # Partition by designated tier(s).
    for tup, subdf in tqdm(transc_rel_df.groupby(partition_tiers), desc="Saving grouped DataFrames"):
        df_filename = '_'.join(tup) + '_TranscriptionReliabilityAnalysis.xlsx'
        df_path = os.path.join(transc_rel_dir, *tup, df_filename)
        try:
            subdf.to_excel(df_path, index=False)
            logging.info(f"Saved reliability analysis DataFrame to: {df_path}")
        except Exception as e:
            logging.error(f"Failed to write DataFrame to {df_path}: {e}")
        
        if test:
            results.append(subdf)
    
    if test:
        return results
