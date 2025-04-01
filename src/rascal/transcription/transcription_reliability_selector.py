import os
import random
import logging
import pandas as pd
from tqdm import tqdm

def select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    """
    Selects transcription reliability samples from CHAT files and writes empty CHAT files with headers preserved.

    Parameters:
    - tiers (dict): A dictionary of tier objects used for partitioning.
    - chats (list): A list of CHAT file paths.
    - frac (float): Fraction of files to be selected for reliability.
    - output_dir (str): Directory where the reliability samples will be saved.

    Returns:
    - None. Saves selected reliability samples and empty CHAT files.
    """
    
    logging.info("Starting transcription reliability sample selection.")
    
    # Sort chat files by unique combinations of partition tiers.
    partitions = {}
    for cha_file in chats:
        partition_tiers = [t.match(cha_file) for t in tiers.values() if t.partition]
        partition_tiers = [pt for pt in partition_tiers if pt is not None]
        
        if partition_tiers:
            partition_key = tuple(partition_tiers)
            if partition_key in partitions:
                partitions[partition_key].append(cha_file)
            else:
                partitions[partition_key] = [cha_file]
        else:
            logging.warning(f"Could not find partition tiers in '{cha_file}'")
    
    # Make transcription reliability folder.
    transc_rel_dir = os.path.join(output_dir, 'TranscriptionReliability')
    try:
        os.makedirs(transc_rel_dir, exist_ok=True)
        logging.info(f"Created transcription reliability directory: {transc_rel_dir}")
    except Exception as e:
        logging.error(f"Failed to create transcription reliability directory {transc_rel_dir}: {e}")
        return

    # Randomly select a fraction of files and write empty .cha files preserving header.
    columns = ['file'] + list(tiers.keys())
    for partition_tiers, cha_files in tqdm(partitions.items(), desc="Selecting transcription reliability subsets by partition."):
        rows = []
        partition_path = os.path.join(transc_rel_dir, *partition_tiers)
        try:
            os.makedirs(partition_path, exist_ok=True)
            logging.info(f"Created partition directory: {partition_path}")
        except Exception as e:
            logging.error(f"Failed to create partition directory {partition_path}: {e}")
            continue
        
        # Ensure no duplicates and at least one file is selected.
        subset_size = max(1, round(frac * len(cha_files)))  
        subset = random.sample(cha_files, k=subset_size)
        logging.info(f"Selected {subset_size} files for partition: {partition_tiers}")

        for cha_file in subset:
            labels = [t.match(cha_file) for t in tiers.values() if t.match(cha_file)]
            row = [cha_file] + labels
            rows.append(row)

            # Keep header but no utterances and write to output path.
            try:
                chat_data = chats[cha_file]
                strs = next(chat_data.to_strs())
                strs = ['@Begin'] + strs.split('\n') + ['@End']
                new_filename = cha_file.replace('.cha', '_Reliability.cha')
                filepath = os.path.join(partition_path, new_filename)
                with open(filepath, 'w') as f:
                    for line in strs:
                        if line.startswith('@'):
                            f.write(line + '\n')
                    logging.info(f"Written blank CHAT file with header: {filepath}")
            except Exception as e:
                logging.error(f"Failed to write blank CHAT file for {cha_file}: {e}")
    
        # Write the transcription reliability DataFrame.
        try:
            transc_rel_df = pd.DataFrame(rows, columns=columns)
            df_filepath = os.path.join(partition_path, f"{'_'.join(partition_tiers)}_TranscriptionReliabilitySamples.xlsx")
            transc_rel_df.to_excel(df_filepath, index=False)
            logging.info(f"Transcription reliability samples saved to: {df_filepath}")
        except Exception as e:
            logging.error(f"Failed to write transcription reliability samples DataFrame: {e}")
