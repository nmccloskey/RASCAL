import os
import random
import logging
import pandas as pd
from tqdm import tqdm

def select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    """
    Selects transcription reliability samples from CHAT files and writes empty CHAT files with headers preserved.
    """
    logging.info("Starting transcription reliability sample selection.")

    # Create a mapping of partition keys to files
    partitions = {}
    has_partition = any(t.partition for t in tiers.values())

    for cha_file in chats:
        if has_partition:
            partition_tiers = [t.match(cha_file) for t in tiers.values() if t.partition]
            partition_tiers = [pt for pt in partition_tiers if pt is not None]
            if not partition_tiers:
                logging.warning(f"No partition tiers matched for '{cha_file}', skipping.")
                continue
            partition_key = tuple(partition_tiers)
        else:
            partition_key = ('ALL',)  # fallback group

        partitions.setdefault(partition_key, []).append(cha_file)

    # Create output directory
    transc_rel_dir = os.path.join(output_dir, 'TranscriptionReliability')
    try:
        os.makedirs(transc_rel_dir, exist_ok=True)
        logging.info(f"Created transcription reliability directory: {transc_rel_dir}")
    except Exception as e:
        logging.error(f"Failed to create transcription reliability directory {transc_rel_dir}: {e}")
        return

    columns = ['file'] + list(tiers.keys())

    for partition_tiers, cha_files in tqdm(partitions.items(), desc="Selecting transcription reliability subsets"):
        rows = []

        if partition_tiers == ('ALL',):
            partition_path = transc_rel_dir
        else:
            partition_path = os.path.join(transc_rel_dir, *partition_tiers)

        try:
            os.makedirs(partition_path, exist_ok=True)
            logging.info(f"Created partition directory: {partition_path}")
        except Exception as e:
            logging.error(f"Failed to create partition directory {partition_path}: {e}")
            continue

        subset_size = max(1, round(frac * len(cha_files)))
        subset = random.sample(cha_files, k=subset_size)
        logging.info(f"Selected {subset_size} files for partition: {partition_tiers}")

        for cha_file in subset:
            labels = [t.match(cha_file) for t in tiers.values()]
            row = [cha_file] + labels
            rows.append(row)

            try:
                chat_data = chats[cha_file]
                strs = next(chat_data.to_strs())
                strs = ['@Begin'] + strs.split('\n') + ['@End']
                new_filename = os.path.basename(cha_file).replace('.cha', '_Reliability.cha')
                filepath = os.path.join(partition_path, new_filename)
                with open(filepath, 'w') as f:
                    for line in strs:
                        if line.startswith('@'):
                            f.write(line + '\n')
                    logging.info(f"Written blank CHAT file with header: {filepath}")
            except Exception as e:
                logging.error(f"Failed to write blank CHAT file for {cha_file}: {e}")

        try:
            transc_rel_df = pd.DataFrame(rows, columns=columns)
            suffix = '_'.join(partition_tiers) if partition_tiers != ('ALL',) else 'ALL'
            df_filepath = os.path.join(partition_path, f"{suffix}_TranscriptionReliabilitySamples.xlsx")
            transc_rel_df.to_excel(df_filepath, index=False)
            logging.info(f"Transcription reliability samples saved to: {df_filepath}")
        except Exception as e:
            logging.error(f"Failed to write transcription reliability samples DataFrame: {e}")
