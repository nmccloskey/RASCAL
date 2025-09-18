import os
import random
import logging
import pandas as pd
from tqdm import tqdm


def select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    """
    Select transcription reliability samples from CHAT transcripts and save 
    both the selected subset and the full set of candidate files.

    Operation
    ---------
    1. Collect CHAT files into groups defined by partition tiers (if any).
       - If at least one tier has `partition=True`, files are divided into
         subgroups by the matching tier values.
       - If no partition tiers exist, all files are grouped together in the
         top-level directory without any fallback label like "ALL".
    2. For each group of files:
       - Randomly select a fraction of CHAT files (at least 1).
       - Write empty `.cha` reliability files with headers preserved 
         (metadata lines only).
       - Construct two DataFrames:
         * "Reliability": the sampled subset.
         * "AllTranscripts": all files in the group with their tier labels.
       - Save these DataFrames into a single Excel workbook with two sheets.
    3. Each partition/group is written to its own subdirectory of 
       `output_dir/TranscriptionReliability`.

    Parameters
    ----------
    tiers : dict[str, Tier]
        Mapping of tier labels to Tier objects, each of which provides a 
        `.match(filename)` method and a `.partition` attribute.
    chats : dict[str, ChatFile]
        Mapping of CHAT file paths to parsed CHAT objects. Each CHAT object
        must implement `.to_strs()` yielding full transcript text.
    frac : float
        Fraction of files per group to select for reliability sampling 
        (e.g., 0.2 selects 20% of files, minimum of 1).
    output_dir : str
        Path to the output directory. A subfolder called 
        `TranscriptionReliability` will be created here.

    Output
    ------
    For each partition/group:
    - Blank CHAT files ending with `_Reliability.cha`.
    - An Excel file with two sheets:
      * "Reliability": sampled subset.
      * "AllTranscripts": all CHAT files with tier labels.
    """
    logging.info("Starting transcription reliability sample selection.")

    # Determine whether partitions are in play
    has_partition = any(t.partition for t in tiers.values())
    partitions = {}

    for cha_file in chats:
        if has_partition:
            partition_tiers = [t.match(cha_file) for t in tiers.values() if t.partition]
            partition_tiers = [pt for pt in partition_tiers if pt is not None]
            if not partition_tiers:
                logging.warning(f"No partition tiers matched for '{cha_file}', skipping.")
                continue
            partition_key = tuple(partition_tiers)
        else:
            partition_key = tuple()  # single "no-partition" group

        partitions.setdefault(partition_key, []).append(cha_file)

    # Create output directory
    transc_rel_dir = os.path.join(output_dir, 'TranscriptionReliability')
    os.makedirs(transc_rel_dir, exist_ok=True)

    columns = ['file'] + list(tiers.keys())

    for partition_tiers, cha_files in tqdm(partitions.items(), desc="Selecting reliability subsets"):
        rows_all = []
        rows_subset = []

        # Directory for this partition
        partition_path = os.path.join(transc_rel_dir, *partition_tiers) if partition_tiers else transc_rel_dir
        os.makedirs(partition_path, exist_ok=True)

        # Select subset
        subset_size = max(1, round(frac * len(cha_files)))
        subset = random.sample(cha_files, k=subset_size)
        logging.info(f"Selected {subset_size} files for partition {partition_tiers or 'root'}.")

        for cha_file in cha_files:
            labels = [t.match(cha_file) for t in tiers.values()]
            row = [cha_file] + labels
            rows_all.append(row)
            if cha_file in subset:
                rows_subset.append(row)

                # Write blank CHAT file with headers only
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

        # Write Excel with two sheets
        try:
            df_all = pd.DataFrame(rows_all, columns=columns)
            df_subset = pd.DataFrame(rows_subset, columns=columns)
            suffix = '_'.join(partition_tiers) if partition_tiers else 'TranscriptionReliabilitySamples'
            df_filepath = os.path.join(partition_path, f"{suffix}.xlsx")

            with pd.ExcelWriter(df_filepath) as writer:
                df_subset.to_excel(writer, sheet_name="Reliability", index=False)
                df_all.to_excel(writer, sheet_name="AllTranscripts", index=False)

            logging.info(f"Reliability Excel saved to: {df_filepath}")
        except Exception as e:
            logging.error(f"Failed to write reliability Excel for partition {partition_tiers}: {e}")
