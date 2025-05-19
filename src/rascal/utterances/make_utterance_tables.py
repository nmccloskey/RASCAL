import os
import logging
import pandas as pd
from tqdm import tqdm


def prepare_utterance_dfs(tiers, chats, output_dir, test=False):
    """
    Process CHAT files and create DataFrames of utterances with blind codes.

    Parameters:
    - tiers (dict): A dictionary of tier information used for matching file names.
    - chats (dict): A dictionary of chat data extracted from CHAT files.
    - output_dir (str): The directory where output files will be saved.

    Returns:
    - None. Saves DataFrames of utterances to Excel files. (Except for testing.)
    """
    
    # Make Utterances folder.
    utterance_dir = os.path.join(output_dir, 'Utterances')
    try:
        os.makedirs(utterance_dir, exist_ok=True)
        logging.info(f"Created directory: {utterance_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {utterance_dir}: {e}")
        return

    # Utterance DataFrame columns.
    dfcols = ['file'] + list(tiers.keys()) + ['sampleID', 'speaker', 'utterance', 'comment']
    rows = []

    # Partition by designated tier(s).
    partition_tiers = [t.name for t in tiers.values() if t.partition]
    logging.info(f"Partitioning by tiers: {partition_tiers}")

    # Iterate through each chat file.
    for i, chat_file in enumerate(chats):
        logging.info(f"Processing chat file: {chat_file}")

        # Extract tier info from file name.
        labels = [t.match(chat_file) for t in tiers.values()]
        logging.debug(f"Extracted labels: {labels}")

        # Make sample ID (blind code) from partition tier.
        partition_labels = [t.match(chat_file) for t in tiers.values() if t.partition]
        sampleID = f"{''.join(partition_labels)}S{i}"
        logging.debug(f"Generated sampleID: {sampleID}")
        
        chat_data = chats[chat_file]

        # Extract utterances from chat data.
        for line in chat_data.utterances():
            speaker = line.participant
            if speaker not in line.tiers:
                logging.warning(f"Speaker '{speaker}' not found in line tiers for file '{chat_file}'.")
            utterance = line.tiers.get(speaker, "")
            comment = line.tiers.get('%com', None)
            row = [chat_file] + labels + [sampleID, speaker, utterance, comment]
            rows.append(row)
    
    # Create DataFrame from collected rows.
    logging.info("Creating DataFrame from collected rows.")
    utterance_df = pd.DataFrame(rows, columns=dfcols)

    # Store results for test.
    results = []

    # Write utterance tables to output directory.
    logging.info(f"Writing Utterance tables to {utterance_dir}")
    if not partition_tiers:
        # No partitioning — just write all rows to a single file
        utterance_df.insert(0, column='UtteranceID', value='U' + utterance_df.reset_index().index.astype(str))
        filename = os.path.join(utterance_dir, 'Utterances.xlsx')
        logging.info(f"Writing file: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            utterance_df.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write file {filename}: {e}")
        if test:
            results.append(utterance_df)
    else:
        for tup, subdf in tqdm(utterance_df.groupby(partition_tiers), desc="Writing utterance files"):
            subdf.insert(0, column='UtteranceID', value=''.join(tup) + 'U' + subdf.reset_index().index.astype(str))
            output_path = [utterance_dir] + list(tup) if len(partition_tiers) > 1 else [utterance_dir]
            filename = os.path.join(*output_path, '_'.join(tup) + '_Utterances.xlsx')
            logging.info(f"Writing file: {filename}")
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                subdf.to_excel(filename, index=False)
            except Exception as e:
                logging.error(f"Failed to write file {filename}: {e}")
            if test:
                results.append(subdf)
            
    if test:
        return results
