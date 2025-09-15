import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def make_timesheets(tiers, input_dir, output_dir, test=False):
    """
    Make excel files for recording speaking times.
    """
    
    # Make timesheet file path.
    timesheet_dir = os.path.join(output_dir, 'TimeSheets')
    logging.info(f"Writing time sheet files to {timesheet_dir}")

    # Store results for test.
    results = []

    utterance_files = list(Path(input_dir).rglob("*_Utterances.xlsx")) + list(Path(output_dir).rglob("*_Utterances.xlsx"))

    # Convert utterance files to CU coding files.
    for file in tqdm(utterance_files, desc="Generating time table files"):
        logging.info(f"Processing file: {file}")
        
        # Extract partition tier info from file name.
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]
        logging.debug(f"Extracted labels: {labels}")

        # Read utterances.
        try:
            uttdf = pd.read_excel(str(file))
            logging.info(f"Successfully read file: {file}")
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")
            continue

        time_df = uttdf.drop(columns=['UtteranceID', 'speaker','utterance','comment'])
        logging.debug("Dropped CU-specific columns.")
        time_df.drop_duplicates(inplace=True)

        empty_col = [np.nan for _ in range(len(time_df))]
        for col in ['total_time', 'clinician_time', 'client_time']:
            time_df = time_df.assign(**{col: empty_col})
        
        # Sort by tiers.
        time_df.sort_values(by=list(tiers.keys()), inplace=True)

        # Write file.
        filename = os.path.join(timesheet_dir, *labels, '_'.join(labels) + '_SpeakingTimes.xlsx')
        logging.info(f"Writing speaking times file: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            time_df.to_excel(filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write speaking times file {filename}: {e}")
        
        if test:
            results.append(time_df)
        
    if test:
        return results
