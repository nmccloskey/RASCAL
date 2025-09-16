import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def make_timesheets(tiers, input_dir, output_dir):
    """
    Generate blank speaking-time entry sheets from utterance-level files.

    Workflow
    --------
    1. Search `input_dir` and `output_dir` recursively for files matching
       "*Utterances.xlsx".
    2. For each utterance file:
       - Extract partition labels from the filename using `tiers`.
       - Read the utterance DataFrame.
       - Drop non-time-relevant columns: 'UtteranceID','speaker','utterance','comment'.
       - Drop duplicate rows (leaving one row per unique sample/partition).
       - Add empty columns:
           * 'total_time'
           * 'clinician_time'
           * 'client_time'
         (all initialized to NaN).
       - Sort rows by the order of `tiers.keys()`.
       - Save to Excel in:
           "<output_dir>/TimeSheets[/<partition_labels...>]/<labels>_SpeakingTimes.xlsx"

    Parameters
    ----------
    tiers : dict[str, Any]
        Tier objects with `.match(filename, return_None=True)` and `.partition`.
        Used for extracting labels from filenames and sorting.
    input_dir : str | os.PathLike
        Directory containing utterance-level Excel files.
    output_dir : str | os.PathLike
        Directory where time sheet Excel files are written.

    Outputs
    -------
    - One Excel file per input utterance file, under "TimeSheets/".
    - Each sheet includes sample/tier identifiers and empty columns for
      time coding.

    Returns
    -------
    None
        Saves results to disk; does not return a value.

    Notes
    -----
    - If an utterance file cannot be read, it is skipped with an error logged.
    - Partition labels determine subdirectory and file prefix.
    """
    
    # Make timesheet file path.
    timesheet_dir = os.path.join(output_dir, 'TimeSheets')
    logging.info(f"Writing time sheet files to {timesheet_dir}")

    utterance_files = list(Path(input_dir).rglob("*Utterances.xlsx")) + list(Path(output_dir).rglob("*Utterances.xlsx"))

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

        time_df = uttdf.drop(columns=['utterance_id', 'speaker','utterance','comment'])
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
