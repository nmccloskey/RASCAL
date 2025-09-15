import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def percent_difference(value1, value2):
    """
    Calculates the percentage difference between two values.

    Args:
        value1 (float): The first value.
        value2 (float): The second value.

    Returns:
        float: The percentage difference, or infinity if either value is zero.
    """
    if value1 == 0 or value2 == 0:
        logging.warning("One of the values is zero, returning 100%.")
        return 100
    elif value1 == value2 == 0:
        return 0

    diff = abs(value1 - value2)
    avg = (value1 + value2) / 2
    return round((diff / avg) * 100, 2)

def agreement(row):
    abs_diff = abs(row['wordCount_org'] - row['wordCount_rel'])
    if abs_diff <= 1:
        return 1
    else:
        perc_diff = percent_difference(row['wordCount_org'], row['wordCount_rel'])
        perc_sim = 100 - perc_diff
        return 1 if perc_sim >= 85 else 0

def calculate_icc(data):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for two raters.
    
    Args:
        data (pd.DataFrame): A dataframe with two columns: 'wordCount_org' and 'wordCount_rel'.
        
    Returns:
        float: ICC(2,1) value.
    """
    # Number of subjects and raters
    n = data.shape[0]  # subjects (utterances)
    k = data.shape[1]  # raters (original, reliability)

    # Mean per subject (row mean)
    mean_per_subject = data.mean(axis=1)
    
    # Mean per rater (column mean)
    mean_per_rater = data.mean(axis=0)

    # Grand mean
    grand_mean = data.values.flatten().mean()

    # Between-subject sum of squares (SSB)
    ss_between = np.sum((mean_per_subject - grand_mean)**2) * k

    # Within-subject sum of squares (SSW)
    ss_within = np.sum((data.values - mean_per_subject.values[:, None])**2)

    # Between-rater sum of squares (SSR)
    ss_rater = np.sum((mean_per_rater - grand_mean)**2) * n

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / ((n * (k - 1)))
    ms_rater = ss_rater / (k - 1)

    # ICC(2,1) formula
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within + (k / n) * (ms_rater - ms_within))

    return round(icc, 4)

def analyze_word_count_reliability(tiers, input_dir, output_dir):
    """
    Analyze word count reliability by comparing coder-1 word counts with
    coder-2 reliability word counts.

    Workflow
    --------
    1. Collect all "*_WordCounting.xlsx" (coding) and
       "*_WordCountingReliability.xlsx" (reliability) files under `input_dir`.
    2. For each reliability file, find the coding file with matching tier labels.
    3. Read both DataFrames and clean the reliability frame to
       ['utterance_id','WCrelCom','wordCount'], dropping NaN word counts.
    4. Merge on 'utterance_id' with suffixes (_org for coding, _rel for reliability).
    5. For each utterance, compute:
         - AbsDiff  : raw difference (org − rel)
         - PercDiff : percent difference (using `percent_difference`)
         - PercSim  : 100 − PercDiff
         - AG       : binary agreement (1 if abs diff ≤1 or percSim ≥85)
    6. Save merged results to
         "<output_dir>/WordCountReliability/<partition_labels>/<labels>_WordCountingReliabilityResults.xlsx"
    7. Compute ICC(2,1) across utterances (using `calculate_icc`).
    8. Write a plain-text report:
         "<labels>_WordCountReliabilityReport.txt"
       with number/percent of utterances agreed and ICC value.

    Parameters
    ----------
    tiers : dict[str, Any]
        Mapping of tier name -> tier object, each with:
          - .match(filename, return_None=True) → label string
          - .partition flag → whether included in output path.
    input_dir : str | os.PathLike
        Directory searched recursively for coding and reliability files.
    output_dir : str | os.PathLike
        Directory where results are written under "WordCountReliability/".

    Outputs
    -------
    - Excel file with merged utterance-level reliability results and agreement.
    - Text report with summary agreement and ICC.

    Returns
    -------
    None
        Results are written to disk; function has no return value.

    Notes
    -----
    - Logs warnings if file read/merge fails or row counts mismatch.
    - Agreement rule: abs diff ≤1 OR percent similarity ≥85%.
    """

    # Make Word Count Reliability folder
    WordCountReliability_dir = os.path.join(output_dir, 'WordCountReliability')
    try:
        os.makedirs(WordCountReliability_dir, exist_ok=True)
        logging.info(f"Created directory: {WordCountReliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {WordCountReliability_dir}: {e}")
        return

    # Collect relevant files
    coding_files = [f for f in Path(input_dir).rglob('*_WordCounting.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*_WordCountingReliability.xlsx')]

    # Match word counting and reliability files
    for rel in tqdm(rel_files, desc="Analyzing word count reliability..."):
        # Extract tier info from file name
        rel_labels = [t.match(rel.name, return_None=True) for t in tiers.values()]

        for cod in coding_files:
            cod_labels = [t.match(cod.name, return_None=True) for t in tiers.values()]

            if rel_labels == cod_labels:
                try:
                    WCdf = pd.read_excel(cod)
                    WCreldf = pd.read_excel(rel)
                    logging.info(f"Processing coding file: {cod} and reliability file: {rel}")
                except Exception as e:
                    logging.error(f"Failed to read files {cod} or {rel}: {e}")
                    continue

                # Clean and filter the reliability DataFrame
                WCreldf = WCreldf.loc[:, ['utterance_id', 'WCrelCom', 'wordCount']]
                WCreldf = WCreldf[~np.isnan(WCreldf['wordCount'])]

                # Merge on utterance_id
                try:
                    WCmerged = pd.merge(WCdf, WCreldf, on="utterance_id", how="inner", suffixes=('_org', '_rel'))
                    logging.info(f"Merged reliability file with coding file for {rel.name}")
                except Exception as e:
                    logging.error(f"Failed to merge {cod.name} with {rel.name}: {e}")
                    continue

                if len(WCreldf) != len(WCmerged):
                    logging.error(f"Length mismatch between reliability and joined files for {rel.name}.")

                # Calculate percent difference
                WCmerged['AbsDiff'] = WCmerged.apply(lambda row: row['wordCount_org'] - row['wordCount_rel'], axis=1)
                WCmerged['PercDiff'] = WCmerged.apply(lambda row: percent_difference(row['wordCount_org'], row['wordCount_rel']), axis=1)
                WCmerged['PercSim'] = 100 - WCmerged['PercDiff']
                WCmerged['AG'] = WCmerged.apply(agreement, axis=1)

                # Create output directory
                partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]
                output_path = os.path.join(WordCountReliability_dir, *partition_labels)
                try:
                    os.makedirs(output_path, exist_ok=True)
                    logging.info(f"Created partition directory: {output_path}")
                except Exception as e:
                    logging.error(f"Failed to create partition directory {output_path}: {e}")
                    continue

                # Write tables.
                filename = os.path.join(output_path, '_'.join(partition_labels) + '_WordCountingReliabilityResults.xlsx')
                logging.info(f"Writing word counting reliability results file: {filename}")
                try:
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    WCmerged.to_excel(filename, index=False)
                except Exception as e:
                    logging.error(f"Failed to write word count reliability results file {filename}: {e}")
                
                # Subset the data for ICC calculation
                icc_data = WCmerged[['wordCount_org', 'wordCount_rel']].dropna()

                # Calculate ICC
                icc_value = calculate_icc(icc_data)
                logging.info(f"Calculated ICC(2,1) for {rel.name}: {icc_value}")

                # Write reliability report
                num_samples_AG = np.nansum(WCmerged['AG'])
                perc_samples_AG = round((num_samples_AG / len(WCmerged)) * 100, 1)
                report_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_WordCountReliabilityReport.txt")
                try:
                    with open(report_path, 'w') as report:
                        report.write(f"Word Count Reliability Report for {' '.join(partition_labels)}\n\n")
                        report.write(f"Coders have 90% similarity in {num_samples_AG} out of {len(WCmerged)} total samples: {perc_samples_AG}%\n\n")
                        report.write(f"Intraclass Correlation Coefficient (ICC(2,1)): {icc_value}\n")
                    logging.info(f"Reliability report written to {report_path}")
                except Exception as e:
                    logging.error(f"Failed to write reliability report {report_path}: {e}")
