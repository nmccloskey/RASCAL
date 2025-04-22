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
        logging.warning("One of the values is zero, returning NA.")
        return np.nan
    elif value1 == value2 == 0:
        return 100

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

def analyze_word_count_reliability(tiers, input_dir, output_dir, test=False):
    """
    Analyzes word count reliability between original and reliability files.

    Args:
        tiers (dict): Dictionary containing tier information and matching criteria.
        input_dir (str): Directory containing input files.
        output_dir (str): Directory to save output results.
        test (bool): If True, the function will return results for testing. Default is False.

    Returns:
        list: Results of the analysis (only if `test=True`).
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

    # Store results for testing
    results = []

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
                WCreldf = WCreldf.loc[:, ['UtteranceID', 'WCrelCom', 'wordCount']]
                WCreldf = WCreldf[~np.isnan(WCreldf['wordCount'])]

                # Merge on UtteranceID
                try:
                    WCmerged = pd.merge(WCdf, WCreldf, on="UtteranceID", how="inner", suffixes=('_org', '_rel'))
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

    if test:
        return results
