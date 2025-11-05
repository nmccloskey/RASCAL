import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rascal.utils.logger import logger


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
        logger.warning("One of the values is zero, returning 100%.")
        return 100
    elif value1 == value2 == 0:
        return 0

    diff = abs(value1 - value2)
    avg = (value1 + value2) / 2
    return round((diff / avg) * 100, 2)

def agreement(row):
    abs_diff = abs(row['word_count_org'] - row['word_count_rel'])
    if abs_diff <= 1:
        return 1
    else:
        perc_diff = percent_difference(row['word_count_org'], row['word_count_rel'])
        perc_sim = 100 - perc_diff
        return 1 if perc_sim >= 85 else 0

def calculate_icc(data):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for two raters.
    
    Args:
        data (pd.DataFrame): A dataframe with two columns: 'word_count_org' and 'word_count_rel'.
        
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

def evaluate_word_count_reliability(tiers, input_dir, output_dir):
    """
    Analyze word count reliability by comparing coder-1 word counts with
    coder-2 reliability word counts.

    Workflow
    --------
    1. Collect all "*word_counting*.xlsx" (coding) and
       "*word_counting_reliability*.xlsx" (reliability) files under `input_dir`.
    2. For each reliability file, find the coding file with matching tier labels.
    3. Read both DataFrames and clean the reliability frame to
       ['utterance_id','WCrelCom','word_count'], dropping NaN word counts.
    4. Merge on 'utterance_id' with suffixes (_org for coding, _rel for reliability).
    5. For each utterance, compute:
         - abs_diff  : raw difference (org − rel)
         - perc_diff : percent difference (using `percent_difference`)
         - perc_sim  : 100 − perc_diff
         - agmt       : binary agreement (1 if abs diff ≤1 or percSim ≥85)
    6. Save merged results to
         "<output_dir>/word_count_reliability/<partition_labels>/<labels>_word_counting_reliability_results.xlsx"
    7. Compute ICC(2,1) across utterances (using `calculate_icc`).
    8. Write a plain-text report:
         "<labels>_word_count_reliability_report.txt"
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
        Directory where results are written under "word_count_reliability/".

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
    word_count_reliability_dir = output_dir / 'word_count_reliability'
    try:
        word_count_reliability_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {word_count_reliability_dir}")
    except Exception as e:
        logger.error(f"Failed to create directory {word_count_reliability_dir}: {e}")
        return

    # Collect relevant files
    coding_files = [f for f in Path(input_dir).rglob('*word_counting*.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*word_counting_reliability*.xlsx')]

    # Match word counting and reliability files
    for rel in tqdm(rel_files, desc="Analyzing word count reliability..."):
        # Extract tier info from file name
        rel_labels = [t.match(rel.name, return_None=True) for t in tiers.values()]

        for cod in coding_files:
            cod_labels = [t.match(cod.name, return_None=True) for t in tiers.values()]

            if rel_labels == cod_labels:
                try:
                    wc_df = pd.read_excel(cod)
                    wc_rel_df = pd.read_excel(rel)
                    logger.info(f"Processing coding file: {cod} and reliability file: {rel}")
                except Exception as e:
                    logger.error(f"Failed to read files {cod} or {rel}: {e}")
                    continue

                # Clean and filter the reliability DataFrame
                wc_rel_df = wc_rel_df.loc[:, ['utterance_id', 'WCrelCom', 'word_count']]
                wc_rel_df = wc_rel_df[~np.isnan(wc_rel_df['word_count'])]

                # Merge on utterance_id
                try:
                    wc_merged = pd.merge(wc_df, wc_rel_df, on="utterance_id", how="inner", suffixes=('_org', '_rel'))
                    logger.info(f"Merged reliability file with coding file for {rel.name}")
                except Exception as e:
                    logger.error(f"Failed to merge {cod.name} with {rel.name}: {e}")
                    continue

                if len(wc_rel_df) != len(wc_merged):
                    logger.error(f"Length mismatch between reliability and joined files for {rel.name}.")

                # Calculate percent difference
                wc_merged['abs_diff'] = wc_merged.apply(lambda row: row['word_count_org'] - row['word_count_rel'], axis=1)
                wc_merged['perc_diff'] = wc_merged.apply(lambda row: percent_difference(row['word_count_org'], row['word_count_rel']), axis=1)
                wc_merged['perc_sim'] = 100 - wc_merged['perc_diff']
                wc_merged['agmt'] = wc_merged.apply(agreement, axis=1)

                # Create output directory
                partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]
                output_path = Path(word_count_reliability_dir, *partition_labels)
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created partition directory: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to create partition directory {output_path}: {e}")
                    continue

                # Write tables.
                lab_str = '_'.join(partition_labels) + '_' if partition_labels else ''
                filename = Path(output_path, lab_str + 'word_counting_reliability_results.xlsx')
                logger.info(f"Writing word counting reliability results file: {filename}")
                try:
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    wc_merged.to_excel(filename, index=False)
                except Exception as e:
                    logger.error(f"Failed to write word count reliability results file {filename}: {e}")
                
                # Subset the data for ICC calculation
                icc_data = wc_merged[['word_count_org', 'word_count_rel']].dropna()

                # Calculate ICC
                icc_value = calculate_icc(icc_data)
                logger.info(f"Calculated ICC(2,1) for {rel.name}: {icc_value}")

                # Write reliability report
                num_samples_agmt = np.nansum(wc_merged['agmt'])
                perc_samples_agmt = round((num_samples_agmt / len(wc_merged)) * 100, 1)
                report_path = Path(output_path, lab_str + "word_count_reliability_report.txt")
                try:
                    with open(report_path, 'w') as report:
                        report.write(f"Word Count Reliability Report for {' '.join(partition_labels)}\n\n")
                        report.write(f"Coders have 90% similarity in {num_samples_agmt} out of {len(wc_merged)} total samples: {perc_samples_agmt}%\n\n")
                        report.write(f"Intraclass Correlation Coefficient (ICC(2,1)): {icc_value}\n")
                    logger.info(f"Reliability report written to {report_path}")
                except Exception as e:
                    logger.error(f"Failed to write reliability report {report_path}: {e}")
