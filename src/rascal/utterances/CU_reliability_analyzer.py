import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define helper functions for aggregation.
def utt_ct(x):
    """Count number of utterances."""
    no_utt = len(x.dropna())
    return no_utt if no_utt > 0 else np.nan

def ptotal(x):
    """Count number of positive scores."""
    return sum(x.dropna()) if len(x.dropna()) > 0 else np.nan

def ag_check(x):
    """Check agreement: at least 80% is in agreement."""
    total_CUs = len(x.dropna())
    if total_CUs > 0:
        return 1 if (sum(x == 1) / total_CUs) >= 0.8 else 0
    else:
        return np.nan

def analyze_CU_reliability(tiers, input_dir, output_dir, test=False):
    """
    Analyzes CU reliability coding by comparing coder results and generating summary statistics.

    Parameters:
    - tiers (dict): Dictionary of tier information used for matching file names.
    - input_dir (str): Directory containing the input CU coding and reliability files.
    - output_dir (str): Directory where the output reliability summaries will be saved.

    Returns:
    - None. Saves reliability summaries and analyzed DataFrames to output directory.
    """
    
    # Make CU Reliability folder.
    CUReliability_dir = os.path.join(output_dir, 'CUReliability')
    try:
        os.makedirs(CUReliability_dir, exist_ok=True)
        logging.info(f"Created directory: {CUReliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {CUReliability_dir}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*_CUCoding.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*_CUReliabilityCoding.xlsx')]

    # Store results for testing.
    results = []

    # Match CU coding and reliability files.
    for rel in tqdm(rel_files, desc="Analyzing CU reliability coding..."):
        # Extract tier info from file name.
        rel_labels = [t.match(rel.name, return_None=True) for t in tiers.values()]

        for cod in coding_files:
            cod_labels = [t.match(cod.name, return_None=True) for t in tiers.values()]

            if rel_labels == cod_labels:
                try:
                    CUcod = pd.read_excel(cod)
                    CUrel = pd.read_excel(rel)
                    logging.info(f"Processing coding file: {cod} and reliability file: {rel}")
                except Exception as e:
                    logging.error(f"Failed to read files {cod} or {rel}: {e}")
                    continue

                # Shed coder 1 and comment columns.
                CUcod = CUcod.loc[:, ['UtteranceID', 'sampleID', 'c2SV', 'c2REL']]
                CUrel = CUrel.loc[:, ['UtteranceID', 'c3SV', 'c3REL']]

                # Merge on UtteranceID.
                try:
                    CUrelcod = pd.merge(CUcod, CUrel, on="UtteranceID", how="inner")
                    logging.info(f"Merged reliability file with coding file for {rel.name}")
                except Exception as e:
                    logging.error(f"Failed to merge {cod.name} with {rel.name}: {e}")
                    continue
                
                if len(CUrel) != len(CUrelcod):
                    logging.error(f"Length mismatch between reliability and joined files for {rel.name}.")

                c2CU, c3CU = [], []
                # Check agreement and that coders are consistent with neutrality.
                for i in range(len(CUrelcod)):
                    # Extract coder scores.
                    c2SV, c2REL, c3SV, c3REL = [CUrelcod.at[i, col] for col in ['c2SV', 'c2REL', 'c3SV', 'c3REL']]
                    # Check consistency for coder 2.
                    if (np.isnan(c2SV) and not np.isnan(c2REL)) or (np.isnan(c2REL) and not np.isnan(c2SV)):
                        logging.error(f"Neutrality inconsistency in coder 2 for utterance {CUrelcod.at[i, 'UtteranceID']}")
                        c2CU.append(np.nan)
                    # Code CU for coder 2.
                    elif np.isnan(c2REL) and np.isnan(c2SV):
                        c2CU.append(np.nan)
                    elif c2SV == c2REL == 1:
                        c2CU.append(1)
                    else:
                        c2CU.append(0)
                    # Check consistency for coder 3.
                    if (np.isnan(c3SV) and not np.isnan(c3REL)) or (np.isnan(c3REL) and not np.isnan(c3SV)):
                        logging.error(f"Neutrality inconsistency in coder 3 for utterance {CUrelcod.at[i, 'UtteranceID']}")
                    # Code CU for coder 3.
                    elif np.isnan(c3REL) and np.isnan(c3SV):
                        c3CU.append(np.nan)
                    elif c3SV == c3REL == 1:
                        c3CU.append(1)
                    else:
                        c3CU.append(0)
                CUrelcod['c2CU'] = c2CU
                CUrelcod['c3CU'] = c3CU

                # Calculate agreement columns: 1 if same value or both NA, else 0.
                CUrelcod['AGSV'] = CUrelcod.apply(lambda row: int((row['c2SV'] == row['c3SV']) or (np.isnan(row['c2SV']) and np.isnan(row['c3SV']))), axis=1)
                CUrelcod['AGREL'] = CUrelcod.apply(lambda row: int((row['c2REL'] == row['c3REL']) or (np.isnan(row['c2REL']) and np.isnan(row['c3REL']))), axis=1)
                CUrelcod['AGCU'] = CUrelcod.apply(lambda row: int((row['c2CU'] == row['c3CU']) or (np.isnan(row['c2CU']) and np.isnan(row['c3CU']))), axis=1)

                # Save utterance-level reliability file.
                partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]
                output_path = os.path.join(CUReliability_dir, *partition_labels)
                try:
                    os.makedirs(output_path, exist_ok=True)
                    logging.info(f"Created partition directory: {output_path}")
                except Exception as e:
                    logging.error(f"Failed to create partition directory {output_path}: {e}")
                    continue

                file_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCoding_ByUtterance.xlsx")
                try:
                    CUrelcod.to_excel(file_path, index=False)
                    logging.info(f"Saved CU reliability coding by utterance to: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to write CU reliability file {file_path}: {e}")

                # Summarize reliability coding by sample.
                CUrelsum = CUrelcod.copy()
                CUrelsum.drop(columns=['UtteranceID'], inplace=True)
                try:
                    CUrelsum = CUrelsum.groupby(['sampleID']).agg(
                        no_utt2=('c2CU', utt_ct),
                        pSV2=('c2SV', ptotal),
                        mSV2=('c2SV', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        pREL2=('c2REL', ptotal),
                        mREL2=('c2REL', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        CU2=('c2CU', ptotal),
                        percCU2=('c2CU', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
                        no_utt3=('c3CU', utt_ct),
                        pSV3=('c3SV', ptotal),
                        mSV3=('c3SV', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        pREL3=('c3REL', ptotal),
                        mREL3=('c3REL', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        CU3=('c3CU', ptotal),
                        percCU3=('c3CU', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
                        totAGSV=('AGSV', ptotal),
                        percAGSV=('AGSV', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),
                        totAGREL=('AGREL', ptotal),
                        percAGREL=('AGREL', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),
                        totAGCU=('AGCU', ptotal),
                        percAGCU=('AGCU', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),
                        sampleAGSV=('AGSV', ag_check),
                        sampleAGREL=('AGREL', ag_check),
                        sampleAGCU=('AGCU', ag_check)
                    ).reset_index()  # Keep 'sampleID' in the output by resetting the index.
                    logging.info(f"Successfully aggregated reliability data for {rel.name}")
                except Exception as e:
                    logging.error(f"Failed during aggregation: {e}")
                    continue
                
                # Generate reliability report.
                num_samples_AG = np.nansum(CUrelsum['sampleAGCU'])
                perc_samples_AG = round(num_samples_AG/len(CUrelsum)*100, 2)
                report_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCodingReport.txt")
                with open(report_path, 'w') as report:
                    report.write(f"CU Reliability Coding Report for {' '.join(partition_labels)}\n\n")
                    report.write(f"Coders agree on at least 80% of CUs in {num_samples_AG} out of {len(CUrelsum)} total samples: {perc_samples_AG}%\n\n")
                    report.write(f"Average agreement on SV: {round(np.nanmean(CUrelsum['percAGSV']), 3)}\n")
                    report.write(f"Average agreement on REL: {round(np.nanmean(CUrelsum['percAGREL']), 3)}\n")
                    report.write(f"Average agreement on CU: {round(np.nanmean(CUrelsum['percAGCU']), 3)}\n")

                # Save summary reliability file.
                summary_file_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCoding_BySample.xlsx")
                try:
                    CUrelsum.to_excel(summary_file_path, index=False)
                    logging.info(f"Saved CU reliability summary to: {summary_file_path}")
                except Exception as e:
                    logging.error(f"Failed to write CU reliability summary file {summary_file_path}: {e}")
                
                if test:
                    results.append(CUrelsum)
    
    if test:
        return results

def analyze_CU_coding(tiers, input_dir, output_dir, test=False):
    """
    Analyzes CU coding by summarizing coder results and generating summary statistics.

    Parameters:
    - tiers (dict): Dictionary of tier information used for matching file names.
    - input_dir (str): Directory containing the input CU coding and reliability files.
    - output_dir (str): Directory where the output reliability summaries will be saved.

    Returns:
    - None. Saves CU coding summaries and analyzed DataFrames to output directory.
    """
    
    # Make CU analysis folder.
    CUanalysis_dir = os.path.join(output_dir, 'CUCodingAnalysis')
    try:
        os.makedirs(CUanalysis_dir, exist_ok=True)
        logging.info(f"Created directory: {CUanalysis_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {CUanalysis_dir}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*_CUCoding.xlsx')]

    # Store results for testing.
    results = []

    # Match CU coding and reliability files.
    for cod in tqdm(coding_files, desc="Analyzing CU coding..."):
        try:
            CUcod = pd.read_excel(cod)
            logging.info(f"Processing coding file: {cod}")
        except Exception as e:
            logging.error(f"Failed to read file {cod}: {e}")
            continue

        # Shed coder 1 and ID columns.
        CUcod.drop(columns=['c1ID','c1SV','c1REL','c1com','c2ID'], inplace=True)

        # Code CUs based on SV and REL.
        c2CU = []
        # Check and that coder2 is consistent with neutrality.
        for i in range(len(CUcod)):
            # Convert values to numeric in case they are stored as strings
            c2SV = pd.to_numeric(CUcod.at[i, 'c2SV'], errors='coerce')
            c2REL = pd.to_numeric(CUcod.at[i, 'c2REL'], errors='coerce')
            # Check consistency for coder 2.
            if (np.isnan(c2SV) and not np.isnan(c2REL)) or (np.isnan(c2REL) and not np.isnan(c2SV)):
                logging.error(f"Neutrality inconsistency in coder 2 for utterance {CUcod.at[i, 'UtteranceID']}")
                c2CU.append(np.nan)
            # Code CU for coder 2.
            elif np.isnan(c2REL) and np.isnan(c2SV):
                c2CU.append(np.nan)
            elif c2SV == c2REL == 1:
                c2CU.append(1)
            else:
                c2CU.append(0)
        CUcod['c2CU'] = c2CU

        # Save utterance-level analysis file.
        partition_labels = [t.match(cod.name) for t in tiers.values() if t.partition]
        output_path = os.path.join(CUanalysis_dir, *partition_labels)
        try:
            os.makedirs(output_path, exist_ok=True)
            logging.info(f"Created partition directory: {output_path}")
        except Exception as e:
            logging.error(f"Failed to create partition directory {output_path}: {e}")
            continue

        file_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUCoding_ByUtterance.xlsx")
        try:
            CUcod.to_excel(file_path, index=False)
            logging.info(f"Saved CU final coding by utterance file to: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write CU reliability file {file_path}: {e}")

        # Shed non-aggregating columns.
        CUcod = CUcod.loc[:, ['sampleID', 'c2SV', 'c2REL','c2CU']]
        # Ensure numeric types before aggregation
        CUcod[['c2SV', 'c2REL', 'c2CU']] = CUcod[['c2SV', 'c2REL', 'c2CU']].apply(pd.to_numeric, errors='coerce')
       
        # Summarize reliability coding by sample.
        CUcodsum = CUcod.copy()
        try:
            CUcodsum = CUcodsum.groupby(['sampleID']).agg(
                no_utt2=('c2CU', utt_ct),
                pSV2=('c2SV', ptotal),
                mSV2=('c2SV', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                pREL2=('c2REL', ptotal),
                mREL2=('c2REL', lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                CU2=('c2CU', ptotal),
                percCU2=('c2CU', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
            ).reset_index()  # Keep 'sampleID' in the output by resetting the index.
            logging.info(f"Successfully aggregated reliability data for {cod.name}")
        except Exception as e:
            logging.error(f"Failed during aggregation: {e}")
            continue

        # Save summary coding file.
        summary_file_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUCoding_BySample.xlsx")
        try:
            CUcodsum.to_excel(summary_file_path, index=False)
            logging.info(f"Saved CU coding summary to: {summary_file_path}")
        except Exception as e:
            logging.error(f"Failed to write CU reliability summary file {summary_file_path}: {e}")
        
        if test:
            results.append(CUcodsum)
    
    if test:
        return results
