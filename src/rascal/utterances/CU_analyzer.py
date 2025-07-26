import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random


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

def compute_CU_column(row):
    """
    Compute CU value for a single coder based on SV and REL columns.

    Returns:
    - 1 if SV==REL==1
    - 0 if SV!=1 or REL!=1 but both are 0 or 0-like
    - np.nan if both are NaN
    - Logs error and returns np.nan if only one is NaN
    """
    sv, rel = row.iloc[0], row.iloc[1]

    if (pd.isna(sv) and not pd.isna(rel)) or (pd.isna(rel) and not pd.isna(sv)):
        logging.error(f"Neutrality inconsistency in CU computation: SV={sv}, REL={rel}")
        return np.nan
    elif pd.isna(sv) and pd.isna(rel):
        return np.nan
    elif sv == rel == 1:
        return 1
    else:
        return 0

def summarize_CU_reliability(CUrelcod, sv2, rel2, sv3, rel3):
    """
    Summarize CU reliability metrics at the sample level.

    Parameters:
    - CUrelcod (DataFrame): Merged CU coding and reliability dataframe with c2CU and c3CU added.
    - sv2, rel2: Column names for coder 2's SV and REL fields.
    - sv3, rel3: Column names for coder 3's SV and REL fields.

    Returns:
    - CUrelsum (DataFrame): Aggregated reliability stats grouped by sampleID.
    """
    CUrelsum = CUrelcod.copy()
    CUrelsum.drop(columns=['UtteranceID'], inplace=True, errors='ignore')

    try:
        CUrelsum = CUrelsum.groupby(['sampleID']).agg(
            no_utt2=('c2CU', utt_ct),
            pSV2=(sv2, ptotal),
            mSV2=(sv2, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            pREL2=(rel2, ptotal),
            mREL2=(rel2, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            CU2=('c2CU', ptotal),
            percCU2=('c2CU', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            no_utt3=('c3CU', utt_ct),
            pSV3=(sv3, ptotal),
            mSV3=(sv3, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            pREL3=(rel3, ptotal),
            mREL3=(rel3, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
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
        ).reset_index()
        logging.info("Successfully aggregated CU reliability data.")
        return CUrelsum
    except Exception as e:
        logging.error(f"Failed during CU reliability aggregation: {e}")
        return pd.DataFrame()  # Fail-safe return

def write_reliability_report(CUrelsum, report_path, partition_labels=None):
    """
    Writes a plain text summary report of CU reliability to the given file path.

    Parameters:
    - CUrelsum (DataFrame): Summary stats by sample, including agreement columns.
    - report_path (str): Full path to the output .txt file.
    - partition_labels (list or None): Optional list of tier labels to include in the header.
    """
    try:
        num_samples_AG = np.nansum(CUrelsum['sampleAGCU'])
        perc_samples_AG = round(num_samples_AG / len(CUrelsum) * 100, 2)

        with open(report_path, 'w') as report:
            if partition_labels:
                report.write(f"CU Reliability Coding Report for {' '.join(partition_labels)}\n\n")
            else:
                report.write("CU Reliability Coding Report\n\n")

            report.write(f"Coders agree on at least 80% of CUs in {num_samples_AG} out of {len(CUrelsum)} total samples: {perc_samples_AG}%\n\n")
            report.write(f"Average agreement on SV: {round(np.nanmean(CUrelsum['percAGSV']), 3)}\n")
            report.write(f"Average agreement on REL: {round(np.nanmean(CUrelsum['percAGREL']), 3)}\n")
            report.write(f"Average agreement on CU: {round(np.nanmean(CUrelsum['percAGCU']), 3)}\n")

        logging.info(f"Successfully wrote CU reliability report to {report_path}")
    except Exception as e:
        logging.error(f"Failed to write reliability report to {report_path}: {e}")

def analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms, test=False):
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

                # Determine paradigms to iterate
                if len(CU_paradigms) >= 2:
                    paradigms_to_run = CU_paradigms
                else:
                    paradigms_to_run = [None]  # Original columns

                for paradigm in paradigms_to_run:
                    # --- Column selection ---
                    if paradigm:
                        sv2, rel2, sv3, rel3 = f'c2SV_{paradigm}', f'c2REL_{paradigm}', f'c3SV_{paradigm}', f'c3REL_{paradigm}'
                        out_subdir = os.path.join(CUReliability_dir, paradigm)
                    else:
                        sv2, rel2, sv3, rel3 = 'c2SV', 'c2REL', 'c3SV', 'c3REL'
                        out_subdir = CUReliability_dir

                    CUcod_sub = CUcod.loc[:, ['UtteranceID', 'sampleID', sv2, rel2]].copy()
                    CUrel_sub = CUrel.loc[:, ['UtteranceID', sv3, rel3]].copy()

                    try:
                        CUrelcod = pd.merge(CUcod_sub, CUrel_sub, on="UtteranceID", how="inner")
                    except Exception as e:
                        logging.error(f"Merge failed for paradigm {paradigm} on {rel.name}: {e}")
                        continue

                    # Validate length
                    if len(CUrel_sub) != len(CUrelcod):
                        logging.error(f"Length mismatch for {paradigm or 'default'}: {rel.name}")

                    # --- CU computation ---
                    CUrelcod['c2CU'] = CUrelcod[[sv2, rel2]].apply(compute_CU_column, axis=1)
                    CUrelcod['c3CU'] = CUrelcod[[sv3, rel3]].apply(compute_CU_column, axis=1)

                    # Calculate agreement columns: 1 if same value or both NA, else 0.
                    CUrelcod['AGSV'] = CUrelcod.apply(lambda row: int((row[sv2] == row[sv3]) or (pd.isna(row[sv2]) and pd.isna(row[sv3]))), axis=1)
                    CUrelcod['AGREL'] = CUrelcod.apply(lambda row: int((row[rel2] == row[rel3]) or (pd.isna(row[rel2]) and pd.isna(row[rel3]))), axis=1)
                    CUrelcod['AGCU'] = CUrelcod.apply(lambda row: int((row['c2CU'] == row['c3CU']) or (pd.isna(row['c2CU']) and pd.isna(row['c3CU']))), axis=1)

                    # Partition subfolder path
                    partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]
                    output_path = os.path.join(out_subdir, *partition_labels)

                    try:
                        os.makedirs(output_path, exist_ok=True)
                    except Exception as e:
                        logging.error(f"Failed to make output folder {output_path}: {e}")
                        continue

                    # Save utterance-level results
                    utterance_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCoding_ByUtterance.xlsx")
                    CUrelcod.to_excel(utterance_path, index=False)

                    # Summary + report + save (unchanged)
                    # Just use CUrelcod and column names as they are
                    CUrelsum = summarize_CU_reliability(CUrelcod, sv2, rel2, sv3, rel3)
                    report_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCodingReport.txt")
                    write_reliability_report(CUrelsum, report_path)
                    summary_path = os.path.join(output_path, f"{'_'.join(partition_labels)}_CUReliabilityCoding_BySample.xlsx")
                    CUrelsum.to_excel(summary_path, index=False)
                    
                    if test:
                        results.append(CUrelsum)
    
    if test:
        return results


def analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms=None, test=False):
    """
    Analyzes CU coding by summarizing coder results and generating summary statistics.

    Parameters:
    - tiers (dict): Tier definitions used to match file names.
    - input_dir (str): Path to directory containing *_CUCoding.xlsx files.
    - output_dir (str): Path to save analyzed results.
    - CU_paradigms (list): List of dialect codes (e.g., ['SAE', 'AAE']). If None or 1, use default columns.
    - test (bool): If True, returns results for testing.
    """
    CUanalysis_dir = os.path.join(output_dir, 'CUCodingAnalysis')
    try:
        os.makedirs(CUanalysis_dir, exist_ok=True)
        logging.info(f"Created directory: {CUanalysis_dir}")
    except Exception as e:
        logging.error(f"Failed to create CU analysis directory {CUanalysis_dir}: {e}")
        return

    coding_files = list(Path(input_dir).rglob('*_CUCoding.xlsx'))
    results = []

    for cod in tqdm(coding_files, desc="Analyzing CU coding..."):
        try:
            CUcod_orig = pd.read_excel(cod)
            logging.info(f"Processing CU coding file: {cod}")
        except Exception as e:
            logging.error(f"Failed to read CU coding file {cod}: {e}")
            continue

        # Determine which paradigms to process
        if CU_paradigms is None or len(CU_paradigms) <= 1:
            paradigms_to_run = [None]
        else:
            paradigms_to_run = CU_paradigms

        for paradigm in paradigms_to_run:
            CUcod = CUcod_orig.copy()

            drop_cols = ['c1ID', 'c1com', 'c2ID']

            # Remove all SV/REL/Compare columns that belong to *other* paradigms
            if paradigm:
                # Explicitly preserve only current paradigm columns
                all_cols = CUcod.columns
                for col in all_cols:
                    if any(
                        col.endswith(f"_{other}")
                        for other in CU_paradigms
                        if other != paradigm
                    ):
                        drop_cols.append(col)
            else:
                # If using default mode, drop any suffixed columns (multi-paradigm leftovers)
                drop_cols += [col for col in CUcod.columns if '_' in col and col.endswith(tuple(CU_paradigms or []))]

            CUcod.drop(columns=[col for col in drop_cols if col in CUcod.columns], inplace=True, errors='ignore')

            # Define target SV and REL columns
            sv_col = f'c2SV_{paradigm}' if paradigm else 'c2SV'
            rel_col = f'c2REL_{paradigm}' if paradigm else 'c2REL'

            # Compute CU
            CUcod['c2CU'] = CUcod[[sv_col, rel_col]].apply(compute_CU_column, axis=1)

            # Output directory
            partition_labels = [t.match(cod.name) for t in tiers.values() if t.partition]
            out_dir = os.path.join(CUanalysis_dir, *(partition_labels + ([paradigm] if paradigm else [])))

            try:
                os.makedirs(out_dir, exist_ok=True)
                logging.info(f"Created output directory: {out_dir}")
            except Exception as e:
                logging.error(f"Failed to create output subdirectory {out_dir}: {e}")
                continue

            # Save utterance-level CU codes
            utterance_path = os.path.join(out_dir, f"{'_'.join(partition_labels)}_CUCoding_ByUtterance.xlsx")
            try:
                CUcod.to_excel(utterance_path, index=False)
                logging.info(f"Saved CU utterance-level analysis to: {utterance_path}")
            except Exception as e:
                logging.error(f"Failed to save utterance-level CU analysis: {e}")

            # Select relevant columns for aggregation
            agg_df = CUcod[['sampleID', sv_col, rel_col, 'c2CU']].copy()
            agg_df[[sv_col, rel_col, 'c2CU']] = agg_df[[sv_col, rel_col, 'c2CU']].apply(pd.to_numeric, errors='coerce')

            try:
                CUcodsum = agg_df.groupby('sampleID').agg(
                    no_utt2=('c2CU', utt_ct),
                    pSV2=(sv_col, ptotal),
                    mSV2=(sv_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                    pREL2=(rel_col, ptotal),
                    mREL2=(rel_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                    CU2=('c2CU', ptotal),
                    percCU2=('c2CU', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
                ).reset_index()
                logging.info(f"Aggregated CU coding stats for: {cod.name}")
            except Exception as e:
                logging.error(f"Aggregation failed for {cod.name}: {e}")
                continue

            summary_path = os.path.join(out_dir, f"{'_'.join(partition_labels)}_CUCoding_BySample.xlsx")
            try:
                CUcodsum.to_excel(summary_path, index=False)
                logging.info(f"Saved CU coding summary to: {summary_path}")
            except Exception as e:
                logging.error(f"Failed to write CU summary file: {e}")

            if test:
                results.append(CUcodsum)

    if test:
        return results


def reselect_CU_reliability(input_dir, output_dir, coder3='3', frac=0.2, test=False):
    """
    Reselects new CU reliability samples from previously unused samples,
    avoiding overlap with the original reliability set.

    Parameters:
    - input_dir (str): Directory containing *_CUCoding and *_CUReliabilityCoding files.
    - output_dir (str): Directory to save new reliability files.
    - coder3 (str): Name or ID of third coder for reliability coding.
    - frac (float): Fraction of samples to reselect for reliability (e.g., 0.2 for 20%).
    - test (bool): If True, returns output DataFrames for testing.

    Returns:
    - None or list of DataFrames if test=True.
    """

    random.seed(88)

    # Create output CU coding directory
    reselected_CU_reliability_dir = os.path.join(output_dir, 'reselected_CU_reliability')
    try:
        os.makedirs(reselected_CU_reliability_dir, exist_ok=True)
        logging.info(f"Created directory: {reselected_CU_reliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {reselected_CU_reliability_dir}: {e}")
        return

    # Gather CU coding files using Path.glob
    coding_files = [f for f in Path(input_dir).rglob('*_CUCoding.xlsx')]
    logging.info(f"Found {coding_files} in {input_dir}.")
    results = []

    for cu_file in tqdm(coding_files, desc="Reselecting CU reliability samples"):
        try:
            rel_file = cu_file.with_name(cu_file.name.replace('_CUCoding', '_CUReliabilityCoding'))

            if not rel_file.exists():
                logging.warning(f"No reliability file found for {cu_file.name}. Skipping.")
                continue

            df_cu = pd.read_excel(cu_file)
            df_rel = pd.read_excel(rel_file)

            used_sample_ids = set(df_rel['sampleID'].unique())
            all_sample_ids = set(df_cu['sampleID'].unique())
            available_ids = list(all_sample_ids - used_sample_ids)

            if len(available_ids) == 0:
                logging.warning(f"No available samples to reselect for {cu_file.name}. Skipping.")
                continue

            num_to_select = max(1, round(len(all_sample_ids) * frac))
            if len(available_ids) < num_to_select:
                logging.warning(f"Not enough unused samples in {cu_file.name} to meet {frac*100:.0f}% threshold. Selecting {len(available_ids)} instead of {num_to_select}.")
                num_to_select = len(available_ids)

            reselected_ids = random.sample(available_ids, k=num_to_select)

            # Construct new reliability DataFrame
            rel_columns = ['c2ID', 'c2SV', 'c2REL', 'c2com']
            rename_map = {'c2ID': 'c3ID', 'c2SV': 'c3SV', 'c2REL': 'c3REL', 'c2com': 'c3com'}
            shared_cols = ['UtteranceID', 'site', 'narrative', 'sampleID', 'speaker', 'utterance', 'comment']

            df_new_rel = df_cu[df_cu['sampleID'].isin(reselected_ids)].copy()
            df_new_rel = df_new_rel[shared_cols + rel_columns]
            df_new_rel.rename(columns=rename_map, inplace=True)
            df_new_rel['c3ID'] = coder3
            df_new_rel['c3com'] = np.nan  # Remove original coder comments

            try:
                os.makedirs(reselected_CU_reliability_dir, exist_ok=True)
                logging.info(f"Created directory: {reselected_CU_reliability_dir}")
            except Exception as e:
                logging.error(f"Failed to create directory {reselected_CU_reliability_dir}: {e}")
                continue

            base_name = cu_file.stem.replace('_CUCoding', '')
            out_file = os.path.join(reselected_CU_reliability_dir, f"{base_name}_reselected_CUReliabilityCoding.xlsx")
            df_new_rel.to_excel(out_file, index=False)
            logging.info(f"Saved reselected CU reliability file: {out_file}")

            if test:
                results.append(df_new_rel)

        except Exception as e:
            logging.error(f"Unexpected error with file {cu_file.name}: {e}")

    if test:
        return results
