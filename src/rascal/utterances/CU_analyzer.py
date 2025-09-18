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

def compute_CU_column(row):
    """
    Compute a single coder's CU value from paired SV/REL fields.

    Input
    -----
    row : pd.Series
        A two-element series ordered as [SV_col, REL_col] containing values {1, 0, NaN}.

    Returns
    -------
    int | float
        1  -> when SV == 1 and REL == 1 (coder marked the utterance as a CU on both dimensions)
        0  -> when SV and REL are both non-1 but present (e.g., 0/0, 0/0-like)
        NaN -> when both entries are NaN

    Notes
    -----
    - If exactly one of (SV, REL) is NaN while the other is not, an error is logged
      (neutrality inconsistency) and NaN is returned.
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
    Aggregate utterance-level CU reliability to the sample level.

    Input
    -----
    CUrelcod : pd.DataFrame
        Merged utterance-level dataframe containing:
        - identifiers: 'utterance_id', 'sample_id'
        - coder-2 columns: sv2, rel2 (names passed in), and computed 'c2CU'
        - coder-3 columns: sv3, rel3 (names passed in), and computed 'c3CU'
        - agreement flags: 'AGSV', 'AGREL', 'AGCU' (1 if equal or both NaN, else 0)

    sv2, rel2, sv3, rel3 : str
        Column names for the respective SV/REL fields used in aggregation.

    Returns
    -------
    pd.DataFrame
        One row per sample_id with:
        - Counts per coder (no_utt2/no_utt3, CU2/CU3, pSV*/mSV*, pREL*/mREL*)
        - Percent CU per coder (percCU2, percCU3)
        - Percent agreement on SV/REL/CU (percAGSV, percAGREL, percAGCU)
        - Binary sample-level agreement indicators (sampleAGSV, sampleAGREL, sampleAGCU),
          where 1 indicates ≥80% agreement across utterances, 0 otherwise.

    Notes
    -----
    - Agreement thresholds use the 'ag_check' helper: ≥0.8 proportion => 1, else 0.
    - Returns an empty DataFrame on aggregation failure (with error logged).
    """
    CUrelsum = CUrelcod.copy()
    CUrelsum.drop(columns=['utterance_id'], inplace=True, errors='ignore')

    try:
        CUrelsum = CUrelsum.groupby(['sample_id']).agg(
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
    Write a plain-text CU reliability summary.

    Input
    -----
    CUrelsum : pd.DataFrame
        Sample-level summary produced by `summarize_CU_reliability`, including:
        'sampleAGCU', 'percAGSV', 'percAGREL', 'percAGCU'.
    report_path : str | os.PathLike
        Destination .txt filepath.
    partition_labels : list[str] | None
        Optional tier labels (e.g., site/test/participant) to render in the header.

    Side Effects
    ------------
    Creates a text file with:
      - Count and percent of samples meeting ≥80% CU agreement (sampleAGCU == 1)
      - Average SV/REL/CU percent agreement across samples

    Logging
    -------
    Logs success or an error if the report cannot be written.
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

def analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms):
    """
    Analyze Complete Utterance (CU) reliability by pairing coder-2 coding with coder-3
    reliability coding, computing utterance-level agreement, and summarizing by sample.

    Inputs
    ------
    tiers : dict[str, Any]
        Mapping of tier name -> tier object. Each tier object is expected to provide:
          - .match(filename, ...) -> label string used for partitioning
          - .partition : bool indicating whether this tier participates in the output path
        Example: site/test/participant tiers derived from filenames.

    input_dir : str | os.PathLike
        Root directory searched (recursively) for:
          - "*CUCoding.xlsx" (coder 2)
          - "*CUReliabilityCoding.xlsx" (coder 3)
    output_dir : str | os.PathLike
        Base directory where outputs are written under:
          "<output_dir>/CUReliability[/<PARADIGM>]/<partition_labels...>/"

    CU_paradigms : list[str]
        List of paradigm labels (e.g., ["SAE", "AAE"]). **Behavior by case:**
          - **0 paradigms ([])**: run **once** using the **base columns**
            'c2SV', 'c2REL', 'c3SV', 'c3REL'; outputs go directly under
            "<output_dir>/CUReliability/<partition_labels...>/".
          - **1 paradigm (['X'])**: treated the **same as 0**—still uses base columns
            (no suffix). This matches RASCAL's convention: when only one paradigm is in
            use, columns remain unsuffixed and no per-paradigm subfolder is created.
          - **2+ paradigms (['X','Y',...])**: for each paradigm `p`, expect **suffixed**
            columns in both files:
              'c2SV_{p}', 'c2REL_{p}', 'c3SV_{p}', 'c3REL_{p}'
            Results for each `p` are written to:
              "<output_dir>/CUReliability/{p}/<partition_labels...>/".

    Outputs (per matched coding/reliability pair and paradigm)
    ---------------------------------------------------------
    - Utterance-level Excel:
        ".../<labels>[_<PARADIGM>]_CUReliabilityCoding_ByUtterance.xlsx"
      (contains c2/c3 CU flags and AGSV/AGREL/AGCU indicators)
    - Sample-level Excel:
        ".../<labels>[_<PARADIGM>]_CUReliabilityCoding_BySample.xlsx"
    - Text report:
        ".../<labels>[_<PARADIGM>]_CUReliabilityCodingReport.txt"

    Pairing & Validation
    --------------------
    - Files are paired when all tier labels extracted from filenames match.
    - A failed merge or read logs an error and skips that pair.
    - If the merge reduces row count relative to the reliability file, a length
      mismatch is logged.

    Returns
    -------
    None
        All results are written to disk; nothing is returned.

    Notes
    -----
    - The function assumes `CU_paradigms` is a list (possibly empty). Upstream code
      should pass an empty list rather than None (e.g., `config.get('CU_paradigms', []) or []`).
    - Agreement on SV/REL/CU is defined as exact equality or both NaN.
    """
    
    # Make CU Reliability folder.
    CUReliability_dir = os.path.join(output_dir, 'CUReliability')
    try:
        os.makedirs(CUReliability_dir, exist_ok=True)
        logging.info(f"Created directory: {CUReliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {CUReliability_dir}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*CUCoding.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*CUReliabilityCoding.xlsx')]

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

                    CUcod_sub = CUcod.loc[:, ['utterance_id', 'sample_id', sv2, rel2]].copy()
                    CUrel_sub = CUrel.loc[:, ['utterance_id', sv3, rel3]].copy()

                    try:
                        CUrelcod = pd.merge(CUcod_sub, CUrel_sub, on="utterance_id", how="inner")
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
                    paradigm_str = f"_{paradigm}" if paradigm else ""
                    utterance_path = os.path.join(output_path, f"{'_'.join(partition_labels)}{paradigm_str}_CUReliabilityCoding_ByUtterance.xlsx")
                    CUrelcod.to_excel(utterance_path, index=False)

                    # Summary + report + save (unchanged)
                    # Just use CUrelcod and column names as they are
                    CUrelsum = summarize_CU_reliability(CUrelcod, sv2, rel2, sv3, rel3)
                    report_path = os.path.join(output_path, f"{'_'.join(partition_labels)}{paradigm_str}_CUReliabilityCodingReport.txt")
                    write_reliability_report(CUrelsum, report_path)
                    summary_path = os.path.join(output_path, f"{'_'.join(partition_labels)}{paradigm_str}_CUReliabilityCoding_BySample.xlsx")
                    CUrelsum.to_excel(summary_path, index=False)
                    

def analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms=[]):
    """
    Summarize coder-2 Complete Utterance (CU) coding at the sample level, optionally
    across multiple CU paradigms, and write both utterance-level and sample-level outputs.

    Parameters
    ----------
    tiers : dict[str, Any]
        Mapping of tier name -> tier object. Each tier object should provide:
          - .match(filename, ...) -> label string used for partitioning outputs, and
          - .partition : bool indicating inclusion in the output path.
        Example partitioning: site/test/participant labels derived from filenames.

    input_dir : str | os.PathLike
        Root directory searched (recursively) for "*CUCoding.xlsx" files.

    output_dir : str | os.PathLike
        Base directory where outputs are written under:
          "<output_dir>/CUCodingAnalysis/<partition_labels...>/"

    CU_paradigms : list[str] | None, default []
        CU paradigm labels (e.g., ["SAE","AAE"]). Behavior:
          - If None or [], paradigms are **inferred** from columns that start with "c2SV_".
            * If any are found, each such suffix is treated as a paradigm (e.g., "SAE").
            * If none are found, a single pass is run using the **base columns**
              'c2SV' and 'c2REL' (internally represented as paradigm == None).
              Note: summary column names will be suffixed with the string "None" (e.g.,
              'percCU_None'), reflecting current implementation.

    Outputs
    -------
    Two Excel files per coding file (and per partition path):
      1. Utterance-level: "<labels>_CUCoding_ByUtterance.xlsx"
         - Full coding frame with newly computed CU column per paradigm.
      2. Sample-level combined summary: "<labels>_CUCoding_BySample.xlsx"
         - If multiple paradigms are present, their per-sample summaries are merged on 'sample_id'.
         - For each paradigm P (or "None" for base), the summary includes:
             'no_utt_P', 'pSV_P', 'mSV_P', 'pREL_P', 'mREL_P', 'CU_P', 'percCU_P'

    Details
    -------
    - For each paradigm P:
        sv_col  = f"c2SV_{P}"  (or "c2SV" if P is None)
        rel_col = f"c2REL_{P}" (or "c2REL" if P is None)
        cu_col  = f"c2CU_{P}"  (or "c2CU" if P is None)
      The CU column is computed via `compute_CU_column(SV, REL)`, which returns:
        1 if SV==REL==1; 0 if both present but not both 1; NaN if both NaN; logs error if only one is NaN.

    - The function drops benign columns if present: ['c1ID', 'c1com', 'c2ID'].

    - Partition path uses tier labels from filename for tiers with `.partition == True`.
      Example: "<output_dir>/CUCodingAnalysis/<site>/"

    Returns
    -------
    None
        All artifacts are saved to disk; the function does not return a value.

    Notes
    -----
    - If a paradigm’s required columns are missing in a file, that paradigm is skipped with a warning.
    - Summary percentages are rounded to 3 decimals.
    """
    CUanalysis_dir = os.path.join(output_dir, 'CUCodingAnalysis')
    try:
        os.makedirs(CUanalysis_dir, exist_ok=True)
        logging.info(f"Created directory: {CUanalysis_dir}")
    except Exception as e:
        logging.error(f"Failed to create CU analysis directory {CUanalysis_dir}: {e}")
        return

    coding_files = list(Path(input_dir).rglob('*CUCoding.xlsx'))

    for cod in tqdm(coding_files, desc="Analyzing CU coding..."):
        try:
            CUcod = pd.read_excel(cod)
            logging.info(f"Processing CU coding file: {cod}")
        except Exception as e:
            logging.error(f"Failed to read CU coding file {cod}: {e}")
            continue

        # Clean base columns
        drop_cols = ['c1ID', 'c1com', 'c2ID']
        CUcod.drop(columns=[col for col in drop_cols if col in CUcod.columns], inplace=True, errors='ignore')

        # Determine paradigms from columns (if not provided)
        if not CU_paradigms:
            CU_paradigms = sorted(set(col.split('_')[-1] for col in CUcod.columns if col.startswith('c2SV_')))
            if not CU_paradigms:  # default fallback
                CU_paradigms = [None]

        summary_list = []

        for paradigm in CU_paradigms:
            sv_col = f'c2SV_{paradigm}' if paradigm else 'c2SV'
            rel_col = f'c2REL_{paradigm}' if paradigm else 'c2REL'
            cu_col = f'c2CU_{paradigm}' if paradigm else 'c2CU'

            if sv_col not in CUcod.columns or rel_col not in CUcod.columns:
                logging.warning(f"Skipping paradigm {paradigm}: columns missing in {cod.name}")
                continue

            # Compute CU column
            CUcod[cu_col] = CUcod[[sv_col, rel_col]].apply(compute_CU_column, axis=1)

            # Create summary stats
            agg_df = CUcod[['sample_id', sv_col, rel_col, cu_col]].copy()
            agg_df[[sv_col, rel_col, cu_col]] = agg_df[[sv_col, rel_col, cu_col]].apply(pd.to_numeric, errors='coerce')

            try:
                CUcodsum = agg_df.groupby('sample_id').agg(
                    **{
                        f'no_utt_{paradigm}': (cu_col, utt_ct),
                        f'pSV_{paradigm}': (sv_col, ptotal),
                        f'mSV_{paradigm}': (sv_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        f'pREL_{paradigm}': (rel_col, ptotal),
                        f'mREL_{paradigm}': (rel_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        f'CU_{paradigm}': (cu_col, ptotal),
                        f'percCU_{paradigm}': (cu_col, lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
                    }
                ).reset_index()
                summary_list.append(CUcodsum)
            except Exception as e:
                logging.error(f"Aggregation failed for {cod.name}, paradigm {paradigm}: {e}")
                continue

        # Save full utterance-level file
        partition_labels = [t.match(cod.name) for t in tiers.values() if t.partition]
        out_dir = os.path.join(CUanalysis_dir, *partition_labels)

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create output directory {out_dir}: {e}")
            continue

        utterance_path = os.path.join(out_dir, f"{'_'.join(partition_labels)}_CUCoding_ByUtterance.xlsx")
        try:
            CUcod.to_excel(utterance_path, index=False)
            logging.info(f"Saved utterance-level CU analysis: {utterance_path}")
        except Exception as e:
            logging.error(f"Failed to save utterance-level file: {e}")

        # Merge all paradigm summaries
        if summary_list:
            try:
                CUcodsum_all = summary_list[0]
                for df in summary_list[1:]:
                    CUcodsum_all = pd.merge(CUcodsum_all, df, on='sample_id', how='outer')

                summary_path = os.path.join(out_dir, f"{'_'.join(partition_labels)}_CUCoding_BySample.xlsx")
                CUcodsum_all.to_excel(summary_path, index=False)
                logging.info(f"Saved combined CU summary: {summary_path}")

            except Exception as e:
                logging.error(f"Failed to merge and save summary files: {e}")
