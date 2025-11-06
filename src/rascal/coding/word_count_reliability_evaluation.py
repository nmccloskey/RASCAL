import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rascal.utils.logger import logger, _rel


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

def calculate_icc(data: pd.DataFrame) -> float:
    """
    Calculate the Intraclass Correlation Coefficient (ICC[2,1]) safely.

    Parameters
    ----------
    data : pd.DataFrame
        Two-column DataFrame with numeric scores (e.g., 'word_count_org', 'word_count_rel').

    Returns
    -------
    float | np.nan
        ICC(2,1) value rounded to 4 decimals, or NaN if undefined.
    """
    if data is None or data.empty:
        logger.warning("ICC calculation skipped: empty or None data.")
        return np.nan

    data = data.dropna()
    n, k = data.shape
    if n < 2 or k < 2:
        logger.warning(f"ICC calculation skipped: insufficient data (n={n}, k={k}).")
        return np.nan

    try:
        # Means
        mean_per_subject = data.mean(axis=1)
        mean_per_rater = data.mean(axis=0)
        grand_mean = data.values.flatten().mean()

        # Sum of squares
        ss_between = np.sum((mean_per_subject - grand_mean) ** 2) * k
        ss_within = np.sum((data.values - mean_per_subject.values[:, None]) ** 2)
        ss_rater = np.sum((mean_per_rater - grand_mean) ** 2) * n

        # Mean squares (protect against zero denom)
        ms_between = ss_between / (n - 1) if (n - 1) > 0 else np.nan
        ms_within = ss_within / (n * (k - 1)) if (n * (k - 1)) > 0 else np.nan
        ms_rater = ss_rater / (k - 1) if (k - 1) > 0 else np.nan

        denom = ms_between + (k - 1) * ms_within + (k / n) * (ms_rater - ms_within)
        if denom == 0 or np.isnan(denom):
            logger.warning("ICC denominator is zero or NaN — returning NaN.")
            return np.nan

        icc = (ms_between - ms_within) / denom
        return round(icc, 4)

    except Exception as e:
        logger.error(f"ICC calculation failed: {e}")
        return np.nan

def _write_word_rel_outputs(wc_merged, out_dir, partition_labels, rel_name):
    """Write Excel + report for word-count reliability with relative-path logging."""
    label_str = "_".join(partition_labels)
    lab_prefix = f"{label_str}_" if label_str else ""

    # Guard against empty DataFrame
    if wc_merged.empty:
        logger.warning(f"No merged rows found for {_rel(out_dir)}; skipping output.")
        return

    # Excel results
    results_path = out_dir / f"{lab_prefix}word_counting_reliability_results.xlsx"
    try:
        wc_merged.to_excel(results_path, index=False)
        logger.info(f"Wrote word-count reliability results: {_rel(results_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability results {_rel(results_path)}: {e}")
        return

    # ICC calculation
    icc_data = wc_merged[["word_count_org", "word_count_rel"]].dropna()
    icc_value = calculate_icc(icc_data)
    logger.info(f"Calculated ICC(2,1) for {rel_name}: {icc_value}")

    # Agreement summary (avoid divide by zero)
    total = len(wc_merged)
    n_agree = int(np.nansum(wc_merged.get("agmt", []))) if total else 0
    perc_agree = round((n_agree / total) * 100, 1) if total > 0 else np.nan

    report_path = out_dir / f"{lab_prefix}word_count_reliability_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write(f"Word Count Reliability Report for {' '.join(partition_labels)}\n\n")
            if total > 0:
                f.write(f"Utterances in agreement: {n_agree}/{total} ({perc_agree}%)\n")
            else:
                f.write("No valid utterances available for agreement calculation.\n")
            f.write(f"ICC(2,1): {icc_value}\n")
        logger.info(f"Successfully wrote reliability report to {_rel(report_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability report {_rel(report_path)}: {e}")

def evaluate_word_count_reliability(tiers, input_dir, output_dir):
    """
    Evaluate word count reliability by comparing coder-1 and coder-2 counts.

    Behavior
    --------
    • Finds and pairs *word_counting*.xlsx (coding) with
      *word_counting_reliability*.xlsx (reliability) files sharing tier labels.
    • Merges on 'utterance_id', computes:
        abs_diff, perc_diff, perc_sim, agmt (≤1 word or ≥85 % similar)
    • Calculates ICC(2,1) across utterances.
    • Writes merged table and plain-text report under:
        <output_dir>/word_count_reliability/<tier_labels>/

    Parameters
    ----------
    tiers : dict[str, Tier]
    input_dir, output_dir : Path or str

    Notes
    -----
    Agreement = 1 if |diff| ≤ 1 or percent similarity ≥ 85 %.
    """
    word_rel_dir = Path(output_dir) / "word_count_reliability"
    word_rel_dir.mkdir(parents=True, exist_ok=True)

    coding_files = list(Path(input_dir).rglob("*word_counting*.xlsx"))
    rel_files = list(Path(input_dir).rglob("*word_counting_reliability*.xlsx"))

    for rel in tqdm(rel_files, desc="Analyzing word count reliability..."):
        rel_labels = [t.match(rel.name, return_None=True) for t in tiers.values()]
        for cod in coding_files:
            cod_labels = [t.match(cod.name, return_None=True) for t in tiers.values()]
            if rel_labels != cod_labels:
                continue

            try:
                wc_df = pd.read_excel(cod)
                wc_rel_df = pd.read_excel(rel)
                logger.info(f"Processing pair: {_rel(cod)} + {_rel(rel)}")
            except Exception as e:
                logger.error(f"Failed reading {_rel(cod)} or {_rel(rel)}: {e}")
                continue

            try:
                wc_rel_df = wc_rel_df[["utterance_id", "wc_rel_com", "word_count"]].dropna(subset=["word_count"])
                wc_merged = pd.merge(wc_df, wc_rel_df, on="utterance_id", how="inner",
                                     suffixes=("_org", "_rel"))
                if len(wc_rel_df) != len(wc_merged):
                    logger.warning(f"Row mismatch after merge on {_rel(rel)}")
            except Exception as e:
                logger.error(f"Failed merging {_rel(cod)} and {_rel(rel)}: {e}")
                continue

            wc_merged["abs_diff"] = wc_merged["word_count_org"] - wc_merged["word_count_rel"]
            wc_merged["perc_diff"] = wc_merged.apply(
                lambda r: percent_difference(r["word_count_org"], r["word_count_rel"]), axis=1
            )
            wc_merged["perc_sim"] = 100 - wc_merged["perc_diff"]
            wc_merged["agmt"] = wc_merged.apply(agreement, axis=1)

            partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]
            out_dir = Path(word_rel_dir, *partition_labels)
            out_dir.mkdir(parents=True, exist_ok=True)

            _write_word_rel_outputs(wc_merged, out_dir, partition_labels, rel.name)
            