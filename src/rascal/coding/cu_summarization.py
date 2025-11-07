import pandas as pd
from pathlib import Path
from rascal.utils.logger import logger, _rel
from rascal.utils.auxiliary import (
    extract_transcript_data,
    find_files,
)

def _apply_blinding(df, tiers):
    """
    Apply tier-based blind codes to a merged utterance dataframe.

    Returns
    -------
    blind_df : pd.DataFrame
    blind_codes : dict
        Mapping {tier_name: {raw_label: blind_code, ...}}.
    """
    blind_df = df.copy()
    blind_codes = {}

    remove_tiers = [t.name for t in tiers.values() if not t.blind]
    blind_df.drop(columns=["file"] + remove_tiers, errors="ignore", inplace=True)
    blind_columns = [t.name for t in tiers.values() if t.blind]

    for tier_name in blind_columns:
        tier = tiers[tier_name]
        try:
            codes = tier.make_blind_codes()
            col = tier.name
            if col in blind_df:
                blind_df[col] = blind_df[col].map(codes[tier.name])
                blind_codes.update(codes)
            logger.debug(f"Applied blinding to {col}")
        except Exception as e:
            logger.warning(f"Failed to blind column {tier_name}: {e}")

    logger.info(f"Blinding applied; total columns blinded: {len(blind_columns)}")
    return blind_df, blind_codes

def _aggregate_sample_level(merged_utts, wc_by_utt, cu_by_sample, tiers, blind_codes_output, file):
    """
    Aggregate merged utterance data to the sample level, add WPM,
    and produce blinded equivalents.

    Returns
    -------
    merged_samples, blind_samples : pd.DataFrame, pd.DataFrame
    """
    utt_data = merged_utts.drop(columns=["utterance_id", "speaker", "word_count"], errors="ignore").drop_duplicates()
    wc_samples = wc_by_utt.groupby("sample_id", dropna=False)["word_count"].sum().reset_index()

    merged_samples = (
        utt_data.merge(cu_by_sample, on="sample_id", how="inner")
        .merge(wc_samples, on="sample_id", how="inner")
    )
    logger.info(f"Aggregated to sample-level for {_rel(file)} — {len(merged_samples)} samples")

    # --- Compute WPM safely ---
    if "speaking_time" in merged_samples:
        merged_samples["wpm"] = merged_samples.apply(
            lambda r: round(r["word_count"] / (r["speaking_time"] / 60), 2)
            if r.get("speaking_time", 0) not in [0, None]
            else pd.NA,
            axis=1,
        )
        logger.debug("Calculated words per minute (WPM).")
    else:
        merged_samples["wpm"] = pd.NA
        logger.warning(f"No 'speaking_time' column found for {_rel(file)}")

    # --- Create blinded sample-level table ---
    remove_tiers = [t.name for t in tiers.values() if not t.blind]
    blind_cols = [t.name for t in tiers.values() if t.blind]
    blind_samples = merged_samples.drop(columns=remove_tiers, errors="ignore").copy()

    for tier_name in blind_cols:
        tier = tiers[tier_name]
        col = tier.name
        if col in blind_samples:
            blind_samples[col] = blind_samples[col].map(blind_codes_output.get(tier.name, {}))

    logger.info(f"Created blinded sample-level data for {_rel(file)}.")
    return merged_samples, blind_samples

def _process_cu_file(file, utt_df, tiers, input_dir):
    """
    Merge utterance-level CU and word-count data for one transcript file,
    then generate both blinded and sample-level versions.

    Parameters
    ----------
    file : Path
        Transcript table file.
    utt_df : pd.DataFrame
        Utterance-level data extracted from the transcript table.
    tiers : dict[str, Tier]
        Tier objects with `.match()`, `.partition`, and `.blind` attributes.
    input_dir : Path
        Directory where CU and word-count files reside.

    Returns
    -------
    tuple | None
        (merged_utts, blind_utts, merged_samples, blind_samples, blind_codes_output),
        or None if processing fails.
    """
    match_tiers = [t.match(file.name) for t in tiers.values() if t.partition]
    if not match_tiers:
        logger.warning(f"No match tiers found for {_rel(file)} — skipping.")
        return None

    # --- Locate related input files ---
    try:
        cu_by_utt_paths = find_files(match_tiers, input_dir, "cu_coding_by_utterance")
        wc_by_utt_paths = find_files(match_tiers, input_dir, "word_counting")
        cu_by_sample_paths = find_files(match_tiers, input_dir, "cu_coding_by_sample")

        if not all([cu_by_utt_paths, wc_by_utt_paths, cu_by_sample_paths]):
            raise FileNotFoundError("One or more corresponding files could not be found.")
    except Exception as e:
        logger.error(f"Missing or invalid related data for {_rel(file)}: {e}")
        return None

    # --- Read dataframes ---
    try:
        cu_by_utt = pd.read_excel(cu_by_utt_paths[0])
        wc_by_utt = pd.read_excel(wc_by_utt_paths[0])
        cu_by_sample = pd.read_excel(cu_by_sample_paths[0])
        logger.info(f"Loaded CU, WC, and sample data for {_rel(file)}")
    except Exception as e:
        logger.error(f"Error reading related data for {_rel(file)}: {e}")
        return None

    # --- Merge utterance-level tables ---
    try:
        cu_cols = cu_by_utt.columns.tolist()
        c2_idx = cu_cols.index("c2_comment") + 1 if "c2_comment" in cu_cols else len(cu_cols)
        cu_by_utt = cu_by_utt.loc[:, ["sample_id", "utterance_id"] + cu_cols[c2_idx:]]
        wc_by_utt = wc_by_utt.loc[:, ["sample_id", "utterance_id", "word_count"]]
    except Exception as e:
        logger.error(f"Unexpected CU/word-count column structure for {_rel(file)}: {e}")
        return None

    merged_utts = (
        utt_df.drop(columns=["utterance", "comment"], errors="ignore")
        .merge(cu_by_utt, on=["sample_id", "utterance_id"], how="inner")
        .merge(wc_by_utt, on=["sample_id", "utterance_id"], how="left")
    )
    logger.info(f"Merged utterance data for {_rel(file)} — {len(merged_utts)} rows")

    # --- Apply blinding ---
    try:
        blind_utts, blind_codes_output = _apply_blinding(merged_utts, tiers)
    except Exception as e:
        logger.error(f"Blinding failed for {_rel(file)}: {e}")
        blind_utts, blind_codes_output = merged_utts.copy(), {}

    # --- Sample-level aggregation ---
    try:
        merged_samples, blind_samples = _aggregate_sample_level(
            merged_utts, wc_by_utt, cu_by_sample, tiers, blind_codes_output, file
        )
    except Exception as e:
        logger.error(f"Sample-level aggregation failed for {_rel(file)}: {e}")
        return None

    return merged_utts, blind_utts, merged_samples, blind_samples, blind_codes_output

def _write_cu_summary_outputs(out_dir, unblind_utts, blind_utts, unblind_samples, blind_samples, blind_codes):
    """Write CU summary tables and blind-code map with detailed logging."""
    try:
        filename = out_dir / "cu_summaries.xlsx"
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            sheets = {
                "unblind_utterances": pd.concat(unblind_utts, ignore_index=True),
                "blind_utterances": pd.concat(blind_utts, ignore_index=True),
                "unblind_samples": pd.concat(unblind_samples, ignore_index=True),
                "blind_samples": pd.concat(blind_samples, ignore_index=True),
            }
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        logger.info(f"Wrote combined CU summary workbook to {_rel(filename)}")
    except Exception as e:
        logger.error(f"Failed writing CU summaries workbook {_rel(out_dir)}: {e}")

    try:
        blind_codes_file = out_dir / "blind_codes.xlsx"
        pd.DataFrame(blind_codes).to_excel(blind_codes_file, index=True)
        logger.info(f"Blind codes saved to {_rel(blind_codes_file)}")
    except Exception as e:
        logger.error(f"Failed writing blind codes to {_rel(out_dir)}: {e}")


def summarize_cus(tiers, input_dir, output_dir):
    """
    Merge utterance-, sample-, and CU-coding data to produce blinded/unblinded
    summary tables and blind-code maps.

    Behavior
    --------
    • Reads utterance, CU, word-count, and speaking-time files under `input_dir`.
    • Merges utterance-level data, computes CU + word metrics, and aggregates by sample.
    • Applies tier-based blind codes to create blinded versions.
    • Writes a combined Excel file with four sheets plus a separate blind-code map.

    Parameters
    ----------
    tiers : dict[str, Tier]
        Tier objects with `.partition`, `.blind`, and `.make_blind_codes()`.
    input_dir, output_dir : Path or str
        Root directories for input and summarized outputs.
    """
    out_dir = Path(output_dir) / "cu_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CU summary outputs will be written to {_rel(out_dir)}")

    try:
        transcript_tables = find_files(directories=[input_dir, output_dir],
                                                    search_base="transcript_tables")
        utt_tables = {tt: extract_transcript_data(tt) for tt in transcript_tables}
        unblind_utt_dfs, blind_utt_dfs = [], []
        unblind_sample_dfs, blind_sample_dfs = [], []
        blind_codes_output = {}

        for file, utt_df in utt_tables.items():
            try:
                result = _process_cu_file(file, utt_df, tiers, input_dir)
                if result is None:
                    continue
                merged_utts, blind_utts, merged_samples, blind_samples, blind_codes = result
                unblind_utt_dfs.append(merged_utts)
                blind_utt_dfs.append(blind_utts)
                unblind_sample_dfs.append(merged_samples)
                blind_sample_dfs.append(blind_samples)
                blind_codes_output.update(blind_codes)
            except Exception as e:
                logger.error(f"Failed processing {_rel(file)}: {e}")

        if not unblind_utt_dfs:
            logger.warning("No CU summary data collected — nothing to write.")
            return

        _write_cu_summary_outputs(
            out_dir,
            unblind_utt_dfs,
            blind_utt_dfs,
            unblind_sample_dfs,
            blind_sample_dfs,
            blind_codes_output,
        )

    except Exception as e:
        logger.error(f"CU summarization failed: {e}")
        raise
