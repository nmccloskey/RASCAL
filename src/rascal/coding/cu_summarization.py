import pandas as pd
from pathlib import Path
from rascal.utils.logger import logger, _rel
from rascal.utils.auxiliary import extract_transcript_data, find_files


def _aggregate_sample_level(merged_utts, wc_by_utt, cu_by_sample, file):
    """
    Aggregate merged utterance data to the sample level and compute WPM.

    Returns
    -------
    merged_samples : pd.DataFrame
        Unblind sample-level table.
    """
    coder_cols = [c for c in merged_utts.columns if c.startswith(("c1", "c2"))]
    utt_data = (
        merged_utts.drop(columns=["utterance_id", "speaker", "word_count"] + coder_cols, errors="ignore")
                  .drop_duplicates()
    )
    wc_samples = wc_by_utt.groupby("sample_id", dropna=False)["word_count"].sum().reset_index()

    merged_samples = (
        utt_data.merge(cu_by_sample, on="sample_id", how="inner")
                .merge(wc_samples, on="sample_id", how="inner")
    )
    logger.info(f"Aggregated to sample-level for {_rel(file)} — {len(merged_samples)} samples")

    # WPM
    if "speaking_time" in merged_samples:
        merged_samples["wpm"] = merged_samples.apply(
            lambda r: round(r["word_count"] / (r["speaking_time"] / 60), 2)
            if r.get("speaking_time", 0) not in [0, None, 0.0]
            else pd.NA,
            axis=1,
        )
    else:
        merged_samples["wpm"] = pd.NA
        logger.warning(f"No 'speaking_time' column found for {_rel(file)}")

    return merged_samples


def _process_cu_file(file, utt_df, tiers, input_dir):
    """
    Merge utterance-level CU and word-count data for one transcript file.

    Returns
    -------
    tuple | None
        (merged_utts, merged_samples) or None on failure.
    """
    match_tiers = [t.match(file.name) for t in tiers.values() if t.partition]
    if not match_tiers:
        logger.warning(f"No match tiers found for {_rel(file)} — skipping.")
        return None

    # Locate related inputs
    try:
        cu_by_utt_paths = find_files(match_tiers, input_dir, "cu_coding_by_utterance")
        wc_by_utt_paths = find_files(match_tiers, input_dir, "word_counting")
        cu_by_sample_paths = find_files(match_tiers, input_dir, "cu_coding_by_sample")
        if not all([cu_by_utt_paths, wc_by_utt_paths, cu_by_sample_paths]):
            raise FileNotFoundError("One or more corresponding files could not be found.")
    except Exception as e:
        logger.error(f"Missing or invalid related data for {_rel(file)}: {e}")
        return None

    # Read
    try:
        cu_by_utt = pd.read_excel(cu_by_utt_paths[0])
        wc_by_utt = pd.read_excel(wc_by_utt_paths[0])
        cu_by_sample = pd.read_excel(cu_by_sample_paths[0])
        logger.info(f"Loaded CU, WC, and sample data for {_rel(file)}")
    except Exception as e:
        logger.error(f"Error reading related data for {_rel(file)}: {e}")
        return None

    # Merge utterance-level
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

    # Sample-level (still UNBLIND)
    try:
        merged_samples = _aggregate_sample_level(
            merged_utts, wc_by_utt, cu_by_sample, file
        )
    except Exception as e:
        logger.error(f"Sample-level aggregation failed for {_rel(file)}: {e}")
        return None

    return merged_utts, merged_samples


def _apply_blinding(df, tiers):
    """
    Apply tier-based blind codes to a merged utterance dataframe (concatenated).

    Returns
    -------
    blind_df : pd.DataFrame
    blind_codes : dict  # {tier_name: {raw_label: blind_code}}
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


def _apply_blind_codes_to_samples(samples_df, tiers, blind_codes):
    """
    Apply precomputed blind codes to a concatenated sample-level dataframe.

    Returns
    -------
    blind_samples : pd.DataFrame
    """
    remove_tiers = [t.name for t in tiers.values() if not t.blind]
    blind_cols = [t.name for t in tiers.values() if t.blind]

    blind_samples = samples_df.drop(columns=["file"] + remove_tiers, errors="ignore").copy()
    for tier_name in blind_cols:
        col = tiers[tier_name].name
        if col in blind_samples:
            blind_samples[col] = blind_samples[col].map(blind_codes.get(col, {}))
    return blind_samples


def _write_cu_summary_outputs(out_dir, unblind_utts, blind_utts, unblind_samples, blind_samples, blind_codes):
    """
    Write CU summary tables and blind-code map.

    Parameters
    ----------
    unblind_utts, blind_utts, unblind_samples, blind_samples : pd.DataFrame
        Already-concatenated dataframes.
    """
    try:
        filename = out_dir / "cu_summaries.xlsx"
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            unblind_utts.to_excel(writer, sheet_name="unblind_utterances", index=False)
            blind_utts.to_excel(writer,   sheet_name="blind_utterances",   index=False)
            unblind_samples.to_excel(writer, sheet_name="unblind_samples", index=False)
            blind_samples.to_excel(writer,   sheet_name="blind_samples",   index=False)
        logger.info(f"Wrote combined CU summary workbook to {_rel(filename)}")
    except Exception as e:
        logger.error(f"Failed writing CU summaries workbook {_rel(out_dir)}: {e}")

    try:
        blind_codes_file = out_dir / "blind_codes.xlsx"
        # flatten dict-of-dicts for readability
        bc_rows = []
        for tier_name, mapping in blind_codes.items():
            for raw, code in mapping.items():
                bc_rows.append({"tier": tier_name, "raw": raw, "blind_code": code})
        pd.DataFrame(bc_rows).to_excel(blind_codes_file, index=False)
        logger.info(f"Blind codes saved to {_rel(blind_codes_file)}")
    except Exception as e:
        logger.error(f"Failed writing blind codes to {_rel(out_dir)}: {e}")


def summarize_cus(tiers, input_dir, output_dir):
    """
    Produce CU summary tables with proper blinding workflow.

    Steps
    -----
    1) Build per-file UNBLIND utterance & sample tables.
    2) Concatenate all UNBLIND tables.
    3) Apply one blinding pass to concatenated utterances; reuse codes for samples.
    4) Write outputs (no concatenation in the writer).
    """
    out_dir = Path(output_dir) / "cu_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CU summary outputs will be written to {_rel(out_dir)}")

    try:
        transcript_tables = find_files(directories=[input_dir, output_dir], search_base="transcript_tables")
        utt_tables = {tt: extract_transcript_data(tt) for tt in transcript_tables}

        unblind_utt_dfs, unblind_sample_dfs = [], []

        for file, utt_df in utt_tables.items():
            try:
                result = _process_cu_file(file, utt_df, tiers, input_dir)
                if result is None:
                    continue
                merged_utts, merged_samples = result
                unblind_utt_dfs.append(merged_utts)
                unblind_sample_dfs.append(merged_samples)
            except Exception as e:
                logger.error(f"Failed processing {_rel(file)}: {e}")

        if not unblind_utt_dfs:
            logger.warning("No CU summary data collected — nothing to write.")
            return

        # 2) Concatenate unblind
        all_unblind_utts = pd.concat(unblind_utt_dfs, ignore_index=True)
        all_unblind_samples = pd.concat(unblind_sample_dfs, ignore_index=True)

        # 3) One blinding pass; reuse codes for samples
        blind_utts, blind_codes_output = _apply_blinding(all_unblind_utts, tiers)
        blind_samples = _apply_blind_codes_to_samples(all_unblind_samples, tiers, blind_codes_output)

        # 4) Write
        _write_cu_summary_outputs(
            out_dir,
            all_unblind_utts,
            blind_utts,
            all_unblind_samples,
            blind_samples,
            blind_codes_output,
        )

    except Exception as e:
        logger.error(f"CU summarization failed: {e}")
        raise
