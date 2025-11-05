import pandas as pd
from pathlib import Path
from rascal.utils.logger import logger
from rascal.utils.auxiliary import find_transcript_tables, extract_transcript_data, find_corresponding_file


def summarize_cus(tiers, input_dir, output_dir):
    """
    Build unblinded and blinded summaries by merging utterances, CU coding,
    word counts, and speaking-time data; then export both utterance-level and
    sample-level tables plus the blind-code key.

    Workflow
    --------
    1) Read and vertically concat the following from `input_dir`:
         - "*Utterances.xlsx" (expects at least: 'utterance_id','sample_id',
           'file','speaker','utterance','comment', and any tier columns by name)
         - "*cu_coding_by_utterance*.xlsx" (expects: 'utterance_id','sample_id',
           'comment', and CU/coder columns to the **right** of 'comment')
         - "*word_counting*.xlsx" (expects: 'utterance_id','sample_id','word_count','WCcom')
         - "*SpeakingTimes.xlsx" (expects: 'sample_id','speaking_time')
         - "*cu_coding_by_sample.xlsx" (sample-level CU metrics; merged later)
    2) Merge utterance-level tables on ['utterance_id','sample_id'] and add speaking time.
       Save as "Summaries/unblind_utterance_data.xlsx".
    3) Produce a **blinded** utterance table by:
         - Dropping "file" and any tier columns whose tier.blind == False,
         - Mapping each blind tier's labels via `tier.make_blind_codes()`.
       Save as "Summaries/blind_utterance_data.xlsx" and retain the mapping(s).
    4) Build a sample-level table:
         - From utterances, drop ['utterance_id','speaker','utterance','comment'] and dedupe,
         - Merge with "*cu_coding_by_sample.xlsx", summed word counts per sample,
           and speaking time,
         - Compute words-per-minute (wpm) = word_count / (speaking_time / 60).
       Save as "Summaries/unblind_sample_data.xlsx".
    5) Produce a **blinded** sample table by dropping non-blind tiers and applying
       the same blind-code mapping(s). Save as "Summaries/blind_sample_data.xlsx".
    6) Export the blind-code key as "Summaries/blind_codes.xlsx".

    Parameters
    ----------
    tiers : dict[str, Any]
        Mapping of tier name → tier object. Each tier object must provide:
          - .name : str  (column name present in the data)
          - .blind : bool (True if this tier should be blinded/mapped)
          - .make_blind_codes() -> dict[str, dict[str, str]]
                Returns { tier.name : { raw_label : blind_code, ... } }
    input_dir : str | os.PathLike
        Root directory searched recursively for the input Excel files listed above.
    output_dir : str | os.PathLike
        Base directory where outputs are written under "<output_dir>/Summaries/".

    Outputs
    -------
    Summaries/unblind_utterance_data.xlsx
    Summaries/blind_utterance_data.xlsx
    Summaries/unblind_sample_data.xlsx
    Summaries/blind_sample_data.xlsx
    Summaries/blind_codes.xlsx

    Returns
    -------
    None

    Notes
    -----
    - Blinding only touches columns for tiers with tier.blind == True.
      Non-blind tier columns are removed from the blinded outputs.
    - If required columns are missing or a merge fails, an error is logged and
      the exception is re-raised by the outer try/except.
    - wpm is computed as word_count / (speaking_time / 60) and rounded to 2 decimals.
    """
    try:
        # Specify subfolder and create directory
        output_dir = output_dir, 'Summaries'
        output_dir.mkdir(parents=True, exist_ok=True)

        # identify partition tiers to complete composite PK
        partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]

        # Read utterance data
        transcript_tables = find_transcript_tables(input_dir, output_dir)
        utt_tables = {tt:extract_transcript_data(tt) for tt in transcript_tables}

        for file, ut in utt_tables:
            # tier values on which to match
            match_tiers = [t.match(file.name) for t in tiers if t.partition]
            # CU data
            cu_df = find_corresponding_file(match_tiers=match_tiers,
                                            directory=input_dir,
                                            search_base="cu_coding_by_utterance")
            cu_by_utt = cu_df.loc[:, ["sample_id", "utterance_id"] + list(cu_df.iloc[:, cu_df.columns.to_list().index('comment')+1:].columns)]
            # word count data
            wc_df = find_corresponding_file(match_tiers=match_tiers,
                                            directory=input_dir,
                                            search_base="word_counting")
            wc_by_utt = wc_df.loc[:, ["sample_id", "utterance_id", "word_count"]]
            df = cu_by_utt.merge(wc_by_utt, on=["sample_id", "utterance_id"], how="outer")


            

        # utts = pd.concat([extract_transcript_data(tt) for tt in transcript_tables], ignore_index=True, sort=False)

        # Read CU data
        cu_by_utt = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*cu_coding_by_utterance*.xlsx')])
        cu_by_utt = cu_by_utt.loc[:, ['utterance_id', 'sample_id'] + list(cu_by_utt.iloc[:, cu_by_utt.columns.to_list().index('comment')+1:].columns)]
        logger.info("CU utterance data loaded successfully.")

        # Read word count data
        word_counts = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*word_counting*.xlsx')])
        word_counts = word_counts.loc[:, ['utterance_id', 'sample_id', 'word_count', 'wc_comment']]

        # Merge datasets
        merged_utts = utts.copy()
        merged_utts = pd.merge(merged_utts, cu_by_utt, on=['utterance_id', 'sample_id'], how='inner')
        merged_utts = pd.merge(merged_utts, word_counts, on=['utterance_id', 'sample_id'], how='inner')
        logger.info("Utterance data merged successfully.")

        # Save unblinded utterances
        unblinded_utts = output_dir / 'unblind_utterance_data.xlsx'
        merged_utts.to_excel(unblinded_utts, index=False)
        logger.info(f"Unblinded utterances saved to {unblinded_utts}.")

        # Prepare blind codes and blinded utterances
        remove_tiers = [t.name for t in tiers.values() if not t.blind]
        blind_utts = merged_utts.drop(columns=["file"]+remove_tiers)
        blind_codes_output = {}
        blind_columns = [t.name for t in tiers.values() if t.blind]
        for tier_name in blind_columns:
            tier = tiers[tier_name]
            blind_codes = tier.make_blind_codes()
            column_name = tier.name
            if column_name in blind_utts.columns:
                blind_utts[column_name] = blind_utts[column_name].map(blind_codes[tier.name])
                blind_codes_output.update(blind_codes)
        logger.info("Blinded utterance data prepared successfully.")

        # Save blinded utterances
        blind_utts_file = output_dir / 'blind_utterance_data.xlsx'
        blind_utts.to_excel(blind_utts_file, index=False)
        logger.info(f"Blinded utterances saved to {blind_utts_file}.")

        # Aggregate by sample - first filter utterance data.
        utts = utts.drop(columns=['utterance_id', 'speaker', 'utterance', 'comment']).drop_duplicates(keep='first')
        logger.info("Utterance data loaded and preprocessed successfully.")
    
        # Load sample CU data.
        cu_by_sample = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*cu_coding_by_sample*.xlsx')])
        
        # Sum word counts.
        word_counts = word_counts.groupby(['sample_id']).agg(word_count=('word_count', 'sum'))
        logger.info("Word count data aggregated successfully.")

        merged_samples = utts.copy()
        merged_samples = pd.merge(merged_samples, cu_by_sample, on='sample_id', how='inner')
        merged_samples = pd.merge(merged_samples, word_counts, on='sample_id', how='inner')
        logger.info("Sample data merged successfully.")

        # Calculate words per minute
        merged_samples['wpm'] = merged_samples.apply(lambda row: round(row['word_count'] / (row['speaking_time'] / 60), 2), axis=1)
        logger.info("Words per minute calculated successfully.")

        # Save unblinded summary
        unblinded_sample_path = output_dir / 'unblind_sample_data.xlsx'
        merged_samples.to_excel(unblinded_sample_path, index=False)
        logger.info(f"Unblinded summary saved to {unblinded_sample_path}.")

        # Prepare blinded samples
        blind_samples = merged_samples.copy()
        blind_samples = blind_samples.drop(columns=remove_tiers)
        for tier_name in blind_columns:
            tier = tiers[tier_name]
            column_name = tier.name
            if column_name in blind_samples.columns:
                blind_samples[column_name] = blind_samples[column_name].map(blind_codes_output[tier.name])
        logger.info("Blinded utterance data prepared successfully.")

        # Save blinded summary
        blinded_samples_path = output_dir / 'blind_sample_data.xlsx'
        blind_samples.to_excel(blinded_samples_path, index=False)
        logger.info(f"Blinded summary saved to {blinded_samples_path}.")

        # Save blind codes separately
        blind_codes_file = output_dir / 'blind_codes.xlsx'
        pd.DataFrame(blind_codes_output).to_excel(blind_codes_file, index=True)
        logger.info(f"Blind codes saved to {blind_codes_file}.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
