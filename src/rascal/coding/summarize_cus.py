import logging
import pandas as pd
from pathlib import Path
from rascal.utils.support_funcs import find_transcript_tables, extract_transcript_data


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
         - "*CUCoding_ByUtterance.xlsx" (expects: 'utterance_id','sample_id',
           'comment', and CU/coder columns to the **right** of 'comment')
         - "*WordCounting.xlsx" (expects: 'utterance_id','sample_id','wordCount','WCcom')
         - "*SpeakingTimes.xlsx" (expects: 'sample_id','client_time')
         - "*CUCoding_BySample.xlsx" (sample-level CU metrics; merged later)
    2) Merge utterance-level tables on ['utterance_id','sample_id'] and add speaking time.
       Save as "Summaries/unblindUtteranceData.xlsx".
    3) Produce a **blinded** utterance table by:
         - Dropping "file" and any tier columns whose tier.blind == False,
         - Mapping each blind tier's labels via `tier.make_blind_codes()`.
       Save as "Summaries/blindUtteranceData.xlsx" and retain the mapping(s).
    4) Build a sample-level table:
         - From utterances, drop ['utterance_id','speaker','utterance','comment'] and dedupe,
         - Merge with "*CUCoding_BySample.xlsx", summed word counts per sample,
           and speaking time,
         - Compute words-per-minute (wpm) = wordCount / (client_time / 60).
       Save as "Summaries/unblindSampleData.xlsx".
    5) Produce a **blinded** sample table by dropping non-blind tiers and applying
       the same blind-code mapping(s). Save as "Summaries/blindSampleData.xlsx".
    6) Export the blind-code key as "Summaries/blindCodes.xlsx".

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
    Summaries/unblindUtteranceData.xlsx
    Summaries/blindUtteranceData.xlsx
    Summaries/unblindSampleData.xlsx
    Summaries/blindSampleData.xlsx
    Summaries/blindCodes.xlsx

    Returns
    -------
    None

    Notes
    -----
    - Blinding only touches columns for tiers with tier.blind == True.
      Non-blind tier columns are removed from the blinded outputs.
    - If required columns are missing or a merge fails, an error is logged and
      the exception is re-raised by the outer try/except.
    - wpm is computed as wordCount / (client_time / 60) and rounded to 2 decimals.
    """
    try:
        # Specify subfolder and create directory
        output_dir = output_dir, 'Summaries'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read utterance data
        transcript_tables = find_transcript_tables(input_dir, output_dir)
        utts = pd.concat([extract_transcript_data(tt) for tt in transcript_tables], ignore_index=True, sort=False)

        # Read CU data
        CUbyUtts = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*CUCoding_ByUtterance.xlsx')])
        CUbyUtts = CUbyUtts.loc[:, ['utterance_id', 'sample_id'] + list(CUbyUtts.iloc[:, CUbyUtts.columns.to_list().index('comment')+1:].columns)]
        logging.info("CU utterance data loaded successfully.")

        # Read word count data
        WCs = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*WordCounting.xlsx')])
        WCs = WCs.loc[:, ['utterance_id', 'sample_id', 'word_count', 'wc_com']]

        # Read speaking time data
        times = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*SpeakingTimes.xlsx')])
        times = times.loc[:, ['sample_id', 'client_time']]
        logging.info("Speaking time data loaded successfully.")

        # Merge datasets
        merged_utts = utts.copy()
        merged_utts = pd.merge(merged_utts, CUbyUtts, on=['utterance_id', 'sample_id'], how='inner')
        merged_utts = pd.merge(merged_utts, WCs, on=['utterance_id', 'sample_id'], how='inner')
        merged_utts = pd.merge(merged_utts, times, on='sample_id', how='inner')
        logging.info("Utterance data merged successfully.")

        # Save unblinded utterances
        unblinded_utts = output_dir / 'unblindUtteranceData.xlsx'
        merged_utts.to_excel(unblinded_utts, index=False)
        logging.info(f"Unblinded utterances saved to {unblinded_utts}.")

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
        logging.info("Blinded utterance data prepared successfully.")

        # Save blinded utterances
        blind_utts_file = output_dir / 'blindUtteranceData.xlsx'
        blind_utts.to_excel(blind_utts_file, index=False)
        logging.info(f"Blinded utterances saved to {blind_utts_file}.")

        # Aggregate by sample - first filter utterance data.
        utts = utts.drop(columns=['utterance_id', 'speaker', 'utterance', 'comment']).drop_duplicates(keep='first')
        logging.info("Utterance data loaded and preprocessed successfully.")
    
        # Load sample CU data.
        CUbySample = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*CUCoding_BySample.xlsx')])
        
        # Sum word counts.
        WCs = WCs.groupby(['sample_id']).agg(wordCount=('word_count', 'sum'))
        logging.info("Word count data aggregated successfully.")

        merged_samples = utts.copy()
        merged_samples = pd.merge(merged_samples, CUbySample, on='sample_id', how='inner')
        merged_samples = pd.merge(merged_samples, WCs, on='sample_id', how='inner')
        merged_samples = pd.merge(merged_samples, times, on='sample_id', how='inner')
        logging.info("Sample data merged successfully.")

        # Calculate words per minute
        merged_samples['wpm'] = merged_samples.apply(lambda row: round(row['word_count'] / (row['client_time'] / 60), 2), axis=1)
        logging.info("Words per minute calculated successfully.")

        # Save unblinded summary
        unblinded_sample_path = output_dir / 'unblindSampleData.xlsx'
        merged_samples.to_excel(unblinded_sample_path, index=False)
        logging.info(f"Unblinded summary saved to {unblinded_sample_path}.")

        # Prepare blinded samples
        blind_samples = merged_samples.copy()
        blind_samples = blind_samples.drop(columns=remove_tiers)
        for tier_name in blind_columns:
            tier = tiers[tier_name]
            column_name = tier.name
            if column_name in blind_samples.columns:
                blind_samples[column_name] = blind_samples[column_name].map(blind_codes_output[tier.name])
        logging.info("Blinded utterance data prepared successfully.")

        # Save blinded summary
        blinded_samples_path = output_dir / 'blindSampleData.xlsx'
        blind_samples.to_excel(blinded_samples_path, index=False)
        logging.info(f"Blinded summary saved to {blinded_samples_path}.")

        # Save blind codes separately
        blind_codes_file = output_dir / 'blindCodes.xlsx'
        pd.DataFrame(blind_codes_output).to_excel(blind_codes_file, index=True)
        logging.info(f"Blind codes saved to {blind_codes_file}.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
