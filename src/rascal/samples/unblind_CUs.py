import os
import logging
import pandas as pd
from pathlib import Path


def unblind_CUs(tiers, input_dir, output_dir, test=False):
    """
    Unblinds participant utterances and prepares both an unblinded and blinded summary.

    This function reads in participant, utterance, CU data, word counts, and speaking times,
    merges them into a comprehensive DataFrame, calculates words per minute, and prepares a blinded version.

    Args:
        tiers (list): List of Tier objects defining the columns to be blinded.
        input_dir (str): Directory containing the input files.
        output_dir (str): Directory to save the output files.
        test (bool): If True, the function will return results for testing.

    Returns:
        None or pd.DataFrame: Returns the merged DataFrame if `test` is True.
    """
    try:
        # Specify subfolder and create directory
        output_dir = os.path.join(output_dir, 'Summaries')
        os.makedirs(output_dir, exist_ok=True)
        
        # Read participant data
        try:
            pdata = pd.read_excel(os.path.join(input_dir, 'ParticipantData.xlsx'))
            logging.info("Participant data loaded successfully.")
        except FileNotFoundError:
            pdata = None
            logging.warning("No participant data available.")
        except Exception as e:
            logging.error(f"No participant data provided: {e}")
            pdata = None

        # Read utterance data
        utts = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*_Utterances.xlsx')])

        # Read CU data
        CUbyUtts = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*_CUCoding_ByUtterance.xlsx')])
        CUbyUtts.drop(columns=['site', 'narrative', 'sampleID', 'speaker', 'utterance', 'comment'], inplace=True)
        logging.info("CU utterance data loaded successfully.")

        # Read word count data
        WCs = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*_WordCounting.xlsx')])
        WCs = WCs.loc[:, ['UtteranceID', 'sampleID', 'wordCount', 'WCcom']]

        # Read speaking time data
        times = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*_SpeakingTimes.xlsx')])
        times = times.loc[:, ['sampleID', 'client_time']]
        logging.info("Speaking time data loaded successfully.")

        # Merge datasets
        if pdata is not None:
            merged_utts = pd.merge(pdata, utts, on='participantID', how='inner')
        else:
            merged_utts = utts.copy()
        merged_utts = pd.merge(merged_utts, CUbyUtts, on='UtteranceID', how='inner')
        merged_utts = pd.merge(merged_utts, WCs, on=['UtteranceID', 'sampleID'], how='inner')
        merged_utts = pd.merge(merged_utts, times, on='sampleID', how='inner')
        logging.info("Utterance data merged successfully.")

        # Save unblinded utterances
        unblinded_utts = os.path.join(output_dir, 'unblindUtteranceData.xlsx')
        merged_utts.to_excel(unblinded_utts, index=False)
        logging.info(f"Unblinded utterances saved to {unblinded_utts}.")

        # Prepare blind codes and blinded utterances
        blind_utts = merged_utts.copy()
        blind_codes_output = {}
        for tier_name in ['site', 'test']:
            tier = tiers[tier_name]
            blind_codes = tier.make_blind_codes()
            column_name = tier.name
            if column_name in blind_utts.columns:
                blind_utts[column_name] = blind_utts[column_name].map(blind_codes[tier.name])
                blind_codes_output.update(blind_codes)
        logging.info("Blinded utterance data prepared successfully.")

        # Save blinded utterances
        blind_utts_file = os.path.join(output_dir, 'blindUtteranceData.xlsx')
        blind_utts.to_excel(blind_utts_file, index=False)
        logging.info(f"Blinded utterances saved to {blind_utts_file}.")

        # Aggregate by sample - first filter utterance data.
        utts = utts.drop(columns=['UtteranceID', 'speaker', 'utterance', 'comment']).drop_duplicates(keep='first')
        logging.info("Utterance data loaded and preprocessed successfully.")
    
        # Load sample CU data.
        CUbySample = pd.concat([pd.read_excel(f) for f in Path(input_dir).rglob('*_CUCoding_BySample.xlsx')])
        
        # Sum word counts.
        WCs = WCs.groupby(['sampleID']).agg(wordCount=('wordCount', 'sum'))
        logging.info("Word count data aggregated successfully.")

        merged_samples = pd.merge(pdata, utts, on='participantID', how='inner')
        merged_samples = pd.merge(merged_samples, CUbySample, on='sampleID', how='inner')
        merged_samples = pd.merge(merged_samples, WCs, on='sampleID', how='inner')
        merged_samples = pd.merge(merged_samples, times, on='sampleID', how='inner')
        logging.info("Sample data merged successfully.")

        # Calculate words per minute
        merged_samples['wpm'] = merged_samples.apply(lambda row: round(row['wordCount'] / (row['client_time'] / 60), 2), axis=1)
        logging.info("Words per minute calculated successfully.")

        # Save unblinded summary
        unblinded_sample_path = os.path.join(output_dir, 'unblindSampleData.xlsx')
        merged_samples.to_excel(unblinded_sample_path, index=False)
        logging.info(f"Unblinded summary saved to {unblinded_sample_path}.")

        # Prepare blinded samples
        blind_samples = merged_samples.copy()
        for tier_name in ['site', 'test']:
            tier = tiers[tier_name]
            column_name = tier.name
            if column_name in blind_samples.columns:
                blind_samples[column_name] = blind_samples[column_name].map(blind_codes_output[tier.name])
        logging.info("Blinded utterance data prepared successfully.")

        # Save blinded summary
        blinded_samples_path = os.path.join(output_dir, 'blindSampleData.xlsx')
        blind_samples.to_excel(blinded_samples_path, index=False)
        logging.info(f"Blinded summary saved to {blinded_samples_path}.")

        # Save blind codes separately
        blind_codes_file = os.path.join(output_dir, 'blindCodes.xlsx')
        pd.DataFrame(blind_codes_output).to_excel(blind_codes_file, index=True)
        logging.info(f"Blind codes saved to {blind_codes_file}.")

        if test:
            return blind_samples, blind_utts, merged_samples, merged_utts 
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
