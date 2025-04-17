import os
import re
import time
import logging
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def reformat(text: str, identifier: str, alternative_participles: list) -> str:
    """
    Prepares a transcription text string for comparison metrics.

    Args:
        text (str): The transcription text to be formatted.
        identifier (str): An identifier for tracking errors.
        alternative_participles (list): List of alternative participles to standardize.

    Returns:
        str: The cleaned and formatted transcription text.
    """
    logging.info("Starting transcription reformatting.")
    
    try:
        text = text.lower().strip('\n').strip(' ')
        text = re.sub(r"(?<=(he|it))'s got", ' has got', text)
        text = ' '.join([contractions.fix(w) for w in text.split(' ')])
        
        for alt in alternative_participles:
            text = re.sub(r'\b{}\b'.format(alt), alt + 'g', text)
        
        text = text.replace('\xa0', '')
        text = re.sub(r'(^|\b)(u|e)+?(h|m|r)+?(\b|$)', '', text)
        text = re.sub(r'(^|\b)h*m+h*m*\b', '', text)
        text = re.sub(r'(^|\b|\b.)x+(\b|$)', '', text)
        
        open_par_count = text.count('(')
        closed_par_count = text.count(')')
        
        if open_par_count == closed_par_count:
            for _ in range(open_par_count):
                text = re.sub(r'\([^\(]*?\)', '', text)
        else:
            logging.warning(f"{identifier} has mismatched number of parentheses.")
        
        text = re.sub(r'\[.+?\]', '', text)
        text = re.sub(r'\*.+?\*', '', text)
        text = re.sub(r'\d+', lambda x: n2w.num2words(int(x.group(0))), text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\bd\b', '', text)
        text = re.sub(r'(\bcl\b)', '', text)
        text = re.sub(r'\s*=+\s*', '=', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\t', '', text)
        
        logging.info("Transcription reformatted successfully.")
        return text.strip(' ')
    
    except Exception as e:
        logging.error(f"An error occurred while reformatting: {e}")
        return ""


def webapp(pID: str, scene_name: str, time_duration: str, transc: str, downloads_path: str):
    """
    Automates interaction with the web application.

    Args:
        pID (str): Participant ID.
        scene_name (str): Name of the scene to select.
        time_duration (str): Duration of the session.
        transc (str): Transcript text.
        downloads_path (str): Path to the downloads folder.
    
    Returns:
        tuple: (num_data, token_data) extracted from the downloaded Excel file.
    """
    logging.info("Starting web automation process.")
    
    try:
        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.implicitly_wait(10)
        logging.info("Chrome WebDriver initialized.")
        
        driver.get('https://rb-cavanaugh.shinyapps.io/coreLexicon/')
        logging.info("Navigated to web application.")
        
        driver.find_element(By.ID, 'glide_next1').click()
        driver.find_element(By.ID, 'name').send_keys(pID)
        
        if scene_name != 'Broken Window':
            dropdown_element = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div/div/div/div[2]/div/div[1]/div/div[2]/div/div/div[1]')
            time.sleep(2)
            dropdown_element.click()
            time.sleep(5)
            
            dropdown = driver.find_element(By.XPATH, f"//*[contains(text(), '{scene_name}')]")
            time.sleep(3)
            dropdown.click()
            time.sleep(5)
            logging.info(f"Selected scene: {scene_name}")
        
        time_box = driver.find_element(By.ID, 'time')
        time_box.clear()
        time_box.send_keys(time_duration)
        
        driver.find_element(By.ID, 'glide_next2').click()
        time.sleep(6)
        
        transc_box = driver.find_element(By.ID, 'transcr')
        transc_box.click()
        transc_box.send_keys(transc)
        time.sleep(2)
        
        driver.find_element(By.ID, 'start').click()
        time.sleep(2)
        
        driver.find_element(By.ID, 'go_to_results').click()
        time.sleep(5)
        logging.info("Navigated to results page.")
        
        data_box = WebDriverWait(driver, 13).until(
            EC.visibility_of_element_located((By.ID, 'downloadData'))
        )
        data_box.click()
        logging.info("Downloaded results data.")
        time.sleep(8)
        
        file_path = os.path.join(downloads_path, '_MC_summary.xlsx')
        
        if not os.path.exists(file_path):
            logging.error("Downloaded file not found.")
            driver.quit()
            return None, None
        
        data = pd.read_excel(file_path, sheet_name=None)
        token_data = data.get('Sheet 2')
        num_data = data.get('Sheet 3')
        
        os.remove(file_path)
        logging.info("Extracted data and deleted downloaded file.")
        
        return num_data, token_data
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None
    
    finally:
        driver.quit()
        logging.info("WebDriver closed.")


def extract_digit(t: str):
    """
    Extracts the first numerical value (integer or float) from a string.

    Args:
        t (str): The text containing a numerical value.

    Returns:
        float: The extracted numerical value or NaN if not found.
    """
    match = re.search(r'\d+\.?\d*', t)
    return float(match.group(0)) if match else np.nan


def get_nums(num_data: pd.DataFrame) -> list:
    """
    Extracts numerical information from a given DataFrame.

    Args:
        num_data (pd.DataFrame): Data containing numerical scores and percentiles.

    Returns:
        list: Extracted numerical values in the order:
              [ncw, ncw_pwa_ptile, ncw_ctrl_ptile, cwpm, cwpm_pwa_ptile, cwpm_ctrl_ptile]
    """
    logging.info("Extracting numerical values from data.")
    
    try:
        ncw = int(extract_digit(num_data.loc[0, 'Score']))
        ncw_pwa_ptile = float(extract_digit(num_data.loc[0, 'Aphasia Percentile']))
        ncw_ctrl_ptile = float(extract_digit(num_data.loc[0, 'Control Percentile']))
        cwpm = float(extract_digit(num_data.loc[1, 'Score']))
        cwpm_pwa_ptile = float(extract_digit(num_data.loc[1, 'Aphasia Percentile']))
        cwpm_ctrl_ptile = float(extract_digit(num_data.loc[1, 'Control Percentile']))
        
        nums = [ncw, ncw_pwa_ptile, ncw_ctrl_ptile, cwpm, cwpm_pwa_ptile, cwpm_ctrl_ptile]
        logging.info("Extraction successful: %s", nums)
        return nums
    
    except Exception as e:
        logging.error(f"Error extracting numerical values: {e}")
        return []


def make_timestamp() -> str:
    """Returns a timestamp in YYMMDD format."""
    return datetime.now().strftime('%y%m%d')

# Define scenes
scenes = {
    'BrokenWindow': 'Broken Window',
    'RefusedUmbrella': 'Refused Umbrella',
    'CatRescue': 'Cat Rescue'
}

# Define base columns
base_columns = [
    'sampleID', 'ncw', 'ncw_pwa_ptile', 'ncw_ctrl_ptile',
    'cwpm', 'cwpm_pwa_ptile', 'cwpm_ctrl_ptile'
]

# Define tokens for each scene
scene_tokens = {
    'BrokenWindow': ["a", "and", "ball", "be", "boy", "break", "go", "he", "in", "it", 
                        "kick", "lamp", "look", "of", "out", "over", "play", "sit", "soccer", 
                        "the", "through", "to", "up", "window"],
    'CatRescue': ["a", "and", "bark", "be", "call", "cat", "climb", "come", "dad",
                    "department", "dog", "down", "father", "fire", "fireman", "get",
                    "girl", "go", "have", "he", "in", "ladder", "little", "not", "out",
                    "she", "so", "stick", "the", "their", "there", "to", "tree", "up", "with"],
    'RefusedUmbrella': ["a", "and", "back", "be", "boy", "do", "get", "go", "have", "he", "home",
                        "i", "in", "it", "little", "mom", "mother", "need", "not", "out", "rain",
                        "say", "school", "she", "so", "start", "take", "that", "the", "then",
                        "to", "umbrella", "walk", "wet", "with", "you"]
}

# Define alternative participles
alternative_participles = [
    'bein', 'breakin', 'goin', 'kickin', 'lookin', 'playin', 'barkin', 'callin', 'climbin', 'comin',
    'gettin', 'havin', 'stickin', 'doin', 'needin', 'rainin', 'sayin', 'startin', 'takin', 'walkin'
]

def run_corelex(input_dir, output_dir):
    """
    Runs CoreLex analysis on utterance data and saves the results.

    Args:
        input_dir (str): The directory where input files may be located.
        output_dir (str): The directory where output files will be saved.
    """
    logging.info("Starting CoreLex processing.")

    token_cols = [f"{scene[:3]}_{w}" for scene, words in scene_tokens.items() for w in words]
    corelexdf = pd.DataFrame(columns=base_columns + token_cols)
    timestamp = make_timestamp()
    downloads_path = os.path.join(Path.home(), "Downloads")

    corelex_dir = os.path.join(output_dir, 'CoreLex')
    os.makedirs(corelex_dir, exist_ok=True)
    logging.info(f"Output directory created: {corelex_dir}")

    utt_data_path = os.path.join(output_dir, 'Summaries', 'unblindUtteranceData.xlsx')
    if not os.path.exists(utt_data_path):
        utt_data_path = os.path.join(input_dir, 'Summaries', 'unblindUtteranceData.xlsx')
        if not os.path.exists(utt_data_path):
            logging.error(f"File not found: {utt_data_path}")
            return

    utt_df = pd.read_excel(utt_data_path)
    utt_df = utt_df[utt_df['narrative'].isin(scenes.keys())]
    utt_df = utt_df[~np.isnan(utt_df['c2CU'])]

    for sample in tqdm(set(utt_df['sampleID'])):
        subdf = utt_df[utt_df['sampleID'] == sample]
        scene_name = scenes[subdf['narrative'].iloc[0]]
        rel_cols = [col for col in corelexdf.columns if col.startswith(scene_name[:3])]

        pID = subdf['participantID'].iloc[0]
        time = str(subdf['client_time'].iloc[0])
        text = ' '.join(subdf['utterance'])
        text = reformat(text, pID, alternative_participles)

        be_forms = re.findall(r'\b(be|is|are|were|been|being|was|am)\b', text)
        text = re.sub(r'\b(be|is|are|were|been|being|was)\b', 'am', text)

        num_data, token_data = None, None

        for attempt in range(5):
            try:
                num_data, token_data = webapp(pID, scene_name, time, text, downloads_path)
                if num_data is not None and token_data is not None:
                    break
                logging.warning(f"Attempt {attempt+1} failed for sample {sample}.")
            except Exception as e:
                logging.error(f"Exception during webapp call on attempt {attempt+1} for sample {sample}: {e}")

        if num_data is None or token_data is None:
            logging.error(f"All attempts failed for sample {sample}. Filling with NaNs.")
            nums = [np.nan] * len(base_columns[1:])
            tokens = [np.nan] * len(rel_cols)
        else:
            nums = get_nums(num_data)
            tokens = [t if t != '-' else np.nan for t in token_data['Token Produced']]

        row_data = [sample] + nums + tokens
        row_idx = len(corelexdf)
        for c, d in zip(base_columns + rel_cols, row_data):
            corelexdf.loc[row_idx, c] = d

        if be_forms:
            corelexdf.loc[row_idx, f"{scene_name[:3]}_be"] = ', '.join(set(be_forms))

    for blind_type in ['unblind', 'blind']:
        sample_data_path = os.path.join(output_dir, 'Summaries', f'{blind_type}SampleData.xlsx')
        if not os.path.exists(sample_data_path):
            sample_data_path = os.path.join(input_dir, 'Summaries', f'{blind_type}SampleData.xlsx')
            if not os.path.exists(sample_data_path):
                logging.warning(f"Missing sample data file: {sample_data_path}")
                continue

        sample_df = pd.read_excel(sample_data_path)
        merged = pd.merge(sample_df, corelexdf, on='sampleID', how='inner')
        output_file = os.path.join(corelex_dir, f'{blind_type}CoreLexData{timestamp}.xlsx')
        merged.to_excel(output_file)
        logging.info(f"Saved: {output_file}")

    logging.info("CoreLex processing complete.")
