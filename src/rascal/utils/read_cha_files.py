import random
import logging
import pylangacq
from tqdm import tqdm
from pathlib import Path


def read_cha_files(input_dir, shuffle=False):
    """
    Reads CHAT (.cha) files from the specified input directory and extracts data.

    Parameters:
    - input_dir (str): The directory containing input `.cha` files.
    - shuffle (bool): Whether to shuffle the list of files before reading. Default is False.

    Returns:
    - dict: A dictionary where keys are filenames and values are CHAT data objects.
    """
    
    # Retrieve all .cha files in the input directory and subdirectories
    cha_files = list(Path(input_dir).rglob("*.cha"))

    if shuffle:
        logging.info("Shuffling the list of .cha files.")
        random.shuffle(cha_files)

    chats = {}

    logging.info(f'Reading .cha files from directory: {input_dir}')
    for cha in tqdm(cha_files, desc="Reading .cha files..."):
        try:
            chat_data = pylangacq.read_chat(str(cha))
            filename = cha.name
            chats[filename] = chat_data
        except Exception as e:
            logging.error(f"Failed to read {cha}: {e}")

    logging.info(f"Successfully read {len(chats)} .cha files from {input_dir}.")
    
    return chats
