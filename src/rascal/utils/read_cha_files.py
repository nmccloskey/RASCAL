import random
import pylangacq
from tqdm import tqdm
from pathlib import Path
from rascal.utils.logger import logger


def read_cha_files(input_dir, shuffle=False):
    """
    Reads CHAT (.cha) files from the specified input directory and extracts data.

    Parameters:
    - input_dir (str): The directory containing input `.cha` files.
    - shuffle (bool): Whether to shuffle the list of files before reading.

    Returns:
    - dict[str, pylangacq.Reader]: keys are filenames, values are CHAT reader objects.
    """
    cha_files = list(Path(input_dir).rglob("*.cha"))

    if shuffle:
        logger.info("Shuffling the list of .cha files.")
        random.shuffle(cha_files)

    chats = {}

    logger.info(f"Reading .cha files from directory: {input_dir}")
    for cha in tqdm(cha_files, desc="Reading .cha files..."):
        try:
            chat_data = pylangacq.read_chat(str(cha))
            filename = cha.name
            chats[filename] = chat_data
        except Exception as e:
            logger.error(f"Failed to read {cha}: {e}")

    logger.info(f"Successfully read {len(chats)} .cha files from {input_dir}.")
    return chats
