import random
import pylangacq
from tqdm import tqdm
from rascal.utils.logger import logger, _rel


def read_cha_files(input_dir, shuffle=False):
    """
    Read CHAT (.cha) files from the given input directory and return
    a dict of {filename: pylangacq.Reader} objects.
    """
    cha_files = list(input_dir.rglob("*.cha"))

    if shuffle:
        logger.info("Shuffling the list of .cha files.")
        random.shuffle(cha_files)

    logger.info(f"Reading .cha files from directory: {_rel(input_dir)}")
    chats = {}

    for cha in tqdm(cha_files, desc="Reading .cha files..."):
        try:
            chat_data = pylangacq.read_chat(str(cha))
            chats[cha.name] = chat_data
        except Exception as e:
            logger.error(f"Failed to read {_rel(cha)}: {e}")

    logger.info(f"Successfully read {len(chats)} .cha files from {_rel(input_dir)}.")
    return chats
