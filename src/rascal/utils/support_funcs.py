import yaml
import logging
from pathlib import Path

def as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def load_config(config_file):
    """Load configuration settings from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def find_utt_files(input_dir, output_dir):
    logging.info("Searching for *Utterances*.xlsx files")
    utterance_files = list(Path(input_dir).rglob("*Utterances*.xlsx")) + \
        list(Path(output_dir).rglob("*Utterances*.xlsx"))
    logging.info(f"Found {len(utterance_files)} utterance file(s)")
    return utterance_files