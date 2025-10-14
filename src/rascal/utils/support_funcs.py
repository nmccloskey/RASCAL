import yaml
from pathlib import Path

def as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def load_config(config_file):
    """Load configuration settings from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
