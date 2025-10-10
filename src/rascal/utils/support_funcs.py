import re
import logging
import pandas as pd
from pathlib import Path

def as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

