import yaml
from pathlib import Path
import pandas as pd
from rascal.utils.logger import logger


def as_path(p: str | Path) -> Path:
    """
    Resolve a string or Path to an absolute, expanded Path object.

    Parameters
    ----------
    p : str or Path
        A file or directory path (may include ~ for user home).

    Returns
    -------
    pathlib.Path
        Resolved absolute Path object.
    """
    try:
        resolved = Path(p).expanduser().resolve()
        logger.debug(f"Resolved path: {resolved}")
        return resolved
    except Exception as e:
        logger.error(f"Failed to resolve path {p}: {e}")
        raise


def find_config_file(base_dir: Path, user_arg: str | None = None) -> Path | None:
    """
    Find a YAML configuration file.
    Priority:
      1. User-specified path via --config
      2. config.yaml in current directory
      3. Any *.yaml file under input/config/
    """
    if user_arg:
        cfg = Path(user_arg)
        if cfg.exists():
            return cfg.resolve()
        else:
            raise FileNotFoundError(f"Specified config not found: {cfg}")

    # Default: search in current working directory
    cwd_cfg = Path("config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg.resolve()

    # Fallback: search recursively under input/config/
    for p in Path("input/config").rglob("*.yaml"):
        return p.resolve()  # first match

    raise FileNotFoundError("No configuration file found. Use --config to specify one.")

def load_config(config_file: str | Path) -> dict:
    """
    Load configuration settings from a YAML file.

    Parameters
    ----------
    config_file : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary loaded from YAML.
    """
    config_file = find_config_file(Path.cwd(), config_file)
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {config_file}: {e}")
        raise


def find_transcript_tables(input_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """
    Locate transcript table Excel files in given directories.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input data.
    output_dir : str or Path
        Directory containing generated transcript tables.

    Returns
    -------
    list of Path
        All matching *transcript_tables*.xlsx files.
    """
    input_dir, output_dir = as_path(input_dir), as_path(output_dir)
    logger.info("Searching for *transcript_tables*.xlsx files")

    try:
        transcript_tables = list(input_dir.rglob("*transcript_tables*.xlsx")) + \
                            list(output_dir.rglob("*transcript_tables*.xlsx"))
        logger.info(f"Found {len(transcript_tables)} transcript table file(s)")
        return transcript_tables
    except Exception as e:
        logger.error(f"Error while searching for transcript tables: {e}")
        raise


def extract_transcript_data(
    transcript_table_path: str | Path,
    type: str = "joined"
) -> pd.DataFrame:
    """
    Load data from a transcript table Excel file.

    Parameters
    ----------
    transcript_table_path : str or Path
        Path to an Excel file produced by `make_transcript_tables`.
    type : {'utterance', 'sample', 'joined'}, default='joined'
        Which dataset to return:
          - 'utterance': utterance-level data
          - 'sample': sample-level metadata
          - 'joined': merged table of both (inner join on 'sample_id')

    Returns
    -------
    pandas.DataFrame
        The requested DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the `type` argument is invalid.
    """
    path = as_path(transcript_table_path)
    if not path.exists():
        logger.error(f"Transcript table not found: {path}")
        raise FileNotFoundError(f"Transcript table not found: {path}")

    try:
        # Read available sheets once to avoid multiple disk I/O operations
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = [s.lower() for s in xls.sheet_names]

        sample_df = pd.read_excel(xls, sheet_name="samples") if "samples" in sheet_names else None
        utt_df = pd.read_excel(xls, sheet_name="utterances") if "utterances" in sheet_names else None
        xls.close()

        if type == "sample":
            if sample_df is None:
                raise ValueError("Sample sheet not found in transcript table.")
            logger.info(f"Loaded sample data from {path}")
            return sample_df

        elif type == "utterance":
            if utt_df is None:
                raise ValueError("Utterance sheet not found in transcript table.")
            logger.info(f"Loaded utterance data from {path}")
            return utt_df

        elif type == "joined":
            if sample_df is None or utt_df is None:
                raise ValueError("Both sheets required for joined type are missing.")
            joined = sample_df.merge(utt_df, on="sample_id", how="inner")
            logger.info(f"Loaded joined transcript data from {path}")
            return joined

        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'sample', 'utterance', or 'joined'.")

    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        raise


def find_corresponding_file(match_tiers=[], directory=Path.cwd(), search_base="", search_ext=".xlsx"):
    files = Path(directory).rglob(f"*{search_base}*{search_ext}")
    matching_files = [f for f in files if all((mt in f for mt in match_tiers))]
    if len(matching_files) == 1:
        return next(matching_files)
    else:
        logger.warning(f"Multiple {len(matching_files)} matching files detected - returning list.")
        return matching_files
