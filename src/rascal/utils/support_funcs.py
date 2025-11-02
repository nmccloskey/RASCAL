import yaml
import logging
from pathlib import Path
import pandas as pd


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
        logging.debug(f"Resolved path: {resolved}")
        return resolved
    except Exception as e:
        logging.error(f"Failed to resolve path {p}: {e}")
        raise


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
    config_file = as_path(config_file)
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error in {config_file}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading {config_file}: {e}")
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
    logging.info("Searching for *transcript_tables*.xlsx files")

    try:
        transcript_tables = list(input_dir.rglob("*transcript_tables*.xlsx")) + \
                            list(output_dir.rglob("*transcript_tables*.xlsx"))
        logging.info(f"Found {len(transcript_tables)} transcript table file(s)")
        return transcript_tables
    except Exception as e:
        logging.error(f"Error while searching for transcript tables: {e}")
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
        logging.error(f"Transcript table not found: {path}")
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
            logging.info(f"Loaded sample data from {path}")
            return sample_df

        elif type == "utterance":
            if utt_df is None:
                raise ValueError("Utterance sheet not found in transcript table.")
            logging.info(f"Loaded utterance data from {path}")
            return utt_df

        elif type == "joined":
            if sample_df is None or utt_df is None:
                raise ValueError("Both sheets required for joined type are missing.")
            joined = sample_df.merge(utt_df, on="sample_id", how="inner")
            logging.info(f"Loaded joined transcript data from {path}")
            return joined

        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'sample', 'utterance', or 'joined'.")

    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        raise
