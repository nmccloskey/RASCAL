import yaml
from pathlib import Path
import pandas as pd
from rascal.utils.logger import logger, get_root, early_log, _rel
import argparse


# -------------------------------------------------------------
# Omnibus mappings
# -------------------------------------------------------------
OMNIBUS_MAP = {
    "1": ["1a"],
    "4": ["4a", "4b"],
    "7": ["7a", "7b"],
    "10": ["10a", "10b"],
}

COMMAND_MAP = {
    "1a": "transcripts select",
    "3a": "transcripts evaluate",
    "3b": "transcripts reselect",
    "4a": "transcripts make",
    "4b": "cus make",
    "6a": "cus evaluate",
    "6b": "cus reselect",
    "7a": "cus analyze",
    "7b": "words make",
    "9a": "words evaluate",
    "9b": "words reselect",
    "10a": "cus summarize",
    "10b": "corelex analyze",
}

# -------------------------------------------------------------
# CLI setup utilities
# -------------------------------------------------------------
def build_arg_parser():
    """Construct and return the argument parser used by both main.py and cli.py."""
    parser = argparse.ArgumentParser(
        description=(
            "RASCAL command-line interface.\n\n"
            "Examples:\n"
            "  rascal 3b\n"
            "  rascal transcripts reselect\n"
            "  rascal 4\n"
            "  rascal 4a,4b\n"
            "  rascal utterances make, cus make, timesheets make\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="+",
        help="Command(s) to run (comma-separated or space-separated)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )

    # ---- Help text for expansions ----
    help_lines = ["\nAvailable Commands:\n"]
    for short, long in COMMAND_MAP.items():
        help_lines.append(f"  {short:<4}  →  {long}")
    help_lines.append("\nOmnibus Commands:\n")
    for omni, subs in OMNIBUS_MAP.items():
        expansions = [f"{s} ({COMMAND_MAP[s]})" for s in subs]
        help_lines.append(f"  {omni:<4}  →  {', '.join(expansions)}")
    parser.epilog = "\n".join(help_lines)

    return parser

def project_path(*parts) -> Path:
    """Return an absolute path anchored to the project root."""
    return (Path.cwd().resolve() / Path(*parts)).resolve()

def as_path(p: str | Path) -> Path:
    """
    Normalize a path to be relative to the current working directory (project root).

    If the target lies outside the working directory, returns its resolved absolute path.
    This ensures all internal references stay project-root–relative without failures.
    """
    try:
        p = Path(p).expanduser()
        cwd = Path.cwd().resolve()
        resolved = (cwd / p).resolve() if not p.is_absolute() else p.resolve()

        # Try to make relative if possible
        try:
            rel = resolved.relative_to(cwd)
            logger.debug(f"Resolved relative path: {rel}")
            return rel
        except ValueError:
            # Not under cwd — fine, just return absolute
            logger.debug(f"Resolved absolute path (outside cwd): {resolved}")
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
    config_file = find_config_file(get_root(), config_file)
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        early_log("info", f"Loaded configuration from {_rel(config_file)}")
        return config
    except FileNotFoundError:
        early_log("error", f"Configuration file not found: {_rel(config_file)}")
        raise
    except yaml.YAMLError as e:
        early_log("error", f"YAML parsing error in {_rel(config_file)}: {e}")
        raise
    except Exception as e:
        early_log("error", f"Unexpected error loading {_rel(config_file)}: {e}")
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


def find_corresponding_file(match_tiers=None, directory=Path.cwd(), search_base="", search_ext=".xlsx"):
    """
    Find file(s) in `directory` matching all tier labels and a given base pattern.

    Behavior
    --------
    • Recursively searches `directory` for files containing both `search_base`
      and all stringified tier labels from `match_tiers`.
    • Returns:
        - a single Path if exactly one match,
        - a list[Path] if multiple,
        - None if none found.
    • Logs warnings when multiple or no matches are found.

    Parameters
    ----------
    match_tiers : list[str] | None
        Tier labels (usually from `tier.match(filename)`), e.g. ["AC", "PreTx"].
        None or empty entries are ignored.
    directory : Path or str
        Root directory to search recursively.
    search_base : str
        Core pattern string (e.g., "cu_coding_by_utterance").
    search_ext : str, default ".xlsx"
        File extension to filter (including dot).

    Returns
    -------
    Path | list[Path] | None
        Matching file(s), or None if no matches.

    Notes
    -----
    - Converts all tier labels to lowercase strings for matching.
    - Logs relative paths via `_rel()` for readability.
    - Catches and logs filesystem errors without interrupting the pipeline.
    """
    directory = Path(directory)
    match_tiers = [str(mt).lower() for mt in (match_tiers or []) if mt]

    try:
        files = list(directory.rglob(f"*{search_base}*{search_ext}"))
        if not files:
            logger.warning(f"No files found for pattern '*{search_base}*{search_ext}' in {_rel(directory)}.")
            return None

        matching_files = []
        for f in files:
            fname = f.name.lower()
            if all(mt in fname for mt in match_tiers):
                matching_files.append(f)

        if not matching_files:
            logger.warning(
                f"No files matched tier values {match_tiers} for base '{search_base}' in {_rel(directory)}."
            )
            return None
        elif len(matching_files) == 1:
            match = matching_files[0]
            logger.info(f"Matched file for {search_base}: {_rel(match)}")
            return match
        else:
            logger.warning(
                f"Multiple ({len(matching_files)}) files matched '{search_base}' and {match_tiers}; returning list."
            )
            for f in matching_files:
                logger.debug(f"  - {_rel(f)}")
            return matching_files

    except Exception as e:
        logger.error(f"Error while searching for '{search_base}' in {_rel(directory)}: {e}")
        return None
