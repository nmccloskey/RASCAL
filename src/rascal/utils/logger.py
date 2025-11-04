from pathlib import Path
import json
import logging
from datetime import datetime


# ---------------------------------------------------------------------
# Global Logger Configuration
# ---------------------------------------------------------------------

logger = logging.getLogger("RASCALLogger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Stream to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------
# Logger Lifecycle Functions
# ---------------------------------------------------------------------

def initialize_logger(start_time: datetime, out_dir: Path) -> Path:
    """
    Initialize file-based logging for a RASCAL run.

    Creates a timestamped log file within the run's output directory
    and records the initialization event.

    Parameters
    ----------
    start_time : datetime
        Timestamp marking the start of the run.
    out_dir : Path
        Output directory for this RASCAL run.

    Returns
    -------
    Path
        Full path to the created log file.
    """
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = start_time.strftime("%y%m%d_%H%M")
    log_path = log_dir / f"rascal_{timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"=== RASCAL run initialized at {start_time.isoformat()} ===")
    logger.info(f"Log file created at: {log_path}")
    return log_path


def record_run_metadata(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: dict,
    start_time: datetime,
    end_time: datetime,
) -> Path:
    """
    Record structured metadata describing the RASCAL run, including
    recursive listings of all files and subdirectories in input/output.

    Parameters
    ----------
    input_dir : Path
        Directory containing the input data.
    output_dir : Path
        Directory containing this run's outputs.
    config_path : Path
        Path to the configuration file used.
    config : dict
        result from load_config(args.config)
    start_time, end_time : datetime
        Datetime objects marking the start and end of the run.

    Returns
    -------
    Path
        Path to the saved metadata JSON file.
    """
    def list_dir_structure(base: Path) -> dict:
        """Recursively list all files and subdirectories from a base path."""
        structure = {"path": str(base.resolve()), "folders": [], "files": []}
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if p.is_dir():
                structure["folders"].append(str(rel))
            else:
                structure["files"].append(str(rel))
        return structure

    runtime_seconds = round((end_time - start_time).total_seconds(), 2)

    metadata = {
        "header": f"RASCAL run {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "timestamp": end_time.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "paths": {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "config_file": str(config_path.resolve()),
        },
        "configuration": config,
        "directory_snapshot": {
            "input_contents": list_dir_structure(input_dir),
            "output_contents": list_dir_structure(output_dir),
        },
    }

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    meta_path = log_dir / f"rascal_{start_time.strftime('%y%m%d_%H%M')}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Run metadata saved at {meta_path}")
    return meta_path


def terminate_logger(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: dict,
    start_time: datetime,
):
    """
    Finalize logging for a RASCAL run.

    Records completion time and metadata, reports total runtime,
    and gracefully closes all file handlers.

    Parameters
    ----------
    input_dir : Path
        Input directory used for this run.
    output_dir : Path
        Output directory containing generated files.
    config_path : Path
        Path to the YAML configuration used.
    config : dict
        result from load_config(args.config)
    start_time : datetime
        Timestamp recorded when the run began.
    """
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    logger.info(f"=== RASCAL run completed at {end_time.isoformat()} ===")
    logger.info(f"Total runtime: {elapsed:.2f} seconds")

    # Write structured metadata
    record_run_metadata(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        start_time=start_time,
        end_time=end_time,
    )

    # Close all file handlers cleanly
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
