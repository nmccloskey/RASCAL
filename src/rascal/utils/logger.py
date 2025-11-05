from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
import json

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
logger = logging.getLogger("RASCALLogger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Early-log buffer and root directory
_early_logs: list[tuple[str, str]] = []
_root_dir: Path = Path.cwd().resolve()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Safely return path relative to project root (or absolute fallback)."""
    try:
        return str(path.resolve().relative_to(_root_dir))
    except Exception:
        return str(path.resolve())


def early_log(level: str, message: str):
    """Queue a log message before the file logger is initialized."""
    _early_logs.append((level, message))
    print(f"[{level.upper()}] {message}")  # still visible on console


def flush_early_logs():
    """Replay buffered early logs into the main logger."""
    for level, msg in _early_logs:
        getattr(logger, level.lower(), logger.info)(msg)
    _early_logs.clear()


# ---------------------------------------------------------------------
# Initialization and termination
# ---------------------------------------------------------------------
def initialize_logger(start_time: datetime, out_dir: Path):
    """
    Initialize file-based logging for a RASCAL run.
    """
    global _root_dir
    _root_dir = Path.cwd().resolve()

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = start_time.strftime("%y%m%d_%H%M")
    log_path = (log_dir / f"rascal_{timestamp}.log").resolve()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"=== RASCAL run initialized at {start_time.isoformat()} ===")
    logger.info(f"Working directory: {_root_dir}")
    logger.info(f"Log file created: {_rel(log_path)}")

    flush_early_logs()
    return log_path


def record_run_metadata(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: dict,
    start_time: datetime,
    end_time: datetime,
) -> Path:
    """Record structured metadata including configuration and directory snapshots."""
    runtime_seconds = round((end_time - start_time).total_seconds(), 2)

    def list_dir_structure(base: Path) -> dict:
        files, folders = [], []
        for p in sorted(base.rglob("*")):
            rel = _rel(p)
            (folders if p.is_dir() else files).append(rel)
        return {"base": _rel(base), "folders": folders, "files": files}

    metadata = {
        "header": f"RASCAL run {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "timestamp": end_time.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "root_directory": str(_root_dir),
        "paths": {
            "input_dir": _rel(input_dir),
            "output_dir": _rel(output_dir),
            "config_file": _rel(config_path),
        },
        "configuration": config,
        "directory_snapshot": {
            "input_contents": list_dir_structure(input_dir),
            "output_contents": list_dir_structure(output_dir),
        },
    }

    meta_path = (output_dir / "logs" /
                 f"rascal_{start_time.strftime('%y%m%d_%H%M')}_metadata.json").resolve()
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Run metadata saved at {_rel(meta_path)}")
    return meta_path


def terminate_logger(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: dict,
    start_time: datetime,
):
    """Finalize logging: record metadata and close file handlers."""
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    logger.info(f"=== RASCAL run completed at {end_time.isoformat()} ===")
    logger.info(f"Total runtime: {elapsed:.2f} seconds")

    record_run_metadata(input_dir, output_dir, config_path, config, start_time, end_time)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
