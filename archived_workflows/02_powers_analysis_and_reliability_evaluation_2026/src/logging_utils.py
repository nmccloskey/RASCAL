from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "powers_prep",
    log_dir: Path | str = "logs",
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """
    Create a timestamped logger for the POWERS preparation mini-program.

    Writes logs to:
        logs/powers_prep_YYYYMMDD_HHMMSS.log

    Reuses an existing logger if already configured, preventing duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if also_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Logger initialized.")
    logger.info("Log file: %s", log_path)

    return logger