#!/usr/bin/env python3
from .main import main as main_core
from rascal.utils.logger import logger
from rascal.utils.auxiliary import build_arg_parser


def main():
    """Entry point for rascal CLI wrapper."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        main_core(args)
    except Exception as e:
        logger.error(f"RASCAL execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
