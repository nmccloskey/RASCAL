#!/usr/bin/env python3
import sys
import os

# Add src to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rascal.main import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RASCAL CLI")
    parser.add_argument('step', type=str, help="Which step(s) to run (e.g., 'abc', '3', or '1gk')")
    args = parser.parse_args()

    main(args)
