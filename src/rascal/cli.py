#!/usr/bin/env python3
import argparse
from .main import main as main_core

def main():
    parser = argparse.ArgumentParser(description="RASCAL CLI")
    parser.add_argument('step', type=str, help="Which step(s) to run (e.g., 'abc', '3', or '1gk')")
    args = parser.parse_args()
    main_core(args)
