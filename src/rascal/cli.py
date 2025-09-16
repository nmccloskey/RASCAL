#!/usr/bin/env python3
import argparse
from .main import main as main_core

def main():
    parser = argparse.ArgumentParser(description="RASCAL CLI")
    parser.add_argument('step', type=str, help="Specify the step or function(s) (e.g., '1' or 'abc').")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()
    main_core(args)
