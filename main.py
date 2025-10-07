#!/usr/bin/env python3
import sys
from src.main_cli import run

if __name__ == "__main__":
    # Check if the script is being run in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("---")
        print("WARNING: It is highly recommended to run this application in a virtual environment.")
        print("Please create and activate a venv, then reinstall dependencies with 'pip install -r requirements.txt'.")
        print("---\n")

    run()