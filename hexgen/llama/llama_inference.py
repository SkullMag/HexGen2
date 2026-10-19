"""Added 2026-10-19: explicit entry point for the FABRIC reproduction harness.

Use --help for supported options. Historical prototype launch options are not
silently translated into a different topology or model configuration.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hexgen2.run import main

if __name__ == "__main__":
    main()
