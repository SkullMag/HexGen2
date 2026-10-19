"""Added 2026-10-19: lossless replacement for the missing artifact module.

The historical filename does not imply 2-bit quantization. No quantization is used.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hexgen2.transport import recv_data, send_data

__all__ = ["send_data", "recv_data"]
