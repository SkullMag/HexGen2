"""Modified 2026-10-19: replace the artifact's non-invertible int8 transport.

Use the same lossless, shape/dtype-aware protocol as kv_cache_sendrecv_int2.
Both sender and receiver must run this version; the old wire format is incompatible.
"""
from kv_cache_sendrecv_int2 import recv_data, send_data

__all__ = ["send_data", "recv_data"]
