import unittest
from types import SimpleNamespace
from unittest.mock import patch
from hexgen2.nccl_runtime import runtime_version

class RuntimeVersionTest(unittest.TestCase):
    def test_native_api_is_authoritative_and_failures_are_checked(self):
        def get_version(output):output._obj.value=22605;return 0
        with patch('hexgen2.nccl_runtime.ctypes.CDLL',return_value=SimpleNamespace(ncclGetVersion=get_version)):
            self.assertEqual(runtime_version(),(2,26,5))
        def failed(output):return 3
        with patch('hexgen2.nccl_runtime.ctypes.CDLL',return_value=SimpleNamespace(ncclGetVersion=failed)):
            with self.assertRaises(RuntimeError):runtime_version()
