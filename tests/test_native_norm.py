import importlib.util
from pathlib import Path
import unittest

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class NativeNormTest(unittest.TestCase):
    def test_unfused_norm_matches_hf_without_cuda_extensions(self):
        path = Path(__file__).resolve().parents[1] / "hexgen/hexgen_core/modules/rms_norm.py"
        spec = importlib.util.spec_from_file_location("native_norm", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            native = module.TorchRMSNorm(128, dtype=dtype)
            reference = LlamaRMSNorm(128).to(dtype)
            data = torch.randn(1, 7, 128).to(dtype)
            torch.testing.assert_close(native(data), reference(data), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
