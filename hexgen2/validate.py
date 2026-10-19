"""Compare split inference to full-context HF logits/tokens for the same weights."""
import argparse
import json

import torch
from transformers import LlamaForCausalLM

from .model import Engine


def validate(model_dir, engine, device, dtype, fp32_diagnostic=False):
    implementation = Engine(model_dir, engine, device, dtype)
    reference = LlamaForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                               use_safetensors=True, torch_dtype=implementation.dtype,
                                               attn_implementation="eager").to(implementation.device).eval()
    high_precision = None
    if fp32_diagnostic:
        high_precision = LlamaForCausalLM.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, attn_implementation="eager").eval()
    rows = []
    for length, count in [(1, 1), (4, 4), (7, 3)]:
        prompt = list(range(1, length + 1))
        first, cache, logits = implementation.prefill(prompt, count)
        tokens = implementation.decode(first, cache, length, count)
        expected = []
        ids = torch.tensor([prompt], dtype=torch.long, device=implementation.device)
        with torch.inference_mode():
            reference_logits = reference(ids).logits[:, -1]
            for _ in range(count):
                token = reference(ids).logits[:, -1].argmax(-1)
                expected.append(int(token.item()))
                ids = torch.cat((ids, token[:, None]), dim=1)
        reference_logits = reference_logits.float().cpu()
        error = float((logits.float().cpu() - reference_logits).abs().max())
        matched = tokens == expected
        rows.append({"prompt_tokens": length, "max_new_tokens": count, "tokens": tokens,
                     "reference_tokens": expected, "tokens_match": matched, "max_logit_error": error})
        if high_precision is not None:
            with torch.inference_mode():
                fp32_logits = high_precision(torch.tensor([prompt], dtype=torch.long)).logits[:, -1]
            rows[-1]["fp32_diagnostic_max_logit_error"] = float((logits.float().cpu() - fp32_logits).abs().max())
        if not matched or not torch.allclose(logits.float().cpu(), reference_logits, atol=0.03, rtol=0.03):
            raise AssertionError(json.dumps(rows))
    return {"engine": engine, "device": device, "dtype": dtype, "status": "passed",
            "reference": {"engine": "huggingface-eager", "device": str(implementation.device), "dtype": dtype},
            "fp32_comparison": "diagnostic only; acceptance uses matching dtype/device" if fp32_diagnostic else None,
            "model_identity": implementation.identity, "checks": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engine", choices=["native", "hf-reference"], default="native")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--fp32-diagnostic", action="store_true", help="Also retain CPU FP32 drift without using it as the FP16 acceptance check")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(validate(args.model_dir, args.engine, args.device, args.dtype, args.fp32_diagnostic), indent=2))
