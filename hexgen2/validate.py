"""Compare split inference to full-context HF logits/tokens for the same weights."""
import argparse
import json

import torch
from transformers import LlamaForCausalLM

from .model import Engine


def validate(model_dir, engine, device, dtype):
    implementation = Engine(model_dir, engine, device, dtype)
    reference = LlamaForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                               use_safetensors=True, torch_dtype=torch.float32,
                                               attn_implementation="eager").eval()
    rows = []
    for length, count in [(1, 1), (4, 4), (7, 3)]:
        prompt = list(range(1, length + 1))
        first, cache, logits = implementation.prefill(prompt, count)
        tokens = implementation.decode(first, cache, length, count)
        expected = []
        ids = torch.tensor([prompt], dtype=torch.long)
        with torch.inference_mode():
            reference_logits = reference(ids).logits[:, -1]
            for _ in range(count):
                token = reference(ids).logits[:, -1].argmax(-1)
                expected.append(int(token.item()))
                ids = torch.cat((ids, token[:, None]), dim=1)
        error = float((logits.float().cpu() - reference_logits).abs().max())
        matched = tokens == expected
        rows.append({"prompt_tokens": length, "max_new_tokens": count, "tokens": tokens,
                     "reference_tokens": expected, "tokens_match": matched, "max_logit_error": error})
        if not matched or not torch.allclose(logits.float().cpu(), reference_logits, atol=0.03, rtol=0.03):
            raise AssertionError(json.dumps(rows))
    return {"engine": engine, "device": device, "dtype": dtype, "status": "passed",
            "model_identity": implementation.identity, "checks": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engine", choices=["native", "hf-reference"], default="native")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(validate(args.model_dir, args.engine, args.device, args.dtype), indent=2))
