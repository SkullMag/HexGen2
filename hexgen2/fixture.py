"""Create an explicitly synthetic tiny Llama checkpoint for correctness tests."""
import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def create(folder, seed=17):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    LlamaForCausalLM(config).eval().save_pretrained(folder, safe_serialization=True)
    (folder / "fixture.json").write_text(json.dumps({"kind": "synthetic-random-weights",
                                                  "seed": seed, "purpose": "correctness only"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    options = parser.parse_args()
    create(options.output, options.seed)
