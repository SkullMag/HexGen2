"""Download a public safetensors checkpoint at an immutable Hugging Face commit."""
import argparse
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download
from .record import model_identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use an immutable full commit SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    source = {"repo": args.repo, "revision": args.revision, "public_download": True}
    (args.output / "download-source.json").write_text(json.dumps(source, indent=2) + "\n")
    snapshot_download(args.repo, revision=args.revision, local_dir=args.output, token=False,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "README.md", "LICENSE"])
    # The identity file is outside the checkpoint to avoid a self-referential hash.
    identity = model_identity(args.output)
    args.output.with_suffix(".identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(source | {"sha256": identity["sha256"]}))
