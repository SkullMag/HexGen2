"""Read PyTorch traces while preserving and documenting a known exporter defect."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def read_profile(path):
    original = Path(path).read_bytes()
    raw = gzip.decompress(original) if str(path).endswith(".gz") else original
    text = raw.decode("utf-8")
    repairs = 0
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        # Torch 2.5 can emit a missing NCCL process-group description value.
        # Only this exact field is repaired; all other malformed JSON still fails.
        text, repairs = re.subn(r'("Process Group Description"\s*:)\s*(?=[,}])', r'\1 null', text)
        if not repairs:
            raise
        document = json.loads(text)
    events = document["traceEvents"]
    kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
    flash = [name for name in kernels if "flash" in name.lower()]
    return document, {
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "original_json_valid": repairs == 0,
        "missing_process_group_descriptions_replaced_with_null": repairs,
        "kernel_count": len(kernels), "flash_kernel_count": len(flash),
        "flash_kernel_names": sorted(set(flash)),
        "policy": "Preserve raw trace; repair only missing Process Group Description values in a separate copy",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New normalized JSON or JSON.gz file")
    args = parser.parse_args()
    document, report = read_profile(args.source)
    encoded = (json.dumps(document, allow_nan=False) + "\n").encode()
    if str(args.output).endswith(".gz"):
        encoded = gzip.compress(encoded, mtime=0)
    with args.output.open("xb") as stream:
        stream.write(encoded)
    report["normalized_sha256"] = hashlib.sha256(encoded).hexdigest()
    print(json.dumps(report, indent=2))
