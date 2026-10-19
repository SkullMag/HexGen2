#!/usr/bin/env python3
"""Verify the imported ICLR baseline; requires only Python's standard library."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Also check a downloaded source ZIP")
    parser.add_argument("--original", action="store_true", help="Compare strictly to the sanitized initial import")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/upstream.json").read_text())
    adaptations_path = root / "provenance/adaptations.json"
    adaptations = {} if args.original or not adaptations_path.exists() else {
        item["path"]: item for item in json.loads(adaptations_path.read_text())["modified_upstream_files"]}
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append("missing: " + entry["path"])
        else:
            adaptation = adaptations.get(entry["path"])
            expected = entry["imported_sha256"]
            if adaptation:
                if adaptation["baseline_sha256"] != expected:
                    failures.append("adaptation baseline mismatch: " + entry["path"])
                expected = adaptation["adapted_sha256"]
            if sha256(path) != expected:
                failures.append("changed: " + entry["path"])
    if args.archive:
        if not args.archive.is_file():
            failures.append("archive does not exist: " + str(args.archive))
        elif sha256(args.archive) != manifest["archive"]["sha256"]:
            failures.append("archive checksum mismatch")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} imported upstream files.")
    if adaptations:
        print(f"Including {len(adaptations)} explicitly documented runtime adaptations.")
    if args.archive:
        print("Verified original archive SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
