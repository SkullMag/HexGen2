"""Verify captured bundles and summarize serial-pilot requests without hiding failures."""

import argparse
import json
from pathlib import Path
import statistics

from .record import digest


def verify(folder):
    sums = folder / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("Incomplete bundle: missing SHA256SUMS in " + str(folder))
    expected = set()
    for line in sums.read_text().splitlines():
        sha, name = line.split("  ", 1)
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Unsafe bundle path")
        if not path.is_file() or digest(path) != sha:
            raise ValueError("Bundle checksum failed: " + name)
        expected.add(name)
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p != sums}
    if expected != actual:
        raise ValueError("Bundle files differ from checksum inventory")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def summarize(run):
    nodes = sorted(p for p in run.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    if not nodes:
        raise ValueError("No recorded node bundles")
    manifests, requests, events = [], [], []
    for node in nodes:
        verify(node)
        manifests.append(json.loads((node / "manifest.json").read_text()))
        requests.extend(jsonl(node / "runtime/requests.jsonl"))
        for path in (node / "runtime").glob("rank-*-events.jsonl"):
            events.extend(jsonl(path))
    records = [row for row in requests if not row["warmup"]]
    starts = {e["id"] for e in events if e.get("event") == "request_start" and not e["id"].startswith("__warmup_")}
    workloads = list(run.glob("*/runtime/rank-0-workload.json"))
    offered = [r for p in workloads for r in json.loads(p.read_text()) if not r.get("warmup", False)]
    successes = [r for r in records if r["status"] == "success"]
    completed_ranks = sorted(e["rank"] for e in events if e.get("event") == "complete")
    expected_ids = {r["id"] for r in offered}
    success_ids = {r["id"] for r in successes}
    timed = [r for r in successes if not r.get("profiled", False)]
    return {
        "kind": "serial-correctness-pilot", "source": str(run),
        "integrity": "passed", "node_statuses": {m["node"]: m["status"] for m in manifests},
        "classifications": sorted({m["classification"] for m in manifests}),
        "complete": all(m["status"] == "success" for m in manifests) and completed_ranks == [0, 1]
                    and bool(offered) and success_ids == expected_ids and len(successes) == len(offered),
        "offered": len(offered), "started": len(starts & expected_ids), "successful": len(successes),
        "profiled_successful": len(successes) - len(timed), "timing_samples": len(timed),
        "unsuccessful_or_incomplete": len(expected_ids - success_ids),
        "warmup_records": sum(r["warmup"] for r in requests),
        "correctness_failures": sum(r["status"] == "correctness_failure" for r in records),
        "failures": [e for e in events if e.get("event") == "failure"],
        "roundtrip_ms": {"median": statistics.median([r["roundtrip_ms"] for r in timed]),
                         "max": max(r["roundtrip_ms"] for r in timed)} if timed else None,
        "client_ttft": None, "serving_throughput": None,
        "scope": "TP=1, PP=1 per phase, batch=1, serial fixed-length greedy decoding; no serving/SLO claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
