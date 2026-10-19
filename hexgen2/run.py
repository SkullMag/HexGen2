"""Two-rank, serial prefill/decode correctness pilot, not a throughput server."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from .model import Engine
from .transport import recv_packet, send_packet


def stamp():
    return {"utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def workload(path, model_dir, context_limit):
    rows, seen, tokenizer = [], set(), None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row.get("id"), str) or not row["id"] or row["id"].startswith("__warmup_")
                or row["id"] in seen):
            raise ValueError("Every request must have a unique string id")
        if set(row) - {"id", "prompt", "input_ids", "max_new_tokens", "warmup"}:
            raise ValueError("Unsupported workload fields; this pilot does not schedule arrivals or sampling")
        if ("prompt" in row) == ("input_ids" in row) or type(row.get("warmup", False)) is not bool:
            raise ValueError("Specify exactly one of prompt/input_ids and a boolean warmup flag")
        seen.add(row["id"])
        if "input_ids" not in row:
            if not isinstance(row.get("prompt"), str):
                raise ValueError("Each request needs input_ids or a prompt")
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            row["input_ids"] = tokenizer.encode(row["prompt"], add_special_tokens=True)
        ids, count = row["input_ids"], row.get("max_new_tokens")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("input_ids must be a nonempty list of token IDs")
        if type(count) is not int or count < 1 or len(ids) + count > context_limit:
            raise ValueError("Invalid output length or context overflow")
        rows.append(row)
    if not rows:
        raise ValueError("Workload is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=os.environ.get("HEXGEN2_RUN_DIR"))
    parser.add_argument("--engine", choices=["native", "hf-reference"], default="native")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--check-colocated", action="store_true",
                        help="Compare every output to the same engine without KV transfer")
    args = parser.parse_args()
    if args.output is None:
        parser.error("Use the record wrapper or supply --output")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "-1"))
    event_path = args.output / f"rank-{rank}-events.jsonl"
    if event_path.exists():
        raise FileExistsError("Refusing to overwrite rank records; use a fresh run directory")
    append(event_path, {"event": "start", "rank": rank, "pid": os.getpid(), "hostname": socket.gethostname(),
                        "run_id": os.environ.get("HEXGEN2_RUN_ID"), "arguments": vars(args) | {
        "model_dir": str(args.model_dir), "trace": str(args.trace), "output": str(args.output)}, **stamp()})
    try:
        if int(os.environ.get("WORLD_SIZE", "0")) != 2 or args.warmup < 0:
            raise ValueError("Exactly two ranks and a nonnegative warmup count are required")
        if args.engine == "native" and (args.device != "cuda" or args.dtype == "float32"):
            raise ValueError("Native FlashAttention uses CUDA with float16 or bfloat16")
        if args.device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.set_num_threads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=args.timeout))
        model = Engine(args.model_dir, args.engine, args.device, args.dtype)
        rows = workload(args.trace, args.model_dir, model.config.max_position_embeddings)
        if any(max(row["input_ids"]) >= model.config.vocab_size for row in rows):
            raise ValueError("Input token ID exceeds model vocabulary")
        canonical = json.dumps(rows, sort_keys=True).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        local_config = {"trace_sha256": digest, "engine": args.engine, "dtype": args.dtype,
                        "seed": args.seed, "warmup": args.warmup,
                        "model_identity": model.identity}
        hashes = [None, None]
        dist.all_gather_object(hashes, hashlib.sha256(json.dumps(local_config, sort_keys=True).encode()).hexdigest())
        if hashes[0] != hashes[1]:
            raise ValueError("Ranks disagree on workload, model config, or run settings")
        (args.output / f"rank-{rank}-workload.json").write_text(json.dumps(rows, indent=2) + "\n")
        append(event_path, {"event": "ready", "rank": rank, "trace_sha256": digest,
                            "model_identity": model.identity,
                            "gpu": ({"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                                     "local_rank": int(os.environ["LOCAL_RANK"]), "allocated_bytes": torch.cuda.memory_allocated()}
                                    if args.device == "cuda" else None), **stamp()})
        attempts = [dict(rows[0], id=f"__warmup_{i}", warmup=True) for i in range(args.warmup)] + rows
        for index, row in enumerate(attempts):
            dist.barrier()
            model.synchronize()
            started = stamp()
            append(event_path, {"event": "request_start", "id": row["id"], "index": index, **started})
            if rank == 0:
                first, cache, _ = model.prefill(row["input_ids"], row["max_new_tokens"])
                model.synchronize()
                prefill_end = stamp()
                packet_bytes = send_packet({"first": first, **{str(k): v for k, v in cache.items()}},
                                          {"id": row["id"], "prompt_length": len(row["input_ids"])}, dst=1)
                model.synchronize()
                sent = stamp()
                tensors, reply = recv_packet(src=1)
                ended = stamp()
                tokens = tensors["tokens"].cpu().tolist()
                if reply["id"] != row["id"] or len(tokens) != row["max_new_tokens"]:
                    raise ValueError("Request correlation or exact output-token count failed")
                record = {"id": row["id"], "index": index, "warmup": bool(row.get("warmup", False)),
                          "engine": args.engine, "input_tokens": len(row["input_ids"]),
                          "output_tokens": len(tokens), "token_ids": tokens,
                          "start": started, "prefill_end": prefill_end, "send_end": sent, "end": ended,
                          "prefill_ms": (prefill_end["monotonic_ns"] - started["monotonic_ns"]) / 1e6,
                          "send_ms": (sent["monotonic_ns"] - prefill_end["monotonic_ns"]) / 1e6,
                          "roundtrip_ms": (ended["monotonic_ns"] - started["monotonic_ns"]) / 1e6,
                          "kv_packet_bytes": packet_bytes, "decode": reply, "status": "success",
                          "client_first_token_time": None,
                          "timing_note": "Serial pilot; no streaming client TTFT or throughput-serving claim"}
                if args.check_colocated:
                    local_first, local_cache, _ = model.prefill(row["input_ids"], row["max_new_tokens"])
                    expected = model.decode(local_first, local_cache, len(row["input_ids"]), row["max_new_tokens"])
                    record["colocated_token_ids"] = expected
                    record["colocated_matches"] = expected == tokens
                    if expected != tokens:
                        record["status"] = "correctness_failure"
                append(args.output / "requests.jsonl", record)
                if record["status"] != "success":
                    raise ValueError("Transferred-cache output differs from colocated inference")
                del cache
            else:
                tensors, metadata = recv_packet(src=0)
                model.synchronize()
                received = stamp()
                if metadata != {"id": row["id"], "prompt_length": len(row["input_ids"])}:
                    raise ValueError("Prefill/decode request metadata mismatch")
                token_events = []
                def on_token(token):
                    model.synchronize()
                    token_events.append({"token_id": token, **stamp()})
                tokens = model.decode(tensors["first"], {int(k): v for k, v in tensors.items() if k != "first"},
                                      len(row["input_ids"]), row["max_new_tokens"], on_token)
                done = stamp()
                reply = {"id": row["id"], "receive_end": received, "decode_end": done,
                         "token_events": token_events,
                         "decode_ms": (done["monotonic_ns"] - received["monotonic_ns"]) / 1e6}
                append(event_path, {"event": "decode_complete", **reply})
                send_packet({"tokens": torch.tensor(tokens, dtype=torch.long, device=tensors["first"].device)}, reply, dst=0)
                del tensors
            append(event_path, {"event": "request_complete", "id": row["id"],
                                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                                **stamp()})
        dist.barrier()
        append(event_path, {"event": "complete", "rank": rank, **stamp()})
    except BaseException as error:
        append(event_path, {"event": "failure", "rank": rank, "error_type": type(error).__name__,
                            "message": str(error), "traceback": traceback.format_exc(), **stamp()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
