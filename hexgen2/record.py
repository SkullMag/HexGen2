"""Run a bounded command and retain source, inputs, diagnostics and telemetry."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = {"GLOO_SOCKET_IFNAME", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MPLCONFIGDIR",
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "TORCH_DISTRIBUTED_DEBUG",
            "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def model_files(model_dir):
    return sorted(p for p in Path(model_dir).iterdir() if p.is_file() and (
        p.suffix in {".safetensors", ".json"} or p.name == "tokenizer.model"))


def model_identity(model_dir):
    hashes = {p.name: digest(p) for p in model_files(model_dir)}
    if not any(name.endswith(".safetensors") for name in hashes):
        raise ValueError("No safetensors checkpoint in model directory")
    return {"files": hashes, "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command_result(command, timeout=20, cwd=ROOT):
    start = time.monotonic_ns()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"command": command, "utc": utc(), "returncode": result.returncode,
                "duration_ms": (time.monotonic_ns() - start) / 1e6,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "utc": utc(), "unavailable": type(error).__name__, "message": str(error)}


def capture_source(folder):
    revision = command_result(["git", "rev-parse", "HEAD"])
    status = command_result(["git", "status", "--porcelain"])
    write(folder / "git.json", {"revision": revision, "status": status})
    with (folder / "working-tree.patch").open("wb") as stream:
        subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, stdout=stream, check=True)
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    hashes = {}
    with tarfile.open(folder / "working-source.tar.gz", "w:gz") as archive:
        for name in sorted(set(filter(None, paths))):
            path = ROOT / name
            if path.is_symlink():
                raise ValueError("Source snapshot refuses symlinks: " + name)
            if path.is_file():
                hashes[name] = digest(path)
                archive.add(path, arcname=name, recursive=False)
    write(folder / "source-files.json", hashes)
    return {"commit": revision.get("stdout", "").strip(), "dirty": bool(status.get("stdout", "")),
            "snapshot_sha256": digest(folder / "working-source.tar.gz"), "files": len(hashes)}


def telemetry(path, stop, child, interval):
    with path.open("w") as stream:
        while not stop.is_set():
            started = time.monotonic()
            row = {"utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                   "cpu_percent": psutil.cpu_percent(), "memory": psutil.virtual_memory()._asdict(),
                   "network": {k: v._asdict() for k, v in psutil.net_io_counters(pernic=True).items()},
                   "disk_io": (psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None),
                   "processes": []}
            try:
                parent = psutil.Process(child.pid)
                for process in [parent] + parent.children(recursive=True):
                    try:
                        row["processes"].append({"pid": process.pid, "cpu_seconds": process.cpu_times()._asdict(),
                                                 "memory": process.memory_info()._asdict(),
                                                 "io": process.io_counters()._asdict() if hasattr(process, "io_counters") else None})
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.Error, OSError) as error:
                row["process_collection_error"] = {"type": type(error).__name__, "message": str(error)}
            row["gpu"] = command_result(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"], timeout=5)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(max(0, interval - (time.monotonic() - started)))


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def collect_telemetry(folder, stop, child, interval):
    try:
        telemetry(folder / "telemetry.jsonl", stop, child, interval)
    except BaseException as error:
        write(folder / "telemetry-error.json", {"type": type(error).__name__, "message": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--classification", choices=["scheduler-modeled", "cpu-reference", "native-pilot", "native-comparison"], required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[], help="Additional immutable input/config file")
    parser.add_argument("--network-dir", type=Path, help="Raw pre-run network measurements")
    parser.add_argument("--root", type=Path, default=ROOT / "runs")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.interval <= 0:
        parser.error("Provide a command and positive timeout/interval")
    for name in (args.run_id, args.node):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or name in {".", ".."}:
            parser.error("Run and node IDs must be safe path components")
    if re.search(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|--(?:token|password|api-key)(?:=|$)", " ".join(command)):
        parser.error("Keep credentials in the environment, not recorded command arguments")
    folder = args.root.resolve() / args.run_id / args.node
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("source", "environment", "inputs", "logs", "runtime"):
        (folder / name).mkdir()
    manifest = {"schema_version": 1, "run_id": args.run_id, "node": args.node,
                "classification": args.classification, "start_utc": utc(), "status": "collecting",
                "argv": command, "cwd": str(ROOT), "hostname": platform.node(),
                "platform": platform.platform(), "python": sys.version,
                "timeout_seconds": args.timeout, "telemetry_interval_seconds": args.interval,
                "environment": {k: v for k, v in os.environ.items() if k in ENV_KEYS or k.startswith("NCCL_")}}
    write(folder / "manifest.json", manifest)
    child, worker, stop = None, None, threading.Event()
    result = 1
    try:
        placement = json.loads(args.placement.read_text())
        write(folder / "inputs/placement.json", placement)
        if args.trace:
            (folder / "inputs/trace.jsonl").write_bytes(args.trace.read_bytes())
            manifest["trace_sha256"] = digest(args.trace)
        manifest["additional_inputs"] = []
        for index, path in enumerate(args.input):
            target = folder / "inputs" / f"extra-{index}-{path.name}"
            shutil.copyfile(path, target)
            manifest["additional_inputs"].append({"path": str(path), "sha256": digest(target)})
        if args.network_dir:
            shutil.copytree(args.network_dir, folder / "network")
        if args.model_dir:
            manifest["model"] = model_identity(args.model_dir)
        manifest["source"] = capture_source(folder / "source")
        diagnostics = {
            "pip-freeze": [sys.executable, "-m", "pip", "freeze", "--all"],
            "pip-check": [sys.executable, "-m", "pip", "check"],
            "gpu": ["nvidia-smi", "-q"], "kernel-messages": ["sudo", "-n", "dmesg", "--time-format", "iso"], "gpu-topology": ["nvidia-smi", "topo", "-m"],
            "cpu": ["lscpu", "-J"], "pci": ["lspci", "-nn"],
            "links": ["ip", "-j", "-s", "link"], "addresses": ["ip", "-j", "addr"],
            "routes": ["ip", "-j", "route"], "clock": ["timedatectl", "show"],
            "clock-offset": ["chronyc", "tracking"], "kernel": ["uname", "-a"],
            "disk": ["df", "-h"], "limits": ["sh", "-c", "ulimit -a"],
            "torch": [sys.executable, "-c", "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version() if torch.cuda.is_available() else None,'cuda_available':torch.cuda.is_available()}))"],
        }
        for name, cmd in diagnostics.items():
            write(folder / f"environment/{name}.json", command_result(cmd))
        if (ROOT / "artifacts/setup").is_dir():
            shutil.copytree(ROOT / "artifacts/setup", folder / "environment/setup")
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(key, "1")
        env.setdefault("PYTHONHASHSEED", "123")
        env["MPLCONFIGDIR"] = str(folder / "environment/matplotlib-cache")
        env.update(HEXGEN2_RUN_ID=args.run_id, HEXGEN2_RUN_DIR=str(folder / "runtime"))
        if "NCCL_DEBUG" in env:
            env["NCCL_DEBUG_FILE"] = str(folder / "logs/nccl-%h-%p.log")
            manifest["environment"]["NCCL_DEBUG_FILE"] = env["NCCL_DEBUG_FILE"]
        manifest["environment"].update({k: v for k, v in env.items() if k in ENV_KEYS})
        manifest.update(status="running", command_start_utc=utc(), command_start_monotonic_ns=time.monotonic_ns())
        write(folder / "manifest.json", manifest)
        with (folder / "logs/stdout.log").open("wb") as stdout, (folder / "logs/stderr.log").open("wb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            manifest["child_pid"] = child.pid
            write(folder / "manifest.json", manifest)
            worker = threading.Thread(target=collect_telemetry, args=(folder, stop, child, args.interval), daemon=True)
            worker.start()
            try:
                result = child.wait(timeout=args.timeout)
                manifest["status"] = "success" if result == 0 else "failed"
            except subprocess.TimeoutExpired:
                terminate(child)
                result, manifest["status"] = 124, "timed_out"
    except KeyboardInterrupt:
        if child:
            terminate(child)
        result, manifest["status"] = 130, "interrupted"
    except BaseException as error:
        if child:
            terminate(child)
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        stop.set()
        if worker:
            worker.join(timeout=8)
        manifest["telemetry_error"] = json.loads((folder / "telemetry-error.json").read_text()) if (folder / "telemetry-error.json").exists() else None
        samples = [json.loads(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()] if (folder / "telemetry.jsonl").is_file() else []
        gaps = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]
        manifest["telemetry"] = {"samples": len(samples), "max_gap_seconds": max(gaps, default=None),
                                 "process_unavailable_samples": sum("process_collection_error" in row for row in samples),
                                 "gpu_unavailable_samples": sum(row["gpu"].get("returncode") != 0 for row in samples)}
        if worker and (worker.is_alive() or manifest["telemetry_error"]):
            manifest["status"] = "capture_incomplete"
            if result == 0:
                result = 2
        if child and not samples and result == 0:
            manifest["status"], result = "capture_incomplete", 2
        for name, cmd in {"gpu-final": ["nvidia-smi", "-q"], "links-final": ["ip", "-j", "-s", "link"], "clock-offset-final": ["chronyc", "tracking"], "kernel-messages-final": ["sudo", "-n", "dmesg", "--time-format", "iso"]}.items():
            write(folder / f"environment/{name}.json", command_result(cmd))
        manifest.update(exit_code=result, end_utc=utc(), end_monotonic_ns=time.monotonic_ns(),
                        child_returncode=child.returncode if child else None)
        write(folder / "manifest.json", manifest)
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (folder / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(folder)}\n" for p in files))
        print(json.dumps({"bundle": str(folder), "status": manifest["status"], "exit_code": result}))
    return result


if __name__ == "__main__":
    sys.exit(main())
