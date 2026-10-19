# Supported FABRIC pilot

## Scope and repairs

The supported entry point is `python -m hexgen2.run`, also exposed through
`hexgen/llama/llama_inference.py`. Historical experiment scripts are retained as
reference and are not claimed to work with their old CLI options or hardcoded
30B/60-layer layouts.

| Original problem | Reproduction implementation |
| --- | --- |
| Missing `kv_cache_sendrecv_int2.py`; supplied int8 helper loses min/max information and decodes before completion | A lossless, self-describing, blocking tensor protocol; both historical module names use the same protocol. The `int2` name does not indicate quantization. |
| Missing `llama_inference.py`; character-based prompt length and fixed cache shapes | Explicit entry point, actual tokenizer/token IDs, checkpoint-derived layer/head/cache shapes and exact requested output count |
| Some prototypes disable loading weights | Mandatory local safetensors checkpoint, strict native state loading, peer weight/config/trace identity handshake |
| Hardcoded coordinator/network paths and RDMA HCA names | Direct two-rank pilot, explicit FABRIC data interface/address, NCCL over TCP, raw link measurements |
| Python/PyTorch/FlashAttention incompatibilities | Pinned reproduction dependency list; PyTorch 2.5 FSDP import repair; native FlashAttention 2.7.4.post1; explicitly unfused RMSNorm/dense/MLP |
| `hexgen-head` Makefile target is empty | Corrected head-only target; new pilot does not require OCF, Megatron initialization or DistServe |
| Scheduler only seeds NumPy and can exit successfully with no solution | Explicit Python/NumPy seed, structured modeled result and nonzero no-solution status |

This is a **serial correctness pilot**, with one prefill GPU and one decode GPU,
TP=1 and PP=1 per phase. It supports Llama architectures without RoPE scaling,
including GQA. Validate any new checkpoint numerically before use. Larger models
that do not fit per GPU, TP/PP across a phase, concurrency, scheduler-to-runtime
placement, OCF serving, and paper workload/SLO reproduction remain separate work.
There is no CPU attention fallback in native mode.

## Controller and resource plan

Use a separate controller environment with `fabrictestbed-extensions==2.0.9` and
an existing FABRIC configuration. Credentials stay outside Git and run bundles.
`configs/fabric-a30-pilot.json` is an **unreserved candidate**, not proof of access
or current availability: CERN `cern-w3` for prefill and `cern-w6` for decode,
one A30 and 8-vCPU/64-GiB/100-GB VM each, separate physical hosts, one L2Bridge.

Read current capacity first:

```bash
CONTROLLER_PYTHON -m hexgen2.fabric plan \
  --fabric-rc /path/to/fabric_rc --plan configs/fabric-a30-pilot.json \
  --state artifacts/fabric/provision.json
```

After reviewing the exact placement, allocation is an explicit separate action:

```bash
CONTROLLER_PYTHON -m hexgen2.fabric create \
  --fabric-rc /path/to/fabric_rc --plan configs/fabric-a30-pilot.json \
  --state artifacts/fabric/provision.json --name UNIQUE_SLICE_NAME --lease-hours 4
```

The controller rechecks capacity, preserves the slice ID before waiting, refuses
to overwrite an existing allocation, configures the data addresses, and exports
actual site/host/VM/component/interface/lease metadata in `allocation.json`.
`status` refreshes the actual allocation. Partial allocations are preserved for
inspection. The tool never automatically deletes, renews or modifies other slices.

## VM preparation

Clone the exact tested repository commit on both fresh Ubuntu 22.04 x86_64 nodes.
Save controller provisioning records and the actual allocation JSON on each node.
Run `bash scripts/fabric/setup.sh` on each node. This installs system packages and
may request a reboot with exit 42 if required for the GPU driver; it does not
reboot automatically. Run it again after that dedicated VM is rebooted.

The script installs PyTorch 2.5.1/CUDA 12.4 and builds FlashAttention from commit
`5231d95fe13733fb534c01895f7ea88c6a6c7793` (2.7.4.post1). Alternatively set
`HEXGEN2_WHEEL_DIR` to compatible reviewed wheels plus `SHA256SUMS`; their hashes
are checked and retained. Extra fused dense/norm kernels are not needed in this
TP=1 pilot. Exact installed package/OS/kernel/driver versions and setup outputs
are saved under `artifacts/setup/` and copied into recorded run bundles.

Setup ends with native numerical checks against full-context Hugging Face logits
and tokens for the identical tiny checkpoint. **Do not advance if they fail.**
`models/tiny` is a correctness fixture, not a pretrained language model or a
performance workload. For pretrained work, use a pinned, locally downloaded
safetensors checkpoint and tokenizer, and retain its immutable repository revision
and checksums. Model access/redistribution terms still apply.

## Network and launch

From the actual allocation, identify each guest's data interface. Bring up the
recorded private addresses (the controller does this), confirm reachability, then
run these concurrently before inference, each with a fresh output directory:

```bash
# Prefill node (server exits after 90 seconds):
bash scripts/fabric/network.sh server 10.92.10.3 artifacts/network-pilot-001
# Decode node:
bash scripts/fabric/network.sh client 10.92.10.2 artifacts/network-pilot-001
```

Wait for both network collectors to finish. They preserve ping, forward/reverse
iperf results, addresses/routes, interface counters, clock information and exit
status. The selected data interface must match that exported in the allocation.
The pilot uses NIC_Basic/TCP; it does not assume RDMA is available.

Set these on **both** nodes, selecting `NODE_RANK=0` for prefill and `1` for decode:

```bash
export RUN_ID=pilot-001 NODE_RANK=0
export MASTER_ADDR=10.92.10.2
export DATA_INTERFACE=ACTUAL_GUEST_DATA_INTERFACE
export PLACEMENT="$PWD/artifacts/fabric/allocation.json"
export NETWORK_DIR="$PWD/artifacts/network-pilot-001"
export MODEL_DIR="$PWD/models/tiny"
export TRACE="$PWD/configs/tiny-trace.jsonl"
bash scripts/fabric/run_node.sh
```

Use the same run ID, checkpoint bytes and trace on both nodes; retain each node's
`runs/<run-id>/<role>/` bundle. The recorded NCCL logs, process logs and token/KV
stage events identify ranks and requests. Client-visible streaming TTFT is
unavailable in this serial pilot and is never inferred from server timestamps.

Collect both node bundles into `runs/<run-id>/` on the controller without merging
or overwriting their role directories. Verify and summarize:

```bash
python -m hexgen2.summarize runs/pilot-001
```

A summary is complete only if both ranks finish, node commands succeed, all offered
measured requests succeed, and all captured files match their checksum inventories.
The colocated token comparison is outside each measured transfer round trip.
Instrumented pilot timings are not throughput/SLO performance claims.

## CPU scheduler

The artifact's machine configuration is a modeled example, not FABRIC inventory:

```bash
.venv/bin/python -m hexgen2.record \
  --run-id scheduler-001 --node local --classification scheduler-modeled \
  --placement configs/local-reference-placement.json \
  --input benchmark/scheduling_algorithm_documents/machine_amounts.json --timeout 240 -- \
  .venv/bin/python -u -m hexgen2.scheduler --model-size llama-70b \
  --machine_config_path benchmark/scheduling_algorithm_documents/machine_amounts.json \
  --batch_size 4 --inter_bw 0.3 --num_clusters 4 --log_interval 20 \
  --niter 1 --seq_in 512 --seq_out 128 --seed 123
```

The bundle includes stdout/stderr and `runtime/scheduler-result.json`. Retain all
search parameters and seeds. The original cost formulas/units require calibration
before using their outputs to make measured FABRIC performance claims.
