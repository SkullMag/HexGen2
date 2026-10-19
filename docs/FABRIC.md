# Initial FABRIC deployment plan

Status on 2026-10-19: source import and integrity verification only. A working
FABRIC deployment, GPU inference, and paper-result reproduction remain unverified.

## Confirmed artifact gaps

These observations come from the downloaded ICLR supplement:

| Observation | Implication / next action |
| --- | --- |
| Several `hexgen/llama/llama_disaggregate_*.py` files import `kv_cache_sendrecv_int2`, which is absent from the archive. | Resolve the intended implementation and record any adaptation before validating KV-cache transfer. |
| Several launchers and README instructions refer to `llama_inference.py`, which is absent from the archive. | Trace the available entry points and make an explicit, reviewed launcher correction. |
| The `Makefile` declares `hexgen-head` phony but attaches its recipe dependencies to `hexgen` again. | Validate or repair the head-node build target. |
| The build pins PyTorch 2.0.1 and FlashAttention 2.0.8, but the Python requirements and cloned FlashAttention source are otherwise unpinned. | Resolve a compatible environment and lock all installed versions and external commits. |
| The archive's `.gitmodules` names DistServe, but its source is absent. | Pin the backend commit if that path is used; this import has no DistServe gitlink or recoverable submodule commit. |
| The supplement includes a CPython 3.8 Linux Megatron helper binary while the README requests Python 3.11+. | Rebuild native components for the chosen environment; do not rely on the bundled binary. |

The scheduler's modeled throughput is distinct from measured serving throughput.
Passing a scheduler smoke test will establish only that the scheduling code runs.

## Staged work

1. **Scheduler baseline.** Pin a Python environment and dependencies. Capture the
   exact command, machine configuration, software versions, seeds, stdout/stderr,
   exit status, timing, and scheduler output. Check Python and NumPy random-state
   handling before claiming repeatability.
2. **Runtime baseline.** Resolve missing entry points and KV-cache dependencies.
   Record every source change. Validate one model and one explicit prefill/decode
   configuration before adding the coordinator or multiple replicas.
3. **FABRIC placement.** Query current inventory, then record sites, physical
   hosts, VM resources, GPU models/memory, and network paths. Confirm model-memory
   fit for each replica and attention-kernel support for the chosen GPUs.
   Availability must be checked again before a reservation.
4. **Communication checks.** Measure latency, bandwidth, and collective/KV-transfer
   behavior on the actual placement. Save commands, full outputs, interface/MTU
   settings, and topology; do not substitute advertised link rates.
5. **Serving smoke test.** Run a small request trace and verify successful outputs,
   actual output-token counts, prefill/decode placement, and transfer behavior.
   Capture client and worker logs together.
6. **Experiment sweeps.** Fix trace, arrivals, seeds, warm-up, duration, timeout,
   and repetitions. Vary documented factors and preserve unsuccessful runs.

Use [EXPERIMENT_RECORDS.md](EXPERIMENT_RECORDS.md) for every stage. Identify each
result as modeled, measured, or an adapted comparison. Document departures from
the paper's settings before interpreting differences.
