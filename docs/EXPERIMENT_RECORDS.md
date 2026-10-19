# Experiment records

Give every attempt a unique UTC run ID. Keep its raw outputs, including failures,
under `runs/<run-id>/`. Write a manifest before launching and finish it with the
actual start/end times, exit status, and an outcome note. Never overwrite an
earlier attempt when retrying.

## Minimum run bundle

| Record | Required contents |
| --- | --- |
| `manifest.json` | Run ID, purpose, modeled/measured classification, UTC times, outcome, repository commit, dirty status, exact commands and working directories, environment-lock and artifact references |
| `source/` | Git diff if dirty; source/archive hashes; external dependency commits; container digest or image/build recipe |
| `environment/` | OS/kernel, CPU/RAM/disk, Python package lock, PyTorch/CUDA/compiler/driver versions, attention kernel, NCCL and relevant non-secret environment settings |
| `placement/` | FABRIC site and physical host mapping, VM cores/RAM/disk, GPU model/UUID/memory, topology, replica roles, TP/PP groups, network path/interface/MTU, resource allocation and lease timestamps |
| `network/` | Actual bandwidth/latency and collective/KV-transfer measurements, commands, and outputs |
| `workload/` | Model and tokenizer revisions, dtype/quantization, weight checksums; dataset revision/hash, request IDs/order, input/output lengths, arrival timestamps/process, rate, seeds, warm-up, duration, batching, concurrency, timeout, SLO definition, repetition index |
| `requests.jsonl` | Each offered request's ID, scheduled/submitted time, first-token time, token times where available, completion time, input/output token counts, outcome/error, retry identity, and replica/worker correlation IDs |
| `logs/` | Client, coordinator, scheduler and worker stdout/stderr, rank/PID mapping, exit codes, and bounded system/network diagnostics |
| `telemetry/` | Timestamped GPU utilization/memory/power/temperature, CPU/RAM, network throughput/errors, sampling interval and missing-sample notes |
| `analysis/` | Aggregation commands, formulas, figure data, exclusions with reasons, summary metrics, and anomaly notes tied to request IDs and timestamps |
| `SHA256SUMS` | Relative paths and hashes for bundle payloads, excluding the checksum file itself |

Synchronize host clocks and record clock offsets/uncertainty. Use monotonic clocks
for within-process durations and UTC timestamps for cross-host correlation. Treat
unavailable timestamps as missing, rather than estimating first-token or per-token
latency from total completion time.

## Comparison rules

- Keep offered, successful, failed, cancelled, and timed-out request counts.
  Record retries separately. State the measurement window and denominator for
  throughput and SLO attainment, including how failures are counted.
- Report latency distributions and tail percentiles with sample counts. Report
  TTFT and per-token/inter-token latency only where actually measured.
- Use the same request traces and arrival schedules for paired comparisons.
  Preserve seed values, warm-up boundaries, repetitions, and raw measurements.
- Record complete topology and resource differences between configurations;
  matching GPU names alone does not establish a controlled comparison.
- Separate scheduler predictions, measurements, published reference values, and
  adapted configurations. A smoke test is not reproduction of the paper's plots.

## Storage and publication

`runs/` is ignored by Git to keep credentials, model weights, and large raw data
out of routine commits. Store reviewed bundles in durable storage or GitHub
release assets, and commit their checksums, stable locations, and analysis code.
Do not treat a local-only directory as a published reproducibility artifact.

Retain run settings through an explicit allowlist; do not dump the entire process
environment. Exclude authentication tokens, SSH keys, and FABRIC access tokens.
Respect model/dataset redistribution terms; where data cannot be redistributed,
retain exact revisions, hashes, access instructions, and the limitation.

## Implemented recorder and limits

`python -m hexgen2.record --help` wraps commands with a wall-clock deadline and
creates a fresh `runs/<run-id>/<node>/` bundle. It copies tracked and non-ignored
untracked working source into a tarball, records the base commit/dirty patch and
per-file hashes, snapshots explicit trace/placement/additional inputs and all
local safetensors/config/tokenizer hashes, and captures stdout/stderr plus sampled
system/process/network/GPU state. Setup logs and pre-run network measurements are
copied when provided. NCCL debug files are placed inside each node's bundle.
Dependency freezes, OS/kernel/driver/GPU topology, interfaces/routes, clock and
resource-limit diagnostics retain return codes and unavailable-probe reasons.
Only an allowlist of environment variables is retained; credentials are never
collected from the entire environment.

Timeouts terminate the newly launched process group, preserve its outputs, and
finalize manifest/checksums with exit 124. Interrupts use exit 130. Nonzero child
exits remain failures. Sampler crashes mark the capture incomplete; access-denied
process counters are recorded as unavailable. Do not overwrite or discard these
attempts when retrying. Host crashes/SIGKILL can leave a `running` manifest or
missing checksum file; the verifier treats those bundles as incomplete.

`python -m hexgen2.summarize runs/<run-id>` verifies all node files before computing
serial-pilot statistics. It includes unfinished/failed offered requests in the
outcome counts. Warm-ups are retained and identified explicitly. Per-token events
are server-side observations; no cross-host monotonic timestamps are subtracted.
Clock offsets and their uncertainty must be considered for cross-host UTC traces.

The current pilot is batch=1, serial, fixed-length greedy decoding. There is no
arrival process, online server or concurrency sweep yet; unknown workload options
are rejected. Model/trace/run-setting identities must match across ranks. EOS does
not shorten fixed-length runs. Runtime identity is `native` or `hf-reference` in
every request record. GPU telemetry covers all visible GPUs and records unsupported
queries; CPU reference runs explicitly lack GPU telemetry. Inspect sampler gaps,
clock availability, and every diagnostic's exit code before reporting results.

A captured environment is reconstructable from the retained package versions,
source/weight hashes, wheel hashes, and OS package inventory. Availability of old
FABRIC images/driver packages and gated model artifacts is an external dependency;
record any inability to reacquire them. The bootstrap's image/package tags alone
are not an immutable environment lock; retain each actual run's complete bundle.
