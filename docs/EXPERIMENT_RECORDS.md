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
