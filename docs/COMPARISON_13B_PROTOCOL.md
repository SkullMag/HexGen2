# Four-GPU 13B comparison protocol

This is a scaled FABRIC adaptation inspired by Figure 6, comparing colocated
HexGen with disaggregated HexGen-2. It is not a numerical reproduction of the
paper's 70B model, RunPod fleet, optimized serving runtime, or full placement search.
DistServe is excluded.

## Frozen system and workload

- Model: `openlm-research/open_llama_13b`, revision
  `b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8`, FP16, greedy decoding, exact output
  counts, no early EOS termination and no prefix caching. All source weights and
  losslessly converted per-layer weights are hashed.
- Fleet: exactly two A30s, one VM each on CERN-w3 and CERN-w6, and two T4s in one
  VM on UTAH-w5. Both methods use the same reservation and private FABRIC network.
  The allocated VM capacities, GPU UUIDs, placement, clocks and network paths are
  captured, including the requested versus allocated disk difference.
- Both complete model groups use TP=1, PP=2, equal layer counts. The colocated
  method has two complete serving replicas. The disaggregated method assigns
  one complete group to prefill and the other to decode, transferring each
  stage's initialized FP16 KV prefix to its corresponding decode stage.
- Each method imports native attention/MLP/block code from its own pinned artifact.
  A common bounded HTTP/queue/microbatch adapter replaces the unavailable complete
  coordinator. Unfused RMSNorm/dense/MLP settings are shared. A common import
  alias resolves the renamed, unused FSDP training dependency under PyTorch 2.5. Ampere uses
  FlashAttention 2; Turing uses the recorded Turing extension. This adapter does
  not implement the paper's paged attention or arbitrary continuous admission.
  Model computation is serial in a dedicated service loop, independent of the
  HTTP response threads; every reply is explicitly flushed. The released NATS
  async handler is not used. Earlier HexGen curves from the blocking NATS stack
  are excluded. A queued T4 delivery diagnostic must pass before measurement;
  retain generation-end, completion-ready, reply-start and reply-flush times.
  Device durations use a single host's monotonic clock. Cross-host callback
  latency is not inferred by subtracting unrelated monotonic timestamps.
  Cache transport has two credits: one active decode plus one prefetched prefix.
  Compute-stream synchronization allows reception during decoding without waiting
  for a subsequent unfulfilled receive. Qualify concurrency, shutdown, numerical
  correctness and memory separately before accepting this configuration.
- Azure Conversation trace commit
  `790921015d50dd6aae7f7e47f39ba0e235ad6b08`, CSV SHA-256
  `2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8`.
  Preserve actual paired input/output lengths. Keep positive lengths, input at
  most 1536, output at most 512 and total at most 2048. Retain exclusion counts.
  Azure provides no prompt text: pack public LMSYS token segments deterministically,
  with one BOS, and record source row/segment identifiers and bank/tokenizer pins.
- Heavy input is >512 tokens; heavy output is >128. Figure columns are HPLD,
  HPHD, LPHD, LPLD and Online. One heterogeneous fleet means one figure row.
- SHA256(Azure row index) modulo 10 selects calibration rows (0) versus held-out
  rows (1–9). Measured seeds are 20261020, 20261021 and 20261022. Independent
  calibration anchors isolate memory, batching, phase and transfer costs.

## Qualification and calibration gates

Before measurement, require both GPU attention backends to pass numerical checks
against float32 SDPA, native tiny-model outputs to match an independent eager HF
FP16 reference, and the 13B greedy outputs/logits to match the same independent
reference under a recorded tolerance. Check both cache-transfer directions,
initialized-prefix size, dtype, layer ownership, rank/source identity and shutdown.
Profiling/diagnostic outputs are separate from measurements.

Profile both role orientations on calibration inputs and freeze a choice before
held-out measurements. The four-card restriction leaves one prefill-to-decode
edge; a calibrated max-flow calculation can select its orientation, but does not
exercise the paper's larger graph partition/refinement search. Record every
candidate, its inputs and measured/modelled capacities separately.

## Measurement and analysis

Use three repetitions per method per column: 30 measured cells. Warmups are
recorded separately. Offline requests maintain a bounded backlog on each active
endpoint; baseline replicas have dedicated lanes so slower replicas cannot starve
faster ones. Online preserves Azure length correlation and arrival-gap order,
scaled to a common 75% of the smaller calibrated capacity. Freeze baseline
capacity-weighted routing before online measurements and reuse arrivals/seeds
across methods. Never select a rate from held-out results.

The primary load-generator throughput is successful completed output tokens
received within a fixed 600-second measurement window divided by 600. Requests
crossing the boundary remain in the drain records and do not silently enlarge the
window. Also report offered, successful, failed, cancelled and incomplete counts,
full-drain throughput, client latency and submission lag. These HTTP endpoints
return completed responses; they do not measure streaming client TTFT. Native
per-token/phase/cache events support separate server-side analysis.

Plot five zero-based panels, two bars with three repetition points and variability.
Keep every repetition, including anomalies. Investigate GPU utilization, memory,
clocks, temperatures, power, network counters, queue/credit wait, native phase
cost, KV bytes/transfer timing, endpoint routing, client lag, failures and clock
uncertainty. Cross-host UTC differences are not used as exact phase latency.

Each recorder captures source commit plus exact working bytes, model/config/trace
hashes, launch commands, rank mapping, environment, dependency freeze, kernel/driver/
CUDA/NCCL versions, raw stdout/stderr/NCCL logs, per-second process/GPU/network
telemetry, per-request IDs/inputs/output tokens and failure finalization. Copy and
verify all evidence before deleting only this experiment's slice; retain a final
release receipt. Publish the report, plotting code, CSV, figures and reviewed raw
bundles in durable versioned locations.
