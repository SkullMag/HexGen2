# FABRIC native pilot — 19 October 2026

**Outcome:** native HexGen-2 prefill, lossless cross-node KV transfer, and decode
worked on two FABRIC A30s. All successful requests matched colocated inference
exactly. Both VMs were released; FABRIC confirmed the slice `Dead` at
**17:16:10 UTC**. The deletion request was sent at 17:15:38 UTC.

The runtime commit was `32ed184618f6971541365faecb095f2089edd4e5`, with clean
working trees on both guests. Later commits add evidence, documentation, and a
profiler analysis helper; the native runtime is unchanged. The public
[release assets](https://github.com/SkullMag/HexGen2/releases/tag/fabric-pilot-20261019)
contain the original source snapshots and all attempts. Machine-readable results:
[fabric-pilot.json](../evidence/2026-10-19/fabric-pilot.json),
[allocation](../evidence/2026-10-19/fabric-allocation.json),
[release receipt](../evidence/2026-10-19/resource-release.json).

## Configuration actually tested

| Item | Actual value |
| --- | --- |
| Slice | `hexgen2-pilot-20261019` / `04489373-4521-4a9d-93fd-f24583689fbb` |
| Prefill | CERN `cern-w3.fabric-testbed.net`, one A30, 8 vCPU / 64 GiB RAM / 100 GB disk |
| Decode | CERN `cern-w6.fabric-testbed.net`, one A30, same VM resources; distinct physical host |
| Network | L2Bridge / NIC_Basic, `enp7s0`, `10.92.10.2` ↔ `10.92.10.3`; NCCL TCP, IB disabled |
| Software | Ubuntu 22.04, Python 3.10, PyTorch 2.5.1+cu124, Transformers 4.36.2, FlashAttention 2.7.4.post1 |
| Runtime | Imported native `GPTLMHeadModel` / attention / cache; explicit unfused Torch norm, dense and MLP |
| Parallelism | One GPU per phase; TP=1, PP=1 per phase; batch=1; serial requests |
| Generation | FP16, greedy, fixed requested lengths, seed 123; EOS does not shorten runs |
| Pretrained checkpoint | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Weight SHA-256 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| Model identity | `5af1228c692f6d503157d678697cd6f12a751a1e0bdcfa54fb52972618dfc409` (all selected weights/config/tokenizer/source metadata) |

The downloader verified upstream LFS hashes. Both guests had identical model
identities, trace identities, and runtime settings. Strict loading rejects missing
or extra native checkpoint parameters. Native logits and greedy tokens were
checked against full-context Hugging Face eager inference at matching dtype and
device on each GPU before cross-node runs.

## Completed attempts

| Run ID | Measured requests | Output tokens | Outcome |
| --- | ---: | ---: | --- |
| `native-fixture-20261019-001` | 3 | 8 | Passed; seeded synthetic GQA checkpoint |
| `native-pretrained-20261019-r1` | 5 | 81 | Passed; first measured request profiled on both ranks |
| `native-pretrained-20261019-r2` | 5 | 81 | Passed; unprofiled |
| `native-pretrained-20261019-r3` | 5 | 81 | Passed; unprofiled |
| `native-missing-peer-20261019-001` | 0 submitted; 3 planned | 0 | Intentional rendezvous timeout, exit 124; no surviving child processes |

Each successful attempt also retained one warm-up request. All 18 measured
successful requests and all four warm-ups had exact requested token counts and
identical transferred-cache versus colocated tokens. The three pretrained
repetitions produced identical token sequences for every request. All node bundles
passed their checksum inventories, with both ranks completing on successful runs.
The missing-peer attempt is explicitly incomplete for inference and is never
counted as a successful serving attempt.

Profiler CUDA events show **154 FlashAttention kernel launches on decode** and
**198 on prefill** for the profiled request. The prefill trace includes the
colocated correctness computation outside the measured transfer round trip.
Kernel names and trace identities are retained in the JSON report.

The separate CPU/reference tests and seeded modeled scheduler results are also
retained. All **8 local tests pass**, including the new narrowly scoped profiler
repair test. The tested runtime commit passed Linux CI. A local sandbox attempt
failed because OpenMP shared memory was unavailable; the unrestricted rerun and
both logs are preserved. This was a local test-environment failure.

## Anomalies retained and investigated

1. **Precision mismatch in the original validator.** Comparing FP16 native logits
   to a CPU FP32 reference failed on the pretrained BOS input, with maximum
   absolute drift `0.1270275`, despite matching greedy tokens. Hugging Face FP16
   versus its own FP32 reference showed that same largest drift. Native versus
   Hugging Face FP16 passed at the original `atol=0.03, rtol=0.03`, with maxima
   `0.015625`, `0.033203125`, and `0.02734375` across the three numerical cases
   and matching token sequences. Acceptance now compares matching dtype/device;
   optional FP32 drift remains diagnostic. Original failed checks, errors, exit
   codes, and the three-way diagnostic were preserved rather than overwritten.
2. **Malformed PyTorch profiler export.** Each original compressed trace contains
   28 missing values for `Process Group Description`. Raw bytes and their hashes
   remain unchanged. `python -m hexgen2.profile` replaces only those missing
   values with JSON `null` in a separate normalized copy and rejects unrelated
   JSON errors. Both original and normalized traces, repair counts and hashes
   are published. This affects trace parsing, not recorded generated tokens.
3. **Cold multi-token synthetic request.** The fixture's `short` request took
   approximately 555.6 ms round trip, including 545.2 ms on decode. Its first
   warm-up generated only one token and did not exercise the decode loop. The
   data are retained; the precise cause of the first-use delay was not isolated.
   Future timing studies should warm the actual decode path and relevant shapes.
4. **Network and sampling variation.** TCP baseline received bandwidth was about
   15.59 Gb/s forward and 16.45 Gb/s reverse, with 1,280 and 3,163 sender
   retransmissions respectively. Decode-to-prefill ping averaged 0.082 ms with
   0/10 packets lost. These are a short pre-run baseline, not an established
   bandwidth ceiling or a causal explanation of request timing. Nominal 1-second
   telemetry had gaps up to 1.681 seconds; no native-run GPU query or process
   collection failures were recorded. Raw samples/counters are available.

Clock records show NTP synchronization and approximately 26 ms root delay plus
1–2 ms dispersion. The small reported local clock corrections do not establish
submillisecond cross-host timestamp accuracy. Durations use each process's own
monotonic clock; cross-host monotonic values are never subtracted.

## Reproduce and inspect

Follow [FABRIC.md](FABRIC.md) for allocation, exact checkpoint download, environment
setup, network probes, and launch commands. Use the tested runtime commit when
reproducing these runs. The release includes the exact compatible CPython 3.10
Linux FlashAttention wheel and its SHA-256; setup can verify/install that wheel
or rebuild from the recorded source commit.

Download the release assets and verify `SHA256SUMS` before extracting. Extract
`hexgen2-evidence-prefill.tgz` and `hexgen2-evidence-decode.tgz` into **separate**
node directories: each contains its own `artifacts/`, model provenance, setup
history, and `runs/<run-id>/<role>/`. To reaggregate, copy only those role
subdirectories under the repository's `runs/<run-id>/`, preserving their bytes:

```bash
.venv/bin/python -m hexgen2.summarize runs/native-pretrained-20261019-r2
```

Each bundle includes a source archive, clean Git revision, dependency and OS
inventories, exact commands, model/input hashes, actual placement, network/clock
probes, stdout/stderr, NCCL logs, per-rank/request/token timestamps, generated
and colocated token IDs, GPU/CPU/process/network telemetry, and `SHA256SUMS`.
The original failed numerical validations live in each node's `artifacts/setup/`
and were also copied into successful run bundles. The controller archive includes
the exact launch commands, collection analysis, source/wheel identities, and
resource-release receipt. No pretrained weight duplication is necessary: retrieve
the immutable checkpoint and verify its retained hashes. The tiny fixture weights
and pretrained tokenizer/config/source metadata are included.

From the current repository, normalize a raw profile into a fresh output file:

```bash
.venv/bin/python -m hexgen2.profile \
  runs/native-pretrained-20261019-r1/decode/runtime/rank-1-trace.json.gz \
  --output decode-trace.normalized.json.gz
```

Profiled requests are excluded from the summarizer's timing statistics; treat the
entire profiled run as diagnostic when making comparisons. A single five-request
trace and two unprofiled repetitions are insufficient for tail-latency or SLO
claims. There is no streaming client, arrival process, concurrency sweep, automatic
scheduler-to-runtime placement, multi-GPU phase, heterogeneous comparison, or
30B/70B paper-scale reproduction in this pilot. Server-side observations are
retained, but client TTFT and serving throughput are explicitly unavailable.
