# HexGen-2 on FABRIC

Reproduction work for **HexGen-2: Disaggregated Generative Inference of LLMs in
Heterogeneous Environment** (Youhe Jiang, Ran Yan, Binhang Yuan; ICLR 2025).
The goal is to run HexGen-2 on FABRIC and preserve the evidence needed to reproduce
results and investigate anomalies.

The original code comes from the [official ICLR supplement](https://proceedings.iclr.cc/paper_files/paper/2025/file/0b941a1e5fbce23fe46b049999d04ed0-Supplemental-Conference.zip).
The [paper page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0b941a1e5fbce23fe46b049999d04ed0-Abstract-Conference.html)
links the paper and artifact. Original documentation is in
[README.upstream.md](README.upstream.md); it contains historical prototype
instructions, not the supported FABRIC procedure.

## Current implementation

- A two-rank pilot separates prefill and decode using the artifact's native
  `GPTLMHeadModel`, attention, MLP and KV-cache implementation. It strictly loads
  actual safetensors weights, transfers only live cache positions without
  quantization, and verifies exact output-token counts and colocated equivalence.
- The native pilot uses FlashAttention 2.7.4.post1, unfused PyTorch RMSNorm/dense/MLP,
  one GPU per phase, TP=1, PP=1, batch=1, fixed-length greedy decoding. These are
  explicit adaptations, not a reconstruction of all published configurations.
- A separate `hf-reference` CPU backend validates the transport and harness.
  Its results are labelled CPU reference, never native HexGen-2 GPU performance.
- The supplied scheduler has an explicit random seed, structured modeled output,
  and nonzero failure status when no feasible plan is found. Its cost estimates
  are not calibrated to FABRIC or automatically applied to the runtime pilot.
- Each recorded attempt retains source, input/weight hashes, dependency versions,
  placement, raw logs, per-rank events, telemetry, failure/timeout state, and checksums.

**Validation:** CPU reference inference, real two-process lossless transfer, exact
output lengths, failure/timeout capture, and bundle integrity have passed locally.
FABRIC provisioning and native CUDA inference remain pending live validation.
See [docs/FABRIC.md](docs/FABRIC.md) and [docs/EXPERIMENT_RECORDS.md](docs/EXPERIMENT_RECORDS.md).

## Local correctness run

Python 3.11 is used for local validation; the GPU setup uses Ubuntu 22.04/Python 3.10.
From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/pip install pip==24.3.1 -r requirements-reproduction.txt
.venv/bin/python -m hexgen2.fixture --output models/tiny
.venv/bin/python -m hexgen2.validate --model-dir models/tiny --engine hf-reference --device cpu --dtype float32
.venv/bin/python -m hexgen2.record \
  --run-id local-reference-001 --node local --classification cpu-reference \
  --placement configs/local-reference-placement.json --model-dir models/tiny \
  --trace configs/tiny-trace.jsonl --timeout 60 -- \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m hexgen2.run --model-dir models/tiny --trace configs/tiny-trace.jsonl \
  --engine hf-reference --device cpu --dtype float32 --check-colocated
.venv/bin/python -m hexgen2.summarize runs/local-reference-001
.venv/bin/python -m unittest discover -s tests -v
```

`models/tiny` contains seeded random weights for correctness only. A new attempt
must use a fresh run ID; neither the recorder nor the rank logger overwrites runs.

## Source integrity

```bash
python3 scripts/verify_upstream.py
```

[provenance/upstream.json](provenance/upstream.json) records the original ZIP URL,
SHA-256, and all 415 imported file hashes. MacOS metadata was omitted and embedded
Hugging Face credentials were removed from four examples before the initial
public commit. [provenance/adaptations.json](provenance/adaptations.json) records
subsequent changes to imported files. `--original` compares strictly to the
sanitized initial import; `--archive path/to/supplement.zip` also verifies the ZIP.
New harness files are versioned in Git and copied into every recorded source bundle.

## Layout

| Path | Contents |
| --- | --- |
| `hexgen2/` | Supported recorder, two-rank pilot, reference validation, scheduler wrapper and FABRIC controller |
| `hexgen/` | Imported native model/cache/parallelism code, with documented adaptations |
| `benchmark/scheduling_algorithm_documents/` | Authors' scheduling and cost-model implementation |
| `scripts/fabric/` | GPU environment setup, network measurements and per-node launcher |
| `configs/` | Tiny correctness trace, CPU placement description, unreserved FABRIC candidate |
| `tests/` | Real multi-process transfer and runtime/capture correctness tests |
| `runs/` | Local run bundles, ignored by Git; publish reviewed bundles with durable locations |

## License and citation

[Apache-2.0](LICENSE); included third-party notices are preserved. New reproduction
code/documentation also uses Apache-2.0. This is an independent artifact import;
no upstream Git history was present in the ZIP. The lab's newer
[HexGen-3 repository](https://github.com/Relaxed-System-Lab/HexGen-3) is a separate runtime.

```bibtex
@inproceedings{jiang2025hexgen2,
  title={HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment},
  author={Jiang, Youhe and Yan, Ran and Yuan, Binhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=Cs6MrbFuMq}
}
```
