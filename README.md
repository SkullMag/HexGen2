# HexGen-2 on FABRIC

This repository starts a reproduction of **HexGen-2: Disaggregated Generative
Inference of LLMs in Heterogeneous Environment** (ICLR 2025), by Youhe Jiang,
Ran Yan, and Binhang Yuan.

The project goal is to get HexGen-2 running on FABRIC and retain enough source,
environment, placement, workload, and raw measurement data to reproduce findings
and investigate anomalies.

## Source and status

The baseline comes from the authors' [official ICLR 2025 source supplement](https://proceedings.iclr.cc/paper_files/paper/2025/file/0b941a1e5fbce23fe46b049999d04ed0-Supplemental-Conference.zip).
See the [paper page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0b941a1e5fbce23fe46b049999d04ed0-Abstract-Conference.html),
[paper PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/0b941a1e5fbce23fe46b049999d04ed0-Paper-Conference.pdf),
and [arXiv record](https://arxiv.org/abs/2502.07903).

- Imported 415 files; omitted 538 macOS metadata files.
- Preserved the supplied source and license files. Replaced embedded Hugging Face
  credentials in four example files with `HF_TOKEN` environment-variable references.
- Preserved the authors' original top-level instructions in
  [README.upstream.md](README.upstream.md).
- Recorded the archive URL, SHA-256, file hashes, and import changes in
  [provenance/upstream.json](provenance/upstream.json).
- **FABRIC deployment and GPU inference have not been validated.**

The lab's [HexGen-3 repository](https://github.com/Relaxed-System-Lab/HexGen-3)
describes a newer runtime. This project uses the ICLR HexGen-2 artifact as its
baseline rather than substituting that runtime.

## Repository map

| Path | Contents |
| --- | --- |
| `benchmark/scheduling_algorithm_documents/` | Scheduler, cost model, and sample machine configuration |
| `hexgen/` | Supplied inference, parallelism, and KV-cache code |
| `benchmark/hexgen-2_documents/` | HexGen-2 launcher examples and backend instructions |
| `benchmark/send_request/` | Supplied request-generation utilities |
| `third_party/` | Dependencies included in the supplement |
| `docs/FABRIC.md` | Initial deployment work and confirmed artifact gaps |
| `docs/EXPERIMENT_RECORDS.md` | Required records for each experiment |
| `provenance/upstream.json` | Import provenance and file-level checksums |
| `runs/` | Local run bundles, excluded from Git by default |

## Verify the imported source

From the repository root, using Python 3 with no extra packages:

```bash
python3 scripts/verify_upstream.py
```

Optionally download the original archive and verify its checksum too:

```bash
mkdir -p artifacts
curl -fL 'https://proceedings.iclr.cc/paper_files/paper/2025/file/0b941a1e5fbce23fe46b049999d04ed0-Supplemental-Conference.zip' \
  -o artifacts/iclr2025-hexgen2-supplement.zip
python3 scripts/verify_upstream.py --archive artifacts/iclr2025-hexgen2-supplement.zip
```

The verifier checks the imported baseline, including the documented credential
replacements. A later source modification will fail this comparison; retain its
Git diff as part of the experiment record rather than silently updating the baseline.

## Next steps

1. Establish a pinned environment and run the CPU scheduler with captured inputs
   and logs.
2. Resolve the supplied runtime and launcher gaps listed in
   [the FABRIC plan](docs/FABRIC.md).
3. Inspect current FABRIC GPU placements and validate one small disaggregated
   inference run.
4. Add workload replay and measurement capture before performance sweeps.

The original setup commands remain available in `README.upstream.md`; they have
not yet been adapted into a validated FABRIC installation procedure.

## License and citation

The supplement includes the [Apache-2.0 license](LICENSE). Included third-party
notices and licenses remain in place. New reproduction documentation and tooling
are provided under Apache-2.0 as well.

```bibtex
@inproceedings{jiang2025hexgen2,
  title={HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment},
  author={Jiang, Youhe and Yan, Ran and Yuan, Binhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=Cs6MrbFuMq}
}
```
