#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
wheel=$(find artifacts/nccl-runtime-wheel -name '*.whl' -type f)
test -n "$wheel"
printf '%s  %s\n' ea5ed3e053c735f16809bee7111deac62ac35b10128a8c102960a0462ce16cbe "$wheel" | sha256sum -c -
test ! -e artifacts/nccl-runtime
.venv/bin/pip install --no-deps --no-compile --target artifacts/nccl-runtime "$wheel"
export LD_PRELOAD="$PWD/artifacts/nccl-runtime/nvidia/nccl/lib/libnccl.so.2"
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
import torch
assert torch.cuda.nccl.version()==(2,26,5),torch.cuda.nccl.version()
libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnccl.so' in line})
assert len(libraries)==1 and '/artifacts/nccl-runtime/' in libraries[0],libraries
identity={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'nccl_runtime':list(torch.cuda.nccl.version()),'libraries':[{'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in libraries],'mechanism':'LD_PRELOAD of checksum-pinned NVIDIA runtime; Torch package dependencies remain installed unchanged'}
Path('artifacts/setup/nccl-runtime.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
PY
