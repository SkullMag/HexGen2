#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
wheel=$(find artifacts/nccl-runtime-wheel -name '*.whl' -type f)
test -n "$wheel"
printf '%s  %s\n' ea5ed3e053c735f16809bee7111deac62ac35b10128a8c102960a0462ce16cbe "$wheel" | sha256sum -c -
if [ ! -e artifacts/nccl-runtime ]; then
 .venv/bin/pip install --no-deps --no-compile --target artifacts/nccl-runtime "$wheel"
fi
export LD_PRELOAD="$PWD/artifacts/nccl-runtime/nvidia/nccl/lib/libnccl.so.2"
.venv/bin/python - <<'PY'
import hashlib,json,zipfile
from pathlib import Path
import torch
from hexgen2.nccl_runtime import runtime_version,mapped_libraries
assert runtime_version()==(2,26,5),runtime_version()
libraries=mapped_libraries()
assert len(libraries)==1 and '/artifacts/nccl-runtime/' in libraries[0],libraries
actual=hashlib.sha256(Path(libraries[0]).read_bytes()).hexdigest()
with zipfile.ZipFile(next(Path('artifacts/nccl-runtime-wheel').glob('*.whl'))) as archive:
 name=next(n for n in archive.namelist() if n.endswith('/libnccl.so.2'))
 h=hashlib.sha256()
 with archive.open(name) as source:
  for block in iter(lambda:source.read(1<<20),b''):h.update(block)
 assert h.hexdigest()==actual,'Installed library differs from the verified wheel'
identity={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'nccl_torch_build':list(torch.cuda.nccl.version()),'nccl_runtime':list(runtime_version()),'libraries':[{'path':libraries[0],'sha256':actual}],'installed_library_matches_verified_wheel':True,'mechanism':'LD_PRELOAD of checksum-pinned NVIDIA runtime; Torch package dependencies remain installed unchanged'}
Path('artifacts/setup/nccl-runtime.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
PY
