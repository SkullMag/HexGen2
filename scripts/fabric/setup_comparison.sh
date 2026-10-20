#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/setup
exec > >(tee -a artifacts/setup/install.log) 2>&1
date -u
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a OMP_NUM_THREADS=1
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3.10-venv python3.10-dev build-essential git curl pciutils iputils-ping iperf3 chrony
if ! nvidia-smi >/dev/null 2>&1; then
 sudo apt-get install -y "linux-headers-$(uname -r)" nvidia-driver-550-server
 sudo modprobe -r nouveau || true
 sudo modprobe nvidia || true
 sudo modprobe nvidia_uvm || true
 if ! nvidia-smi >/dev/null 2>&1; then
  printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/hexgen2-nouveau.conf
  sudo update-initramfs -u
  exit 42
 fi
fi
python3.10 -m venv .venv
.venv/bin/pip install pip==24.3.1
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-reproduction.txt accelerate==0.25.0 requests==2.32.3 aiohttp==3.11.18
(cd artifacts/wheels && sha256sum -c SHA256SUMS)
.venv/bin/pip install --no-deps artifacts/wheels/*.whl
.venv/bin/pip check
.venv/bin/python - <<'PY'
from hexgen2 import fa_adapter
import json,torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() in [(7,5),(8,0)]
fa_adapter.install()
print(json.dumps(fa_adapter.validate(),indent=2))
PY
.venv/bin/pip freeze --all > artifacts/setup/pip-freeze.txt
dpkg-query -W > artifacts/setup/dpkg-packages.txt
nvidia-smi -q > artifacts/setup/gpu.txt
uname -a > artifacts/setup/kernel.txt
echo COMPARISON_KERNELS_READY
