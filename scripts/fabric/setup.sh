#!/usr/bin/env bash
# Explicit VM setup; run on a dedicated FABRIC Ubuntu 22.04 x86_64 node.
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$task_repo"
mkdir -p artifacts/setup
exec > >(tee -a artifacts/setup/install.log) 2>&1
date -u
test "$(uname -m)" = x86_64
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
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
    echo 'Driver setup requires a reboot. Reboot this dedicated VM, then rerun this script.'
    exit 42
  fi
fi
python3.10 -m venv .venv
.venv/bin/pip install pip==24.3.1
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-reproduction.txt
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.get_device_capability()[0] >= 8, 'This baseline requires Ampere or newer'
print(torch.__version__, torch.cuda.get_device_name(0))
PY

# Either provide reviewed CPython 3.10 wheels with SHA256SUMS, or build from source.
if [ -n "${HEXGEN2_WHEEL_DIR:-}" ]; then
  (cd "$HEXGEN2_WHEEL_DIR" && sha256sum -c SHA256SUMS)
  .venv/bin/pip install --no-deps "$HEXGEN2_WHEEL_DIR"/*.whl
  cp "$HEXGEN2_WHEEL_DIR/SHA256SUMS" artifacts/setup/native-wheel-SHA256SUMS
else
  curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o artifacts/setup/cuda-keyring.deb
  sudo dpkg -i artifacts/setup/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-4
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="$task_repo/.venv/bin:$CUDA_HOME/bin:$PATH"
  export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0
  if [ ! -d artifacts/flash-attention/.git ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git artifacts/flash-attention
  fi
  git -C artifacts/flash-attention checkout --detach 5231d95fe13733fb534c01895f7ea88c6a6c7793
  git -C artifacts/flash-attention submodule update --init --recursive
  test "$(git -C artifacts/flash-attention rev-parse HEAD)" = 5231d95fe13733fb534c01895f7ea88c6a6c7793
  mkdir -p artifacts/wheels
  .venv/bin/pip wheel artifacts/flash-attention --no-build-isolation --no-deps -w artifacts/wheels
  # This TP=1 pilot uses unfused dense/MLP/RMSNorm; extra CUDA extensions are not required.
  .venv/bin/pip install --no-deps artifacts/wheels/*.whl
  sha256sum artifacts/wheels/*.whl > artifacts/setup/native-wheel-SHA256SUMS
fi
.venv/bin/pip check
.venv/bin/pip freeze --all > artifacts/setup/pip-freeze.txt
dpkg-query -W > artifacts/setup/dpkg-packages.txt
uname -a > artifacts/setup/kernel.txt
nvidia-smi -q > artifacts/setup/gpu.txt
if [ ! -d models/tiny ]; then .venv/bin/python -m hexgen2.fixture --output models/tiny; fi
.venv/bin/python -m hexgen2.validate --model-dir models/tiny > artifacts/setup/native-numerical-validation.json
echo 'NATIVE_NUMERICAL_VALIDATION_PASSED'
