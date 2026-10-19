#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$task_repo"
: "${RUN_ID:?Use the same unique RUN_ID on both nodes}"
: "${NODE_RANK:?Set 0 for prefill or 1 for decode}"
: "${MASTER_ADDR:?Use the prefill node data-plane address}"
: "${DATA_INTERFACE:?Select the actual FABRIC data-plane interface}"
: "${PLACEMENT:?Path to the actual slice/host/GPU/VM/network allocation JSON}"
: "${MODEL_DIR:?Local safetensors checkpoint directory on both nodes}"
: "${TRACE:?Same request trace on both nodes}"
: "${NETWORK_DIR:?Directory containing completed raw pre-run network measurements}"
case "$NODE_RANK" in 0) task_role=prefill;; 1) task_role=decode;; *) exit 2;; esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 PYTHONHASHSEED=123
export NCCL_SOCKET_IFNAME="=$DATA_INTERFACE" GLOO_SOCKET_IFNAME="$DATA_INTERFACE"
# FABRIC NIC_Basic networking uses TCP here; no unverified RDMA/HCA assumptions.
export NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
mkdir -p artifacts
exec .venv/bin/python -m hexgen2.record \
  --run-id "$RUN_ID" --node "$task_role" --classification native-pilot \
  --placement "$PLACEMENT" --model-dir "$MODEL_DIR" --trace "$TRACE" --network-dir "$NETWORK_DIR" --timeout 900 -- \
  .venv/bin/python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29620}" \
  -m hexgen2.run --model-dir "$MODEL_DIR" --trace "$TRACE" --check-colocated
