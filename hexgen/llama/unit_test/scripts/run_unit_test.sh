export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=1,2 torchrun --master_port 996 --nproc_per_node=2 unit_test.py
