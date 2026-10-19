export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 unit_test_multi_to_one.py
