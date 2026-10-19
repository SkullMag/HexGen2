export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port 9996 unit_test_batch_isend_irecv.py
