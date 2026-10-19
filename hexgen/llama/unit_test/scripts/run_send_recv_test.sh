export NCCL_P2P_DISABLE=1
  
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 9996 send_recv_test.py
