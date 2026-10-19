export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 9992 llama_1_to_1.py \
--model_size llama-7b \
--use-flash-attn \
--prefill_size 1 \
--decode_size 1 \
