export NCCL_P2P_DISABLE=1
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 9992 llama_4_to_4.py \
--model_size llama-7b \
--use-flash-attn \
--prefill_size 4 \
--decode_size 4 \
