CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 9992 llama_prefill_decode.py \
--model_size llama-7b \
--use-flash-attn \
--hetero_config 1 \
--pp_partition 32 \
