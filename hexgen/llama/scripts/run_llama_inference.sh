CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 llama_inference.py \
--model_size llama-7b \
--use-flash-attn \
--hetero_config 4 \
--pp_partition 32 \
