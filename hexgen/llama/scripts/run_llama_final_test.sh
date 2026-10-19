export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 9992 llama_final_test.py \
--model_size llama-30b \
--use-flash-attn \
--prefill_size 1 \
--decode_size 1 \
