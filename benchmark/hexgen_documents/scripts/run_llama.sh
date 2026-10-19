export NCCL_P2P_DISABLE=1
export PORT=9991
export DEVICES=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.launch --nproc_per_node=2 --master_port $PORT _llama_worker.py \
--model_size llama-30b \
--use-flash-attn \
--hetero_config 1 \
--pp_partition 60 \
--model_name "Llama-2-7b-chat-hf" \
# Modify the IP below before execution
--head_node 'http://xxx.xxx.xx.xxx:xxxx' \
--group_id 0 \
