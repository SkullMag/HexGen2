# Modified 2026-10-19: replaced an upstream embedded credential with HF_TOKEN.
export INITIAL_PEERS=$1

python3 _petals.py --model-name meta-llama/Llama-2-70b-chat-hf --token "${HF_TOKEN:?Set HF_TOKEN in your environment}" --init-peers $INITIAL_PEERS