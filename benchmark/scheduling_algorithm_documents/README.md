## Scheduling Algorithm in HexGen2

HexGen-2 employs a sophisticated scheduling module that optimizes the deployment of LLMs across heterogeneous GPUs by dynamically allocating resources for prefill and decoding phases. Here’s a brief outline of the scheduling process:

### Problem Formalization

The scheduling problem involves:
1. **Group Partition:** Segmenting GPUs into groups designated for either prefill or decoding tasks.
2. **Group Type:** Assigning specific operational roles to each group.
3. **Parallel Strategy and KV Cache Communication:** Determining parallel processing strategies and managing key-value cache communications between groups.

### Graph Partition

- **Initial Partitioning:** Utilizes spectral partitioning techniques to minimize inter-group connections and balance computational loads.
- **Refinement:** Adjusts partitions to maximize efficiency in KV cache communication.

### Max Flow Optimization

- **Flow Setup:** Constructs a directed graph to simulate data flows and computational demands.
- **Flow Optimization:** Employs the max-flow algorithm to optimize resource distribution and data transfer paths for efficient execution.

### Iterative Refinement

- **Continuous Improvement:** Repeatedly refines the graph partition and flow paths to enhance overall system performance, ensuring optimal alignment with dynamic workload demands.

### Run Experiment

```
python3 main.py \
--model-size llama-70b \
--inter_bw 0.3 \
--batch_size 4 \
--machine_config_path machine_amounts.json \
--log_interval 20 \
--niter 10 \
--seq_in 512 \
--seq_out 128 \
```
