# HexGen-2: Disaggregated Generative Inference Framework

HexGen-2 is an advanced distributed system designed to optimize the deployment of large language models (LLMs) in heterogeneous GPU environments. This framework improves upon its predecessor by introducing a novel scheduling algorithm that efficiently handles the prefill and decoding phases of LLM inference, ensuring high throughput and low latency across diverse hardware setups.

## Overview

HexGen-2 disaggregates the generative inference of LLMs, such as OPT and LLAMA-2, by effectively managing computation and communication across a range of GPU types. It utilizes a constraint optimization-based scheduling approach, leveraging graph partitioning and max-flow algorithms to optimize resource allocation and parallel execution strategies.

### Key Features

- Disaggregated Inference Paradigm: Separates the prefill and decoding phases to reduce interference and enhance parallelism.
- Heterogeneity-Aware Scheduling: Tailors resource allocation and parallelism strategies to accommodate the capabilities and limitations of heterogeneous GPU setups.
- Enhanced Throughput and Reduced Latency: Demonstrates significant improvements in throughput and latency compared to homogeneous and previous heterogeneous systems.

----------

## Content

- [Building Environment](#building-environment)
    - [Establish A Personal Head Node Coordinator](#establish-a-personal-head-node-coordinator)
    - [Incorporate Additional Worker Nodes](#incorporate-additional-worker-nodes)
- [Loading Model Parameters for LlaMA Models](#loading-model-parameters-for-llama-models)
    - [Create Separate Model State Dicts](#create-separate-model-state-dicts)
    - [Load Model Parameters](#load-model-parameters)
- [Starting HexGen2](#starting-hexgen2)
    - [Activating Head Node Coordinator](#activating-head-node-coordinator)
    - [Activating Worker Nodes](#activating-worker-nodes)
    - [Activating Independent Inference Process](#activating-independent-inference-process)
- [Scheduling Algorithm in HexGen2](#scheduling-algorithm-in-hexgen2)
    - [Problem Formalization](#problem-formalization)
    - [Graph Partition](#graph-partition)
    - [Max Flow Optimization](#max-flow-optimization)
    - [Iterative Refinement](#iterative-refinement)
- [Performance Results of HexGen2](#performance-results-of-hexgen2)


## Building Environment

HexGen-2 stipulates the utilization of CUDA version 11.8 and Python version 3.11 or above. The assembly of HexGen-2 is designed to be efficient and accessible:

### Only Establish A Personal Head Node Coordinator

```bash
make hexgen-head
```

### Incorporate Additional Worker Nodes

```bash 
make hexgen
```

## Loading Model Parameters for LlaMA Models

Navigate to the `hexgen/llama/load_model_parameters_utils` directory. Here, you will initiate the process of setting up parameters for the model.

### Create Separate Model State Dicts

For scenarios where specific custom paths are required, modifications to the `create_separate_state_dicts_llama_7b.py` script are necessary. In this script, locate the function call to `save_model_components`. You can then alter the paths according to your specific requirements. For instance:

```python
save_model_components(
    config_path='../llama-config/',
    checkpoint_name='llama-7b',
    checkpoint_path='/path/to/Llama-2-7b-chat-hf/',
    num_layers=32,
    save_dir='./separate_state_dicts/'
)
```

Here, your sole requirement is to specify the `checkpoint_path`, as the other parameters have been pre-defined and supplied for your convenience. You can download the model checkpoints from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

A recommended way to download is:

```bash
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --local-dir Llama-2-7b-chat-hf --token <your token>
```

To create the separate state dictionaries for the Llama-7b model, run the following command in the terminal:

```bash
python3 create_separate_state_dicts_llama_7b.py
```

This script will automatically generate and save the state dictionaries in the appropriate directory.

### Load Model Parameters

In the `llama_inference.py` file, add the following code snippet to load the parameters for Llama-7b. Adjust the paths as per your setup:

```python
# Load model checkpoints with respect to hetero_config
tp_ranks_whole_model = hetero_groups['tp_ranks_whole_model']
tp_group_list = hetero_groups['tp_rank_groups']
state_dicts_path = "./load_model_parameters_utils/"
load_model_parameters(model, config, state_dicts_path, tp_ranks_whole_model, tp_group_list, rank)
```

## Starting HexGen2

### Activating Head Node Coordinator

HexGen-2 can be launched in head node coordinator modes by:

```bash
bash scripts/run_head.sh
```

### Activating Worker Nodes

HexGen-2 can be launched in worker modes by a similar command, except that you should modify the file `./third_party/ocf/ocf-core/config/cfg.yaml`, the p2p addr should be as similar format as `"/ip4/{Pubilc_IP}/tcp/43905/p2p/{Peer_ID}"`, you could replace `{Public_IP}` as your own head coordinator's IP address and `{Peer_ID}` as its peer ID:

```bash
bash scripts/run_worker.sh
```

### Activating Independent Inference Process

To initiate an independent inference process without involving the coordinator, navigate to the `hexgen/llama` directory and run the scripts.

You have the flexibility to customize various inputs to tailor your inference task according to your specific requirements. The `model_msg` object can be adjusted with different parameters, as shown in the example below:

```python
model_msg = {
    'prompt': "Do you like yourself ?",  # Define your own prompt here
    'max_new_tokens': 128,               # Set the maximum number of new tokens
    'temperature': 0.2,                  # Adjust the randomness in response generation
    'top_k': 20,                         # Specify the number of highest probability vocabulary tokens to keep for top-k sampling
    'top_p': 0.9,                        # Set the cumulative probability threshold for top-p (nucleus) sampling
}
```

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

## Performance Results of HexGen2

The performance of HexGen-2 was extensively evaluated against contemporary systems, showcasing its ability to significantly enhance the efficiency of LLM serving in heterogeneous environments. Below are the key findings:

- **Throughput and Latency Improvements:** HexGen-2 consistently outperforms existing homogeneous and heterogeneous LLM serving systems. It achieves up to 2.0× and on average a 1.3× improvement in serving throughput across various workloads and reduces average inference latency by 1.5× compared with state-of-the-art systems.

- **Economic Efficiency:** Even with a 30% lower budget, HexGen-2 matches or surpasses the performance of more costly homogeneous setups, proving its cost-effectiveness.
