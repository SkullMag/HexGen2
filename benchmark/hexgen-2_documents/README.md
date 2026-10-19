## Run HexGen-2 worker

1. HexGen-2 worker can be launched on single machine or multi machines. To start it, simply run the corresponding scripts. Remember to properly specify port, uesd devices and master addr. An example is shown below.

```bash
# for single machine
bash scripts/run_llama.sh
# for multi mahcines
bash scripts/run_cross_node.sh
```

2. Besides for common parameters, HexGen worker requires you to
    - Set `--head_node` as the format of `'http://<ip>:<port>'`, you could replace the IP address as your own head coordinator.
    - Set `--model_name` as the name you wish users to call, for example `--model_name "Llama-2-70b-chat-hf"`.
    - Set `--group_id` uniquely, in case starting multiple services on a single nood. 

3. After started up, a HexGen worker uses coroutine techniques to hang there and wait for incoming requests. When sending requests, it is forced to add a suffix `_0` to the previous declared `--model_name`, i.e. call `"Llama-2-70b-chat-hf_0"` for inference request.

4. If you have multiple service on a single head node, it will dispatch requests by roubin robin method, i.e. the first request goes to the first model replica, the second request goes to the second model replica...

5. If you run into cases that different machines have different number of GPUs and `torch.distributed.launch` doesn't work, please refer to `../../hexgen/llama/scripts/run_llama_p0.sh`.

6. It is also supported to run multiple instances on a single worker, just differentiate them by `group_id`.

## Run HexGen-2 with DistServe Backend

HexGen-2 can be deployed using DistServe, a robust distributed computing framework, designed to optimize resource allocation for both online and offline inference tasks.

HexGen-2's scheduling results are fully compatible with DistServe, enabling seamless deployment of HexGen-2 using DistServe as the backend framework.

### Build & Install

To integrate HexGen-2 with DistServe, follow these detailed steps:

```shell
# Clone the DistServe repository
git clone https://github.com/LLMServe/DistServe.git && cd DistServe

# Set up the Conda environment specified in DistServe's environment.yml
conda env create -f environment.yml && conda activate distserve

# Clone and build the SwiftTransformer library, which is a prerequisite for DistServe
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer
git submodule update --init --recursive

# Compile the SwiftTransformer using CMake
cmake -B build && cmake --build build -j$(nproc)
cd ..

# Install DistServe from the current directory in editable mode
pip install -e .
```

### Offline Inference

#### Run Offline Example

To test offline inference capabilities:

```shell
# Navigate to the `examples` directory and run the script `offline.py` using Python
python offline.py
```

This script demonstrates how to perform inference without the need for active server-client communication, utilizing DistServe's efficient processing capabilities.

### Online Inference

#### Run Online Example

To run online inference, you need to launch the DistServe API server, see the comments in `distserve/api_server/distserve_api_server.py`.

```shell
# Launch the Client: Navigate to the `examples` directory and run the script `online.py` using Python
python online.py
```

This script simulates a client initiating real-time data processing requests to the server, showcasing the dynamic interaction facilitated by DistServe.

---

**Note**: This documentation incorporates information from the DistServe framework, utilized under its licensing terms.

By following these instructions, users can fully leverage the capabilities of HexGen-2 powered by the DistServe backend, ensuring efficient deployment in both development and production environments.
