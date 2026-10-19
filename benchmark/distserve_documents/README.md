## Build && Install
```shell
# clone the project
git clone https://github.com/LLMServe/DistServe.git && cd DistServe

# setup the distserve conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive
cmake -B build && cmake --build build -j$(nproc)
cd ..

# install distserve
pip install -e .
```

## Launching

### Launch Ray Cluster

DistServe relies on [Ray](https://ray.io) to implement distributed workers. If you do not launch a Ray runtime in advance, it will automatically initiate a cluster consisting of all the gpus on the current node. You may need to start the Ray runtime manually in advance if you want to use multiple nodes for inference.

### Run offline example

DistServe requires at least two GPUs to play with. An offline inference example is provided in `examples/offline.py`.

### Run online example

To run online inference, you need to launch the DistServe API server, see the comments in `distserve/api_server/distserve_api_server.py`.

Then launch the client example in `examples/online.py`.
