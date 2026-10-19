import numpy as np
from globals import configs

def create_layer_partition(strategy):
    """
        Assume layers are distributed proportionally, remaining layers are not so many
    """
    global configs
    
    memories = [sum([configs.devices[strategy[i][j]].memory for j in range(len(strategy[i]))])
                  for i in range(len(strategy))]

    ngpus = len(memories)
    
    # machines = [0, 0, 1, 1, 2]
    # memories = [16, 16, 8, 8, 40]

    memory_ranking = np.argsort(memories)[::-1]
    
    quota = [0 for _ in range(ngpus)]


    i = 0
    remainings = configs.L
    overload = [False for _ in range(ngpus)]
    rank_ptr = 0
    while remainings:
        overload[i % ngpus] = (quota[i % ngpus] + 1) / configs.L >  memories[i % len(memories) ] / sum(memories)
        if sum(overload) != ngpus and overload[i % ngpus]:
            i += 1
        elif sum(overload) == ngpus:
            quota[memory_ranking[rank_ptr % len(memory_ranking)] % ngpus] += 1   
            rank_ptr += 1
            remainings -= 1
            i += 1
        else:
            quota[i % ngpus] += 1
            remainings -= 1
            i += 1

    return quota