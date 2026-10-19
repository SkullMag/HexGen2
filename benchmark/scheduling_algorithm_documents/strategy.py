from globals import configs
import numpy as np
from cost_modeling import TimeCost
from typing import List

def get_considered_stragegies(i):
    """
        Do not consider stages over 3
    """
    considered_strategies = {}

    considered_strategies[1] = [[1]]
    considered_strategies[2] = [[2], [1, 1]]
    considered_strategies[3] = [[2, 1], [1, 1, 1]]
    considered_strategies[4] = [[4], [2, 2], [2, 1, 1]]
    considered_strategies[5] = [[4, 1], [2, 2, 1]]
    considered_strategies[6] = [[4, 2], [2, 2, 2]]
    considered_strategies[7] = [[4, 2, 1], [2, 2, 2, 1]]
    considered_strategies[8] = [[8], [4, 4], [4, 2, 2], ]

    return considered_strategies[i]

def refine_strategies(strategies, pp_devices, task_type=None):

    global configs

    ngpu_strategy = []

    def tp_pp_tradeoff():
        global configs

        offsets: List = np.cumsum([sum(possible_strategy[0]) for possible_strategy in strategies]).tolist()
        offsets.insert(0, 0)

        for i in range(len(strategies)):
            local_optimal = None
            local_minimum_cost = 1e20
            possible_strategies = strategies[i]
            for stage_strategy in possible_strategies:
                strategy_offsets: List = np.cumsum(stage_strategy).tolist()
                strategy_offsets.insert(0, 0)

                stage = [pp_devices[offsets[i]: offsets[i + 1]][strategy_offsets[j]: strategy_offsets[j + 1]] for j in range(len(strategy_offsets) - 1)]

                # simulate by fake layer partition
                stage_layer_partition = np.round([configs.L * ndevices // sum(stage_strategy) for ndevices in stage_strategy], decimals=0)
                stage_layer_partition[-1] -= sum(stage_layer_partition) - configs.L
                assert sum(stage_layer_partition) == configs.L

                strategy_cost = TimeCost(pipeline=[stage, stage_layer_partition])

                # update strategy_cost to 1e8 if oom
                
                if local_minimum_cost > strategy_cost.pipeline_cost(task_type):
                    local_minimum_cost = strategy_cost.pipeline_cost(task_type)
                    local_optimal = stage_strategy

            ngpu_strategy.extend(local_optimal)
                
    tp_pp_tradeoff()

    return ngpu_strategy


def gen_strategy(replica_devices, task_type=None):
    replica_devices.sort()

    global configs

    # example: replica_devices = [[1, 3, 6, 8, 9], [2, 4, 11, 12]]
    strategies = []

    parts_machines = [configs.device_machine_map[gpu] for gpu in replica_devices]
    intra_counts = [parts_machines.count(i) for i in set(parts_machines)]
    
    for ngpus in intra_counts:
        strategies.append(get_considered_stragegies(ngpus))

    ngpu_strategy = refine_strategies(strategies, replica_devices, task_type)
    
    strategy = []
    i = 0
    start = 0
    while i < len(ngpu_strategy):
        
        end = start + ngpu_strategy[i]
        strategy.append(replica_devices[start : end])

        start = end
        i += 1
    
    return strategy