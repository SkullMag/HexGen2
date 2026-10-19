import random
from strategy import *
from layer_partition import *
from globals import configs


def evaluate_fitness(G, clusters):
    f = sum(G[node1][node2]['weight'] for cluster in clusters.values() for node1 in cluster for node2 in cluster if G.has_edge(node1, node2))
    return f


def mutate(G, clusters, mutation_rate=0.1, mem_variation=1.05):
    if random.random() < mutation_rate:
        # Get the current memory status of each cluster
        cluster_memories = {k: sum(G.nodes[n]['memory'] for n in clusters[k]) for k in clusters}
        average_memory = np.mean(list(cluster_memories.values()))
        
        # Identify clusters significantly above or below the acceptable range
        max_mem_cluster_id = max(cluster_memories, key=lambda k: cluster_memories[k] if cluster_memories[k] > mem_variation * average_memory else -1)
        min_mem_cluster_id = min(cluster_memories, key=lambda k: cluster_memories[k] if cluster_memories[k] < average_memory / mem_variation else float('inf'))

        # Ensure there's a valid move to make
        if max_mem_cluster_id != -1 and min_mem_cluster_id != float('inf') and len(clusters[max_mem_cluster_id]) > 1:
            node_to_move = random.choice(clusters[max_mem_cluster_id])
            clusters[max_mem_cluster_id].remove(node_to_move)
            clusters[min_mem_cluster_id].append(node_to_move)


def genetic_algorithm(G, initial_clusters, population_size=50, generations=100):
    population = [initial_clusters.copy() for _ in range(population_size)]
    for generation in range(generations):
        fitness_scores = [evaluate_fitness(G, ind) for ind in population]
        parents = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:population_size//2]
        offspring = []
        for _ in range(population_size - len(parents)):
            parent1, parent2 = random.choice(parents)[0], random.choice(parents)[0]
            child1, child2 = parent1.copy(), parent2.copy()
            mutate(G, child1)
            mutate(G, child2)
            offspring.append(child1)
            offspring.append(child2)
        population = [p[0] for p in parents] + offspring[:population_size-len(parents)]
    best_solution = max(population, key=lambda ind: evaluate_fitness(G, ind))

    return best_solution


def evaluate_throughput(optimized_clusters ):
    def comp_throughput(cluster, task_type=None):

        strategy = gen_strategy(cluster, task_type=task_type)
        layer_partition = create_layer_partition(strategy)
        cost = TimeCost([strategy, layer_partition], configs=configs).pipeline_cost(task_type=task_type)

        return cost

    max_latency = 0
    
    for i in range(0, len(optimized_clusters), 2):
        prefill_cluster = optimized_clusters[i]
        decode_cluster = optimized_clusters[i + 1]


        prefill_cost = comp_throughput(prefill_cluster, "prefill")
        decode_cost = comp_throughput(decode_cluster, "decode")

        kv_cost = comp_throughput(prefill_cluster, "kv_comm")

        end_to_end_cost = float(prefill_cost + decode_cost + kv_cost)

        max_latency = max(max_latency, end_to_end_cost)
    
    throughput = configs.S * configs.batch_size / (len(optimized_clusters ) // 2) / max_latency

    return throughput