from graph_partition import *
from maxflow import *
from globals import *
from utils import *
import time
from cost_modeling import check_oom
from evaluate import *
from pure_genetic import genetic_algorithm, evaluate_throughput


if __name__ == "__main__":

    args = get_args()
    start = time.time()

    meta_infos = None
    outputs = None
    max_throughput = 0

    G = create_graph()


    if args.use_genetic:
        max_throughput = 0
        optimized_clusters = None
        for _ in range(configs.niter):
            print(f"Iter: {_}")
            devices_ids = [i for i in range(len(configs.devices))]
            initial_clusters_number = random.randint(2, len(devices_ids) // 2)
            
            clusters_gpus = np.array_split(devices_ids, initial_clusters_number)
            initial_clusters = {k: clusters_gpus[k].tolist() for k in range(len(clusters_gpus))}

            best_solutions = genetic_algorithm(G, initial_clusters)

            if len(best_solutions) % 2 != 0:
                print("Genetic algorithm not successful")
                continue
            
            throughput = evaluate_throughput(best_solutions)

            if max_throughput < throughput:
                max_throughput = throughput
                optimized_clusters = best_solutions
        
        if optimized_clusters is not None:
            print(f"Optimized clusters: {optimized_clusters}")
            print(f"Throughput", round(max_throughput, 4))

    else:
        for _ in range(configs.niter):
            num_clusters = random.choice([i for i in range(1, len(configs.devices)) if i % 2 == 0]) if args.num_clusters is None else args.num_clusters
            print(f"Iter: {_}, num_clusters: {num_clusters}")
    
            clusters = spectral_partition(G, num_clusters=num_clusters)
            fitness_score = evaluate_fitness(G, clusters)

            # Run the genetic algorithm to refine the clusters
            optimized_clusters = graph_refinement(G, clusters)
            fitness_score = evaluate_fitness(G, optimized_clusters)

            prefill_caps, decode_caps, clusters_split = decide_prefill_and_decode(optimized_clusters)
            clusters_comm_matrix = decide_clusters_comm_matrix(clusters_split, optimized_clusters)
            prefill_caps, decode_caps, clusters_comm_matrix = uniform_units(prefill_caps, decode_caps, clusters_comm_matrix)

            for __ in range(args.niter_maxflow):
                G_maxflow = create_flow_network(prefill_caps, decode_caps, clusters_comm_matrix)
                flow_value, flow_dict = nx.maximum_flow(G_maxflow, 'source', 'sink')

                if args.no_maxflow:
                    continue

                edge_labels = {(u, v): float(d['capacity']) / flow_dict[u][v] for u, v, d in G_maxflow.edges(data=True) if flow_dict[u][v] > 0}

                min_burden_node, max_burden_node = extract_min_max_burden_edges(edge_labels)

                prefill_caps, decode_caps, clusters_comm_matrix, optimized_clusters = update_caps(min_burden_node, max_burden_node, prefill_caps, decode_caps, clusters_comm_matrix, 
                                                                                                clusters_split, optimized_clusters)
                

            cluster_id = check_oom(optimized_clusters)

            if cluster_id is not None:
                print(f"    --The {cluster_id} cluster is OOM")
                continue
            
            if flow_value > max_throughput:
                max_throughput = flow_value
                outputs = (G_maxflow, flow_dict)
                meta_infos = {"num_clusters": num_clusters, 
                            "optimized_clusters": optimized_clusters, 
                            "clusters_comm_matrix": clusters_comm_matrix,
                            "clusters_split": clusters_split,
                            "flow_value": round(float(flow_value), 4)}

        if outputs is not None:
            if args.visualize:
                draw_flow_network(*outputs)

            clusters_split = meta_infos['clusters_split']
            meta_infos['throughput'] = calculate_throughput(clusters_split, meta_infos['clusters_comm_matrix'])
            print("Results:", )
            print_fn(meta_infos)


        else:
            print("Ops!!! Nothing is found.")

    end = time.time()
    print("=" * 80)
    print("Consumed Time(s):", round(end-start, 3))
