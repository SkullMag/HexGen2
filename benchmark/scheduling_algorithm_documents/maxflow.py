import networkx as nx
import matplotlib.pyplot as plt
from strategy import *
from layer_partition import *
import numpy as np
import re
from globals import configs
import random

def create_flow_network(prefill_caps, decode_caps, comm_matrix):
    G = nx.DiGraph()

    source = 'source'
    sink = 'sink'
    G.add_node(source, layer=0)
    G.add_node(sink, layer=5)

    prefill_in = [f'p{i}_in' for i in range(len(prefill_caps))]
    prefill_out = [f'p{i}_out' for i in range(len(prefill_caps))]
    decode_in = [f'd{i}_in' for i in range(len(decode_caps))]
    decode_out = [f'd{i}_out' for i in range(len(decode_caps))]

    for i, p_in in enumerate(prefill_in):
        G.add_node(p_in, layer=1)
        G.add_edge(source, p_in, capacity=float('inf'))

    for i, p_out in enumerate(prefill_out):
        G.add_node(p_out, layer=2)
        for j, d_in in enumerate(decode_in):
            G.add_edge(p_out, d_in, capacity=comm_matrix[i][j])

    for i, d_in in enumerate(decode_in):
        G.add_node(d_in, layer=3)
    
    for i, d_out in enumerate(decode_out):
        G.add_node(d_out, layer=4)
        G.add_edge(d_out, sink, capacity=float('inf'))

    for i in range(len(prefill_caps)):
        G.add_edge(prefill_in[i], prefill_out[i], capacity=prefill_caps[i])
    for i in range(len(decode_caps)):
        G.add_edge(decode_in[i], decode_out[i], capacity=decode_caps[i])

    return G


def draw_flow_network(G, flow_dict):
    layers = nx.get_node_attributes(G, 'layer')
    pos = nx.multipartite_layout(G, subset_key='layer')
    edge_labels = {(u, v): f"{round(d['capacity'], 2)} | {round(flow_dict[u][v], 2)}" for u, v, d in G.edges(data=True) if flow_dict[u][v] > 0}

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='darkred')

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, width=2, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_color='green')

    plt.title('Flow Network with Capacities and Flow Values')
    plt.axis('off')
    plt.show()


def coarsen(optimized_clusters):
    global configs
    G = nx.Graph()
    
    coarsened_optimized_clusters = [(key, sum([configs.devices[gpu].memory for gpu in val])) for key, val in optimized_clusters.items()]

    # Add nodes with memory attributes
    G.add_nodes_from([(cluster[0], {'memory': cluster[1]}) for cluster in coarsened_optimized_clusters])
    
    for i in range(len(optimized_clusters)):
        for j in range(i+1, len(optimized_clusters)):
            comm_bw = sum([configs.comm_bws_dict[tuple(sorted([gpu_i, gpu_j]))] for gpu_i in optimized_clusters[i] for gpu_j in optimized_clusters[j]])
            G.add_edge(i, j, weight=comm_bw)

    return G


def decide_prefill_and_decode(optimized_clusters):
    """
        find the best strategy and then find capability of each cluster
    """


    
    def find_prefill_decode_clusters_bipartition(optimized_clusters):
        coarsened_graph = coarsen(optimized_clusters)
        from graph_partition import spectral_partition
        coarsened_clusters = spectral_partition(coarsened_graph, num_clusters=2)  # bi-partition

        while len(coarsened_clusters[0]) > len(coarsened_clusters[1]):
            
            cluster = coarsened_clusters[0].pop()
            coarsened_clusters[1].append(cluster)

        while len(coarsened_clusters[1]) > len(coarsened_clusters[0]):
            cluster = coarsened_clusters[1].pop()
            coarsened_clusters[0].append(cluster)

        return coarsened_clusters[0], coarsened_clusters[1]
    

    def find_prefill_decode_clusters_greedy(optimized_clusters):
        all_cluster_decode_caps = []

        for cluster_id, devices in optimized_clusters.items():

            # prefill_cap = 
            task_type = "decode"
            strategy = gen_strategy(devices, task_type=task_type)
            
            layer_partition = create_layer_partition(strategy)

            cost = TimeCost([strategy, layer_partition]).pipeline_cost(task_type=task_type)
            
            all_cluster_decode_caps.append((cluster_id, cost))

        all_cluster_decode_caps.sort(key=lambda x:x[1],)

        decode_clusters = []
        prefill_clusters = []
        for i in range(len(optimized_clusters)):
            if i < len(all_cluster_decode_caps) // 2:
                decode_clusters.append(all_cluster_decode_caps[i][0])
            else:
                prefill_clusters.append(all_cluster_decode_caps[i][0])
        
        return prefill_clusters, decode_clusters

    prefill_clusters, decode_clusters = find_prefill_decode_clusters_bipartition(optimized_clusters)
    # prefill_clusters, decode_clusters = find_prefill_decode_clusters_greedy(optimized_clusters)

    def find_caps_strategy_and_layer_partition(cluster_ids, task_type=None):
        pass
        caps_strategy_and_layer_partition = []
        for cluster_id, devices in optimized_clusters.items():
            if cluster_id not in cluster_ids:
                continue

            strategy = gen_strategy(devices, task_type=task_type)

            layer_partition = create_layer_partition(strategy)


            cost = TimeCost([strategy, layer_partition]).pipeline_cost(task_type=task_type)
            
            caps_strategy_and_layer_partition.append((cluster_id, cost, strategy, layer_partition))

        return caps_strategy_and_layer_partition

    prefill_caps = find_caps_strategy_and_layer_partition(prefill_clusters, task_type='prefill')
    decode_caps = find_caps_strategy_and_layer_partition(decode_clusters, task_type='decode')
    
    prefill_caps_val = [float(x[1]) for x in prefill_caps]
    decode_caps_val = [float(x[1]) for x in decode_caps]


    clusters_split = {}
    clusters_split["decode_clusters"] = decode_clusters
    clusters_split["prefill_clusters"] = prefill_clusters
    clusters_split['prefill_strategy'] = [x[2] for x in prefill_caps]

    clusters_split['prefill_cost'] = prefill_caps_val
    clusters_split['decode_cost'] = decode_caps_val

    clusters_split['decode_strategy'] = [x[2] for x in decode_caps]

    clusters_split['prefill_layer_partition'] = [x[3] for x in prefill_caps]
    clusters_split['decode_layer_partition'] = [x[3] for x in decode_caps]

    return prefill_caps_val, decode_caps_val, clusters_split
        

def decide_clusters_comm_matrix(clusters_split, optimized_clusters):
    global configs

    prefill_clusters, decode_clusters = clusters_split["prefill_clusters"], clusters_split["decode_clusters"]
    
    nclusters = len(prefill_clusters)
    clusters_comm_matrix = np.zeros(shape=(nclusters, nclusters))

    for i in prefill_clusters:
        for j in decode_clusters:
            clusters_i, clusters_j = optimized_clusters[i], optimized_clusters[j]

            min_bw_node, max_bw_node = detect_bottleneck_connection_edge(optimized_clusters[i], optimized_clusters[j])
            clusters_comm_matrix[prefill_clusters.index(i), decode_clusters.index(j)] =  configs.specs[1][min_bw_node[0]][min_bw_node[1]]

    return clusters_comm_matrix


def extract_min_max_burden_edges(edge_labels):
    
    max_val = -1
    min_burden_node = None
    for (u, v), val in edge_labels.items():
        if max_val < val < float("inf"):
            min_burden_node = (u, v)
            max_val = val

    min_val = float("inf")
    max_burden_node = None

    for (u, v), val in edge_labels.items():
        if val < min_val:
            max_burden_node = (u, v)
            min_val = val

    return min_burden_node, max_burden_node


def uniform_units(prefill_caps, decode_caps, clusters_comm_matrix):
    """
        bandwidth is calculated as time
    """

    global configs

    KV_size = configs.H * configs.H * 2 * configs.L * configs.B_type / 1024 ** 3 * 8
    clusters_comm_matrix = KV_size / clusters_comm_matrix

    return prefill_caps, decode_caps, clusters_comm_matrix


def update_caps(min_burden_node, max_burden_node, prefill_caps, decode_caps, clusters_comm_matrix, 
                clusters_split, optimized_clusters):
    
    def update_optimized_clusters(node):
        i = int(re.findall(r'\d+', string=node[0])[0])
        j = int(re.findall(r'\d+', string=node[1])[0])
        # if prefill out -> decode in is bottleneck, then update the one with higher capability
        if "out" in node[0]:
            choice = random.choice([0, 1])
            if choice:
                first_gpu = optimized_clusters[i].pop(0)
                optimized_clusters[i].append(first_gpu)
            else:
                first_gpu = optimized_clusters[j].pop(0)
                optimized_clusters[j].append(first_gpu)   
        # if prefill in -> prefill in is bottleneck, then update the capability
        elif "in" in node[0] and "p" in node[0]:
            first_gpu = optimized_clusters[i].pop(0)
            optimized_clusters[i].append(first_gpu)
        # if decode in -> decode in is bottleneck, then update the capability
        elif "in" in node[0] and "d" in node[0]:
            first_gpu = optimized_clusters[j].pop(0)
            optimized_clusters[j].append(first_gpu)
        else:
            raise NotImplementedError

    def swap_edges(node):
        swap_from, swap_to = node

        i = int(re.findall(r'\d+', string=swap_from)[0])
        j = int(re.findall(r'\d+', string=swap_to)[0])

        if "p" in swap_from and "d" in swap_to and len(optimized_clusters[clusters_split['prefill_clusters'][i]]) > 1:
            gpu = optimized_clusters[clusters_split['prefill_clusters'][i]].pop()
            optimized_clusters[clusters_split['decode_clusters'][j]].append(gpu)

        # In this case, no edges can be swapped, can only adjust weight
        elif "p" in swap_from and "p" in swap_to:
            pass

        elif "d" in swap_from and "d" in swap_to:
            pass
    
    swap_edges(min_burden_node)
    swap_edges(max_burden_node)


    prefill_caps, decode_caps, clusters_split = decide_prefill_and_decode(optimized_clusters)
    clusters_comm_matrix = decide_clusters_comm_matrix(clusters_split, optimized_clusters)
    prefill_caps, decode_caps, clusters_comm_matrix = uniform_units(prefill_caps, decode_caps, clusters_comm_matrix)

    return prefill_caps, decode_caps, clusters_comm_matrix, optimized_clusters

def detect_bottleneck_connection_edge(nodes_from, nodes_to):

    global configs

    min_bw = 1e8
    max_bw = -1

    min_bw_node = None
    max_bw_node = None

    for node_from in nodes_from:
        for node_to in nodes_to:
            cur_bw = configs.comm_bws_dict[node_from, node_to] if (node_from, node_to) in configs.comm_bws_dict else configs.comm_bws_dict[node_to, node_from]
            if cur_bw < min_bw:
                min_bw = cur_bw
                min_bw_node = (node_from, node_to)
            if cur_bw > max_bw:
                max_bw = cur_bw
                max_bw_node = (node_from, node_to)

    return min_bw_node, max_bw_node


def update_graph(G, min_burden_node, max_burden_node, optimized_clusters, clusters_split):

    def update_graph_by_node(node, adjust_direction=""):
        i = int(re.findall(r'\d+', string=node[0])[0])
        j = int(re.findall(r'\d+', string=node[1])[0])

        from_clusters = optimized_clusters[clusters_split['prefill_clusters'][i]]
        to_clusters = optimized_clusters[clusters_split['decode_clusters'][j]]

        min_bw_node, max_bw_node = detect_bottleneck_connection_edge(from_clusters, to_clusters)

        if adjust_direction == "higher":
            G[max_bw_node[0]][max_bw_node[1]]['weight'] *= 2
        else:
            G[min_bw_node[0]][min_bw_node[1]]['weight'] /= 2

        
    update_graph_by_node(min_burden_node, adjust_direction="lower")
    update_graph_by_node(max_burden_node, adjust_direction="higher")
