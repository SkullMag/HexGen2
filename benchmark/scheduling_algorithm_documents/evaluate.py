from globals import configs
import numpy as np

def calculate_throughput(clusters_split, clusters_comm_matrix):
    global configs

    prefill_cost, decode_cost = clusters_split['prefill_cost'], clusters_split['decode_cost']

    kv_comm_time = [float(clusters_comm_matrix[i][j]) for i in range(len(clusters_comm_matrix)) for j in range(len(clusters_comm_matrix)) if i == j]
    
    inference_cost = np.array(prefill_cost) + np.array(decode_cost) + np.array(kv_comm_time)

    max_inference_cost = float(max(inference_cost))

    num_models = len(prefill_cost)

    throughput = round(configs.batch_size * configs.S * num_models / max_inference_cost , 4)
    
    return throughput