import numpy as np
from globals import configs


def compute_prompt_time_stage(seq_in, batch_size, m_d, c_d, num_layers, tp_degree=1, h_dim=12288, b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / m_d
    layer_compute_time = 24 * batch_size * seq_in * h_dim * h_dim / tp_degree / c_d
    return (layer_scan_time + layer_compute_time) * num_layers


def compute_token_step_time_stage(batch_size, m_d, c_d, num_layers, tp_degree=1, h_dim=12288, b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / m_d
    layer_compute_time = 24 * batch_size * h_dim * h_dim / tp_degree / c_d
    return (layer_scan_time + layer_compute_time) * num_layers


def communicate_prompt_time_stage(seq_in, batch_size, num_layers, device_sets,
                                  delay_matrix, bandwidth_matrix,
                                  h_dim=12288, b_type=2) -> float:
    tp_degree = len(device_sets)
    step_time = 0
    for i in device_sets:
        current_step = 0
        for j in device_sets:
            if i != j:
                current_step += (delay_matrix[i,j]+batch_size*seq_in*h_dim*b_type/tp_degree/bandwidth_matrix[i,j])
                
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers
    
    return result

def communicate_token_step_time_stage(batch_size, num_layers, device_set,
                                      delay_matrix, bandwidth_matrix,
                                      h_dim=12288, b_type=2) -> float:
    tp_degree = len(device_set)
    step_time = 0
    for i in device_set:
        current_step = 0
        for j in device_set:
            if i != j:
                current_step += (delay_matrix[i,j] + batch_size*h_dim*b_type/tp_degree/bandwidth_matrix[i,j])
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers
    return result


def communication_pipeline_prompt_time_cross_stage(seq_in, batch_size, device_set1, device_set2,
                                                   delay_matrix, bandwidth_matrix,
                                                   h_dim=12288, b_type=2) -> float:
    send_scatter_time = None
    tp_degree2 = len(device_set2)
    chunk_size = batch_size * seq_in * h_dim * b_type / tp_degree2
    for i in device_set1:
        for j in device_set2:
            send_time = delay_matrix[i, j] + batch_size * h_dim * b_type / bandwidth_matrix[i,j]
            scatter_time = 0
            for k in device_set2:
                if j != k:
                    scatter_time += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
            if send_scatter_time is None:
                send_scatter_time = send_time + scatter_time
            else:
                send_scatter_time = max(send_scatter_time, send_time + scatter_time)
    all_gather_time = 0
    for i in device_set2:
        current_step = 0
        for j in device_set2:
            if i != j:
                current_step += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
        all_gather_time = max(all_gather_time, current_step)
    result = send_scatter_time + all_gather_time
    return result


def communication_pipeline_token_step_time_cross_stage(batch_size, device_set1, device_set2,
                                                       delay_matrix, bandwidth_matrix,
                                                       h_dim=12288, b_type=2) -> float:
    send_scatter_time = None
    tp_degree2 = len(device_set2)
    chunk_size = batch_size * h_dim * b_type / tp_degree2
    for i in device_set1:
        for j in device_set2:
            send_time = delay_matrix[i, j] + batch_size * h_dim * b_type / bandwidth_matrix[i, j]
            scatter_time = 0
            for k in device_set2:
                if j != k:
                    scatter_time += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
            if send_scatter_time is None:
                send_scatter_time = send_time + scatter_time
            else:
                send_scatter_time = max(send_scatter_time, send_time + scatter_time)
    all_gather_time = 0
    for i in device_set2:
        current_step = 0
        for j in device_set2:
            if i != j:
                current_step += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
        all_gather_time = max(all_gather_time, current_step)
    result = send_scatter_time + all_gather_time
    return result


def communication_pipeline_token_step_time_cross_stage_last(batch_size, device_set1, device_set2,
                                                            delay_matrix, bandwidth_matrix) -> float:
    send_time = None
    chunk_size = batch_size * 4
    for i in device_set1:
        for j in device_set2:
            if send_time is None:
                send_time = delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j]
            else:
                send_time = min(send_time,  delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
    return send_time


def decode_time(batch_size, seq_in, seq_out, stage_device_sets, stage_partitions, device_info,
                    delay_matrix, bandwidth_matrix,
                    h_dim=12288, b_type=2) -> float:
    token_step_compute_time = 0
    token_step_tp_comm_time = 0
    token_step_pp_comm_time = 0
    stage_num = len(stage_partitions)
    for i in range(stage_num):
        device_set = stage_device_sets[i]
        m_d = device_info[device_set[0]]['memory_bandwidth']
        c_d = device_info[device_set[0]]['flops']
        num_layers = stage_partitions[i]
        compute_time = compute_token_step_time_stage(batch_size, m_d, c_d, num_layers, len(device_set),
                                                     h_dim, b_type)
        print(f"Token step phase stage-<{i}> compute time {compute_time}")
        token_step_compute_time += compute_time
        comm_time = communicate_token_step_time_stage(batch_size, num_layers, device_set, delay_matrix,
                                                      bandwidth_matrix, h_dim, b_type)
        print(f"Token step phase stage-<{i}> comm time {comm_time}")
        token_step_tp_comm_time += comm_time
    for i in range(stage_num):
        if i < stage_num - 1:
            comm_time = communication_pipeline_token_step_time_cross_stage(batch_size, stage_device_sets[i],
                                                                           stage_device_sets[i + 1], delay_matrix,
                                                                           bandwidth_matrix, h_dim, b_type)
            print(f"Token_step phase pipeline stage-<{i}, {i + 1}> comm time {comm_time}")
        else:
            if stage_num != 1:
                comm_time = communication_pipeline_token_step_time_cross_stage_last(batch_size, stage_device_sets[i],
                                                                                    stage_device_sets[0], delay_matrix,
                                                                                    bandwidth_matrix)
                print(f"Token_step phase pipeline stage-<{i}, {0}> comm time {comm_time}")
        token_step_pp_comm_time + comm_time
    print(f"Token step, compute: {token_step_compute_time}, tp comm time: {token_step_tp_comm_time}, "
          f"pp comm time: {token_step_pp_comm_time}")
    
    decode_time = (token_step_compute_time + token_step_tp_comm_time + token_step_pp_comm_time) * seq_out

    return decode_time


class TimeCost:
    def __init__(self, pipeline) -> None:
        global configs
        self.batch_size = configs.batch_size
        self.seq_in = configs.seq_in
        self.seq_out = configs.seq_out

        # each pipeline: [strategy, layer_partition, ]
        self.stage_device_sets, self.stage_partitions,  = pipeline[0], pipeline[1]
        self.device_info = configs.devices
        
        self.configs = configs
        self.tensor_cores, self.comm_bws, self.delay_bws= configs.specs

        self.comm_bws = np.array(self.comm_bws)
        self.delay_bws = np.array(self.delay_bws)


    def evaluate(self, consumed_time):
    
        return  consumed_time

    def pipeline_cost(self, task_type):
        
        if task_type == "prefill":
            consumed_time = self.prefill_time()
        elif task_type == "decode":
            consumed_time = self.decode_time()
        elif task_type == "kv_comm":
            consumed_time = self.kv_cache_comm_time()
        else:
            raise NotImplementedError

        return self.evaluate(consumed_time)


    def prefill_time(self, h_dim=12288, b_type=2) -> float:
        batch_size, seq_in, stage_device_sets, stage_partitions, device_info,\
                        delay_matrix, bandwidth_matrix = self.batch_size, self.seq_in, self.stage_device_sets,\
                            self.stage_partitions, self.device_info, self.delay_bws, self.comm_bws
        
        bandwidth_matrix = np.array(bandwidth_matrix) * 1024**3
        h_dim = self.configs.H
        b_type = self.configs.B_type

        prompt_compute_time = 0
        prompt_tp_comm_time = 0
        prompt_pp_comm_time = 0
        stage_num = len(stage_partitions)
        for i in range(stage_num):
            device_set = stage_device_sets[i]
            m_d = device_info[device_set[0]].memory_bw * 1073741824
            c_d = device_info[device_set[0]].tensor_core * 10 ** 12
            num_layers = stage_partitions[i]
            compute_time = compute_prompt_time_stage(seq_in, batch_size, m_d, c_d, num_layers, len(device_set),
                                                    h_dim, b_type)
            
            prompt_compute_time += compute_time
            comm_time = communicate_prompt_time_stage(seq_in, batch_size, num_layers, device_set, delay_matrix,
                                                            bandwidth_matrix, h_dim, b_type)
            
            prompt_tp_comm_time += comm_time
        for i in range(stage_num-1):
            
            comm_time = communication_pipeline_prompt_time_cross_stage(seq_in, batch_size, stage_device_sets[i],
                                                                    stage_device_sets[i+1], delay_matrix,
                                                                    bandwidth_matrix, h_dim, b_type)
            
            prompt_pp_comm_time += comm_time
        
        
        prefill_time = prompt_compute_time + prompt_tp_comm_time + prompt_pp_comm_time

        return prefill_time
    
    def decode_time(self, h_dim=12288, b_type=2) -> float:

        batch_size, seq_out, stage_device_sets, stage_partitions, device_info,\
                        delay_matrix, bandwidth_matrix = self.batch_size, self.seq_out, self.stage_device_sets,\
                            self.stage_partitions, self.device_info, self.delay_bws, self.comm_bws
        
        bandwidth_matrix = np.array(bandwidth_matrix) * 1024**3
        h_dim = self.configs.H
        b_type = self.configs.B_type

        token_step_compute_time = 0
        token_step_tp_comm_time = 0
        token_step_pp_comm_time = 0
        stage_num = len(stage_partitions)
        for i in range(stage_num):
            device_set = stage_device_sets[i]
            m_d = device_info[device_set[0]].memory_bw * 1073741824
            c_d = device_info[device_set[0]].tensor_core * 10 ** 12
            num_layers = stage_partitions[i]
            compute_time = compute_token_step_time_stage(batch_size, m_d, c_d, num_layers, len(device_set),
                                                        h_dim, b_type)
            
            token_step_compute_time += compute_time
            comm_time = communicate_token_step_time_stage(batch_size, num_layers, device_set, delay_matrix,
                                                        bandwidth_matrix, h_dim, b_type)
            
            token_step_tp_comm_time += comm_time
        for i in range(stage_num):
            if i < stage_num - 1:
                comm_time = communication_pipeline_token_step_time_cross_stage(batch_size, stage_device_sets[i],
                                                                            stage_device_sets[i + 1], delay_matrix,
                                                                            bandwidth_matrix, h_dim, b_type)
                
            else:
                if stage_num != 1:
                    comm_time = communication_pipeline_token_step_time_cross_stage_last(batch_size, stage_device_sets[i],
                                                                                        stage_device_sets[0], delay_matrix,
                                                                                        bandwidth_matrix)
                    
            token_step_pp_comm_time += comm_time
        
        decode_time = (token_step_compute_time + token_step_tp_comm_time + token_step_pp_comm_time) * seq_out

        return decode_time
    
    def kv_cache_comm_time(self, ):

        global configs

        kv_bw = configs.inter_bw
        KV_size = configs.H * configs.H * 2 * configs.L * configs.B_type / 1024 ** 3

        return KV_size / kv_bw
    

def check_oom(optimized_clusters):
    global configs

    estimated_memory = 12 * configs.H ** 2 * configs.B_type / 1024 ** 3  * configs.L

    for cluster_id, devices in optimized_clusters.items():
        
        memory_cumsum = 0
        for device in devices:
            memory_cumsum += configs.devices[device].memory

        if memory_cumsum < estimated_memory:

            return cluster_id
    
    return None
