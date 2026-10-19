from device import Device
import json
from config import get_args
from typing import Dict


def create_machines_list():
    args = get_args()
    # hetero experiment
    with open(args.machine_config_path, 'r') as machine_config_file:
        machine_config = json.load(machine_config_file)
    machine_specs = machine_config['machine_specs']


    # sublist is in the same type, sublist has three number, indicating n_same_machine of 2,4,8 gpus
    machine_amounts = machine_config['machine_amounts']

    machines = []
    for name, machine_amount in machine_amounts.items():
        spec = machine_specs[name]
        for ngpu, n in machine_amount.items():
            if n == 0:
                continue
            machines.append({"name": name, "tensor_core": spec["flops"], 
                             "memory": spec["memory_limit"], 
                             "intra_bw": spec["intra_bw"], 
                             "memory_bw": spec["memory_bw"],
                             "ngpus": int(ngpu), 
                             "n_same_machine":  n})

    return machines


def create_specs(devices, inter_bw):
    
    tensor_cores = []   # (n, )
    for device in devices:
        tensor_cores.append(device.tensor_core )

    
    comm_bws_dict = {}
    for i in range(len(devices)):
        for j in range(i+1, len(devices)):
            if i != j:
                bw = inter_bw if devices[i].machine_id != devices[j].machine_id else devices[i].intra_bw 
                bw = bw * 8
                comm_bws_dict[i, j] = bw
        
        
    comm_bws = []   # (n, n)
    delay_bws = []
    for i in range(len(devices)):
        comm_bw = []
        delay_bw = []
        for j in range(len(devices)):
            bw = inter_bw if devices[i].machine_id != devices[j].machine_id else devices[i].intra_bw 
            bw = bw * 8

            comm_bw.append(bw)
            delay_bw.append(0)
        comm_bws.append(comm_bw)
        delay_bws.append(delay_bw)


    return tensor_cores, comm_bws, delay_bws, comm_bws_dict

def create_device_machine_map(devices):
    machine_ids = [d.machine_id for d in devices]
    return machine_ids


def create_devices(machines):
    devices = []

    assigned_id = 0
    for machine in machines:
        for i in range(machine['n_same_machine']):
            for j in range(machine['ngpus']):
                devices.append(Device(name=machine['name'], machine_id=assigned_id, 
                                  tensor_core=machine['tensor_core'], intra_bw=machine['intra_bw'], memory_bw=machine['memory_bw'],
                                  memory=machine['memory'], device_id=j + 1, machine_ngpus=machine['ngpus']))
            assigned_id += 1
    return devices

def print_fn(meta_infos, indent=0):
    if not isinstance(meta_infos, Dict):
        print(" " * indent, meta_infos)
    else:
        for key, val in meta_infos.items():
            if not isinstance(val, Dict):
                print(" "* indent, f"{key}: {val}", )
            else:
                print_fn(val, indent + 4)