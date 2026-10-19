import numpy as np
from typing import List
from utils import *
from device import Device
from config import Config

np.random.seed(10)

configs = Config()

def update_configs(configs):
    machines = create_machines_list()

    devices: List[Device]  = create_devices(machines)

    device_machine_map = create_device_machine_map(devices)
    tensor_cores, comm_bws, delay_bws, comm_bws_dict = create_specs(devices, configs.inter_bw)

    configs.specs = [tensor_cores, comm_bws, delay_bws]
    configs.device_machine_map = device_machine_map
    configs.devices = devices
    configs.comm_bws_dict = comm_bws_dict

    print("Scheduling Input Log============================================================")
    print("Machines:")
    for machine in machines:
        print(f"    *: {machine}")
    print("Total GPUs:", len(devices))
    print(f"Inter Bandwidth: {configs.inter_bw} GB/s", )

    print("=" * 80)

update_configs(configs)



