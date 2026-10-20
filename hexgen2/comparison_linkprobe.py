"""Lossless pairwise NCCL data-plane measurements, outside benchmark windows."""
import argparse,json,os,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300));rows=[]
    def barrier():dist.barrier(device_ids=[device]);torch.cuda.synchronize()
    for left,right in [(0,1),(2,3),(0,2),(0,3),(1,2),(1,3)]:
        for source,target in [(left,right),(right,left)]:
            for size in [1024,1<<20,32<<20]:
                for repeat in range(3):
                    tensor=torch.full((size,),17,device='cuda',dtype=torch.uint8) if rank==source else torch.empty(size,device='cuda',dtype=torch.uint8) if rank==target else None
                    barrier();started=time.perf_counter_ns()
                    if rank==source:dist.send(tensor,dst=target)
                    elif rank==target:dist.recv(tensor,src=source)
                    torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-started)/1e9
                    if rank in [source,target]:
                        if rank==target and not bool((tensor==17).all()):raise ValueError('Lossless link verification failed')
                        row={'rank':rank,'source':source,'target':target,'bytes':size,'repeat':repeat,'seconds':elapsed,'gbps':size*8/elapsed/1e9,'warmup':size==1024,'utc_ns':time.time_ns()}
                        rows.append(row)
                        with (a.output/f'rank-{rank}-linkprobe.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    del tensor
    barrier();(a.output/f'rank-{rank}-linkprobe-complete.json').write_text(json.dumps({'complete':True,'rows':len(rows),'gpu':torch.cuda.get_device_name(),'rank':rank})+'\n');dist.destroy_process_group()
if __name__=='__main__':main()
