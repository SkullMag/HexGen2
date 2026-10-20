"""Launch the heterogeneous four-rank worker placement on one reserved VM."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def placement(path,node):
    rows=json.loads(Path(path).read_text())['ranks']
    if sorted(r['rank'] for r in rows)!=[0,1,2,3]:raise ValueError('Placement must own each rank exactly once')
    if len({(r['node'],r['local_device']) for r in rows})!=4:raise ValueError('GPU device reused by multiple ranks')
    if any(type(r['local_device']) is not int or r['local_device']<0 for r in rows):raise ValueError('Invalid GPU device')
    selected=[(r['rank'],r['local_device'],r['bind']) for r in rows if r['node']==node]
    if not selected:raise ValueError('Node absent from rank placement')
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['gpu0','gpu1','gpu2'],required=True)
    p.add_argument('--task',choices=['worker','linkprobe'],default='worker');p.add_argument('--method',choices=['hexgen','hexgen2'],required=True);p.add_argument('--prefill-group',type=int,choices=[0,1],default=0)
    p.add_argument('--master',default='10.94.10.2');p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--interface',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--baseline-root',default='artifacts/hexgen1')
    p.add_argument('--max-batch',type=int,default=1);p.add_argument('--port',type=int,default=8120)
    p.add_argument('--kv-transport',choices=['nccl','gloo'],default='nccl')
    p.add_argument('--rank-map',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--audit',action='store_true');a=p.parse_args()
    if not a.interface.strip():p.error('An explicit nonempty data-plane interface is required')
    output=a.output or Path(os.environ['HEXGEN2_RUN_DIR']);output.mkdir(parents=True,exist_ok=True)
    ranks=placement(a.rank_map,a.node) if a.rank_map else {'gpu0':[(0,0,'10.94.10.2')],'gpu1':[(1,0,'10.94.10.3')],'gpu2':[(2,0,'10.94.10.4'),(3,1,'10.94.10.4')]}[a.node]
    children=[];logs=[];stopping=False
    def stop(sig=None,frame=None):
        nonlocal stopping
        stopping=True
        for child in children:
            if child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for rank,device,bind in ranks:
            env=os.environ.copy();env.update(RANK=str(rank),LOCAL_RANK=str(device),WORLD_SIZE='4',MASTER_ADDR=a.master,MASTER_PORT=str(a.master_port),
                CUDA_MODULE_LOADING='LAZY',NCCL_LAUNCH_ORDER_IMPLICIT='1',NCCL_LAUNCH_RACE_FATAL='1',GLOO_SOCKET_IFNAME=a.interface,NCCL_SOCKET_IFNAME=a.interface,NCCL_IB_DISABLE='1',NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(output/f'rank-{rank}-nccl-%p.log'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            phase='all' if a.method=='hexgen' else ('prefill' if rank//2==a.prefill_group else 'decode')
            command=[sys.executable,'-u','-m','hexgen2.comparison_worker','--method',a.method,'--phase',phase,'--bind',bind,'--port',str(a.port),
                '--model-dir',a.model_dir,'--baseline-root',a.baseline_root,'--max-batch',str(a.max_batch),'--kv-transport',a.kv_transport,'--output',str(output)]
            if a.audit:command.append('--audit')
            if a.task=='linkprobe':command=[sys.executable,'-u','-m','hexgen2.comparison_linkprobe','--output',str(output)]
            out=(output/f'rank-{rank}-stdout.log').open('wb');err=(output/f'rank-{rank}-stderr.log').open('wb');logs.extend([out,err])
            children.append(subprocess.Popen(command,env=env,stdout=out,stderr=err))
        (output/'launcher.json').write_text(json.dumps({'arguments':vars(a)|{'output':str(output),'rank_map':str(a.rank_map) if a.rank_map else None},'worker_environment':{'CUDA_MODULE_LOADING':'LAZY','NCCL_LAUNCH_ORDER_IMPLICIT':'1','NCCL_LAUNCH_RACE_FATAL':'1','LD_PRELOAD':env.get('LD_PRELOAD')},'ranks':[{'rank':r,'local_device':d,'bind':b,'pid':c.pid} for (r,d,b),c in zip(ranks,children)]},indent=2)+'\n')
        while not all(c.poll() is not None for c in children):
            if stopping or any(c.poll() not in [None,0] for c in children):stop();break
            time.sleep(.2)
        for c in children:
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        returncodes=[c.returncode for c in children]
        (output/'launcher-exit.json').write_text(json.dumps({'returncodes':returncodes,'stopping':stopping})+'\n')
        return 0 if not stopping and all(r==0 for r in returncodes) else 3
    finally:
        stop()
        for f in logs:f.close()
if __name__=='__main__':sys.exit(main())
