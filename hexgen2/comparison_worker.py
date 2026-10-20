"""Instrumented native PP=2 workers for the four-GPU scaled comparison.

The serving adapter uses bounded, length-homogeneous microbatches. It does not
claim paged attention, arbitrary continuous-batch admission or paper-scale search.
"""
import argparse,collections,gc,hashlib,json,os,queue,threading,time,traceback
from datetime import timedelta
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import requests
import torch
import torch.distributed as dist
from .comparison_model import NativeStage
from .transport import send_packet,recv_packet
from . import fa_adapter
from .record import digest
from .comparison_fence import LaunchFence
from .nccl_runtime import runtime_version,mapped_libraries


def fingerprints(tensors):return {k:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for k,t in tensors.items()}

def stamp():return {'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns()}

class Journal:
    def __init__(self,path):self.path=path;self.lock=threading.Lock()
    def write(self,event,**data):
        with self.lock,self.path.open('a') as f:f.write(json.dumps({'event':event,**data,**stamp()},allow_nan=False)+'\n');f.flush()

class Service:
    def __init__(self,args,model,pair,control,cross,readiness,ordering):
        self.args,self.model,self.pair,self.control,self.cross=args,model,pair,control,cross
        self.rank=int(os.environ['RANK']);self.head=self.rank%2==0;self.group=self.rank//2
        self.clock_domain={'hostname':os.uname().nodename,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        self.head_rank=self.group*2;self.tail_rank=self.head_rank+1
        self.peer=self.rank+2 if self.group==0 else self.rank-2
        self.readiness=readiness
        self.ordering=ordering;self.fence=LaunchFence();self.compute_epoch=0;self.transfer_epoch=0
        self.output=args.output;self.journal=Journal(args.output/f'rank-{self.rank}-events.jsonl')
        self.jobs=queue.Queue();self.pending={};self.pending_lock=threading.Lock();self.stash=collections.deque()
        # One active decode plus one prefetched prefix. Synchronize only the
        # compute stream: a device-wide sync would wait for the next blocked recv.
        self.send_credit=threading.Semaphore(2);self.outbox=queue.Queue(maxsize=1);self.inbox=queue.Queue(maxsize=1);self.credit=threading.Semaphore(2)
        self.feedback=queue.Queue();self.threads=[];self.errors=[];self.ready=False;self.stopping=False
        if args.audit:
            model.audit_directory=args.output/f'audit-rank-{self.rank}';model.audit_directory.mkdir(exist_ok=False)
        libraries=mapped_libraries()
        self.journal.write('model_loaded',rank=self.rank,source=model.source,gpu={'name':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'module_loading':os.environ.get('CUDA_MODULE_LOADING'),'allocator':torch.cuda.get_allocator_backend(),'nccl_runtime':list(runtime_version()),'nccl_torch_build':list(torch.cuda.nccl.version()),'nccl_libraries':[{'path':p,'sha256':digest(p)} for p in libraries],'launch_order_implicit':os.environ.get('NCCL_LAUNCH_ORDER_IMPLICIT'),'launch_race_fatal':os.environ.get('NCCL_LAUNCH_RACE_FATAL'),'cache_transport':dist.get_backend(self.cross),'cache_staging':'cpu' if args.kv_transport=='gloo' else 'gpu'},arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})

    def park_compute(self,epoch):
        started=stamp();torch.cuda.current_stream().synchronize()
        self.fence.park(epoch);self.compute_epoch=epoch
        self.journal.write('cache_epoch_model_resumed',rank=self.rank,epoch=epoch,started=started)

    def cooperate(self):
        # Both pipeline stages make the same decision at a completed token
        # boundary. A local flag alone can strand one stage inside the next
        # blocking pipeline operation while the other waits for a cache epoch.
        epoch=torch.tensor([self.fence.pending() if self.args.phase=='decode' else 0],dtype=torch.int64)
        dist.all_reduce(epoch,op=dist.ReduceOp.MAX,group=self.control)
        if epoch.item()>self.compute_epoch:self.park_compute(epoch.item())

    def begin_transfer_epoch(self):
        self.transfer_epoch+=1;epoch=self.transfer_epoch
        self.fence.request(epoch);self.fence.wait_parked(epoch)
        # A dedicated Gloo group is used only by the four transport owners.
        # Every preceding pipeline operation is GPU-complete on all ranks.
        dist.barrier(group=self.ordering)
        self.journal.write('cache_epoch_begin',rank=self.rank,epoch=epoch)
        return epoch

    def release_transfer_epoch(self,epoch):
        # Every rank has enqueued the complete cache packet before any rank
        # may enqueue its next pipeline operation. NCCL 2.26+ then preserves
        # that host launch order across the distinct CUDA streams/communicators.
        dist.barrier(group=self.ordering)
        self.journal.write('cache_epoch_enqueued',rank=self.rank,epoch=epoch)
        self.fence.release(epoch)

    def next_decode(self):
        sentinel=object();item=sentinel
        while True:
            self.cooperate()
            if item is sentinel:
                try:item=self.inbox.get_nowait()
                except queue.Empty:pass
            available=torch.tensor([int(item is not sentinel)],dtype=torch.int64)
            dist.all_reduce(available,op=dist.ReduceOp.MIN,group=self.control)
            if available.item():return item
            time.sleep(.002)

    def exchange_ready(self,sending):
        # Never leave an idle NCCL receive kernel spinning on the GPU. CUDA
        # module loading/allocation may require context synchronization.
        token=torch.ones(1,dtype=torch.int64)
        if sending:
            dist.send(token,dst=self.peer,group=self.readiness);dist.recv(token,src=self.peer,group=self.readiness)
        else:
            dist.recv(token,src=self.peer,group=self.readiness);dist.send(token,dst=self.peer,group=self.readiness)
        if token.item()!=1:raise ValueError('Invalid cache readiness handshake')

    def thread(self,name,fn):
        def guarded():
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            try:fn()
            except BaseException as e:
                self.errors.append(repr(e));self.journal.write('thread_failure',thread=name,error=repr(e),traceback=traceback.format_exc())
                # A failed transport cannot leave an apparently healthy service waiting forever.
                os._exit(3)
        t=threading.Thread(target=guarded,name=name,daemon=True);t.start();self.threads.append(t)

    def sender(self):
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            while True:
                item=self.outbox.get()
                waiting=stamp();self.exchange_ready(True)
                epoch=self.begin_transfer_epoch() if self.args.kv_transport=='nccl' else None
                started=stamp()
                if item is None:
                    send_packet({}, {'stop':True}, self.peer,group=self.cross)
                    if epoch is not None:self.release_transfer_epoch(epoch)
                    stream.synchronize();return
                tensors,meta,event=item;stream.wait_event(event)
                if self.args.audit:meta['sender_payload_sha256']=fingerprints(tensors)
                payload=tensors
                staging_ms=0.
                if self.args.kv_transport=='gloo':
                    # Retain GPU sources through the asynchronous copies. Network
                    # transport receives only exact, initialized FP16 CPU bytes.
                    payload={key:torch.empty_like(tensor,device='cpu',pin_memory=True) for key,tensor in tensors.items()}
                    for key,tensor in tensors.items():payload[key].copy_(tensor,non_blocking=True)
                    stream.synchronize();staging_ms=(stamp()['monotonic_ns']-started['monotonic_ns'])/1e6
                size=send_packet(payload,meta,self.peer,group=self.cross)
                if epoch is not None:self.release_transfer_epoch(epoch)
                stream.synchronize()
                self.journal.write('kv_send_complete',rank=self.rank,ids=meta['ids'],payload_bytes=size,started=started,readiness_wait_ms=(started['monotonic_ns']-waiting['monotonic_ns'])/1e6,payload_sha256=meta.get('sender_payload_sha256'),backend=self.args.kv_transport,staging_ms=staging_ms)
                del payload
                del tensors,item;self.send_credit.release();self.outbox.task_done()

    def receiver(self):
        if self.args.kv_transport=='gloo':
            while True:
                self.credit.acquire()
                waiting=stamp();self.exchange_ready(False);started=stamp()
                tensors,meta=recv_packet(self.peer,group=self.cross)
                if meta.get('stop'):self.inbox.put(None);return
                if self.args.audit:
                    actual=fingerprints(tensors)
                    if actual!=meta.get('sender_payload_sha256'):raise ValueError('KV transfer SHA256 mismatch')
                    self.journal.write('kv_fingerprints_verified',rank=self.rank,ids=meta['ids'],payload_sha256=actual)
                size=sum(t.numel()*t.element_size() for t in tensors.values())
                meta['receiver']={'rank':self.rank,'started':started,'completed':stamp(),'payload_bytes':size,'readiness_wait_ms':(started['monotonic_ns']-waiting['monotonic_ns'])/1e6,'backend':'gloo','staging':'cpu'}
                self.journal.write('kv_receive_complete',**meta['receiver'],ids=meta['ids'])
                self.inbox.put((tensors,meta,None))
            return
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            while True:
                self.credit.acquire()
                waiting=stamp();self.exchange_ready(False);epoch=self.begin_transfer_epoch();started=stamp()
                tensors,meta=recv_packet(self.peer,group=self.cross)
                self.release_transfer_epoch(epoch);stream.synchronize()
                if meta.get('stop'):
                    self.inbox.put(None);return
                if self.args.audit:
                    actual=fingerprints(tensors)
                    if actual!=meta.get('sender_payload_sha256'):raise ValueError('KV transfer SHA256 mismatch')
                    self.journal.write('kv_fingerprints_verified',rank=self.rank,ids=meta['ids'],payload_sha256=actual)
                event=torch.cuda.Event();event.record(stream)
                size=sum(t.numel()*t.element_size() for t in tensors.values())
                meta['receiver']={'rank':self.rank,'started':started,'completed':stamp(),'payload_bytes':size,'readiness_wait_ms':(started['monotonic_ns']-waiting['monotonic_ns'])/1e6}
                self.journal.write('kv_receive_complete',**meta['receiver'],ids=meta['ids'])
                self.inbox.put((tensors,meta,event))

    def deliver_feedback(self):
        session=requests.Session();session.trust_env=False
        while True:
            item=self.feedback.get()
            if item is None:return
            url,data=item
            response=session.post(url+'/finish',json=data,timeout=120);response.raise_for_status()
            self.journal.write('feedback_delivered',ids=[r['id'] for r in data['results']]);self.feedback.task_done()

    def finish(self,results):
        with self.pending_lock:
            for result in results:
                result['completion_ready']=stamp()
                job=self.pending[result['id']];job['result']=result;job['done'].set()
        for result in results:self.journal.write('request_complete',**result)

    def take_batch(self):
        first=self.stash.popleft() if self.stash else self.jobs.get()
        if first is None:return None
        key=(len(first['input_ids']),first['max_new_tokens']);batch=[first];deadline=time.monotonic()+self.args.batch_wait_ms/1000
        while len(batch)<self.args.max_batch:
            timeout=deadline-time.monotonic()
            if timeout<=0:break
            try:item=self.jobs.get(timeout=timeout)
            except queue.Empty:break
            if item is None:self.stash.appendleft(None);break
            if (len(item['input_ids']),item['max_new_tokens'])==key:batch.append(item)
            else:self.stash.append(item)
        return batch

    def infer(self,meta,tensors=None,event=None):
        ids=meta['ids'];count=meta['max_new_tokens'];length=meta['input_tokens'];batch=len(ids)
        torch.cuda.reset_peak_memory_stats();calls=dict(fa_adapter.CALLS);queued=stamp();token_stamps=[]
        self.model.audit_context={'ids':ids,'input_tokens':length}
        if tensors is None and self.args.phase=='prefill':self.send_credit.acquire()
        started=stamp();restore_ms=0.
        def callback(token):
            torch.cuda.current_stream().synchronize();token_stamps.append(stamp())
            if len(token_stamps)%32==0:self.journal.write('decode_progress',rank=self.rank,ids=ids,tokens_observed=len(token_stamps))
            # Deliver an already-computed final token before admitting the
            # next cache epoch; the idle loop will perform the next handoff.
            if len(token_stamps)<count:self.cooperate()
        if tensors is None:
            inputs=torch.tensor(meta['input_ids'],device='cuda',dtype=torch.long)
            first,state,logits=self.model.prefill(inputs,count,self.pair,self.head_rank,self.tail_rank,prefill_only=self.args.phase=='prefill')
            torch.cuda.current_stream().synchronize();prefill_end=stamp()
            if self.args.phase=='prefill':
                live={str(i):v[:,:length].contiguous() for i,v in state.key_value_memory_dict.items()}
                live['first']=first.clone();event=torch.cuda.Event();event.record()
                transferred=dict(meta);transferred['prefill']={'rank':self.rank,'started':started,'ended':prefill_end,'duration_ms':(prefill_end['monotonic_ns']-started['monotonic_ns'])/1e6,'transfer_credit_wait_ms':(started['monotonic_ns']-queued['monotonic_ns'])/1e6,'cache_capacity_tokens':state.max_sequence_len,'cache_storage_bytes':sum(v.untyped_storage().nbytes() for v in state.key_value_memory_dict.values())}
                del state,inputs,logits
                self.outbox.put((live,transferred,event))
                self.journal.write('prefill_complete',ids=ids,batch=batch,input_tokens=length,output_tokens=count,peak_allocated_bytes=torch.cuda.max_memory_allocated(),attention_calls={k:v-calls.get(k,0) for k,v in fa_adapter.CALLS.items()},**transferred['prefill'])
                if self.args.kv_transport=='nccl':self.park_compute(self.compute_epoch+1)
                return
        else:
            if event is not None:torch.cuda.current_stream().wait_event(event)
            restoring=stamp();first,state=self.model.restore(tensors,length,count);del tensors
            torch.cuda.current_stream().synchronize()
            restore_ms=(stamp()['monotonic_ns']-restoring['monotonic_ns'])/1e6
            prefill_end=None
        tokens=self.model.decode(first,state,count,self.pair,self.head_rank,self.tail_rank,callback)
        torch.cuda.current_stream().synchronize();ended=stamp()
        details={'rank':self.rank,'group':self.group,'batch':batch,'started':started,'ended':ended,
            'clock_domain':self.clock_domain,
            'duration_ms':(ended['monotonic_ns']-started['monotonic_ns'])/1e6,'cache_restore_ms':restore_ms,'token_observations':token_stamps,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'attention_calls':{k:v-calls.get(k,0) for k,v in fa_adapter.CALLS.items()}}
        self.journal.write('generation_complete',ids=ids,token_ids=tokens,**details)
        if self.head:
            results=[{'id':rid,'status':'success','input_tokens':length,'output_tokens':len(seq),'token_ids':seq,
                      'native':details,'prefill':meta.get('prefill',{'started':started,'ended':prefill_end}),
                      'receiver':meta.get('receiver'),'method':self.args.method,'phase':self.args.phase} for rid,seq in zip(ids,tokens)]
            if any(len(r['token_ids'])!=count for r in results):raise ValueError('Exact output count failed')
            if self.args.phase=='decode':self.feedback.put((meta['callback'],{'results':results}))
            else:self.finish(results)
        del state,first,tokens
        if self.args.phase=='decode':self.credit.release()

    def loop(self):
        if self.args.phase=='prefill':self.thread('kv_sender',self.sender)
        elif self.args.phase=='decode':self.thread('kv_receiver',self.receiver)
        if self.head and self.args.phase=='decode':self.thread('completion_feedback',self.deliver_feedback)
        self.ready=True;self.journal.write('ready',rank=self.rank,phase=self.args.phase)
        while True:
            item=None
            if self.args.phase=='decode':item=self.next_decode()
            if self.head:
                if self.args.phase=='decode':
                    meta=None if item is None else {k:v for k,v in item[1].items() if k!='input_ids'}
                else:
                    batch=self.take_batch()
                    meta=None if batch is None else {'ids':[r['id'] for r in batch],'input_ids':[r['input_ids'] for r in batch],
                        'input_tokens':len(batch[0]['input_ids']),'max_new_tokens':batch[0]['max_new_tokens'],
                        'callback':f'http://{self.args.bind}:{self.args.port}'}
            else:meta=None
            objects=[meta];dist.broadcast_object_list(objects,src=self.head_rank,group=self.control);meta=objects[0]
            if meta is None:break
            if self.args.phase=='decode':
                if item is None or item[1]['ids']!=meta['ids']:raise ValueError('Stage request/cache correlation failed')
                tensors,local_meta,event=item
                self.infer(local_meta,tensors,event)
                del item,tensors
            else:self.infer(meta)
        if self.args.phase=='prefill':
            self.outbox.put(None)
            if self.args.kv_transport=='nccl':self.park_compute(self.compute_epoch+1)
        for t in self.threads:
            if t.name=='completion_feedback':self.feedback.join();self.feedback.put(None)
            t.join(timeout=90)
            if t.is_alive():raise TimeoutError('Worker thread did not stop: '+t.name)
        self.journal.write('complete',rank=self.rank)

    def handler(self):
        service=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def reply(self,status,data):
                raw=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);self.wfile.flush()
            def do_GET(self):
                if self.path=='/health':self.reply(200,{'ready':service.ready,'rank':service.rank,'phase':service.args.phase,'errors':service.errors,'pending':len(service.pending)})
                else:self.reply(404,{'error':'unknown path'})
            def do_POST(self):
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<2*1024*1024:raise ValueError('Invalid request size')
                    data=json.loads(self.rfile.read(size))
                    if self.path=='/finish':service.finish(data['results']);self.reply(200,{'ok':True});return
                    if self.path=='/shutdown':service.jobs.put(None);self.reply(200,{'ok':True});return
                    if self.path!='/generate' or service.args.phase=='decode':self.reply(404,{'error':'endpoint unavailable'});return
                    if not isinstance(data.get('id'),str) or not data['id'] or len(data['id'])>150:raise ValueError('Invalid request id')
                    values=data['input_ids'];count=data['max_new_tokens']
                    if not values or any(type(v) is not int or not 0<=v<service.model.hf.vocab_size for v in values):raise ValueError('Invalid token ids')
                    if type(count) is not int or count<1 or len(values)+count>service.model.hf.max_position_embeddings:raise ValueError('Context/output bound')
                    with service.pending_lock:
                        if data['id'] in service.pending:raise ValueError('Duplicate id')
                        if sum(not j['done'].is_set() for j in service.pending.values())>=service.args.max_pending:self.reply(429,{'error':'bounded pending queue full'});return
                        job=dict(data,done=threading.Event(),received=stamp());service.pending[data['id']]=job
                    service.journal.write('request_received',id=data['id'],input_ids=values,max_new_tokens=count,received=job['received'])
                    service.jobs.put(job)
                    if not job['done'].wait(service.args.request_timeout):self.reply(504,{'id':data['id'],'status':'timeout'});return
                    result=dict(job['result'],received=job['received'],response_started=stamp(),response_clock_domain=getattr(service,'clock_domain',None))
                    self.reply(200,result);sent=stamp()
                    timing={'completion_ready_to_response_sent_ms':(sent['monotonic_ns']-result['completion_ready']['monotonic_ns'])/1e6}
                    if result['native']['rank']==service.rank:
                        timing['generation_end_to_response_sent_ms']=(sent['monotonic_ns']-result['native']['ended']['monotonic_ns'])/1e6
                    service.journal.write('response_sent',id=data['id'],response_started=result['response_started'],response_flushed=sent,**timing)
                except BaseException as e:
                    service.journal.write('http_failure',error=repr(e),traceback=traceback.format_exc())
                    try:self.reply(400,{'error':repr(e)})
                    except OSError:pass
        return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',action='store_true');p.add_argument('--model-dir',required=True);p.add_argument('--baseline-root',required=True)
    p.add_argument('--kv-transport',choices=['nccl','gloo'],default='nccl')
    p.add_argument('--method',choices=['hexgen','hexgen2'],required=True);p.add_argument('--phase',choices=['all','prefill','decode'],required=True)
    p.add_argument('--bind',required=True);p.add_argument('--port',type=int,default=8120);p.add_argument('--max-batch',type=int,default=2)
    p.add_argument('--batch-wait-ms',type=float,default=10);p.add_argument('--max-pending',type=int,default=512);p.add_argument('--request-timeout',type=float,default=1800)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # JSON event arguments remain portable string values.
    args.output=str(args.output)
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device);torch.set_num_threads(1)
    if runtime_version()<(2,26,0) or os.environ.get('NCCL_LAUNCH_ORDER_IMPLICIT')!='1':
        raise RuntimeError('Ordered comparison requires NCCL >=2.26 and NCCL_LAUNCH_ORDER_IMPLICIT=1')
    torch.manual_seed(123);torch.cuda.manual_seed_all(123);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=1800))
    controls=[dist.new_group(r,backend='gloo',timeout=timedelta(seconds=1800)) for r in [[0,1],[2,3]]]
    pairs=[dist.new_group(r,timeout=timedelta(seconds=1800),group_desc='pipeline-'+str(i)) for i,r in enumerate([[0,1],[2,3]])]
    links=[dist.new_group(r,backend=args.kv_transport,timeout=timedelta(seconds=1800),group_desc='cache-'+str(i)) for i,r in enumerate([[0,2],[1,3]])]
    readiness=[dist.new_group(r,backend='gloo',timeout=timedelta(seconds=1800)) for r in [[0,2],[1,3]]]
    ordering=dist.new_group([0,1,2,3],backend='gloo',timeout=timedelta(seconds=1800))
    final=dist.new_group([0,1,2,3],backend='gloo',timeout=timedelta(seconds=1800))
    model=NativeStage(args.model_dir,rank%2,args.method,args.baseline_root,max_batch=args.max_batch)
    identity={'rank':rank,'method':args.method,'stage':rank%2,'config':digest(Path(args.model_dir)/'config.json'),'origin':digest(Path(args.model_dir)/'verified-source.json')}
    identities=[None]*4;dist.all_gather_object(identities,identity,group=final)
    if any(i['method']!=args.method or i['config']!=identity['config'] or i['origin']!=identity['origin'] for i in identities):raise ValueError('Worker model/method identity mismatch')
    args.output=Path(args.output);service=Service(args,model,pairs[rank//2],controls[rank//2],links[rank%2],readiness[rank%2],ordering)
    service.journal.write('worker_identity_handshake',identities=identities)
    server=None
    if rank%2==0:
        server=ThreadingHTTPServer((args.bind,args.port),service.handler());server.daemon_threads=True
        threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        service.loop();dist.barrier(group=final)
    except BaseException as e:
        service.journal.write('failure',rank=rank,error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if server:server.shutdown();server.server_close()
        dist.destroy_process_group()
if __name__=='__main__':main()
