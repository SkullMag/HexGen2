import threading,time,unittest
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hexgen2.comparison_fence import LaunchFence

def distributed_owner(rank,folder):
    from hexgen2.comparison_worker import Service,Journal
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(seconds=20))
    control=dist.new_group([2,3],backend='gloo',timeout=timedelta(seconds=20))
    ordering=dist.new_group([0,1,2,3],backend='gloo',timeout=timedelta(seconds=20))
    torch.cuda.current_stream=lambda:SimpleNamespace(synchronize=lambda:None)
    service=object.__new__(Service);service.rank=rank;service.args=SimpleNamespace(phase='decode' if rank>=2 else 'prefill')
    service.control=control;service.ordering=ordering;service.fence=LaunchFence(timeout=10)
    service.compute_epoch=0;service.transfer_epoch=0;service.journal=Journal(Path(folder)/f'rank-{rank}.jsonl')
    errors=[]
    def transport():
        try:
            for epoch in range(1,7):
                time.sleep(.002*((rank+epoch)%4))
                actual=service.begin_transfer_epoch();assert actual==epoch
                service.journal.write('mock_packet_enqueued',rank=rank,epoch=epoch)
                service.release_transfer_epoch(epoch)
                # Model progress may resume while GPU completion is pending.
                time.sleep(.005)
        except BaseException as e:errors.append(repr(e))
    thread=threading.Thread(target=transport,daemon=True);thread.start()
    if rank<2:
        for epoch in range(1,7):service.park_compute(epoch)
    else:
        while service.compute_epoch<6:service.cooperate()
    thread.join(timeout=15);assert not thread.is_alive() and not errors,errors
    dist.barrier();dist.destroy_process_group()


class FenceTest(unittest.TestCase):
    def test_all_launches_precede_any_model_resume(self):
        fences=[LaunchFence(timeout=3) for _ in range(4)]
        parked=threading.Barrier(4);enqueued=threading.Barrier(4)
        launched=[];resumed=[];errors=[]
        def transport(rank):
            try:
                for epoch in [1,2]:
                    fence=fences[rank];fence.request(epoch);fence.wait_parked(epoch);parked.wait(timeout=3)
                    time.sleep(.003*rank);launched.append((epoch,rank));enqueued.wait(timeout=3);fence.release(epoch)
            except BaseException as e:errors.append(e)
        def compute(rank):
            try:
                for epoch in [1,2]:
                    fences[rank].park(epoch)
                    resumed.append((epoch,rank,len([r for e,r in launched if e==epoch])))
            except BaseException as e:errors.append(e)
        threads=[threading.Thread(target=fn,args=(rank,)) for fn in [transport,compute] for rank in range(4)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=5)
        self.assertFalse(any(t.is_alive() for t in threads));self.assertEqual(errors,[])
        self.assertEqual(len(resumed),8);self.assertTrue(all(count==4 for _,_,count in resumed))

    def test_launch_handoff_does_not_wait_for_gpu_completion(self):
        fence=LaunchFence(timeout=1);returned=threading.Event();gpu_complete=threading.Event()
        compute=threading.Thread(target=lambda:(fence.park(1),returned.set()))
        compute.start();fence.request(1);fence.wait_parked(1);fence.release(1)
        self.assertTrue(returned.wait(1));self.assertFalse(gpu_complete.is_set());compute.join()

    def test_missing_model_owner_and_skipped_epoch_fail(self):
        fence=LaunchFence(timeout=.01);fence.request(1)
        with self.assertRaises(TimeoutError):fence.wait_parked(1)
        with self.assertRaises(ValueError):fence.release(1)
        with self.assertRaises(ValueError):fence.request(3)

    def test_real_four_rank_idle_and_active_owner_handoffs(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_owner,args=(folder,),nprocs=4,join=True)
            events=[json.loads(line) for p in Path(folder).glob('rank-*.jsonl') for line in p.read_text().splitlines()]
            for epoch in range(1,7):
                self.assertEqual(len([e for e in events if e['event']=='mock_packet_enqueued' and e['epoch']==epoch]),4)
                self.assertEqual(len([e for e in events if e['event']=='cache_epoch_model_resumed' and e['epoch']==epoch]),4)
