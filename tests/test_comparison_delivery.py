"""A blocked next inference must not postpone an already-computed HTTP reply."""
import json,queue,tempfile,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import requests
from hexgen2.comparison_worker import Service,Journal,stamp

class DeliveryTest(unittest.TestCase):
    def test_reply_progresses_while_next_serial_inference_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            service=object.__new__(Service)
            service.rank=0;service.jobs=queue.Queue();service.pending={};service.pending_lock=threading.Lock()
            service.journal=Journal(Path(folder)/'events.jsonl')
            service.args=SimpleNamespace(phase='all',max_pending=16,request_timeout=5)
            service.model=SimpleNamespace(hf=SimpleNamespace(vocab_size=10,max_position_embeddings=32))
            release=threading.Event();next_started=threading.Event();order=[]
            server=ThreadingHTTPServer(('127.0.0.1',0),service.handler());server.daemon_threads=True
            server_thread=threading.Thread(target=server.serve_forever);server_thread.start()
            def compute():
                for i in range(2):
                    job=service.jobs.get(timeout=5);order.append(int(job['id']))
                    if i:next_started.set();release.wait(5)
                    service.finish([{'id':job['id'],'status':'success','token_ids':[4],
                        'native':{'rank':0,'ended':stamp()}}])
            worker=threading.Thread(target=compute);worker.start()
            def request(i):
                session=requests.Session();session.trust_env=False
                return session.post(f'http://127.0.0.1:{server.server_port}/generate',
                    json={'id':str(i),'input_ids':[1,2],'max_new_tokens':1},timeout=5)
            try:
                with ThreadPoolExecutor(2) as pool:
                    futures=[pool.submit(request,0),pool.submit(request,1)]
                    # The next inference can start immediately after completion.
                    self.assertTrue(next_started.wait(2))
                    self.assertEqual(futures[order[0]].result(timeout=1).status_code,200)
                    self.assertFalse(futures[order[1]].done())
                    release.set();self.assertEqual(futures[order[1]].result(timeout=2).status_code,200)
            finally:
                release.set();worker.join(5);server.shutdown();server.server_close();server_thread.join(5)
            events=[json.loads(s) for s in (Path(folder)/'events.jsonl').read_text().splitlines()]
            sent=[e for e in events if e['event']=='response_sent']
            self.assertEqual(len(sent),2)
            self.assertTrue(all(e['generation_end_to_response_sent_ms']<1000 for e in sent))
