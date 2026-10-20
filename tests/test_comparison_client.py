"""Exercise real HTTP completion, fixed-window censoring, and timeout capture."""
import argparse,asyncio,hashlib,json,tempfile,unittest
from pathlib import Path
from aiohttp import web
from hexgen2.comparison_client import execute

class ClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.trace=self.root/'trace.jsonl';self.trace.write_text(''.join(json.dumps({'id':f'p{i}','input_ids':[1,3,4],'max_new_tokens':4})+'\n' for i in range(2)))
        async def handler(request):
            data=await request.json();await asyncio.sleep(.04)
            return web.json_response({'id':data['id'],'status':'success','input_tokens':len(data['input_ids']),'token_ids':[5]*data['max_new_tokens']})
        app=web.Application();app.router.add_post('/generate',handler);self.runner=web.AppRunner(app);await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start();self.endpoint=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    async def asyncTearDown(self):await self.runner.cleanup();self.tmp.cleanup()
    def args(self,mode):
        return argparse.Namespace(output=self.root/'result',trace=self.trace,arrivals=None,routing=None,inflight=2,request_timeout=2,method='test',mode=mode,duration=.01,endpoint=[self.endpoint],run_id='http-test',drain_timeout=2)
    async def test_finite_diagnostic_has_no_fixed_window_rate(self):
        args=self.args('finite');await execute(args);s=json.loads((args.output/'summary.json').read_text())
        self.assertEqual((s['offered'],s['successful'],s['completed_output_tokens']),(2,2,8));self.assertTrue(s['capture_complete'])
        self.assertIsNone(s['throughput_tokens_per_second']);self.assertFalse(s['measurement_window_applicable']);self.assertGreater(s['full_drain_tokens_per_second'],0)
        self.assertEqual(s['trace_sha256'],hashlib.sha256(self.trace.read_bytes()).hexdigest())
    async def test_online_completions_after_window_remain_in_drain(self):
        args=self.args('online');args.arrivals=self.root/'arrivals.json';args.arrivals.write_text(json.dumps({'offsets_s':[0,.005]}));await execute(args)
        s=json.loads((args.output/'summary.json').read_text());self.assertEqual(s['measurement_output_tokens'],0);self.assertEqual(s['throughput_tokens_per_second'],0)
        self.assertEqual((s['offered'],s['successful'],s['completed_output_tokens']),(2,2,8));self.assertTrue(s['capture_complete']);self.assertGreater(s['full_drain_tokens_per_second'],0)
    async def test_http_timeouts_keep_every_offered_outcome(self):
        args=self.args('finite');args.request_timeout=.001
        with self.assertRaises(SystemExit) as e:await execute(args)
        self.assertEqual(e.exception.code,2);s=json.loads((args.output/'summary.json').read_text());self.assertEqual((s['offered'],s['completed'],s['failures']),(2,2,2));self.assertTrue(s['capture_complete'])
        self.assertEqual(len((args.output/'requests.jsonl').read_text().splitlines()),2)
