"""Direct FABRIC data-plane load generator with bounded in-flight work.

Offline maintains a fixed backlog for a fixed window; online replays a frozen
arrival trace. All offered requests, drain outcomes and server records are retained.
"""
import argparse,asyncio,json,time
from .record import digest
from pathlib import Path
import aiohttp

async def execute(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in args.trace.read_text().splitlines() if s.strip()]
    arrivals=json.loads(args.arrivals.read_text()) if args.arrivals else None
    if not rows:raise ValueError('Empty trace')
    routing=json.loads(args.routing.read_text())['endpoint_indices'] if args.routing else list(range(len(args.endpoint)))
    if not routing or any(type(i) is not int or not 0<=i<len(args.endpoint) for i in routing):raise ValueError('Invalid frozen routing')
    results=[];offered=[];origin=time.monotonic();origin_ns=time.time_ns();sequence=0
    semaphore=asyncio.Semaphore(args.inflight)
    timeout=aiohttp.ClientTimeout(total=args.request_timeout)
    connector=aiohttp.TCPConnector(limit=0,force_close=False)
    summary={'method':args.method,'mode':args.mode,'duration_seconds':args.duration,'endpoints':args.endpoint,
             'inflight_limit':args.inflight,'origin_utc_ns':origin_ns,'origin_monotonic':origin,'run_id':args.run_id,
             'trace_sha256':digest(args.trace),'arrivals_sha256':digest(args.arrivals) if args.arrivals else None,
             'routing_sha256':digest(args.routing) if args.routing else None,
             'clock':'Client durations use only this process monotonic clock','client_ttft':None,
             'throughput_definition':'Successful completed output tokens received within the fixed measurement window / window seconds; full drain reported separately'}
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    async with aiohttp.ClientSession(timeout=timeout,connector=connector,trust_env=False) as session:
        async def one(index,endpoint,offset=0,phase='measured'):
            row=rows[index%len(rows)];rid=f'{args.run_id}-{phase}-{index:07}'
            scheduled=origin+offset;await asyncio.sleep(max(0,scheduled-time.monotonic()))
            record={'id':rid,'index':index,'prompt_id':row['id'],'phase':phase,'endpoint':endpoint,
                    'input_tokens':len(row['input_ids']),'requested_output_tokens':row['max_new_tokens'],
                    'scheduled_offset_s':offset,'scheduled_utc_ns':origin_ns+round(offset*1e9),'offered_utc_ns':time.time_ns()}
            offered.append(record)
            with (args.output/'offered.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            try:
                async with semaphore:
                    submitted=time.monotonic();record.update(submitted_utc_ns=time.time_ns(),submission_lag_seconds=submitted-scheduled)
                    async with session.post(endpoint+'/generate',json={'id':rid,'input_ids':row['input_ids'],'max_new_tokens':row['max_new_tokens']}) as response:
                        data=await response.json();record['http_status']=response.status;record['server']=data
                        if response.status!=200 or data.get('status')!='success':raise RuntimeError('Server did not complete request')
                        if len(data['token_ids'])!=row['max_new_tokens'] or data['input_tokens']!=len(row['input_ids']):raise RuntimeError('Token count validation failed')
                        record['success']=True
            except BaseException as error:
                record.update(success=False,error_type=type(error).__name__,error=str(error))
                if isinstance(error,asyncio.CancelledError):record['cancelled']=True
            completed=time.monotonic();record.update(completed_utc_ns=time.time_ns(),completed_offset_s=completed-origin,
                latency_seconds=completed-scheduled,within_measurement_window=completed-origin<=args.duration)
            results.append(dict(record))
            with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            return record
        if args.mode=='finite':
            await asyncio.gather(*(one(i,args.endpoint[routing[i%len(routing)]],0,phase='qualification') for i in range(len(rows))))
        elif args.mode=='offline':
            # Each endpoint has a dedicated share of the fixed global backlog so a
            # slower replica cannot leave a faster baseline replica starved.
            async def lane(lane_id):
                nonlocal sequence
                endpoint=args.endpoint[lane_id%len(args.endpoint)]
                while time.monotonic()<origin+args.duration:
                    index=sequence;sequence+=1
                    await one(index,endpoint,time.monotonic()-origin)
            tasks=[asyncio.create_task(lane(i)) for i in range(args.inflight)]
            try:await asyncio.wait_for(asyncio.gather(*tasks),args.duration+args.drain_timeout)
            except asyncio.TimeoutError:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        else:
            if not arrivals or len(arrivals['offsets_s'])>10000:raise ValueError('Missing or oversized frozen arrivals')
            tasks=[asyncio.create_task(one(i,args.endpoint[routing[i%len(routing)]],offset)) for i,offset in enumerate(arrivals['offsets_s']) if offset<args.duration]
            try:await asyncio.wait_for(asyncio.gather(*tasks),args.duration+args.drain_timeout)
            except asyncio.TimeoutError:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        summary.update(end_utc_ns=time.time_ns(),wall_seconds=time.monotonic()-origin,
            offered=len(offered),completed=len(results),successful=sum(r.get('success',False) for r in results),
            failures=sum(not r.get('success',False) for r in results),
            completed_output_tokens=sum(r['requested_output_tokens'] for r in results if r.get('success')),
            measurement_output_tokens=sum(r['requested_output_tokens'] for r in results if r.get('success') and r['within_measurement_window']))
        summary['throughput_tokens_per_second']=summary['measurement_output_tokens']/args.duration
        summary['full_drain_tokens_per_second']=summary['completed_output_tokens']/summary['wall_seconds']
        summary['capture_complete']=len(offered)==len(results)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        if summary['failures'] or not summary['capture_complete']:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['offline','online','finite'],required=True);p.add_argument('--endpoint',action='append',required=True)
    p.add_argument('--routing',type=Path);p.add_argument('--trace',type=Path,required=True);p.add_argument('--arrivals',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=600);p.add_argument('--inflight',type=int,default=32)
    p.add_argument('--request-timeout',type=float,default=1800);p.add_argument('--drain-timeout',type=float,default=1200)
    a=p.parse_args();asyncio.run(execute(a))
