"""Match diagnostic native logits and exact greedy outputs to the FP16 oracle."""
import argparse,json
from pathlib import Path
import torch
from .record import digest

def check(reference,worker,client,atol,rtol):
    expected={r['id']:r for r in json.loads((reference/'results.json').read_text())};observed={}
    for path in worker.rglob('audit-*.pt'):
        record=torch.load(path,map_location='cpu',weights_only=True)
        for i,rid in enumerate(record['context']['ids']):
            key=(rid,record['sequence_offset'])
            if key in observed:raise ValueError('Duplicate native audit observation: '+str(key))
            observed[key]=(record['logits'][i],str(path))
    requests=[json.loads(l) for l in (client/'requests.jsonl').read_text().splitlines()];results=[]
    for row in requests:
        rid=row['id'];case=expected[row['prompt_id']];assert row['success']
        actual=row['server']['token_ids'];assert actual==case['token_ids'],f'Greedy tokens differ: {rid}'
        logits=torch.load(reference/case['logits_file'],map_location='cpu',weights_only=True)
        errors=[]
        for j,ref in enumerate(logits):
            offset=0 if j==0 else len(case['input_ids'])+j-1
            got,path=observed[(rid,offset)];torch.testing.assert_close(got,ref,atol=atol,rtol=rtol)
            errors.append({'offset':offset,'max_absolute_error':float((got-ref).abs().max()),'rms_error':float((got-ref).square().mean().sqrt()),'native_file':path})
        results.append({'id':rid,'reference_id':case['id'],'exact_greedy_tokens':True,'steps':errors})
    if not results:raise ValueError('No correctness requests found')
    return {'passed':True,'requests':len(results),'atol':atol,'rtol':rtol,'reference_results_sha256':digest(reference/'results.json'),'client_results_sha256':digest(client/'requests.jsonl'),'results':results}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--client',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--atol',type=float,default=.15);p.add_argument('--rtol',type=float,default=.03);a=p.parse_args()
    result=check(a.reference,a.worker,a.client,a.atol,a.rtol);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'requests':result['requests']}))
