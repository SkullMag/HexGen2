"""Freeze Azure joint-length workloads with explicitly constructed public token prompts.

Azure publishes lengths/times, not prompt text. Inputs deterministically pack the
previous experiment's pinned public LMSYS token bank. No prefix caching is used.
"""
import argparse,csv,collections,hashlib,json,random
from datetime import datetime
from pathlib import Path
from .record import digest
CLASSES=['HPLD','HPHD','LPHD','LPLD']
def category(n,m):return ('HP' if n>512 else 'LP')+('HD' if m>128 else 'LD')
def write_rows(path,rows):path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
def build(azure,bank_path,origin,output):
    output.mkdir(parents=True,exist_ok=False);bank=json.loads(bank_path.read_text());prompts=bank['prompts']
    all_rows=list(csv.DictReader(azure.open()));eligible=[];excluded=collections.Counter()
    for index,row in enumerate(all_rows):
        n,m=int(row['ContextTokens']),int(row['GeneratedTokens']);reason=None
        if n<1 or m<1:reason='nonpositive_length'
        elif n>1536:reason='context_above_1536'
        elif m>512:reason='output_above_512'
        elif n+m>2048:reason='total_above_2048'
        if reason:excluded[reason]+=1;continue
        eligible.append({'source_row':index,'input_tokens':n,'max_new_tokens':m,'timestamp':row['TIMESTAMP'],'class':category(n,m)})
    def construct(row,seed,j):
        rng=random.Random(f'{seed}:{j}:{row["source_row"]}');ids=[1];segments=[]
        while len(ids)<row['input_tokens']:
            p=prompts[rng.randrange(len(prompts))];tail=p['input_ids'][1:];take=min(len(tail),row['input_tokens']-len(ids))
            segments.append({'prompt_id':p['prompt_id'],'source_row_index':p['source_row_index'],'token_start':1,'token_count':take});ids.extend(tail[:take])
        return dict(row,id=f'azure-{row["source_row"]}-s{seed}-p{j}',input_ids=ids,prompt_segments=segments)
    # A row-level hash split keeps calibration inputs separate from held-out inputs.
    calibration=[r for r in eligible if int(hashlib.sha256(str(r['source_row']).encode()).hexdigest()[:8],16)%10==0]
    heldout=[r for r in eligible if r not in calibration]
    for seed in [20261020,20261021,20261022]:
        rng=random.Random(seed)
        for name in CLASSES:
            choices=[r for r in heldout if r['class']==name];selected=rng.sample(choices,min(128,len(choices)))
            write_rows(output/f'{name}-{seed}.jsonl',[construct(r,seed,j) for j,r in enumerate(selected)])
        # A contiguous retained trace segment preserves its input/output correlation
        # and the order of Azure arrival gaps; rate scaling happens after calibration.
        begin=rng.randrange(max(1,len(heldout)-1024));selected=heldout[begin:begin+1024]
        rows=[construct(r,seed,j) for j,r in enumerate(selected)];write_rows(output/f'Online-{seed}.jsonl',rows)
        times=[datetime.fromisoformat(r['timestamp'][:26]).timestamp() for r in rows]
        (output/f'online-gaps-{seed}.json').write_text(json.dumps({'source_row_indices':[r['source_row'] for r in rows],'gaps_s':[max(.000001,b-a) for a,b in zip(times,times[1:])]},indent=2)+'\n')
    # Representative anchors isolate phase, batching, and memory effects. These are
    # calibration/diagnostic only; the measured figure uses the Azure samples above.
    anchors=[]
    for j,(name,n,m) in enumerate([('HPLD',1024,32),('HPHD',1024,256),('LPHD',128,256),('LPLD',128,32)]):
        row=dict(source_row=-j-1,input_tokens=n,max_new_tokens=m,timestamp=None,**{'class':name});anchors.append(construct(row,991,j))
    write_rows(output/'anchors.jsonl',anchors)
    write_rows(output/'calibration-mixed.jsonl',[construct(r,991,j) for j,r in enumerate(random.Random(991).sample(calibration,32))])
    qualification=[]
    for j,n in enumerate([8,31,65]):
        row=dict(source_row=-100-j,input_tokens=n,max_new_tokens=4,timestamp=None,**{'class':'qualification'})
        qualification.append(construct(row,917,j))
    write_rows(output/'qualification.jsonl',qualification);(output/'qualification-cases.json').write_text(json.dumps(qualification,indent=2)+'\n')
    metadata={'azure_origin':json.loads(origin.read_text()),'azure_rows':len(all_rows),'eligible_rows':len(eligible),'exclusions':dict(excluded),
        'retained_classes':dict(collections.Counter(r['class'] for r in eligible)),'calibration_rows':len(calibration),'heldout_rows':len(heldout),
        'bounds':{'input_min':1,'input_max':1536,'output_min':1,'output_max':512,'total_max':2048},
        'selection':'SHA256(row index) modulo 10 == 0 is calibration; other rows held out; three fixed seeds',
        'prompt_construction':'One BOS then deterministically selected public LMSYS prompt token segments, removing subsequent BOS tokens; exactly Azure input length; no prefix cache',
        'source_bank_sha256':digest(bank_path),'source_bank_metadata':{k:v for k,v in bank.items() if k!='prompts'},'generator_sha256':digest(__file__),
        'files':{p.name:digest(p) for p in sorted(output.iterdir())}}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:metadata[k] for k in ['azure_rows','eligible_rows','retained_classes','exclusions']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--azure',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--origin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.azure,a.bank,a.origin,a.output)
