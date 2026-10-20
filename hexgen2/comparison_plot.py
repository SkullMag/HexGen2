"""Regenerate the five-panel measured figure from completed controller outcomes."""
import argparse,csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .record import digest
CLASSES=['HPLD','HPHD','LPHD','LPLD','Online'];METHODS=['hexgen2','hexgen']
def plot(plan_path,trials,output):
    plan=json.loads(plan_path.read_text());output.mkdir(parents=True,exist_ok=True);rows=[]
    for cell in plan['cells']:
        path=trials/cell['run_id']/'outcome.json';result=json.loads(path.read_text())
        if result['status']!='success':raise ValueError('Incomplete or failed measured cell: '+cell['run_id'])
        client=result['client']['summary']
        if client['failures'] or not client['capture_complete']:raise ValueError('Invalid capture: '+cell['run_id'])
        if client['duration_seconds']!=plan['duration_seconds']:raise ValueError('Window mismatch')
        rows.append({k:cell[k] for k in ['run_id','workload','method','seed','repetition']}|{k:client[k] for k in ['duration_seconds','throughput_tokens_per_second','full_drain_tokens_per_second','offered','completed','successful','failures','measurement_output_tokens','completed_output_tokens','wall_seconds']}|{'outcome_sha256':digest(path)})
    with (output/'figure6-data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,5,figsize=(14.8,3.6));colors=['#6A86E7','#60A67C'];summary=[]
    for ax,workload in zip(axes,CLASSES):
        for i,method in enumerate(METHODS):
            group=[r for r in rows if r['workload']==workload and r['method']==method]
            if len(group)!=3 or len({r['seed'] for r in group})!=3:raise ValueError('Need all three repetitions per method/workload')
            values=[r['throughput_tokens_per_second'] for r in group];mean=statistics.mean(values);sd=statistics.stdev(values)
            ax.bar(i,mean,width=.62,color=colors[i],yerr=sd,capsize=4,error_kw={'elinewidth':1.2},label=['HexGen-2 (adapted)','HexGen (adapted)'][i])
            ax.scatter(np.linspace(i-.09,i+.09,3),values,s=18,c='#222222',zorder=3)
            summary.append({'workload':workload,'method':method,'mean_output_tokens_per_second':mean,'sample_sd':sd,'repetitions':values})
        ax.set_ylim(bottom=0);ax.set_xticks([]);ax.set_xlabel(workload);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
    axes[0].set_ylabel('Completed output tokens / s')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.91))
    fig.suptitle(plan.get('figure_title','OpenLLaMA-13B FP16 · FABRIC'),y=.99,fontsize=13)
    fig.text(.5,.025,plan.get('runtime_note','Three 600 s windows per bar; mean ± sample SD; dots are individual runs. Scaled native-runtime adaptation; DistServe excluded.'),ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.81))
    for extension in ['png','pdf','svg']:fig.savefig(output/('figure6-13b.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)
    result={'plan_sha256':digest(plan_path),'plot_source_sha256':digest(__file__),'aggregation':'arithmetic mean and sample standard deviation of three fixed-window throughputs; no failed cells or repetitions omitted','summary':summary,'cells':rows}
    (output/'figure6-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--trials',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.plan,a.trials,a.output)
