"""Portable single-figure analysis of the alternative-incorrect-hint experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONDITIONS = [.1,.25,.5,.75,.9,None]


def load_results(root, preview=False, seed_count=5):
    if seed_count not in (3,5):
        raise ValueError("seed_count must be 3 or 5")
    records=[json.loads(p.read_text()) for p in sorted(Path(root).rglob('contrast.json'))]
    if not records:
        raise ValueError('no completed contrast analyses found')
    keys=[(r['hint_probability'],r['seed']) for r in records]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate condition/seed')
    if len({r['study_manifest_sha256'] for r in records})!=1 or len({r['control_manifest_sha256'] for r in records})!=1:
        raise ValueError('mixed study or control banks')
    if not preview and set(keys)!={(c,s) for c in CONDITIONS for s in range(2026092201,2026092201+seed_count)}:
        raise ValueError(f'require all {6*seed_count} matched runs; use --preview for an explicitly labeled partial figure')
    for run in records:
        rows=run['records']
        if [r['step'] for r in rows]!=list(range(101)):
            raise ValueError('incomplete trajectory')
        if [r['step'] for r in rows if r['observed_change'] is not None]!=list(range(0,101,5)):
            raise ValueError('incomplete direct observations')
        if not np.isfinite([r['price_change'] for r in rows]+[r['observed_change'] for r in rows if r['observed_change'] is not None]).all():
            raise ValueError('non-finite contrast')
    return records


def plot_results(root, output, preview=False, seed_count=5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs=load_results(root,preview,seed_count)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.grid':True,'grid.alpha':.18})
    conditions=[c for c in CONDITIONS if any(r['hint_probability']==c for r in runs)]
    columns=min(3,len(conditions));rows=(len(conditions)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(4*columns,3.5*rows),sharex=True,sharey=True,squeeze=False)
    blue,orange='#2463b4','#d86a22'
    all_values=[row[key] for run in runs for row in run['records'] for key in ['price_change','observed_change'] if row[key] is not None]
    lower,upper=min(0.,min(all_values))-.05,max(1.,max(all_values))+.05
    csv_rows=[]
    for ax,c in zip(axes.flat,conditions):
        group=[r for r in runs if r['hint_probability']==c];observed=[];prices=[]
        for run in group:
            records=run['records'];x=[r['step'] for r in records];obs=[r for r in records if r['observed_change'] is not None]
            ox=[r['step'] for r in obs];y=[r['observed_change'] for r in obs];p=[r['price_change'] for r in records]
            observed.append(y);prices.append(p)
            ax.plot(x,p,color=orange,alpha=.22,lw=1);ax.plot(ox,y,color=blue,alpha=.22,lw=1)
            csv_rows.extend(dict(run_id=run['run_id'],hint_accuracy=c,seed=run['seed'],**r) for r in records)
        ax.plot(x,np.mean(prices,axis=0),color=orange,lw=2.5)
        ax.plot(ox,np.mean(observed,axis=0),'o-',color=blue,lw=2.5,ms=3)
        label='No training hint' if c is None else f'{c:.0%} training hint accuracy'
        ax.set_title(label+f' (n={len(group)})')
        ax.set_ylim(lower,upper);ax.axhline(0,color='gray',lw=.6)
        ax.set_xlabel('Training update')
    for ax in axes[:,0]:ax.set_ylabel('Change in wrong-hint influence')
    for ax in list(axes.flat)[len(conditions):]:ax.set_visible(False)
    title='Effect of changing an incorrect hint'
    if preview:title=f'PREVIEW: {len(runs)} of {6*seed_count} runs · '+title
    fig.suptitle(title,fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color=blue,lw=2.5,label='Direct contrast change'),
                        Line2D([0],[0],color=orange,lw=2.5,label='Cumulative Price difference')],
               loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ['pdf','svg','png']:fig.savefig(output/f'hint_contrast.{ext}',dpi=180)
    plt.close(fig)
    with (output/'trajectories.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    (output/'results.json').write_text(json.dumps(runs,indent=2,allow_nan=False)+'\n')
    svg=(output/'hint_contrast.svg').read_text();svg=svg[svg.index('<svg'):]
    (output/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wrong-hint influence</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px;color:#243044}svg{width:100%;height:auto}</style><h1>'''+title+'''</h1><p>The same question is evaluated with two different incorrect hints. Both arms score whether the response selects the option recommended by the original hint. The contrast is the original-hint probability minus the alternative-hint probability; the figure plots its change from initialization.</p><p>Blue: direct contrast change. Orange: the difference between the two cumulative Price covariances. Faded curves: individual seeds. Bold curves: means of the displayed seeds. No bootstrap bands or filtering are applied. Lines connect measured checkpoints without filling missing observations.</p>'''+svg+'</html>')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'analysis')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'figures')
    p.add_argument('--preview',action='store_true')
    p.add_argument('--seed-count',type=int,choices=[3,5],default=5,help='Require the first three or all five matched seeds')
    args=p.parse_args();print(plot_results(args.results,args.output,args.preview,args.seed_count))
