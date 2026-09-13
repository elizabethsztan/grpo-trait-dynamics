"""Frozen binary confirmation analysis and small launch checks; never fits a law."""
import argparse
import base64
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from numpy.polynomial.polynomial import polyval
import yaml

from .io import read_csv, sha256, write_csv

FROZEN_COMMIT = '32b632637905980286f7835f493765c246d5635f'
CONFIG = Path('closure_study/configs/llm_binary_confirmation')
PROTOCOL = Path('closure_study/llm_binary_confirmation_protocol.md')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def verify_specification(root):
    root = Path(root)
    git(root, 'merge-base', '--is-ancestor', FROZEN_COMMIT, 'HEAD')
    frozen = [CONFIG/p for p in ('coefficients.json', 'inputs.json', 'development_pairs.csv', 'confirmation.yaml')]+[PROTOCOL]
    for path in frozen:
        if (root/path).read_bytes() != git(root, 'show', f'{FROZEN_COMMIT}:{path}'):
            raise ValueError(f'frozen specification changed: {path}')
    inputs = json.loads((root/CONFIG/'inputs.json').read_text())
    hashes = [(r['path'], r['sha256']) for r in inputs['code']]
    hashes += [(r[k+'_path'], r[k+'_sha256']) for r in inputs['development_inputs'] for k in ('metrics', 'config')]
    for path, expected in hashes:
        if sha256(root/path) != expected:
            raise ValueError(f'development source hash mismatch: {path}')
    if git(root, 'status', '--porcelain', '--untracked-files=normal', '--', 'closure_study', 'llm-experiments').strip():
        raise ValueError('commit and review the implementation before launching or analyzing confirmation')
    files = git(root, 'ls-files', '--', 'closure_study', 'llm-experiments').decode().splitlines()
    source = {p:sha256(root/p) for p in files if p.endswith(('.py', '.md', '.json', '.yaml'))}
    provenance = dict(frozen_commit=FROZEN_COMMIT, implementation_commit=git(root, 'rev-parse', 'HEAD').decode().strip(),
        frozen_sha256={str(p):sha256(root/p) for p in frozen}, source_sha256=source)
    return (yaml.safe_load((root/CONFIG/'confirmation.yaml').read_text()),
            json.loads((root/CONFIG/'coefficients.json').read_text()),
            json.loads((root/CONFIG/'execution.json').read_text()), provenance)


def reserve_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'refusing populated output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two launches from claiming an empty path.
    (path/'.claim').touch(exist_ok=False)
    return path


def load_measurements(directory, steps, groups, completions):
    """Group-level sufficient statistics preserve paired old/new response scores."""
    directory = Path(directory)
    z, w, wz = [], [], []
    for t in range(steps+1):
        path = directory/'pools'/f'{t:03d}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [(r['group'],r['response']) for r in rows] != list(itertools.product(range(groups),range(completions))):
            raise ValueError('missing or reordered prompt groups/responses')
        for r in rows:
            if r['z'] not in (0,1) or r['z'] != int(r['parsed_choice'] is not None and r['parsed_choice']==r['user_hint']):
                raise ValueError('binary trait disagrees with saved parse/hint')
        traits = np.array([r['z'] for r in rows], dtype=float)
        z.append(traits.reshape(groups,completions).mean(axis=1))
        if t == steps:
            continue
        with np.load(directory/'scores'/f'{t:03d}.npz', allow_pickle=False) as scores:
            if int(scores['step']) != t or int(scores['step_end']) != t+1 or str(scores['pool_sha256']) != sha256(path):
                raise ValueError('successor scores belong to a different pool/transition')
            pre = np.array([r['pre_logprob'] for r in rows])
            post = scores['post_logprob']
            if post.shape != pre.shape or not np.isfinite([pre,post]).all():
                raise ValueError('incomplete or nonfinite response scores')
            with np.errstate(over='ignore'):
                ratio = np.exp(post-pre)
            if not np.isfinite(ratio).all():
                raise ValueError('nonfinite likelihood ratios; preserve this measurement failure')
            w.append(ratio.reshape(groups,completions).mean(axis=1))
            wz.append((ratio*traits).reshape(groups,completions).mean(axis=1))
    return np.asarray(z), np.asarray(w), np.asarray(wz)


def measured(z, w, wz):
    T = z.mean(axis=1)
    return T, wz.mean(axis=1)-w.mean(axis=1)*T[:-1]


def bootstrap(z, w, wz, draws, seed):
    """Independent checkpoint pools, whole groups, joint z/w/wz within a pool."""
    rng = np.random.Generator(np.random.PCG64(seed))
    T = np.empty((draws,len(z)))
    C = np.empty((draws,len(w)))
    for t in range(len(z)):
        indices = rng.integers(0,z.shape[1],size=(draws,z.shape[1]))
        T[:,t] = z[t,indices].mean(axis=1)
        if t < len(w):
            C[:,t] = wz[t,indices].mean(axis=1)-w[t,indices].mean(axis=1)*T[:,t]
    return T, C


def generate(initial, model, steps):
    """Vectorized independent recurrences; retain every first invalid attempt."""
    path = np.full((len(initial),steps+1),np.nan)
    path[:,0] = initial
    failed_step = np.full(len(initial),-1)
    attempted = np.full(len(initial),np.nan)
    for t in range(steps):
        active = failed_step == -1
        mu = path[:,t]
        with np.errstate(over='ignore',invalid='ignore'):
            s = polyval(t if model['predictor']=='step' else mu,model['coefficients'])
            nxt = mu+mu*(1-mu)*s
        bad = active & (~np.isfinite(nxt) | (nxt<0) | (nxt>1))
        failed_step[bad], attempted[bad] = t+1, nxt[bad]
        path[active & ~bad,t+1] = nxt[active & ~bad]
    return path, failed_step, attempted


def row_rmse(x):
    count = np.isfinite(x).sum(axis=1)
    return np.sqrt(np.divide(np.nansum(x*x,axis=1),count,
                   out=np.full(len(x),np.nan),where=count>0))


def evaluate(T, C, models, scoring_steps):
    T, C = np.atleast_2d(T), np.atleast_2d(C)
    v = T[:,:-1]*(1-T[:,:-1])
    S = np.divide(C,v,out=np.full_like(C,np.nan),where=v>0)
    output = {}
    for name, model in models.items():
        law = polyval(np.arange(C.shape[1]) if model['predictor']=='step' else T[:,:-1],model['coefficients'])
        path, failed, attempted = generate(T[:,0],model,C.shape[1])
        c_res, s_res = C-v*law, S-law
        error = path[:,scoring_steps]-T[:,scoring_steps]
        trajectory_rmse = row_rmse(error)
        trajectory_rmse[failed!=-1] = np.nan
        output[name] = dict(path=path, failed_step=failed, attempted_T=attempted,
            C_residual=c_res, S_residual=s_res, S_count=np.isfinite(S).sum(axis=1),
            C_rmse=row_rmse(c_res), S_rmse=row_rmse(s_res), T_rmse=trajectory_rmse,
            terminal_error=path[:,-1]-T[:,-1])
    return output


def interval(x, percentiles):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return (*np.percentile(x,percentiles,method='linear').tolist(), len(x)) if len(x) else (None,None,0)


def analyze_run(directory, output, spec, models, execution, seed):
    m = spec['measurement']
    z,w,wz = load_measurements(directory,len(m['checkpoints'])-1,m['prompt_groups'],m['completions_per_group'])
    T,C = measured(z,w,wz)
    point = evaluate(T,C,models,spec['scoring']['prevalence_checkpoints'])
    bt,bc = bootstrap(z,w,wz,execution['bootstrap_draws'],seed+execution['bootstrap_seed_offset'])
    boot = evaluate(bt,bc,models,spec['scoring']['prevalence_checkpoints'])
    percentiles = execution['percentiles']
    metrics, trajectories, residuals, comparisons, uncertainty = [], [], [], [], []
    for name, result in point.items():
        for metric in ('C_rmse','S_rmse','T_rmse','terminal_error'):
            lo,hi,n = interval(boot[name][metric],percentiles)
            metrics.append(dict(seed=seed,model=name,metric=metric,value=result[metric][0],
                lower=lo,upper=hi,bootstrap_defined=n,bootstrap_draws=len(bt)))
        for t in range(len(T)):
            lo,hi,n = interval(boot[name]['path'][:,t],percentiles)
            trajectories.append(dict(seed=seed,model=name,step=t,T_observed=T[t],T_generated=result['path'][0,t],
                                     lower=lo,upper=hi,bootstrap_defined=n))
        for t in range(len(C)):
            residuals.append(dict(seed=seed,model=name,step=t,step_end=t+1,T=T[t],C=C[t],
                C_residual=result['C_residual'][0,t],S_residual=result['S_residual'][0,t]))
        for metric in ('C_residual','S_residual'):
            for t in range(len(C)):
                lo,hi,n = interval(boot[name][metric][:,t],percentiles)
                uncertainty.append(dict(model=name,quantity=metric,step=t,lower=lo,upper=hi,bootstrap_defined=n))
    for a,b in itertools.combinations(models,2):
        for metric in ('C_rmse','S_rmse','T_rmse'):
            lo,hi,n = interval(boot[a][metric]-boot[b][metric],percentiles)
            comparisons.append(dict(seed=seed,model_a=a,model_b=b,metric=metric,
                difference=point[a][metric][0]-point[b][metric][0],lower=lo,upper=hi,bootstrap_defined=n))
    bv = bt[:,:-1]*(1-bt[:,:-1])
    bs = np.divide(bc,bv,out=np.full_like(bc,np.nan),where=bv>0)
    for quantity,values in [('T',bt),('C',bc),('S',bs)]:
        for t in range(values.shape[1]):
            lo,hi,n = interval(values[:,t],percentiles)
            uncertainty.append(dict(model='measurement',quantity=quantity,step=t,lower=lo,upper=hi,bootstrap_defined=n))
    completion = [dict(seed=seed,model=name,status='complete' if r['failed_step'][0]==-1 else 'failed',
        failed_step=int(r['failed_step'][0]),attempted_T=r['attempted_T'][0],
        completed_updates=int(np.isfinite(r['path'][0,1:]).sum()),S_rows=int(r['S_count'][0]),
        bootstrap_complete=int((boot[name]['failed_step']==-1).sum()),bootstrap_draws=len(bt)) for name,r in point.items()]
    output = reserve_directory(output)
    for filename, rows in [('metrics',metrics),('trajectories',trajectories),('residuals',residuals),
                           ('comparisons',comparisons),('completion',completion),('uncertainty',uncertainty)]:
        write_csv(output/f'{filename}.csv',rows)
    np.savez_compressed(output/'bootstrap.npz',T=bt,C=bc,
        **{f'{name}_{key}':r[key] for name,r in boot.items() for key in
           ('failed_step','attempted_T','C_rmse','S_rmse','T_rmse','terminal_error')})
    return metrics, trajectories, completion, comparisons


def report(output, metrics, trajectories, completion, comparisons, collection):
    import matplotlib.pyplot as plt
    from .fit_controlled import table
    html = ['<meta charset="utf-8"><title>LLM binary confirmation</title>',
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>LLM binary prospective confirmation</h1>',
        '<p>All coefficients are frozen development estimates. Predictions receive only initial prevalence and, for the time law, the clock. Intervals quantify response measurement uncertainty conditional on each trained run and the frozen laws. Training runs are the replication unit; no pooled significance test is supplied.</p>',
        '<p>Intervals for undefined quantities use only defined bootstrap draws, with counts shown. Failed paths retain their first invalid attempt and have no full-horizon score; no clipping or replacement. Trajectory bands are pointwise, conditional on a path remaining defined at that checkpoint.</p>',
        table(collection,['seed','collection_status'])]
    for seed in [r['seed'] for r in collection]:
        ts = [r for r in trajectories if r['seed']==seed]
        html.append(f'<h2>Seed {seed}</h2>')
        if not ts:
            continue
        fig,axes = plt.subplots(2,2,figsize=(11,7))
        ax = axes[0,0]
        for name in dict.fromkeys(r['model'] for r in ts):
            rows = [r for r in ts if r['model']==name]
            ax.plot([r['step'] for r in rows],[r['T_generated'] for r in rows],label=name)
        rows = [r for r in ts if r['model']==ts[0]['model']]
        ax.scatter([r['step'] for r in rows],[r['T_observed'] for r in rows],s=8,c='black',label='measured')
        ax.set(xlabel='optimizer update',ylabel='wrong-hint agreement T',ylim=(0,1)); ax.legend()
        residuals = read_csv(output/f'seed{seed}'/'residuals.csv')
        for name in dict.fromkeys(r['model'] for r in residuals):
            rows = [r for r in residuals if r['model']==name]
            state = np.array([float(r['T']) for r in rows])
            t = np.array([int(r['step']) for r in rows])
            v = state*(1-state)
            selection = np.divide([float(r['C']) for r in rows],v,out=np.full(len(rows),np.nan),where=v>0)
            predicted = selection-np.array([float(r['S_residual']) for r in rows])
            axes[0,1].scatter(state,predicted,s=10,label=name)
            axes[1,0].plot(t,predicted,label=name)
            axes[1,1].plot(t,[float(r['C_residual']) for r in rows],label=name)
        axes[0,1].scatter(state,selection,s=10,c='black',label='measured S')
        axes[1,0].scatter(t,selection,s=10,c='black',label='measured S')
        axes[0,1].set(xlabel='measured T',ylabel='selection S'); axes[0,1].legend()
        axes[1,0].set(xlabel='optimizer update',ylabel='selection S')
        axes[1,1].set(xlabel='optimizer update',ylabel='C minus fitted C'); axes[1,1].axhline(0,c='black',lw=.5)
        path = output/f'seed{seed}.png'; fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
        html.append('<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
        for rows,fields in [(completion,['model','status','completed_updates','failed_step','attempted_T','S_rows','bootstrap_complete','bootstrap_draws']),
            (metrics,['model','metric','value','lower','upper','bootstrap_defined']),
            (comparisons,['model_a','model_b','metric','difference','lower','upper','bootstrap_defined'])]:
            html.append(table([r for r in rows if r['seed']==seed],fields))
    (output/'readout.html').write_text('\n'.join(html))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec,models,execution,provenance = verify_specification(root)
    output = reserve_directory(args.output)
    metrics,trajectories,completion,comparisons,collection = [],[],[],[],[]
    for run in spec['runs']:
        directory = root/run['output']
        status = 'complete' if (directory/'completed.json').exists() else 'incomplete or absent'
        if status == 'complete':
            if json.loads((directory/'provenance.json').read_text()) != provenance:
                raise ValueError('run belongs to a different frozen specification or implementation')
            try:
                if json.loads((directory/'completed.json').read_text()) != dict(updates=60,measurement=True):
                    raise ValueError('completion marker differs from registered collection')
                a,b,c,d = analyze_run(directory,output/f"seed{run['seed']}",spec,models,execution,run['seed'])
                metrics.extend(a); trajectories.extend(b); completion.extend(c); comparisons.extend(d)
            except (ValueError,FileNotFoundError) as error:
                status = f'measurement failure: {error}'
        collection.append(dict(seed=run['seed'],collection_status=status))
    report(output,metrics,trajectories,completion,comparisons,collection)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
