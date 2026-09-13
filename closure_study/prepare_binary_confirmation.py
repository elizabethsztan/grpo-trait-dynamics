"""One calibration of the three registered output-trait laws on development data."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .adapters import output_llm
from .fit_llm import fit_binary_flux
from .io import sha256, write_csv

DEVELOPMENT = (290416, 290417, 290418)
CONFIRMATION = (2026091501, 2026091502, 2026091503)
ARCHIVE = Path('results/trajectory_forecast/hierarchical_20260724/prospective_v1')
MODELS = [('constant', 'T', 0), ('state_affine', 'T', 1), ('time_affine', 'step', 1)]


def extract(root):
    rows, inputs, common = [], [], None
    for seed in DEVELOPMENT:
        directory = ARCHIVE/'runs'/f'hierarchical_above_hook_lr1e5_seed{seed}'
        config_path, metrics_path = directory/'config.yaml', directory/'metrics.jsonl'
        config = yaml.safe_load((root/config_path).read_text())
        assert config['RunConfig']['seed'] == seed
        comparable = {k:v for k,v in config.items() if k != 'RunConfig'}
        if common is not None and comparable != common:
            raise ValueError('development model, training or measurement settings differ')
        common = comparable
        entry = dict(run_id=f'historical-above_hook_lr1e5-seed{seed}', family='llm_binary',
            setting_id='output-above_hook_lr1e5', inspection_status='development',
            metrics_path=str(root/metrics_path), distribution='eval_wrong_hint')
        run_rows, _, _ = output_llm(entry)
        if len(run_rows) != 60:
            raise ValueError('expected all 60 one-update development transitions')
        valid = [r for r in run_rows if r.get('T') is not None]
        if [r['step'] for r in valid] != list(range(0,60,5)):
            raise ValueError('direct old-state observations differ from the registered inputs')
        rows.extend(valid)
        inputs.append(dict(seed=seed, run_id=entry['run_id'], metrics_path=str(metrics_path),
            metrics_sha256=sha256(root/metrics_path), config_path=str(config_path),
            config_sha256=sha256(root/config_path), fit_rows=len(valid),
            excluded_rows_missing_direct_T=len(run_rows)-len(valid)))
    return rows, inputs, config


def freeze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('calibration output already exists; do not refit the frozen laws')
    targets = [Path('results/llm_binary_confirmation')/f'seed{seed}' for seed in CONFIRMATION]
    for target in targets:
        path = root/target
        if path.exists() and any(path.iterdir()):
            raise ValueError(f'confirmation directory is not empty: {target}')
    rows, inputs, original = extract(root)
    coefficients = {}
    # Exactly one application of each registered fit to the same 36 real pairs.
    for name, predictor, degree in MODELS:
        coef, mask, informative = fit_binary_flux(rows, predictor, degree)
        assert mask.all() and informative == len(rows)
        coefficients[name] = dict(predictor=predictor, coefficients=coef.tolist(),
            fit_rows=len(rows), informative_rows=informative)
    base = deepcopy(original)
    base['RunConfig'].update(seed=CONFIRMATION[0], name=f'seed{CONFIRMATION[0]}',
        results_dir='results/llm_binary_confirmation', fail_if_exists=True)
    # The registered dense measurement replaces the legacy sparse evaluation
    # configuration; it is separate from the unchanged model/training setup.
    del base['PriceConfig'], base['ObservedEvalConfig']
    specification = dict(training_config=base,
        runs=[dict(seed=seed, output=str(target), prompt_seed=seed+200000,
                   measurement_generation_seed=seed+100000) for seed,target in zip(CONFIRMATION,targets)],
        measurement=dict(distribution='eval_wrong_hint', checkpoints=list(range(61)),
                         prompt_groups=512, completions_per_group=2, preserve_prompt_groups=True),
        scoring=dict(prevalence_checkpoints=list(range(5,61,5)), closure_source_steps=list(range(60)),
                     autonomous_initial_state_only=True, no_clipping=True, no_score_after_failure=True))
    code_paths = ['closure_study/prepare_binary_confirmation.py','closure_study/adapters.py',
                  'closure_study/fit_llm.py','closure_study/fit_controlled.py']
    provenance = dict(development_inputs=inputs, code=[dict(path=p,sha256=sha256(root/p)) for p in code_paths],
        source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        numpy_version=np.__version__, distribution='eval_wrong_hint',
        objective='sum((C_t - T_t*(1-T_t)*S_hat(T_t or t))**2)',
        solver='numpy.linalg.lstsq: design=T*(1-T)*polyvander(predictor, degree), rcond=None',
        alignment='C for t -> t+1 comes from metrics record t+1; T is direct observed_eval at record t',
        pooling='equal weight per retained covariance row; 12 rows per run; no interpolation, normalization, or trajectory fitting',
        confirmation_directories=[dict(path=str(p),status='absent' if not (root/p).exists() else 'empty') for p in targets])
    output.mkdir(parents=True)
    for filename, data in [('coefficients.json',coefficients),('inputs.json',provenance)]:
        (output/filename).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (output/'confirmation.yaml').write_text(yaml.safe_dump(specification,sort_keys=False))
    write_csv(output/'development_pairs.csv',[
        dict(run_id=r['run_id'],step=r['step'],step_end=r['step_end'],T=r['T'],C=r['C'],
             T_source_record=r['step'],C_source_record=r['step_end']) for r in rows])
    return coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.root,args.output),indent=2))


if __name__=='__main__': main()
