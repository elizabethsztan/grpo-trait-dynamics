"""One controlled confirmation run with frozen closures and exact measurements."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.polynomial import polyval
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_study.fit_controlled import rms, rollout
from closure_study.io import write_csv
from closure_study.statistics import enumerated_transition, moments, probabilities
from run_experiment import build_neural_policy, build_policy

DEFAULT_CONFIG = Path(__file__).parent / 'configs/confirmation.yaml'


def make_policy(config, system, training_seed, initial_quality):
    setup = config[system]
    if system == 'tabular':
        policy = build_policy(config['policy'], config['reward'], config['environment_seed'])
    else:
        policy = build_neural_policy(config['policy'], config['reward'], setup['training'],
                                     config['neural'], config['environment_seed'])
    policy.init_env(training_seed=training_seed, initial_quality=initial_quality)
    return policy


def population(policy):
    with torch.no_grad():
        pi = policy.get_pi()
    if isinstance(pi, torch.Tensor):
        pi = pi.numpy()
    return probabilities(pi.reshape(-1))


def measure(policy, training):
    """Enumerate the same fixed trait before and after every optimizer update."""
    z = np.asarray(policy.s, dtype=float).reshape(-1)
    old = population(policy)
    states = [dict(step=0, **moments(z, old), reward=policy.expected_reward())]
    transitions = []
    for step in range(training['steps']):
        policy.grpo_step(batch_size=training['batch_size'], eta=training.get('eta'))
        new = population(policy)
        row = enumerated_transition(z, old, new)
        transitions.append(dict(step=step, **row))
        states.append(dict(step=step+1, **moments(z, new), reward=policy.expected_reward()))
        old = new
    return states, transitions


def compare(states, transitions, setup, binary):
    """No fitting; observations after initialization enter diagnostics/scoring only."""
    steps = np.arange(len(transitions))
    observed = np.array([[s['mu'], s['V']] for s in states])
    mu, v, beta, m3, q, c = (np.array([r[k] for r in transitions], dtype=float)
                             for k in ('mu', 'V', 'beta', 'M3', 'Q', 'C'))
    initial = observed[0, :1] if binary else observed[0]
    specs = [('state', setup['selection_state'], False, 1.0),
             ('time', setup['selection_time'], True, 1.0)] if binary else [
                 ('corrected', setup['selection_state'], False, setup['kappa']),
                 ('uncorrected', setup['selection_state'], False, 1.0)]
    trajectories, residuals, metrics, failures = [], [], [], []
    for model, selection, time, kappa in specs:
        skewness = None if binary else setup['skewness']
        path, status = rollout(initial, steps, selection, skewness, time, kappa=kappa)
        bhat = polyval(steps if time else mu, selection)
        residual = dict(C_residual=c-v*bhat)
        if binary:
            measured_s = c/(mu*(1-mu))
            residual['S_residual'] = measured_s-bhat
        else:
            mhat = polyval(mu, skewness)*v**1.5
            residual.update(beta_residual=beta-bhat, M3_residual=m3-mhat,
                gamma_residual=m3/v**1.5-polyval(mu, skewness),
                Q_linear_residual=q-beta*m3, Q_flux_residual=q-kappa*beta*m3,
                Q_moment_residual=kappa*beta*(m3-mhat),
                Q_selection_residual=kappa*(beta-bhat)*mhat,
                Q_total_residual=q-kappa*bhat*mhat,
                delta_V_residual=q-c*c-(kappa*bhat*mhat-(bhat*v)**2))
        residuals.extend(dict(model=model, step=int(t), **{k: float(a[t]) for k,a in residual.items()})
                         for t in steps)
        metric = dict(model=model, status=status,
            completed_updates=int(np.isfinite(path[1:,0]).sum()),
            mu_rmse=rms(path[1:,0]-observed[1:,0]) if status=='complete' else None,
            terminal_mu_error=float(path[-1,0]-observed[-1,0]) if status=='complete' else None)
        for key, a in residual.items():
            # Retain the historical 999-transition moment/flux window.
            window = a if binary or key in ('C_residual','beta_residual') else a[:-1]
            metric[key+'_rmse'] = rms(window)
            if not binary:
                metric[key+'_terminal'] = float(a[-1])
                metric[key+'_early_rmse'] = rms(window[:100])
                metric[key+'_late_rmse'] = rms(window[100:])
        if not binary:
            metric.update(V_rmse=rms(path[1:-1,1]-observed[1:-1,1]) if status=='complete' else None,
                terminal_V_error=float(path[-1,1]-observed[-1,1]) if status=='complete' else None)
        metrics.append(metric)
        trajectories.extend(dict(model=model, step=i, mu_observed=float(observed[i,0]),
            mu_generated=float(path[i,0]), V_observed=None if binary else float(observed[i,1]),
            V_generated=None if binary else float(path[i,1])) for i in range(len(states)))
        if status != 'complete':
            t = metric['completed_updates']
            m = path[t,0]
            variance = m*(1-m) if binary else path[t,1]
            b = polyval(t if time else m, selection)
            with np.errstate(over='ignore', invalid='ignore'):
                attempt_mu = m+b*variance
                attempt_v = None if binary else variance+kappa*b*polyval(m,skewness)*variance**1.5-(b*variance)**2
            failures.append(dict(model=model, source_step=t, destination_step=t+1,
                mu=m, V=variance, attempted_mu=attempt_mu, attempted_V=attempt_v, status=status))
    return dict(trajectories=trajectories, residuals=residuals, metrics=metrics, failures=failures)


def run(config, system, arm, seed, output):
    setup, arm_cfg = config[system], config['arms'][arm]
    if seed not in setup['seeds'][:arm_cfg['seed_count']]:
        raise ValueError('seed is not registered for this system/arm')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output/'run.json').write_text(json.dumps(dict(system=system, arm=arm, training_seed=seed,
        environment_seed=config['environment_seed'], initial_quality=arm_cfg['initial_quality']), indent=2))
    policy = make_policy(config, system, seed, arm_cfg['initial_quality'])
    states, transitions = measure(policy, setup['training'])
    write_csv(output/'states.csv', states)
    write_csv(output/'transitions.csv', transitions)
    results = compare(states, transitions, setup, system=='tabular')
    for name, rows in results.items():
        write_csv(output/f'{name}.csv', rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--system', choices=['tabular','neural_continuous'], required=True)
    parser.add_argument('--arm', choices=['replication','perturbed'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    results = run(config, args.system, args.arm, args.seed, args.output)
    print(json.dumps(results['metrics'], indent=2))


if __name__ == '__main__': main()
