"""CPU-only prompt-group uncertainty for saved, completed Price measurements."""
from pathlib import Path
import csv

import numpy as np

from .study import PRICE_DISTRIBUTIONS, digest, read_json, read_rows, verify_artifact, write_json


def resampling_weights(strata, draws, seed):
    """Keep the balanced bank's correct/wrong counts fixed; share draws everywhere."""
    strata = np.asarray(strata, dtype=bool)
    if draws < 2 or not len(strata) or strata.sum() * 2 != len(strata):
        raise ValueError("require at least two draws and an exactly balanced bank")
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(strata)), dtype=np.float64)
    for label in (False, True):
        indices = np.flatnonzero(strata == label)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), draws)
    return weights / len(strata)


def accounting_draws(price_moments, observed, observed_steps, weights):
    """Moments are [transition, question, (omega, trait, omega*trait)]."""
    means = np.einsum('bn,tnk->btk', weights, price_moments)
    covariance = means[:, :, 2] - means[:, :, 0] * means[:, :, 1]
    cumulative = np.column_stack((np.zeros(len(weights)), covariance.cumsum(axis=1)))
    prevalence = weights @ observed.T
    drift = prevalence - prevalence[:, :1]
    return {
        'cov_step': np.column_stack((np.zeros(len(weights)), covariance)),
        'cov_cum': cumulative,
        'prevalence': prevalence,
        'observed_drift': drift,
        'residual': drift - cumulative[:, observed_steps],
    }


def load_measurement(root):
    root = Path(root)
    status = read_json(root / 'status.json')
    if status['state'] != 'complete':
        raise ValueError('measurement must be complete')
    inventory = read_json(verify_artifact(root, status['files']))['files']
    references = {r['path']: r for r in inventory}
    if len(references) != len(inventory):
        raise ValueError('duplicate inventory paths')

    def verified(name):
        return verify_artifact(root, references[name])

    config = read_json(verified('config.json'))
    logged = list(read_rows(verified('metrics.jsonl')))
    n = config['PriceConfig']['prompts_per_distribution']
    steps = config['TrainConfig']['num_steps']
    every = config['ObservedEvalConfig']['eval_every']
    observed_steps = np.array([t for t in range(steps + 1) if t % every == 0 or t == steps])
    if [r['step'] for r in logged] != list(range(steps + 1)):
        raise ValueError('incomplete logged trajectory')
    group_ids, strata, pools = None, None, {}

    def pool(name, distribution, step, completions, price):
        nonlocal group_ids, strata
        rows = list(read_rows(verified(name)))
        if len(rows) != n * completions:
            raise ValueError('incomplete response pool')
        ids, labels, values = [], [], []
        for group in range(n):
            samples = rows[group * completions:(group + 1) * completions]
            ids.append(samples[0]['group_id'])
            label = samples[0]['example']['hint_is_correct']
            labels.append(label)
            for response, row in enumerate(samples):
                if (row['group_index'], row['response_index'], row['group_id'], row['distribution'],
                    row['source_step'], row['target_step'], row['kind']) != (
                    group, response, ids[-1], distribution, step, step + 1 if price else None,
                    'price' if price else 'observed'
                ) or row['example']['hint_is_correct'] != label:
                    raise ValueError('inconsistent response grouping or transition')
                trait = float(row['traits']['output_agreement'])
                omega = np.exp(float(row['post_logprob']) - float(row['pre_logprob'])) if price else 1.0
                values.append((omega, trait, omega * trait))
        if len(set(ids)) != n or (group_ids is not None and ids != group_ids):
            raise ValueError('question groups differ across checkpoints or variants')
        group_ids = ids
        if distribution == 'eval_balanced_hint':
            if any(type(label) is not bool for label in labels) or (strata is not None and labels != strata):
                raise ValueError('balanced-hint assignments differ across checkpoints')
            strata = labels
        values = np.asarray(values).reshape(n, completions, 3)
        if not np.isfinite(values).all():
            raise ValueError('non-finite bootstrap inputs')
        return values.mean(axis=1)

    for distribution in PRICE_DISTRIBUTIONS:
        price = np.array([pool(f'price_{t:04d}_{t+1:04d}_{distribution}.jsonl.gz', distribution, t,
                              config['PriceConfig']['completions_per_prompt'], True) for t in range(steps)])
        observed = np.array([pool(f'observed_{t:04d}_{distribution}.jsonl.gz', distribution, int(t),
                                 config['ObservedEvalConfig']['completions_per_prompt'], False)[:, 1]
                             for t in observed_steps])
        point = accounting_draws(price, observed, observed_steps, np.full((1, n), 1 / n))
        for t in range(1, steps + 1):
            for key in ('cov_step', 'cov_cum'):
                expected = logged[t]['price'][distribution]['output_agreement'][key]
                if not np.isclose(point[key][0, t], expected, atol=1e-10, rtol=1e-10):
                    raise ValueError('raw Price reconstruction differs from logged accounting')
        for index, t in enumerate(observed_steps):
            if not np.isclose(point['prevalence'][0, index], logged[t]['observed_eval'][distribution]['agreement_rate'], atol=1e-12):
                raise ValueError('raw prevalence differs from logged accounting')
        pools[distribution] = (price, observed, point)
    return pools, observed_steps, strata, group_ids


def bootstrap_measurement(measurement, output, draws=2000, seed=2026092200):
    measurement, output = Path(measurement), Path(output)
    if output.exists():
        raise FileExistsError(output)
    pools, observed_steps, strata, group_ids = load_measurement(measurement)
    weights = resampling_weights(strata, draws, seed)
    rows, arrays, final = [], {'question_counts': np.rint(weights * len(strata)).astype(np.int32),
                              'observed_steps': observed_steps}, {}
    for distribution, (price, observed, point) in pools.items():
        samples = accounting_draws(price, observed, observed_steps, weights)
        for metric, values in samples.items():
            arrays[f'{distribution}__{metric}'] = values
            lower, upper = np.quantile(values, [.025, .975], axis=0)
            checkpoints = range(price.shape[0] + 1) if metric.startswith('cov_') else observed_steps
            for index, step in enumerate(checkpoints):
                row = dict(distribution=distribution, step=int(step), metric=metric,
                           estimate=float(point[metric][0, index]), lower_95=float(lower[index]),
                           upper_95=float(upper[index]), half_width=float((upper[index]-lower[index])/2))
                rows.append(row)
                if metric == 'residual' and step == price.shape[0]:
                    final[distribution] = {**row, 'precision_target_met': row['half_width'] <= .05}
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'intervals.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'draws.npz', **arrays)
    write_json(output / 'summary.json', {
        'schema_version': 1, 'draws': draws, 'seed': seed, 'interval': 'pointwise percentile 95%',
        'resampling': 'question groups, stratified by balanced-hint correctness; shared across time and variants',
        'uncertainty_scope': 'evaluation sampling for one trained trajectory; excludes training-seed variability',
        'measurement_status_sha256': digest(measurement / 'status.json'),
        'measurement_inventory_sha256': digest(measurement / 'files.json'),
        'analysis_source_sha256': digest(__file__), 'numpy_version': np.__version__,
        'group_ids': group_ids, 'balanced_hint_correctness': strata, 'final_residuals': final,
        'artifacts': {name: digest(output / name) for name in ('intervals.csv', 'draws.npz')},
    })
    return final
