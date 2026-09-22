import gzip
from pathlib import Path

import numpy as np
import pytest

from src.bootstrap import accounting_draws, bootstrap_measurement, load_measurement, resampling_weights
from src.study import PRICE_DISTRIBUTIONS, artifact, read_json, write_json, write_row


def test_group_bootstrap_matches_explicit_response_duplication_and_joint_time():
    omega = np.array([[.5, 1.5], [.8, 1.2], [.6, 1.4], [.9, 1.1]])
    trait = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    moments = np.stack([np.column_stack((w.mean(1), trait.mean(1), (w*trait).mean(1)))
                        for w in (omega, 2-omega)])
    observed = np.array([[0, .5, 1, 0], [1, .5, 0, 1]])
    counts = np.array([[2, 0, 1, 1], [1, 1, 0, 2]])
    results = accounting_draws(moments, observed, [0, 2], counts/4)
    for draw, row in enumerate(counts):
        indices = np.repeat(np.arange(4), row)
        expected_cov = []
        for w in (omega, 2-omega):
            x, y = w[indices].ravel(), trait[indices].ravel()
            expected_cov.append(np.mean(x*y)-np.mean(x)*np.mean(y))
        np.testing.assert_allclose(results['cov_step'][draw], [0, *expected_cov], atol=1e-15)
        np.testing.assert_allclose(results['cov_cum'][draw], [0, *np.cumsum(expected_cov)], atol=1e-15)
        drift = observed[:, indices].mean(1) - observed[0, indices].mean()
        np.testing.assert_allclose(results['observed_drift'][draw], drift)
        np.testing.assert_allclose(results['residual'][draw], drift - results['cov_cum'][draw, [0, 2]])
    # Opposite increments cancel when the same questions are resampled over time.
    np.testing.assert_allclose(results['cov_cum'][:, -1], 0, atol=1e-15)


def test_draws_preserve_balanced_design_and_are_reproducible():
    strata = np.array([True, False, False, True, True, False])
    weights = resampling_weights(strata, 2000, 72)
    np.testing.assert_array_equal(weights, resampling_weights(strata, 2000, 72))
    assert not np.array_equal(weights, resampling_weights(strata, 2000, 73))
    np.testing.assert_allclose(weights.sum(1), 1)
    np.testing.assert_allclose(weights[:, strata].sum(1), .5)
    np.testing.assert_allclose(weights[:, ~strata].sum(1), .5)
    with pytest.raises(ValueError):
        resampling_weights([True, False, True], 2000, 72)


@pytest.fixture
def measurement(tmp_path):
    root = tmp_path / 'measurement'
    root.mkdir()
    files, logged = [], [{'step': t, 'price': {}, 'observed_eval': {}} for t in range(3)]
    config = {'TrainConfig': {'num_steps': 2},
              'PriceConfig': {'prompts_per_distribution': 4, 'completions_per_prompt': 2},
              'ObservedEvalConfig': {'eval_every': 2, 'completions_per_prompt': 2}}
    write_json(root/'config.json', config)
    files.append(artifact(root, root/'config.json'))
    for distribution in PRICE_DISTRIBUTIONS:
        cumulative = 0
        for kind, steps in (('price', [0, 1]), ('observed', [0, 2])):
            for t in steps:
                name = f'price_{t:04d}_{t+1:04d}_{distribution}' if kind == 'price' else f'observed_{t:04d}_{distribution}'
                path = root/f'{name}.jsonl.gz'
                omega, traits = [], []
                with gzip.open(path, 'xt') as stream:
                    for group in range(4):
                        for response in range(2):
                            w = .6 + .1 * (group+response+t)
                            trait = (group+response+t//2) % 2
                            write_row(stream, dict(group_index=group, group_id=f'q{group}', response_index=response,
                                distribution=distribution, source_step=t, target_step=t+1 if kind=='price' else None,
                                kind=kind, example={'hint_is_correct': group%2==0 if distribution=='eval_balanced_hint' else False},
                                traits={'output_agreement': bool(trait)}, pre_logprob=-3.,
                                post_logprob=-3.+np.log(w) if kind=='price' else None))
                            omega.append(w)
                            traits.append(trait)
                files.append(artifact(root, path))
                if kind == 'price':
                    cov = np.mean(np.array(omega)*traits) - np.mean(omega)*np.mean(traits)
                    cumulative += cov
                    logged[t+1]['price'][distribution] = {'output_agreement': {'cov_step': cov, 'cov_cum': cumulative}}
                else:
                    logged[t]['observed_eval'][distribution] = {'agreement_rate': np.mean(traits)}
    with (root/'metrics.jsonl').open('x') as stream:
        for row in logged:
            write_row(stream, row)
    files.append(artifact(root, root/'metrics.jsonl'))
    write_json(root/'files.json', {'files': files})
    write_json(root/'status.json', {'state': 'complete', 'files': artifact(root, root/'files.json')})
    return root


def test_portable_output_reconstructs_raw_data_and_shares_draws(measurement, tmp_path):
    output = tmp_path/'bootstrap'
    bootstrap_measurement(measurement, output, draws=2000, seed=72)
    summary = read_json(output/'summary.json')
    assert summary['draws'] == 2000 and len(summary['group_ids']) == 4
    with np.load(output/'draws.npz') as samples:
        np.testing.assert_array_equal(samples['observed_steps'], [0, 2])
        for metric in ('cov_step', 'cov_cum', 'prevalence', 'observed_drift', 'residual'):
            np.testing.assert_allclose(samples[f'eval_wrong_hint__{metric}'], samples[f'eval_balanced_hint__{metric}'])
        values = samples['eval_wrong_hint__residual'][:, -1]
        final = summary['final_residuals']['eval_wrong_hint']
        np.testing.assert_allclose([final['lower_95'], final['upper_95']], np.quantile(values, [.025, .975]))
    assert {int(row.split(',')[1]) for row in (output/'intervals.csv').read_text().splitlines()[1:] if ',residual,' in row} == {0, 2}
    with pytest.raises(FileExistsError):
        bootstrap_measurement(measurement, output)


def test_failed_measurement_and_corrupt_raw_data_are_rejected(measurement):
    status = read_json(measurement/'status.json')
    write_json(measurement/'status.json', {**status, 'state': 'failed'}, update=True)
    with pytest.raises(ValueError, match='complete'):
        load_measurement(measurement)
    write_json(measurement/'status.json', status, update=True)
    raw = next(measurement.glob('price_*.gz'))
    with raw.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='mismatch'):
        load_measurement(measurement)
