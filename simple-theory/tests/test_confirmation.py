import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_confirmation import DEFAULT_CONFIG, compare, make_policy, measure, population, run


@pytest.fixture
def config():
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg['environment_seed'] = 41
    cfg['policy'].update(N=24, K=4, G=4)
    cfg['neural'].update(N_eval=16, d=8)
    for system in ('tabular', 'neural_continuous'):
        cfg[system]['training'].update(steps=4, batch_size=8)
        cfg[system]['seeds'] = [71, 72, 73, 74, 75]
    torch.set_num_threads(1)
    return cfg


@pytest.mark.parametrize('system', ['tabular', 'neural_continuous'])
def test_initialization_preserves_environment_and_separates_training(config, system):
    base = make_policy(config, system, 71, 0.0)
    initial = population(base)
    states_a, _ = measure(base, config[system]['training'])
    repeated = make_policy(config, system, 71, 0.0)
    states_repeat, _ = measure(repeated, config[system]['training'])
    other = make_policy(config, system, 72, 0.0)
    states_other, _ = measure(other, config[system]['training'])
    perturbed = make_policy(config, system, 71, 0.5)
    np.testing.assert_array_equal(base.s, other.s)
    np.testing.assert_array_equal(base.s, perturbed.s)
    assert states_a == states_repeat
    assert states_a != states_other
    assert not np.allclose(initial, population(perturbed))
    if system == 'tabular':
        np.testing.assert_array_equal(base.q, perturbed.q)
        np.testing.assert_array_equal(base.q, other.q)
    else:
        for key in ('_h_train','_h_eval','_q_train','_q_eval'):
            np.testing.assert_array_equal(getattr(base,key), getattr(perturbed,key))
            np.testing.assert_array_equal(getattr(base,key), getattr(other,key))
        assert torch.count_nonzero(perturbed._h_eval[:,:,1]) == 0
        assert not perturbed._optimizer.state
        assert float(perturbed._w.detach()[0]) == 0.5


@pytest.mark.parametrize('system', ['tabular', 'neural_continuous'])
def test_exact_measurements_and_initial_state_only_prediction(config, system):
    policy = make_policy(config, system, 71, 0.5)
    states, transitions = measure(policy, config[system]['training'])
    np.testing.assert_allclose(np.diff([s['mu'] for s in states]),
                               [r['C'] for r in transitions], atol=1e-14)
    np.testing.assert_allclose(np.diff([s['V'] for s in states]),
                               [r['Q']-r['C']**2 for r in transitions], atol=1e-14)
    binary = system == 'tabular'
    result = compare(states, transitions, config[system], binary)
    changed = copy.deepcopy(states)
    for s in changed[1:]:
        s['mu'] += 0.1
        s['V'] *= 2
    alternative = compare(changed, transitions, config[system], binary)
    for a, b in zip(result['trajectories'], alternative['trajectories']):
        assert a['mu_generated'] == b['mu_generated']
        assert a['V_generated'] == b['V_generated']
    assert result['metrics'] != alternative['metrics']
    if not binary:
        for r in result['residuals']:
            assert r['Q_total_residual'] == pytest.approx(
                r['Q_flux_residual']+r['Q_moment_residual']+r['Q_selection_residual'], abs=1e-14)


def test_failed_paths_and_output_overwrite_are_not_hidden(config, tmp_path):
    config['tabular']['selection_state'] = [100.0]
    dest = tmp_path/'run'
    result = run(config, 'tabular', 'perturbed', 71, dest)
    failed = next(r for r in result['metrics'] if r['model']=='state')
    assert failed['status'] == 'invalid_probability_at_1'
    assert failed['mu_rmse'] is None
    assert result['failures'][0]['attempted_mu'] > 1
    assert (dest/'transitions.csv').exists()
    with pytest.raises(FileExistsError):
        run(config, 'tabular', 'perturbed', 71, dest)
    with pytest.raises(ValueError, match='not registered'):
        run(config, 'tabular', 'perturbed', 75, tmp_path/'unregistered')
