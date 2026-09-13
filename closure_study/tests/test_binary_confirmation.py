import json
import subprocess

import numpy as np
import pytest
import yaml

from closure_study import binary_confirmation as bc


def test_group_bootstrap_keeps_pairs_and_joint_covariance():
    z = np.array([[0,.5,1],[.5,.5,.5],[.2,.3,.7]])
    w = np.array([[2,1,.5],[.5,2,1]])
    wz = np.array([[0,.7,.5],[.1,.9,.4]])
    T,C = bc.bootstrap(z,w,wz,7,731)
    rng = np.random.Generator(np.random.PCG64(731))
    for t in range(3):
        indices = rng.integers(0,3,size=(7,3))
        np.testing.assert_array_equal(T[:,t],z[t,indices].mean(axis=1))
        if t < 2:
            np.testing.assert_array_equal(C[:,t],wz[t,indices].mean(axis=1)-w[t,indices].mean(axis=1)*T[:,t])
    # Every middle-checkpoint group contains one positive and one negative:
    # group resampling cannot create prevalence uncertainty there.
    np.testing.assert_array_equal(T[:,1],.5)


def test_state_predictions_only_use_initial_state_and_time_uses_source_clock():
    models = dict(state=dict(predictor='T',coefficients=[.3,-.2]),
                  time=dict(predictor='step',coefficients=[.3,-.1]))
    T = np.array([.2,.3,.4,.5])
    a = bc.evaluate(T,np.zeros(3),models,[1,2,3])
    b = bc.evaluate([.2,.9,.1,.8],np.ones(3),models,[1,2,3])
    for name in models:
        np.testing.assert_array_equal(a[name]['path'],b[name]['path'])
        assert a[name]['T_rmse'][0] != b[name]['T_rmse'][0]
    assert a['time']['path'][0,1] == pytest.approx(.2+.2*.8*.3)


def test_failures_and_boundary_selection_remain_visible():
    model = dict(predictor='T',coefficients=[100.])
    T = np.array([[0,0,0],[.5,.5,.5],[1,1,1]])
    r = bc.evaluate(T,np.zeros((3,2)),dict(test=model),[1,2])['test']
    np.testing.assert_array_equal(r['failed_step'],[-1,1,-1])
    assert r['attempted_T'][1] == 25.5
    assert np.isnan(r['T_rmse'][1]) and np.isnan(r['terminal_error'][1])
    np.testing.assert_array_equal(r['S_count'],[0,2,0])
    assert bc.interval(r['T_rmse'],[2.5,97.5]) == (0.,0.,2)
    assert np.isfinite(r['C_rmse']).all()


def make_pool(directory):
    (directory/'pools').mkdir()
    (directory/'scores').mkdir()
    for t in range(2):
        rows=[dict(group=i//2,response=i%2,z=z,parsed_choice='A' if z else None,
                   user_hint='A',pre_logprob=-3.) for i,z in enumerate([0,1,1,1])]
        (directory/'pools'/f'{t:03d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    path = directory/'pools/000.jsonl'
    ratio = np.array([2.,1.,.5,1.5])
    np.savez(directory/'scores/000.npz',step=0,step_end=1,
             pool_sha256=bc.sha256(path),post_logprob=np.log(ratio)-3)
    return path, ratio


def test_pool_statistics_match_raw_response_covariance_and_reject_misalignment(tmp_path):
    path,ratio = make_pool(tmp_path)
    z,w,wz = bc.load_measurements(tmp_path,1,2,2)
    T,C = bc.measured(z,w,wz)
    np.testing.assert_array_equal(T,[.75,.75])
    assert C[0] == pytest.approx(np.mean(ratio*[0,1,1,1])-np.mean(ratio)*.75)
    rows=path.read_text().splitlines()
    path.write_text('\n'.join(reversed(rows))+'\n')
    with pytest.raises(ValueError,match='reordered'):
        bc.load_measurements(tmp_path,1,2,2)


def test_successor_hash_is_checked(tmp_path):
    path,_ = make_pool(tmp_path)
    path.write_text(path.read_text().replace('-3.0','-3.00'))
    with pytest.raises(ValueError,match='different pool'):
        bc.load_measurements(tmp_path,1,2,2)


def test_output_reservation_refuses_reuse_without_touching_data(tmp_path):
    p=tmp_path/'run'
    bc.reserve_directory(p)
    with pytest.raises(FileExistsError): bc.reserve_directory(p)
    (p/'data').write_text('preserve me')
    with pytest.raises(FileExistsError): bc.reserve_directory(p)
    assert (p/'data').read_text()=='preserve me'


@pytest.fixture
def frozen_repo(tmp_path,monkeypatch):
    config=tmp_path/bc.CONFIG
    config.mkdir(parents=True)
    (tmp_path/bc.PROTOCOL).write_text('unrelated synthetic protocol')
    source=tmp_path/'closure_study/test_source.py'
    source.write_text('# fixture\n')
    inputs=dict(code=[dict(path=str(source.relative_to(tmp_path)),sha256=bc.sha256(source))],development_inputs=[])
    for name,value in [('inputs.json',inputs),('coefficients.json',dict(test=dict(coefficients=[.1]))),('execution.json',{})]:
        (config/name).write_text(json.dumps(value))
    (config/'confirmation.yaml').write_text(yaml.safe_dump(dict(runs=[dict(seed=731)])))
    (config/'development_pairs.csv').write_text('synthetic\n')
    def command(*args): return subprocess.check_output(['git','-C',str(tmp_path),*args],stderr=subprocess.DEVNULL)
    command('init','-q'); command('add','.')
    command('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    monkeypatch.setattr(bc,'FROZEN_COMMIT',command('rev-parse','HEAD').decode().strip())
    return tmp_path


def test_launcher_checks_frozen_bytes_sources_and_committed_implementation(frozen_repo):
    root=frozen_repo
    _,_,_,p=bc.verify_specification(root)
    assert p['implementation_commit']==bc.FROZEN_COMMIT
    coefficients=root/bc.CONFIG/'coefficients.json'
    original=coefficients.read_bytes()
    coefficients.write_text('{}')
    with pytest.raises(ValueError,match='frozen specification'): bc.verify_specification(root)
    coefficients.write_bytes(original)
    source=root/'closure_study/test_source.py'
    source.write_text('# altered\n')
    with pytest.raises(ValueError,match='source hash'): bc.verify_specification(root)
    source.write_text('# fixture\n')
    (root/'closure_study/new_runner.py').write_text('# uncommitted\n')
    with pytest.raises(ValueError,match='commit and review'): bc.verify_specification(root)


def test_scratch_analysis_writes_all_models_uncertainty_and_failure_records(tmp_path):
    source=tmp_path/'fixture'; source.mkdir()
    make_pool(source)
    spec=dict(measurement=dict(checkpoints=[0,1],prompt_groups=2,completions_per_group=2),
              scoring=dict(prevalence_checkpoints=[1]))
    models=dict(constant=dict(predictor='T',coefficients=[.1]),
                failing=dict(predictor='T',coefficients=[100.]))
    execution=dict(bootstrap_draws=20,bootstrap_seed_offset=300000,percentiles=[2.5,97.5])
    metrics,trajectories,completion,comparisons=bc.analyze_run(source,tmp_path/'analysis',spec,models,execution,731)
    assert len(metrics)==8 and len(trajectories)==4 and len(completion)==2 and len(comparisons)==3
    assert completion[1]['status']=='failed'
    with np.load(tmp_path/'analysis/bootstrap.npz') as draws:
        assert len(draws['failing_failed_step'])==20
        assert (draws['failing_failed_step']==1).any()
    # Exercise the standalone HTML with a failed path as well as a missing run.
    (tmp_path/'analysis').rename(tmp_path/'seed731')
    bc.report(tmp_path,metrics,trajectories,completion,comparisons,
              [dict(seed=731,collection_status='complete'),dict(seed=732,collection_status='incomplete or absent')])
    html=(tmp_path/'readout.html').read_text()
    assert 'Seed 731' in html and 'Seed 732' in html and 'failed' in html and 'data:image/png' in html
