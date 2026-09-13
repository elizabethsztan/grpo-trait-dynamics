import json
import numpy as np
import pytest
import yaml

from closure_study.prepare_binary_confirmation import ARCHIVE, DEVELOPMENT, CONFIRMATION, extract, freeze
from closure_study.fit_llm import fit_binary_flux


def fixture_archive(root):
    for seed in DEVELOPMENT:
        path=root/ARCHIVE/'runs'/f'hierarchical_above_hook_lr1e5_seed{seed}'
        path.mkdir(parents=True)
        (path/'config.yaml').write_text(yaml.safe_dump(dict(RunConfig=dict(seed=seed),TrainConfig=dict(num_steps=60))))
        rows=[]
        for t in range(61):
            rows.append(dict(step=t,
                observed_eval={'eval_wrong_hint':{'agreement_rate':.1+.01*t}} if t%5==0 else {},
                price={} if t==0 else {'eval_wrong_hint':{'output_agreement':{'cov_step':t/10000}}}))
        (path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_extraction_pairs_direct_old_state_with_following_covariance(tmp_path):
    fixture_archive(tmp_path)
    rows,inputs,_=extract(tmp_path)
    assert len(rows)==36 and all(i['fit_rows']==12 and i['excluded_rows_missing_direct_T']==48 for i in inputs)
    assert [r['step'] for r in rows[:12]]==list(range(0,60,5))
    assert rows[0]['T']==.1 and rows[0]['C']==.0001
    assert rows[1]['T']==pytest.approx(.15) and rows[1]['C']==.0006
    assert all(len(i['metrics_sha256'])==64 for i in inputs)


def test_missing_direct_old_state_is_not_interpolated(tmp_path):
    fixture_archive(tmp_path)
    p=tmp_path/ARCHIVE/'runs'/f'hierarchical_above_hook_lr1e5_seed{DEVELOPMENT[0]}'/'metrics.jsonl'
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    rows[5]['observed_eval']={}
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='direct old-state observations'):
        extract(tmp_path)


def test_flux_objective_is_not_an_unweighted_selection_fit():
    t=np.array([.05,.2,.4,.7,.9])
    c=np.array([.01,.025,.015,-.002,.008])
    rows=[dict(T=a,C=b,step=i) for i,(a,b) in enumerate(zip(t,c))]
    coef,mask,informative=fit_binary_flux(rows,'T',0)
    v=t*(1-t)
    assert mask.all() and informative==5
    assert coef[0]==pytest.approx(v@c/(v@v))
    assert not np.isclose(coef[0],np.mean(c/v))
    for predictor,x in [('T',t),('step',np.arange(5))]:
        coef,_,_=fit_binary_flux(rows,predictor,1)
        design=v[:,None]*np.column_stack([np.ones(5),x])
        np.testing.assert_allclose(design.T@(c-design@coef),0,atol=1e-14)


def test_freeze_refuses_existing_calibration_before_reading_data(tmp_path):
    output=tmp_path/'frozen'
    output.mkdir()
    with pytest.raises(FileExistsError,match='do not refit'):
        freeze(tmp_path,output)


def test_freeze_refuses_confirmation_data_before_reading_development(tmp_path):
    target=tmp_path/'results/llm_binary_confirmation'/f'seed{CONFIRMATION[1]}'
    target.mkdir(parents=True)
    (target/'metrics.jsonl').write_text('reserved output')
    output=tmp_path/'frozen'
    with pytest.raises(ValueError,match='confirmation directory is not empty'):
        freeze(tmp_path,output)
    assert not output.exists()
    assert (target/'metrics.jsonl').read_text()=='reserved output'
