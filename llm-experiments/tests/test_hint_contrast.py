from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.hint_contrast import (prepare_control, control_examples, target_record,
                               measure_control, analyze_contrast, ARM)
from src.price import price_covariance
from src.study import prepare_study, read_json, read_rows
from src import study_runner as runner
from test_study_pipeline import config, tiny_loader


@pytest.mark.parametrize('behavior,expected', [('constant',0.),('uniform',0.),('solver',0.),('copy',1.),('format',0.)])
def test_control_contrast_and_price_identity(behavior, expected):
    # Four answer letters plus an invalid response. Target A; alternative hint D; gold C.
    p=np.array([.05,.05,.05,.05,.8]);z=np.array([1.,0.,0.,0.,0.])
    choices={'constant':([1,0,0,0,0],[1,0,0,0,0]),
             'uniform':([.25]*4+[0],[.25]*4+[0]),
             'solver':([0,0,1,0,0],[0,0,1,0,0]),
             'copy':([1,0,0,0,0],[0,0,0,1,0]),
             'format':([.25]*4+[0],[.25]*4+[0])}
    q0,q1=map(np.array,choices[behavior])
    actual=float(q0@z-q1@z)
    assert actual==pytest.approx(expected)
    predicted=price_covariance(q0/p,z,weights=p)-price_covariance(q1/p,z,weights=p)
    assert predicted==pytest.approx(actual)


@pytest.mark.parametrize('parsed,value', [('A',1),('D',0),(None,0)])
def test_target_is_original_option_not_displayed_hint(parsed,value):
    sample=SimpleNamespace(example=SimpleNamespace(problem_id='q',user_hint='D'),traits=SimpleNamespace(parsed_choice=parsed))
    assert target_record(sample,{'q':'A'})==dict(target_wrong_choice='A',alternative_wrong_choice='D',target_indicator=value)


def test_control_bank_changes_only_wrong_hint_and_is_reproducible(config,tmp_path):
    study=tmp_path/'study';m=prepare_study(config,study,'pilot')
    a=prepare_control(study,tmp_path/'a');b=prepare_control(study,tmp_path/'b')
    assert (a/'bank.jsonl').read_bytes()==(b/'bank.jsonl').read_bytes()
    examples,targets=control_examples(study,a,m,4)
    old=list(read_rows(study/'bank.jsonl'))
    for e,row in zip(examples,old):
        original=row['variants']['eval_wrong_hint']
        assert e.user_hint not in (targets[e.problem_id],e.gold_choice)
        assert e.hint_phrase==original['hint_phrase']
        assert e.options==original['options']
        assert e.problem_text==original['problem_text']
    with pytest.raises(FileExistsError):prepare_control(study,a)
    with (a/'bank.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='mismatch'):control_examples(study,a,m,4)


def test_complete_control_measurement_and_raw_contrast(config,tmp_path,tiny_loader):
    config['PriceConfig']['prompts_per_distribution']=4
    config['ObservedEvalConfig']['prompts_per_distribution']=4
    study=tmp_path/'study';m=prepare_study(config,study,'pilot');run=m['runs'][0]['run_id']
    control=prepare_control(study,tmp_path/'control')
    runner.train_run(study,run)
    runner.measure_run(study,run,'original',question_batch_size=3)
    directory=measure_control(study,control,run,'alternative',question_batch_size=3)
    rows=list(read_rows(directory/f'price_0000_0001_{ARM}.jsonl.gz'))
    assert [r['generation_batch_size'] for r in rows]==[6]*6+[2]*2
    for row in rows:
        assert row['target_indicator']==int(row['traits']['parsed_choice']==row['target_wrong_choice'])
        assert row['target_wrong_choice']!=row['example']['user_hint']
    output=analyze_contrast(study,control,run,tmp_path/'analysis','original','alternative')
    data=read_json(output/'contrast.json');records=data['records']
    assert records[0]['price_change']==records[0]['observed_change']==0
    assert records[1]['observed_change'] is None
    assert records[2]['observed_change']==pytest.approx(records[2]['contrast']-data['baseline_contrast'])
    assert records[2]['price_change']==pytest.approx(records[2]['original_price_sum']-records[2]['alternative_price_sum'])
    assert records[2]['residual']==pytest.approx(records[2]['observed_change']-records[2]['price_change'])
    with pytest.raises(FileExistsError):measure_control(study,control,run,'alternative')
    # Preserve completed raw files; mismatched bank provenance must be rejected.
    status=read_json(directory/'status.json');status['control_manifest_sha256']='invalid'
    (directory/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='control bank'):
        analyze_contrast(study,control,run,tmp_path/'bad','original','alternative')


def test_contrast_plot_requires_complete_collection_unless_preview(tmp_path):
    from plot_hint_contrast import load_results
    rows=[dict(step=t,price_change=0.,observed_change=0. if t%5==0 else None) for t in range(101)]
    record=dict(run_id='test',hint_probability=.1,seed=2026092201,
                study_manifest_sha256='study',control_manifest_sha256='control',records=rows)
    (tmp_path/'contrast.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='30'):load_results(tmp_path)
    assert len(load_results(tmp_path,preview=True))==1
    rows[5]['observed_change']=None
    (tmp_path/'contrast.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='observations'):load_results(tmp_path,preview=True)


def test_contrast_plot_three_seed_design(tmp_path):
    from plot_hint_contrast import load_results, CONDITIONS
    rows=[dict(step=t,price_change=0.,observed_change=0. if t%5==0 else None) for t in range(101)]
    for c in CONDITIONS:
        for seed in range(2026092201,2026092204):
            p=tmp_path/f'{c}_{seed}';p.mkdir()
            (p/'contrast.json').write_text(json.dumps(dict(hint_probability=c,seed=seed,
                study_manifest_sha256='study',control_manifest_sha256='control',records=rows)))
    assert len(load_results(tmp_path,seed_count=3))==18
    with pytest.raises(ValueError,match='30'):load_results(tmp_path)
    (p/'contrast.json').unlink()
    with pytest.raises(ValueError,match='18'):load_results(tmp_path,seed_count=3)
