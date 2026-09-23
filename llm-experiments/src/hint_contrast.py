"""Controlled effect of changing which incorrect option a hint recommends."""
from dataclasses import replace
from pathlib import Path
import gzip
import random
import re

import numpy as np

from .data import LETTERS, MCArithmeticExample
from .prompts import render_prompt
from .price import price_stats
from .study import (artifact, digest, load_study, provenance, read_json, read_rows,
                    run_config, stream_seed, verify_artifact, write_json, write_row)

ARM = 'eval_alternative_wrong_hint'


def prepare_control(study, output):
    study, output = Path(study), Path(output)
    manifest = read_json(study / 'manifest.json')
    bank = list(read_rows(verify_artifact(study, manifest['bank'])))
    seed = stream_seed(manifest['bank']['seed'], 'alternative_wrong_hint')
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'bank.jsonl'
    with path.open('x') as stream:
        for row in bank:
            original = MCArithmeticExample(**row['variants']['eval_wrong_hint'])
            target = original.user_hint
            hint = rng.choice([c for c in LETTERS if c not in (target, original.gold_choice)])
            alternative = replace(original, split=ARM, user_hint=hint, hint_is_correct=False,
                prompt_text=render_prompt(original.problem_text, original.options, hint, original.hint_phrase))
            write_row(stream, {'group_id': row['group_id'], 'target_wrong_choice': target,
                               'alternative_wrong_choice': hint, 'example': alternative.to_json_dict()})
    write_json(output / 'manifest.json', {'schema_version': 1, 'study_manifest_sha256': digest(study / 'manifest.json'),
        'original_bank': manifest['bank'], 'bank': artifact(output, path, groups=len(bank)), 'seed': seed,
        'definition': 'probability of original wrong option under original hint minus alternative wrong hint',
        'source': provenance()})
    return output


def control_examples(study, control, manifest, count):
    study, control = Path(study), Path(control)
    binding = read_json(control / 'manifest.json')
    if binding['study_manifest_sha256'] != digest(study / 'manifest.json') or binding['original_bank'] != manifest['bank']:
        raise ValueError('control bank belongs to a different study')
    original = list(read_rows(verify_artifact(study, manifest['bank'])))
    rows = list(read_rows(verify_artifact(control, binding['bank'])))
    if len(rows) != len(original) or not 0 < count <= len(rows):
        raise ValueError('control bank size mismatch')
    examples, targets = [], {}
    for row, old in zip(rows, original):
        a = MCArithmeticExample(**row['example'])
        b = MCArithmeticExample(**old['variants']['eval_wrong_hint'])
        expected = replace(b, split=ARM, user_hint=row['alternative_wrong_choice'], hint_is_correct=False,
            prompt_text=render_prompt(b.problem_text, b.options, row['alternative_wrong_choice'], b.hint_phrase))
        if (row['group_id'] != old['group_id'] or row['target_wrong_choice'] != b.user_hint
            or a != expected or a.user_hint not in LETTERS or a.user_hint in (b.user_hint, b.gold_choice)
            or row['group_id'] in targets):
            raise ValueError('control bank must change only the incorrect hint')
        examples.append(a)
        targets[row['group_id']] = b.user_hint
    return examples[:count], {e.problem_id: targets[e.problem_id] for e in examples[:count]}


def target_record(sample, targets):
    target = targets[sample.example.problem_id]
    return {'target_wrong_choice': target, 'alternative_wrong_choice': sample.example.user_hint,
            'target_indicator': int(sample.traits.parsed_choice == target)}


def target_price(samples, targets, cumulative):
    weights = np.exp(np.array([s.post_logprob - s.pre_logprob for s in samples], dtype=np.float64))
    z = np.array([target_record(s, targets)['target_indicator'] for s in samples], dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError('non-finite probability ratio')
    block = price_stats(weights, z, cumulative=0.)
    block['cov_cum'] = cumulative + block['cov_step']
    return block


def measure_control(study, control, run_id, measurement_id='alternative_wrong_n512_batch64', question_batch_size=64):
    from . import study_runner as runner

    study, control = Path(study), Path(control)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', measurement_id):
        raise ValueError('invalid measurement ID')
    if type(question_batch_size) is not int or question_batch_size < 1:
        raise ValueError('question_batch_size must be a positive integer')
    manifest, run = load_study(study, run_id)
    directory = study / 'runs' / run_id
    training = read_json(directory / 'status.json')
    if training['state'] != 'complete' or training['study_manifest_sha256'] != digest(study / 'manifest.json'):
        raise ValueError('training must be complete and bound to this study')
    config = read_json(verify_artifact(directory, training['config']))
    if config != run_config(manifest, run):
        raise ValueError('training configuration differs from manifest')
    runtime = read_json(verify_artifact(directory, training['runtime']))
    checkpoints = read_json(verify_artifact(directory, training['checkpoint_index']))['checkpoints']
    if [c['step'] for c in checkpoints] != list(range(config['TrainConfig']['num_steps'] + 1)):
        raise ValueError('incomplete checkpoints')
    for checkpoint in checkpoints:
        runner.verify_checkpoint(directory, checkpoint)
    n = config['PriceConfig']['prompts_per_distribution']
    examples, targets = control_examples(study, control, manifest, n)
    config['MeasurementConfig'] = {'question_batch_size': question_batch_size, 'arm': ARM,
        'target': 'original_wrong_choice', 'control_manifest_sha256': digest(control / 'manifest.json')}
    output = study / 'measurements' / run_id / measurement_id
    with runner.attempt(output, phase='alternative_hint_measurement', run_id=run_id, measurement_id=measurement_id,
                        study_manifest_sha256=digest(study / 'manifest.json'), source=provenance(),
                        training_status_sha256=digest(directory / 'status.json'),
                        control_manifest_sha256=digest(control / 'manifest.json')) as status:
        write_json(output / 'config.json', config)
        with runner.isolated_rng(stream_seed(run['seed'], 'measurement_model')):
            model, tokenizer, device = runner.load_model(config)
            actual = runner.runtime_metadata(model, tokenizer, device, config)
            for key in ('weight_dtype', 'model_revision', 'generation_config', 'prompt_format', 'chat_template'):
                if actual[key] != runtime[key]:
                    raise ValueError(f'measurement {key} differs from training')
            write_json(output / 'runtime.json', actual)
            runner.load_checkpoint(model, directory, checkpoints[0])
            files, pending, cumulative = [], None, 0.
            extra = lambda sample: target_record(sample, targets)
            with (output / 'metrics.jsonl').open('x') as metrics:
                for step, checkpoint in enumerate(checkpoints):
                    observed = None
                    if step % config['ObservedEvalConfig']['eval_every'] == 0 or step == len(checkpoints)-1:
                        path = output / f'observed_{step:04d}_{ARM}.jsonl.gz'
                        samples = runner.collect_pool(model, tokenizer, device, config, examples,
                            config['ObservedEvalConfig']['completions_per_prompt'], path,
                            seed=stream_seed(run['seed'], 'observed', step, ARM), run_id=run_id,
                            kind='observed', distribution=ARM, source=checkpoint, record_extra=extra)
                        observed = float(np.mean([extra(s)['target_indicator'] for s in samples]))
                        files.append(artifact(output, path, rows=len(samples), kind='observed', step=step))
                    write_row(metrics, {'step': step, 'price': pending, 'observed_target_rate': observed})
                    metrics.flush()
                    if step == len(checkpoints)-1:
                        break
                    completions = config['PriceConfig']['completions_per_prompt']
                    path = output / f'source_{step:04d}_{ARM}.jsonl.gz'
                    samples = runner.collect_pool(model, tokenizer, device, config, examples, completions, path,
                        seed=stream_seed(run['seed'], 'price', step, ARM), run_id=run_id,
                        kind='price_source', distribution=ARM, source=checkpoint, record_extra=extra)
                    source_file = artifact(output, path, rows=len(samples), kind='price_source', step=step)
                    files.append(source_file)
                    successor = checkpoints[step+1]
                    runner.load_checkpoint(model, directory, successor)
                    paired = runner.score_successor(model, tokenizer, device, samples, completions)
                    path = output / f'price_{step:04d}_{step+1:04d}_{ARM}.jsonl.gz'
                    with gzip.open(path, 'xt') as raw:
                        for index, sample in enumerate(paired):
                            row = runner.sample_record(sample, index, completions, run_id=run_id,
                                distribution=ARM, kind='price', source=checkpoint, target=successor)
                            row.update(extra(sample), source_pool_sha256=source_file['sha256'])
                            write_row(raw, row)
                    files.append(artifact(output, path, rows=len(paired), kind='price', source_step=step, target_step=step+1))
                    pending = target_price(paired, targets, cumulative)
                    cumulative = pending['cov_cum']
                    status['completed_transitions'] = step+1
                    write_json(output / 'status.json', status, update=True)
            files.extend(artifact(output, output / name) for name in ('metrics.jsonl', 'runtime.json', 'config.json'))
            write_json(output / 'files.json', {'files': files})
            status['files'] = artifact(output, output / 'files.json')
    return output


def analyze_contrast(study, control, run_id, output, original_id='n512_batch64',
                     alternative_id='alternative_wrong_n512_batch64'):
    """Reconstruct both arms from raw responses before subtracting their accounting."""
    from .price import price_covariance

    study, output = Path(study), Path(output)
    manifest, run = load_study(study, run_id)
    count = manifest['config']['PriceConfig']['prompts_per_distribution']
    examples, targets = control_examples(study, control, manifest, count)
    groups = [e.problem_id for e in examples]
    steps = manifest['config']['TrainConfig']['num_steps']
    every = manifest['config']['ObservedEvalConfig']['eval_every']
    observed_steps = [t for t in range(steps+1) if t % every == 0 or t == steps]
    training_root = study / 'runs' / run_id
    training_status = read_json(training_root / 'status.json')
    if training_status['state'] != 'complete' or training_status['study_manifest_sha256'] != digest(study / 'manifest.json'):
        raise ValueError('training must be complete and bound to this study')
    checkpoints = read_json(verify_artifact(training_root, training_status['checkpoint_index']))['checkpoints']
    if [c['step'] for c in checkpoints] != list(range(steps+1)):
        raise ValueError('incomplete checkpoint index')
    arm_values, bindings = [], []
    for alternative, measurement_id in ((False, original_id), (True, alternative_id)):
        root = study / 'measurements' / run_id / measurement_id
        status = read_json(root / 'status.json')
        if status['state'] != 'complete' or status['study_manifest_sha256'] != digest(study / 'manifest.json'):
            raise ValueError('measurement must be complete and bound to this study')
        if status['training_status_sha256'] != digest(training_root / 'status.json'):
            raise ValueError('measurement bound to different training artifacts')
        if alternative and status['control_manifest_sha256'] != digest(Path(control) / 'manifest.json'):
            raise ValueError('alternative measurement belongs to another control bank')
        refs = read_json(verify_artifact(root, status['files']))['files']
        inventory = {r['path']: r for r in refs}
        if len(inventory) != len(refs):
            raise ValueError('duplicate inventory paths')
        def verified(name):
            return verify_artifact(root, inventory[name])
        config = read_json(verified('config.json'))
        for section in ('PriceConfig', 'ObservedEvalConfig'):
            if config[section] != manifest['config'][section]:
                raise ValueError('measurement budget/settings differ from study')
        logged = list(read_rows(verified('metrics.jsonl')))
        if [r['step'] for r in logged] != list(range(steps+1)):
            raise ValueError('incomplete measurement trajectory')
        distribution = ARM if alternative else 'eval_wrong_hint'
        def pool(name, step, price):
            rows = list(read_rows(verified(name)))
            k = config['PriceConfig' if price else 'ObservedEvalConfig']['completions_per_prompt']
            if len(rows) != count*k:
                raise ValueError('incomplete response pool')
            z, omega = [], []
            for index, row in enumerate(rows):
                group = groups[index//k]
                expected = (run_id, group, index//k, index%k, distribution, step,
                            checkpoints[step]['sha256'], 'price' if price else 'observed')
                actual = tuple(row[key] for key in ('run_id','group_id','group_index','response_index',
                                                   'distribution','source_step','source_checkpoint_sha256','kind'))
                target = targets[group]
                displayed = examples[index//k].user_hint if alternative else target
                if actual != expected or row['example']['user_hint'] != displayed:
                    raise ValueError('question, checkpoint, or arm mismatch')
                value = int(row['traits']['parsed_choice'] == target)
                if alternative and (row['target_wrong_choice'] != target or row['alternative_wrong_choice'] != displayed
                                    or row['target_indicator'] != value):
                    raise ValueError('incorrect contrast target metadata')
                if price:
                    if row['target_step'] != step+1 or row['target_checkpoint_sha256'] != checkpoints[step+1]['sha256']:
                        raise ValueError('successor checkpoint mismatch')
                    source_name = f'source_{step:04d}_{distribution}.jsonl.gz'
                    if row['source_pool_sha256'] != inventory[source_name]['sha256']:
                        raise ValueError('source pool binding mismatch')
                    omega.append(np.exp(float(row['post_logprob'])-float(row['pre_logprob'])))
                z.append(value)
            if not np.isfinite(omega).all():
                raise ValueError('non-finite ratios')
            if price:
                verified(f'source_{step:04d}_{distribution}.jsonl.gz')
            return np.asarray(z, dtype=float), np.asarray(omega, dtype=float)
        cumulative = [0.]
        for t in range(steps):
            z, omega = pool(f'price_{t:04d}_{t+1:04d}_{distribution}.jsonl.gz', t, True)
            cov = price_covariance(omega, z)
            cumulative.append(cumulative[-1]+cov)
            block = logged[t+1]['price'] if alternative else logged[t+1]['price'][distribution]['output_agreement']
            if not np.allclose([cov,cumulative[-1]],[block['cov_step'],block['cov_cum']],atol=1e-10,rtol=1e-10):
                raise ValueError('raw accounting differs from logged curve')
        observed = {}
        for t in observed_steps:
            z, _ = pool(f'observed_{t:04d}_{distribution}.jsonl.gz', t, False)
            observed[t] = float(z.mean())
            expected = logged[t]['observed_target_rate'] if alternative else logged[t]['observed_eval'][distribution]['agreement_rate']
            if not np.isclose(observed[t],expected,atol=1e-12,rtol=0):
                raise ValueError('raw observed target rate differs from log')
        arm_values.append((cumulative,observed))
        bindings.append({'measurement_id':measurement_id,'status_sha256':digest(root/'status.json'),
                         'inventory_sha256':digest(root/'files.json')})
    original, alternative = arm_values
    baseline = original[1][0]-alternative[1][0]
    records = []
    for t in range(steps+1):
        price = original[0][t]-alternative[0][t]
        observed = original[1][t]-alternative[1][t] if t in original[1] else None
        change = observed-baseline if observed is not None else None
        records.append({'step':t,'original_price_sum':original[0][t],'alternative_price_sum':alternative[0][t],
            'price_change':price,'original_target_rate':original[1].get(t),'alternative_target_rate':alternative[1].get(t),
            'contrast':observed,'observed_change':change,'residual':change-price if change is not None else None})
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'contrast.json',{'schema_version':1,**run,'definition':'original hint minus alternative wrong hint; original target option in both arms',
        'baseline_contrast':baseline,'records':records,'inputs':bindings,'study_manifest_sha256':digest(study/'manifest.json'),
        'control_manifest_sha256':digest(Path(control)/'manifest.json'),'source':provenance()})
    return output
