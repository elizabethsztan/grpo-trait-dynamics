"""Execute one registered binary confirmation run; no calibration or overrides."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closure_study.binary_confirmation import verify_specification, reserve_directory
from closure_study.io import sha256
from src.generation import configure_tokenizer_and_model
from src.grpo import attach_logprobs, make_examples_for_distribution, sample_rollouts, train_grpo_step
from src.lora_freeze import apply_lora_above_hook
from train_grpo_price import (_generate_train_examples, _resolve_device, _resolve_dtype,
                              _write_jsonl_line, _write_rollout_examples)


class MeasurementRNG:
    """Persistent measurement stream, with the training Torch stream restored."""
    def __init__(self, seed):
        import torch
        self.devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            self.cpu = torch.get_rng_state()
            self.cuda = torch.cuda.get_rng_state_all()

    @contextmanager
    def use(self, model):
        import torch
        training = model.training
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu)
            torch.cuda.set_rng_state_all(self.cuda)
            model.eval()
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                self.cuda = torch.cuda.get_rng_state_all()
                model.train(training)


def initialize(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    seed = config['RunConfig']['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = _resolve_device(config['RunConfig'])
    cfg = config['ModelConfig']
    kwargs = dict(revision=cfg['revision'], trust_remote_code=cfg['trust_remote_code'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model_name'], torch_dtype=_resolve_dtype(config['RunConfig']), **kwargs).to(device)
    configure_tokenizer_and_model(tokenizer, model)
    if cfg['use_gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    model = apply_lora_above_hook(model, config['LoRAConfig']).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['TrainConfig']['learning_rate'], weight_decay=config['TrainConfig']['weight_decay'])
    return model, tokenizer, optimizer, device


def score(model, tokenizer, samples, device, field, batch_size):
    # Same sequence scorer and precision as development, with bounded memory.
    return [s for i in range(0, len(samples), batch_size)
            for s in attach_logprobs(model, tokenizer, samples[i:i+batch_size], device, field)]


def collect(model, tokenizer, config, measurement, prompt_rng, generation_rng, device, batch_size):
    distribution = measurement['distribution']
    examples = make_examples_for_distribution(config['DataConfig']['eval_distributions'][distribution],
        config['DataConfig'], n=measurement['prompt_groups'], seed=prompt_rng.getrandbits(64), split=distribution)
    with generation_rng.use(model):
        samples = sample_rollouts(model, tokenizer, examples, config['GenerationConfig'],
            measurement['completions_per_group'], device)
        return score(model, tokenizer, samples, device, 'pre_logprob', batch_size)


def save_pool(path, samples, completions):
    with path.open('x') as f:
        for i, s in enumerate(samples):
            row = dict(group=i//completions, response=i%completions,
                prompt_ids=s.prompt_ids, completion_ids=s.completion_ids,
                prompt_text=s.example.prompt_text, completion_text=s.generated.completion_text,
                gold_choice=s.example.gold_choice, user_hint=s.example.user_hint,
                parsed_choice=s.traits.parsed_choice, z=int(s.traits.output_agreement),
                pre_logprob=s.pre_logprob)
            f.write(json.dumps(row, allow_nan=False)+'\n')


def train(config, measurement, run, output, execution, model, tokenizer, optimizer, device, *, measure=True):
    """The measure switch is for paired scratch checks; the launcher always measures."""
    output = Path(output)
    steps = config['TrainConfig']['num_steps']
    prompt_rng = random.Random(run['prompt_seed'])
    generation_rng = MeasurementRNG(run['measurement_generation_seed'])
    if measure:
        (output/'pools').mkdir()
        (output/'scores').mkdir()
    samples = None
    for step in range(steps+1):
        if measure:
            samples = collect(model, tokenizer, config, measurement, prompt_rng, generation_rng,
                              device, execution['score_batch_size'])
            pool_path = output/'pools'/f'{step:03d}.jsonl'
            save_pool(pool_path, samples, measurement['completions_per_group'])
        if step == steps:
            break
        cfg = config['TrainConfig']
        summary, training_samples = train_grpo_step(model, tokenizer, optimizer,
            _generate_train_examples(config, step), config['GenerationConfig'],
            group_size=cfg['group_size'], eps=cfg['eps'], max_grad_norm=cfg['max_grad_norm'], device=device)
        _write_jsonl_line(output/'training.jsonl', dict(step=step+1, **summary))
        _write_rollout_examples(output, step, training_samples, config['ExampleLoggingConfig'])
        if measure:
            with generation_rng.use(model):
                post = score(model, tokenizer, samples, device, 'post_logprob', execution['score_batch_size'])
            np.savez(output/'scores'/f'{step:03d}.npz', step=step, step_end=step+1,
                     pool_sha256=sha256(pool_path), post_logprob=[s.post_logprob for s in post])
        print(f'completed update {step+1}/{steps}', flush=True)
    model.save_pretrained(output/'adapter')
    (output/'completed.json').write_text(json.dumps(dict(updates=steps, measurement=measure))+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    spec, coefficients, execution, provenance = verify_specification(ROOT)
    matches = [r for r in spec['runs'] if r['seed'] == args.seed]
    if len(matches) != 1:
        parser.error('seed must be one of the three registered confirmation seeds')
    run = matches[0]
    output = reserve_directory(ROOT/run['output'])
    config = deepcopy(spec['training_config'])
    config['RunConfig'].update(seed=run['seed'], name=f"seed{run['seed']}")
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output/'specification.json').write_text(json.dumps(dict(config=config, run=run,
        measurement=spec['measurement'], scoring=spec['scoring'], coefficients=coefficients,
        execution=execution), indent=2)+'\n')
    (output/'examples').mkdir()
    try:
        model, tokenizer, optimizer, device = initialize(config)
        import torch, transformers, peft
        (output/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,
            transformers=transformers.__version__, peft=peft.__version__, numpy=np.__version__,
            device=device, dtype=str(next(model.parameters()).dtype)), indent=2)+'\n')
        train(config, spec['measurement'], run, output, execution, model, tokenizer, optimizer, device)
    except Exception as error:
        (output/'failed.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
