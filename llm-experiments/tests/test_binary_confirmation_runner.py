from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')

import run_binary_confirmation as runner
from src.logprobs import sequence_logprobs


class TinyTokenizer:
    pad_token='<pad>'
    eos_token='<eos>'
    pad_token_id=0
    eos_token_id=1
    def encode(self,text,add_special_tokens=True): return [2,3]
    def decode(self,ids,skip_special_tokens=True):
        valid=[i for i in ids if i>=2]
        return '<answer>'+('ABCD'[valid[-1]-2] if valid else '?')+'</answer>'


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.randn(6))
        self.generation_config=SimpleNamespace()
    def forward(self,input_ids,attention_mask):
        logits=self.logits+torch.nn.functional.one_hot(input_ids,6)*.3
        return SimpleNamespace(logits=logits)
    def generate(self,input_ids,num_return_sequences,**kwargs):
        next_token=torch.multinomial(torch.softmax(self.logits,dim=0),num_return_sequences,replacement=True)
        return torch.cat([input_ids.repeat(num_return_sequences,1),next_token[:,None]],dim=1)
    def save_pretrained(self,path):
        path.mkdir(); torch.save(self.state_dict(),path/'state.pt')


def test_rng_restores_training_state_and_model_mode_even_on_exception():
    torch.manual_seed(731)
    model=TinyPolicy()
    before=torch.get_rng_state().clone()
    rng=runner.MeasurementRNG(831)
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(RuntimeError):
        with rng.use(model):
            assert not model.training
            torch.rand(20)
            raise RuntimeError('fixture')
    assert model.training and torch.equal(before,torch.get_rng_state())
    with rng.use(model): a=torch.rand(4)
    with rng.use(model): b=torch.rand(4)
    assert not torch.equal(a,b) and torch.equal(before,torch.get_rng_state())


def test_likelihood_includes_completion_eos_excludes_prompt_and_padding():
    torch.manual_seed(731)
    model=TinyPolicy()
    samples=[SimpleNamespace(prompt_ids=[2,3],completion_ids=[4,1]),
             SimpleNamespace(prompt_ids=[5],completion_ids=[2])]
    scores=sequence_logprobs(model,samples,0,device='cpu')
    expected=[]
    for s in samples:
        seq=s.prompt_ids+s.completion_ids
        expected.append(sum(torch.log_softmax(model.logits+torch.nn.functional.one_hot(torch.tensor(seq[i-1]),6)*.3,dim=0)[seq[i]]
                            for i in range(len(s.prompt_ids),len(seq))))
    torch.testing.assert_close(scores,torch.stack(expected))
    for i,s in enumerate(samples):
        torch.testing.assert_close(scores[i],sequence_logprobs(model,[s],0,device='cpu')[0])


def test_actual_training_loop_is_identical_with_measurement_disabled_enabled(tmp_path,monkeypatch):
    config=dict(RunConfig=dict(seed=731),DataConfig=dict(difficulty='medium',
        train_hint_correct_probability=.9,eval_distributions={'eval_wrong_hint':dict(hint_correct_probability=0.,has_hint=True)}),
        TrainConfig=dict(num_steps=3,train_prompts_per_step=4,group_size=8,eps=1e-8,max_grad_norm=1.),
        GenerationConfig={},ExampleLoggingConfig=dict(enabled=False))
    measurement=dict(distribution='eval_wrong_hint',prompt_groups=3,completions_per_group=2)
    run=dict(prompt_seed=200731,measurement_generation_seed=100731)
    original=runner.train_grpo_step
    histories=[]
    for enabled in (False,True):
        torch.manual_seed(731)
        model=TinyPolicy(); tokenizer=TinyTokenizer()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        history=[]
        def record(*args,**kwargs):
            result=original(*args,**kwargs)
            history.append(([s.completion_ids for s in result[1]],model.logits.detach().clone(),torch.get_rng_state().clone()))
            return result
        monkeypatch.setattr(runner,'train_grpo_step',record)
        directory=tmp_path/str(enabled); directory.mkdir()
        runner.train(config,measurement,run,directory,dict(score_batch_size=2),
                     model,tokenizer,optimizer,'cpu',measure=enabled)
        histories.append(history)
    for a,b in zip(*histories):
        assert a[0]==b[0]
        assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
    assert (tmp_path/'False/training.jsonl').read_bytes()==(tmp_path/'True/training.jsonl').read_bytes()
    assert not torch.equal(histories[0][0][1],histories[0][-1][1])
    assert len(list((tmp_path/'True/pools').glob('*.jsonl')))==4
    assert len(list((tmp_path/'True/scores').glob('*.npz')))==3
