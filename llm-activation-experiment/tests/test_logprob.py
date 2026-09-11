"""Run with: python -m unittest discover -s tests -p test_logprob.py"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

spec = importlib.util.spec_from_file_location(
    'activation_logprob', Path(__file__).resolve().parents[1] / 'src/logprob.py')
logprob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logprob)


class ForceTokens(LogitsProcessor):
    def __init__(self, completions, width):
        self.completions, self.width = completions, width

    def __call__(self, ids, scores):
        j = ids.shape[1] - self.width
        tokens = [int(c[j]) if j < len(c) else 0 for c in self.completions]
        return torch.full_like(scores, float('-inf')).scatter_(
            1, torch.tensor(tokens)[:, None], 0.)


class GenerationLogprobTest(unittest.TestCase):
    def test_matches_generation_before_and_after_weight_change(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32,
            n_embd=16, n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2,
            pad_token_id=0)).eval()
        policy = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
        prompts = [torch.tensor(p) for p in ([3, 4], [5, 6, 7, 8], [9])]
        # Immediate EOS, later EOS, and a completion truncated without EOS.
        comps = [torch.tensor(c) for c in ([2], [4, 5, 6, 2], [7, 8, 9, 10, 11])]
        ids = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8], [0, 0, 0, 9]])
        mask = (ids != 0).long()
        pairs = list(zip(prompts, comps))
        expected_runs, scored_runs = [], []
        for changed in (False, True):
            if changed:
                with torch.no_grad():
                    model.transformer.wte.weight.add_(torch.randn_like(model.transformer.wte.weight) * .02)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=mask, do_sample=True, top_k=0,
                    max_new_tokens=5, output_logits=True, return_dict_in_generate=True,
                    logits_processor=[ForceTokens(comps, ids.shape[1])])
            expected = []
            for i, c in enumerate(comps):
                self.assertTrue(torch.equal(out.sequences[i, 4:4 + len(c)], c))
                expected.append(sum(float(torch.log_softmax(out.logits[j][i].float(), -1)[tok])
                                    for j, tok in enumerate(c)))
            rng = torch.random.get_rng_state()
            scored = logprob.generation_logprob_batch(policy, pairs)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            torch.testing.assert_close(torch.tensor(scored), torch.tensor(expected), rtol=0, atol=2e-6)
            expected_runs.append(torch.tensor(expected, dtype=torch.float64))
            scored_runs.append(torch.tensor(scored, dtype=torch.float64))
        torch.testing.assert_close((scored_runs[1] - scored_runs[0]).exp(),
                                   (expected_runs[1] - expected_runs[0]).exp(), rtol=3e-6, atol=0)
        self.assertEqual(logprob.generation_logprob_batch(policy, []), [])
        self.assertEqual(logprob.generation_logprob_batch(policy, [(prompts[0], torch.tensor([], dtype=torch.long))]), [0.])
        mixed = [(prompts[0], torch.tensor([], dtype=torch.long)), *pairs[1:]]
        mixed_scores = logprob.generation_logprob_batch(policy, mixed)
        self.assertEqual(mixed_scores[0], 0.)
        torch.testing.assert_close(torch.tensor(mixed_scores[1:]), scored_runs[-1][1:].float(), rtol=0, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
