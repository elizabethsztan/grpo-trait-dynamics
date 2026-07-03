# LLM Activation Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **No test suite** (per project instruction): each task's cycle is *build → smoke-verify by running it → (optional) commit*, not TDD. Verification is a real run with expected stdout. Correctness is guarded by the in-pipeline assertions from the spec (frozen-trait check, `mean(omega) ~ 1`, verifier spot-checks), not unit tests.
> **Commits are optional checkpoints** — the user does not require them; include or skip per preference.

**Goal:** Empirically test the sampled Price-equation prediction for a frozen Qwen-Scope SAE trait in Qwen3.5-2B-Base under GRPO on GSM8K — does `mean(omega*s) - mean(s)` recover the direct trait drift, and at what sample budget.

**Architecture:** Four staged scripts (capability check → feature discovery → GRPO → Price eval) over shared `src/` modules. The SAE reads the residual stream at layer L; LoRA is attached strictly downstream of the SAE hook so the trait is frozen (Δs=0) and the Price equation reduces to pure selection.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `transformers` + `datasets`, `peft` (LoRA), Qwen-Scope SAE weights, `numpy`, `matplotlib`, `pyyaml`. Run everything with `uv run python -m ...` on a Swirles GPU.

## Global Constraints

- Base model: `Qwen/Qwen3.5-2B-Base` (24 layers, hidden 2048). Run as a base model with a fixed few-shot GSM8K prompt (no chat/thinking template).
- SAE: Qwen-Scope residual SAE for Qwen3.5-2B-Base at layer L (default **L=12**), TopK, width 32K. Load via the convention confirmed in Task 2.
- Sampling for every rollout whose logprobs feed ω: **temperature 1.0, top-p 1.0, no top-k**. Never tune temperature to fix reward variance.
- LoRA is placed **strictly downstream of the exact SAE hook activation** (confirmed in Task 2). This is the load-bearing invariant.
- Trait score: `s(x,a) = max over completion tokens τ of feature activation f_{L,m}(x,a)_τ` (completion tokens only, excluding the prompt).
- ω uses the sequence logprob over the **exact sampled token span** (EOS included iff generation stopped on EOS; only emitted tokens if truncated), computed identically under π_t and π_{t+1}.
- KL penalty (Phase 2): per-token **k3 estimator** against frozen reference `π_ref = π_0` (base model, LoRA disabled), token-averaged, coefficient `kl_coef`.
- All code in `src/`; all outputs under `results/` (gitignored); configs in `experiments/configs/`. `LOGGER = logging.getLogger(__name__)` in every script. Floats rounded to ≤4 dp in JSON/JSONL dumps.
- Reference spec: `llm-activation-experiment/experiment-spec.md`. Theory: `simple-theory/project_claim.md`.

---

## File Structure

```
llm-activation-experiment/
  pyproject.toml                # deps
  README.md                     # one-liner per experiment + run commands
  experiment-spec.md            # (exists) design spec
  implementation-plan.md        # (this file)
  experiments/
    run_capability_check.py     # Phase 0
    run_feature_discovery.py    # Phase 1
    run_grpo.py                 # Phase 2
    run_price_eval.py           # Phase 3
    configs/
      config.yaml               # full run
      config_smoke.yaml         # tiny end-to-end
  src/
    __init__.py
    data.py        # GSM8K load, few-shot prompt builder, train/feat/eval splits
    verifier.py    # answer extraction + exact-match reward
    sae.py         # load & freeze Qwen-Scope SAE at layer L; forward hook; encode
    model.py       # load Qwen base + tokenizer; attach LoRA downstream of hook; ref-disable ctx
    trait.py       # teacher-force a completion, read hook activation, s = max_tau f_{L,m}
    logprob.py     # sequence logprob over exact sampled span (EOS/truncation-aware)
    grpo.py        # GRPO step: advantages + KL + LoRA-only Adam update
    plotting.py    # serif-styled plots shared by the phase scripts
  results/                      # gitignored
```

**Module responsibilities & key interfaces** (locked here; later tasks depend on these exact names):

- `data.py` → `load_gsm8k_splits(cfg) -> dict[str, list[Example]]`; `Example = {"question": str, "gold": str}`; `build_prompt(question, few_shot_examples, n_shots) -> str`.
- `verifier.py` → `extract_answer(completion: str) -> str | None`; `reward(completion: str, gold: str) -> float`.
- `sae.py` → `load_sae(repo: str, layer: int, device) -> SAE`; `SAE.encode(acts: Tensor[..., d_model]) -> Tensor[..., n_features]`; `SAE.hook_name(layer) -> str` (the exact module path to hook).
- `model.py` → `load_policy(cfg, device) -> Policy`; `Policy.model`, `Policy.tokenizer`, `Policy.hook_module(layer)`, `Policy.attach_lora(layer_L, rank)`, `Policy.disable_lora()` (contextmanager → reference model), `Policy.save_lora(path)`, `Policy.load_lora(path)`.
- `trait.py` → `trait_scores(policy, sae, features, prompt_ids, completion_ids, layer) -> Tensor[n_features]` (max over completion tokens); `collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids) -> Tensor[n_features]` (all features).
- `logprob.py` → `sequence_logprob(policy, prompt_ids, completion_ids) -> float` (sum of token logprobs over the completion span).
- `grpo.py` → `grpo_step(policy, ref_logprobs_fn, batch, cfg) -> dict` (metrics: mean_reward, mean_adv, degenerate_frac, kl).
- `plotting.py` → `plot_expected_trait(...)`, `plot_price_check(...)`, `plot_price_convergence(...)`, `plot_grid_price(...)`, `plot_omega_diagnostics(...)`.

---

## Task 1: Project scaffold, dependencies, configs

**Files:**
- Create: `llm-activation-experiment/pyproject.toml`
- Create: `llm-activation-experiment/src/__init__.py`
- Create: `llm-activation-experiment/experiments/configs/config.yaml`
- Create: `llm-activation-experiment/experiments/configs/config_smoke.yaml`
- Create: `llm-activation-experiment/.gitignore`

**Interfaces:**
- Produces: the config schema every phase reads; the `uv` environment.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "llm-activation-experiment"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "transformers>=4.44",
    "datasets>=2.20",
    "peft>=0.12",
    "safetensors>=0.4",
    "huggingface-hub>=0.24",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Create `src/__init__.py`** (empty file).

- [ ] **Step 3: Write `experiments/configs/config.yaml`** — copy the full config block from `experiment-spec.md` §Config verbatim (ModelConfig, DataConfig, GenConfig, CapabilityConfig, FeatureDiscoveryConfig, GRPOConfig, PriceEvalConfig, OutputConfig).

- [ ] **Step 4: Write `experiments/configs/config_smoke.yaml`** — same keys, tiny values:

```yaml
ModelConfig:
  backbone: Qwen/Qwen3.5-2B-Base
  sae_repo: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
  layer_L: 12
  seed: 290402
DataConfig: {dataset: gsm8k, n_train: 50, n_feat: 40, n_eval: 40}
GenConfig: {temperature: 1.0, top_p: 1.0, top_k: 0, max_new_tokens: 128, prompt_style: few_shot, n_shots: 4}
CapabilityConfig: {n_prompts: 40, temperature: 1.0, target_success_band: [0.2, 0.7]}
FeatureDiscoveryConfig: {K: 4, candidate_layers: [12], min_activity: 0.1, top_k_each_sign: 2, val_frac: 0.5, n_controls: 2}
GRPOConfig: {steps: 8, batch_size: 8, G: 4, lr: 1.0e-5, kl_coef: 0.01, clip: null, lora_rank: 8, lora_layers: ">L", checkpoint_every: 1, num_runs: 1, seed: 290402}
PriceEvalConfig: {price_samples: [32, 128], direct_samples: 256, eval_prompts: 32}
OutputConfig: {results_dir: results, name: smoke_L12}
```

- [ ] **Step 5: Write `.gitignore`**

```
.venv/
results/
__pycache__/
*.pyc
```

- [ ] **Step 6: Verify the env builds and config parses**

Run: `cd llm-activation-experiment && uv run python -c "import yaml, torch, transformers, peft; print(yaml.safe_load(open('experiments/configs/config_smoke.yaml'))['ModelConfig']['layer_L'])"`
Expected: first run creates `.venv` and installs deps, then prints `12`.

- [ ] **Step 7 (optional): Commit**

```bash
git add llm-activation-experiment/pyproject.toml llm-activation-experiment/src/__init__.py llm-activation-experiment/experiments/configs llm-activation-experiment/.gitignore
git commit -m "chore: scaffold llm-activation-experiment (deps + configs)"
```

---

## Task 2: Qwen-Scope SAE loader + hook-convention spike (`src/sae.py`)

This is the load-bearing verification from the spec's "Pre-implementation verification": confirm exactly which residual activation the SAE was trained on, so LoRA can be placed strictly downstream.

**Files:**
- Create: `llm-activation-experiment/src/sae.py`

**Interfaces:**
- Produces: `load_sae(repo, layer, device) -> SAE`; `SAE.encode(acts) -> Tensor[..., n_features]`; `SAE.d_model`, `SAE.n_features`; `SAE.hook_name(layer) -> str` returning the transformers module path whose **output** is the trained-on activation (e.g. `model.layers.{layer}` for the post-block residual).

- [ ] **Step 1: Inspect the Qwen-Scope repo to pin the convention**

Run: `cd llm-activation-experiment && uv run python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100')))"`
Expected: a file listing (per-layer safetensors + a config json). Read the repo's model card / config to determine: (a) the hook point (pre-block / post-attn / post-MLP/post-block residual), (b) the weight tensor names (encoder `W_enc`/`b_enc`, decoder, and TopK `k`), (c) per-layer file naming. **Record findings in a top-of-file docstring comment in `sae.py`.**

- [ ] **Step 2: Implement `sae.py` against the confirmed convention**

```python
"""Qwen-Scope residual-stream SAE loader.

HOOK CONVENTION (confirmed via Task 2 Step 1 against
Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100):
  - Trained on the POST-BLOCK residual stream, i.e. the output of
    transformers module `model.layers.{L}`. LoRA must therefore live on
    layers strictly greater than L. Update this docstring if the inspected
    repo says otherwise (e.g. post-attn), and adjust hook_name accordingly.
  - TopK activation with k as stored in the SAE config.
"""
import logging
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

LOGGER = logging.getLogger(__name__)


class SAE(nn.Module):
    def __init__(self, W_enc, b_enc, W_dec, b_dec, k):
        super().__init__()
        self.register_buffer("W_enc", W_enc)   # (d_model, n_features)
        self.register_buffer("b_enc", b_enc)   # (n_features,)
        self.register_buffer("W_dec", W_dec)   # (n_features, d_model)
        self.register_buffer("b_dec", b_dec)   # (d_model,)
        self.k = int(k)
        self.d_model = W_enc.shape[0]
        self.n_features = W_enc.shape[1]

    @torch.no_grad()
    def encode(self, acts):
        # acts: (..., d_model) -> features (..., n_features), TopK sparse.
        pre = (acts - self.b_dec) @ self.W_enc + self.b_enc
        if self.k and self.k < self.n_features:
            topv, topi = torch.topk(pre, self.k, dim=-1)
            out = torch.zeros_like(pre)
            out.scatter_(-1, topi, torch.relu(topv))
            return out
        return torch.relu(pre)

    @staticmethod
    def hook_name(layer):
        return f"model.layers.{layer}"


def load_sae(repo, layer, device):
    # Filenames/keys per the convention recorded in Task 2 Step 1.
    weights_path = hf_hub_download(repo, filename=f"layer_{layer}.safetensors")
    sd = load_file(weights_path)
    sae = SAE(sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"], sd.get("k", 100))
    sae = sae.to(device).eval()
    LOGGER.info(f"loaded SAE repo={repo} layer={layer} d_model={sae.d_model} "
                f"n_features={sae.n_features} k={sae.k} hook={SAE.hook_name(layer)}")
    return sae
```

Adapt filename/key names to whatever Step 1 revealed — the structure stays, only the strings change.

- [ ] **Step 3: Smoke-verify load + encode shapes on GPU**

Run: `cd llm-activation-experiment && uv run python -c "import torch, logging; logging.basicConfig(level=logging.INFO); from src.sae import load_sae; s=load_sae('Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100',12,'cuda'); x=torch.randn(3,5,s.d_model,device='cuda'); f=s.encode(x); print(f.shape, (f>0).float().mean().item())"`
Expected: prints `torch.Size([3, 5, 32768]) <~k/32768>` and the active fraction ≈ k/32768. Confirms load + TopK sparsity.

- [ ] **Step 4 (optional): Commit** `git add src/sae.py && git commit -m "feat(sae): Qwen-Scope loader + confirmed hook convention"`

---

## Task 3: GSM8K data + few-shot prompt (`src/data.py`)

**Files:**
- Create: `llm-activation-experiment/src/data.py`

**Interfaces:**
- Produces: `load_gsm8k_splits(cfg) -> dict` with keys `train/feat/eval`, each a `list[dict]` of `{"question", "gold"}`; `build_prompt(question, shots, n_shots) -> str`; `FEW_SHOT` constant (list of solved examples ending in `Answer: N`).

- [ ] **Step 1: Implement `data.py`**

```python
import logging
from datasets import load_dataset

LOGGER = logging.getLogger(__name__)

# Four fixed worked examples; completions end with the exact "Answer: N" line
# the verifier keys on. Kept short to bound sequence length.
FEW_SHOT = [
    {"q": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell altogether?",
     "a": "In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72.\nAnswer: 72"},
    {"q": "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
     "a": "Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes: 50 * 0.2 = 10.\nAnswer: 10"},
    {"q": "Betty needs $100 for a wallet. She has half. Her parents give $15 and grandparents twice that. How much more does she need?",
     "a": "Betty has 100 / 2 = 50. Grandparents give 2 * 15 = 30. She now has 50 + 15 + 30 = 95. She needs 100 - 95 = 5.\nAnswer: 5"},
    {"q": "James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?",
     "a": "Each time he writes 3 * 2 = 6 pages. Twice a week: 6 * 2 = 12 pages. Per year: 12 * 52 = 624.\nAnswer: 624"},
]


def build_prompt(question, shots=FEW_SHOT, n_shots=4):
    blocks = []
    for ex in shots[:n_shots]:
        blocks.append(f"Question: {ex['q']}\n{ex['a']}")
    blocks.append(f"Question: {question}\n")
    return "\n\n".join(blocks)


def _gold_from_answer(answer_field):
    # GSM8K gold answers end with "#### <number>".
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k_splits(cfg):
    data_cfg = cfg["DataConfig"]
    seed = cfg["ModelConfig"]["seed"]
    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed)
    examples = [{"question": r["question"], "gold": _gold_from_answer(r["answer"])} for r in ds]
    n_train, n_feat, n_eval = data_cfg["n_train"], data_cfg["n_feat"], data_cfg["n_eval"]
    splits = {
        "train": examples[:n_train],
        "feat": examples[n_train:n_train + n_feat],
        "eval": examples[n_train + n_feat:n_train + n_feat + n_eval],
    }
    LOGGER.info(f"splits: train={len(splits['train'])} feat={len(splits['feat'])} eval={len(splits['eval'])}")
    return splits
```

- [ ] **Step 2: Smoke-verify splits are disjoint and prompt is well-formed**

Run: `cd llm-activation-experiment && uv run python -c "import yaml,logging; logging.basicConfig(level=logging.INFO); from src.data import load_gsm8k_splits, build_prompt; c=yaml.safe_load(open('experiments/configs/config_smoke.yaml')); s=load_gsm8k_splits(c); print(s['train'][0]['gold']); print(build_prompt(s['train'][0]['question'])[-200:])"`
Expected: prints a numeric gold and a prompt tail ending with `Question: <...>` then a blank line. Confirms parsing + prompt format.

- [ ] **Step 3 (optional): Commit** `git add src/data.py && git commit -m "feat(data): GSM8K splits + few-shot prompt"`

---

## Task 4: Deterministic verifier (`src/verifier.py`)

**Files:**
- Create: `llm-activation-experiment/src/verifier.py`

**Interfaces:**
- Produces: `extract_answer(completion) -> str | None`; `reward(completion, gold) -> float`.

- [ ] **Step 1: Implement `verifier.py`**

```python
import re

# Completions are prompted to end with "Answer: N". Take the LAST such match
# (the model may echo the pattern mid-reasoning). Fallback: last number.
_ANSWER_RE = re.compile(r"Answer:\s*(-?\$?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(num_str):
    return num_str.replace(",", "").replace("$", "").rstrip(".").strip()


def extract_answer(completion):
    # Only look at the first line containing "Answer:" onward isn't required;
    # take the last "Answer: N" occurrence for robustness.
    matches = _ANSWER_RE.findall(completion)
    if matches:
        return _normalize(matches[-1])
    nums = _NUM_RE.findall(completion)
    return _normalize(nums[-1]) if nums else None


def reward(completion, gold):
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    gold_n = _normalize(gold)
    try:
        return 1.0 if abs(float(pred) - float(gold_n)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if pred == gold_n else 0.0
```

- [ ] **Step 2: Smoke-verify extraction + reward on hand cases**

Run: `cd llm-activation-experiment && uv run python -c "from src.verifier import extract_answer, reward; print(extract_answer('...so 72 clips.\nAnswer: 72')); print(reward('Answer: 72','72'), reward('Answer: 1,200','1200'), reward('no answer here','5'))"`
Expected: prints `72` then `1.0 1.0 0.0`. Confirms parsing, comma handling, and the no-answer→0 path.

- [ ] **Step 3 (optional): Commit** `git add src/verifier.py && git commit -m "feat(verifier): GSM8K answer extraction + reward"`

---

## Task 5: Policy loader + LoRA placement (`src/model.py`)

**Files:**
- Create: `llm-activation-experiment/src/model.py`

**Interfaces:**
- Consumes: `sae.SAE.hook_name(layer)` (the boundary reference).
- Produces: `load_policy(cfg, device) -> Policy`; `Policy.model`, `.tokenizer`; `.attach_lora(layer_L, rank)`; `.disable_lora()` (contextmanager); `.hook_module(layer) -> nn.Module`; `.save_lora(path)`, `.load_lora(path)`.

- [ ] **Step 1: Implement `model.py`**

```python
import logging
from contextlib import contextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger(__name__)

# Attention + MLP projection module leaf names for Qwen2/3 blocks.
_TARGET_LEAVES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Policy:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._has_lora = False

    def hook_module(self, layer):
        # The module whose OUTPUT is the SAE-trained activation (post-block).
        return self.model.get_submodule(f"model.layers.{layer}")

    def attach_lora(self, layer_L, rank):
        # LoRA strictly downstream of the hook: layers > layer_L only.
        n_layers = self.model.config.num_hidden_layers
        target_modules = [
            f"model.layers.{i}.{sub}.{leaf}"
            for i in range(layer_L + 1, n_layers)
            for sub in ("self_attn", "mlp")
            for leaf in _TARGET_LEAVES
            if hasattr(self._maybe(f"model.layers.{i}.{sub}"), leaf)
        ]
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                         target_modules=target_modules, bias="none")
        self.model = get_peft_model(self.model, cfg)
        self._has_lora = True
        LOGGER.info(f"attached LoRA rank={rank} to layers {layer_L + 1}..{n_layers - 1} "
                    f"({len(target_modules)} modules)")

    def _maybe(self, path):
        try:
            return self.model.get_submodule(path)
        except AttributeError:
            return None

    @contextmanager
    def disable_lora(self):
        # Reference policy pi_ref = pi_0 = base model (LoRA off).
        if not self._has_lora:
            yield
            return
        with self.model.disable_adapter():
            yield

    def save_lora(self, path):
        self.model.save_pretrained(path)

    def load_lora(self, path):
        from peft import PeftModel
        base = self.model.get_base_model() if self._has_lora else self.model
        self.model = PeftModel.from_pretrained(base, path)
        self._has_lora = True


def load_policy(cfg, device):
    name = cfg["ModelConfig"]["backbone"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()
    LOGGER.info(f"loaded backbone={name} layers={model.config.num_hidden_layers}")
    return Policy(model, tok)
```

- [ ] **Step 2: Smoke-verify LoRA lands only on layers > L**

Run: `cd llm-activation-experiment && uv run python -c "import yaml,logging; logging.basicConfig(level=logging.INFO); from src.model import load_policy; c=yaml.safe_load(open('experiments/configs/config_smoke.yaml')); p=load_policy(c,'cuda'); p.attach_lora(12,8); names=[n for n,_ in p.model.named_parameters() if 'lora' in n.lower()]; import re; layers=sorted({int(re.search(r'layers\.(\d+)\.',n).group(1)) for n in names}); print('lora layers:', layers); assert min(layers)>12, 'LEAK'"`
Expected: prints `lora layers: [13, 14, ... 23]`, no assertion error. Confirms the downstream-only invariant at the parameter level.

- [ ] **Step 3 (optional): Commit** `git add src/model.py && git commit -m "feat(model): Qwen policy loader + downstream-only LoRA"`

---

## Task 6: Trait scoring via teacher-forced hook capture (`src/trait.py`)

**Files:**
- Create: `llm-activation-experiment/src/trait.py`

**Interfaces:**
- Consumes: `Policy.hook_module`, `SAE.encode`.
- Produces: `collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids) -> Tensor[n_features]` (max over completion tokens, all features); `trait_scores(..., features) -> Tensor[len(features)]` (subset).

- [ ] **Step 1: Implement `trait.py`**

```python
import torch

def _capture_hook_acts(policy, layer, input_ids):
    # Single teacher-forced forward pass; grab the hook module's OUTPUT
    # (post-block residual) for every position.
    captured = {}
    module = policy.hook_module(layer)

    def hook(_m, _inp, out):
        captured["acts"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            policy.model(input_ids=input_ids)
    finally:
        handle.remove()
    return captured["acts"]  # (1, seq, d_model)


def collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids):
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1).unsqueeze(0).to(policy.model.device)
    acts = _capture_hook_acts(policy, layer, input_ids)[0]          # (seq, d_model)
    comp_start = prompt_ids.shape[-1]
    comp_acts = acts[comp_start:].to(sae.W_enc.dtype)              # completion tokens only
    feats = sae.encode(comp_acts)                                  # (n_comp, n_features)
    return feats.max(dim=0).values                                 # (n_features,)


def trait_scores(policy, sae, features, prompt_ids, completion_ids, layer):
    all_scores = collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids)
    return all_scores[features]
```

- [ ] **Step 2: Smoke-verify scores are finite and per-feature**

Run: `cd llm-activation-experiment && uv run python -c "import torch,yaml; from src.model import load_policy; from src.sae import load_sae; from src.trait import collect_feature_scores; c=yaml.safe_load(open('experiments/configs/config_smoke.yaml')); p=load_policy(c,'cuda'); s=load_sae(c['ModelConfig']['sae_repo'],12,'cuda'); pi=p.tokenizer('Question: 2+2?\n',return_tensors='pt').input_ids[0]; ci=p.tokenizer('Answer: 4',return_tensors='pt').input_ids[0]; sc=collect_feature_scores(p,s,12,pi,ci); print(sc.shape, torch.isfinite(sc).all().item(), (sc>0).sum().item())"`
Expected: prints `torch.Size([32768]) True <positive count>`. Confirms hook capture + max-over-tokens produce a per-feature vector.

- [ ] **Step 3 (optional): Commit** `git add src/trait.py && git commit -m "feat(trait): teacher-forced hook capture + max-over-tokens score"`

---

## Task 7: Exact-span sequence logprob (`src/logprob.py`)

**Files:**
- Create: `llm-activation-experiment/src/logprob.py`

**Interfaces:**
- Produces: `sequence_logprob(policy, prompt_ids, completion_ids) -> float` — sum of token logprobs of the completion span under the current policy (LoRA state as set by caller).

- [ ] **Step 1: Implement `logprob.py`**

```python
import torch
import torch.nn.functional as F

def sequence_logprob(policy, prompt_ids, completion_ids):
    # completion_ids are the EXACT sampled tokens (EOS included iff generation
    # stopped on EOS; only emitted tokens if truncated). Logprob is summed over
    # exactly those positions, predicted from the preceding context.
    device = policy.model.device
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = policy.model(input_ids=input_ids).logits[0]        # (seq, vocab)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    comp_start = prompt_ids.shape[-1]
    # token at position i is predicted by logits at i-1
    total = 0.0
    for i in range(comp_start, input_ids.shape[-1]):
        total += logprobs[i - 1, input_ids[0, i]].item()
    return total
```

- [ ] **Step 2: Smoke-verify ω=1 when the policy is unchanged**

Run: `cd llm-activation-experiment && uv run python -c "import torch,yaml,math; from src.model import load_policy; from src.logprob import sequence_logprob; c=yaml.safe_load(open('experiments/configs/config_smoke.yaml')); p=load_policy(c,'cuda'); pi=p.tokenizer('Question: 2+2?\n',return_tensors='pt').input_ids[0]; ci=p.tokenizer('Answer: 4',return_tensors='pt').input_ids[0]; a=sequence_logprob(p,pi,ci); b=sequence_logprob(p,pi,ci); print(a, math.exp(a-b))"`
Expected: prints a negative logprob and `1.0` (identical policy → ω=1). This is the ω sanity primitive.

- [ ] **Step 3 (optional): Commit** `git add src/logprob.py && git commit -m "feat(logprob): exact-span sequence logprob for omega"`

---

## Task 8: GRPO step (`src/grpo.py`)

**Files:**
- Create: `llm-activation-experiment/src/grpo.py`

**Interfaces:**
- Consumes: `Policy` (with LoRA attached), `verifier.reward`, `data.build_prompt`.
- Produces: `sample_completions(policy, prompt_ids, G, gen_cfg) -> list[Tensor]`; `grpo_step(policy, optimizer, batch, sae_unused, cfg) -> dict` returning `{mean_reward, mean_adv, degenerate_frac, kl}`. The step samples G completions/prompt, computes rewards + group-normalised advantages, and applies the policy-gradient + k3-KL loss on LoRA params.

- [ ] **Step 1: Implement `grpo.py`**

```python
import logging
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def sample_completions(policy, prompt_ids, G, gen_cfg):
    device = policy.model.device
    batch = prompt_ids.unsqueeze(0).to(device).expand(G, -1)
    with torch.no_grad():
        out = policy.model.generate(
            batch,
            do_sample=True,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            top_k=(gen_cfg["top_k"] or 0),
            max_new_tokens=gen_cfg["max_new_tokens"],
            pad_token_id=policy.tokenizer.pad_token_id,
        )
    plen = prompt_ids.shape[-1]
    # Exact sampled span per row, trailing pad stripped (keeps EOS if present).
    completions = []
    for row in out:
        comp = row[plen:]
        pad = policy.tokenizer.pad_token_id
        keep = (comp != pad).nonzero()
        end = (keep[-1].item() + 1) if len(keep) else 0
        completions.append(comp[:end].detach().cpu())
    return completions


def _token_logprobs(policy, prompt_ids, completion_ids):
    device = policy.model.device
    ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)
    logits = policy.model(input_ids=ids).logits[0]
    lp = F.log_softmax(logits.float(), dim=-1)
    plen = prompt_ids.shape[-1]
    idx = torch.arange(plen, ids.shape[-1], device=device)
    return lp[idx - 1, ids[0, idx]]                     # (n_comp,) with grad


def grpo_step(policy, optimizer, batch, gen_cfg, cfg):
    grpo_cfg = cfg["GRPOConfig"]
    from src.verifier import reward as reward_fn
    G, kl_coef, eps = grpo_cfg["G"], grpo_cfg["kl_coef"], 1e-8
    losses, rewards_all, advs_all, kls = [], [], [], []
    degenerate = 0

    for ex, prompt_ids in batch:                        # ex has 'gold'; prompt_ids precomputed
        comps = sample_completions(policy, prompt_ids, G, gen_cfg)
        texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in comps]
        r = torch.tensor([reward_fn(t, ex["gold"]) for t in texts])
        rewards_all.extend(r.tolist())
        if r.std() < eps:
            degenerate += 1
            adv = torch.zeros(G)
        else:
            adv = (r - r.mean()) / (r.std() + eps)
        advs_all.extend(adv.tolist())

        for c, a in zip(comps, adv):
            tok_lp = _token_logprobs(policy, prompt_ids, c)          # grad
            with policy.disable_lora():
                ref_lp = _token_logprobs(policy, prompt_ids, c).detach()
            log_ratio = ref_lp - tok_lp
            kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()     # k3, token-avg
            pg = -(a.to(tok_lp.device) * tok_lp.mean())
            losses.append(pg + kl_coef * kl)
            kls.append(kl.item())

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "mean_reward": float(sum(rewards_all) / len(rewards_all)),
        "mean_adv": float(sum(advs_all) / len(advs_all)),
        "degenerate_frac": degenerate / len(batch),
        "kl": float(sum(kls) / len(kls)),
    }
```

- [ ] **Step 2: Smoke-verify a single step runs and reward is in [0,1]**

Run: `cd llm-activation-experiment && uv run python -c "import yaml,torch,logging; logging.basicConfig(level=logging.INFO); from src.model import load_policy; from src.data import load_gsm8k_splits, build_prompt; from src.grpo import grpo_step; c=yaml.safe_load(open('experiments/configs/config_smoke.yaml')); p=load_policy(c,'cuda'); p.attach_lora(12,8); opt=torch.optim.Adam([q for q in p.model.parameters() if q.requires_grad],lr=1e-5); sp=load_gsm8k_splits(c); g=c['GenConfig']; batch=[(e,p.tokenizer(build_prompt(e['question'],n_shots=g['n_shots']),return_tensors='pt').input_ids[0]) for e in sp['train'][:2]]; m=grpo_step(p,opt,batch,g,c); print(m)"`
Expected: prints a metrics dict with `mean_reward` in [0,1], finite `kl`. Confirms the end-to-end update path.

- [ ] **Step 3 (optional): Commit** `git add src/grpo.py && git commit -m "feat(grpo): GRPO step with k3-KL and LoRA-only update"`

---

## Task 9: Phase 0 — capability check (`experiments/run_capability_check.py`)

**Files:**
- Create: `llm-activation-experiment/experiments/run_capability_check.py`

**Interfaces:**
- Consumes: `load_policy`, `load_gsm8k_splits`, `build_prompt`, `sample_completions`, `reward`.
- Produces: `results/<name>/capability_report.json`.

- [ ] **Step 1: Implement the script**

```python
import argparse, json, logging
from pathlib import Path
import yaml
from src.model import load_policy
from src.data import load_gsm8k_splits, build_prompt
from src.grpo import sample_completions
from src.verifier import reward, extract_answer

LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    gen = dict(cfg["GenConfig"]); gen["temperature"] = cfg["CapabilityConfig"]["temperature"]

    policy = load_policy(cfg, "cuda")
    splits = load_gsm8k_splits(cfg)
    prompts = splits["feat"][: cfg["CapabilityConfig"]["n_prompts"]]
    G = cfg["GRPOConfig"]["G"]

    per_prompt_rates, spot = [], []
    for i, ex in enumerate(prompts):
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        comps = sample_completions(policy, pid, G, gen)
        texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in comps]
        rates = [reward(t, ex["gold"]) for t in texts]
        per_prompt_rates.append(sum(rates) / len(rates))
        if i < 5:
            spot.append({"completion": texts[0][-200:], "pred": extract_answer(texts[0]),
                         "gold": ex["gold"], "reward": rates[0]})

    lo, hi = cfg["CapabilityConfig"]["target_success_band"]
    in_band = sum(lo <= p_ <= hi for p_ in per_prompt_rates) / len(per_prompt_rates)
    report = {
        "accuracy": round(sum(per_prompt_rates) / len(per_prompt_rates), 4),
        "frac_prompts_in_band": round(in_band, 4),
        "success_rate_hist": [round(p_, 4) for p_ in per_prompt_rates],
        "temperature": gen["temperature"], "max_new_tokens": gen["max_new_tokens"],
        "spot_check": spot,
    }
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "capability_report.json", "w"), indent=2)
    LOGGER.info(f"accuracy={report['accuracy']} in_band={report['frac_prompts_in_band']} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke config end-to-end**

Run: `cd llm-activation-experiment && uv run python -m experiments.run_capability_check --config experiments/configs/config_smoke.yaml`
Expected: logs `accuracy=<0..1> in_band=<0..1>` and writes `results/smoke_L12/capability_report.json`. **Inspect the spot-check rows** to confirm the verifier reads the model's format. This is the gate: if accuracy is ~0 or ~1, adjust `n_shots`/difficulty before proceeding.

- [ ] **Step 3 (optional): Commit** `git add experiments/run_capability_check.py && git commit -m "feat(phase0): capability check"`

---

## Task 10: Phase 1 — feature discovery (`experiments/run_feature_discovery.py`)

**Files:**
- Create: `llm-activation-experiment/experiments/run_feature_discovery.py`

**Interfaces:**
- Consumes: `load_policy`, `load_sae`, `collect_feature_scores`, `sample_completions`, `reward`.
- Produces: `results/<name>/features.json` — `{layer_L, positive:[...], negative:[...], controls:[...], nuisance:{...}}`, each feature entry `{feature_id, rho_discovery, rho_val, activity}`.

- [ ] **Step 1: Implement the script**

```python
import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.grpo import sample_completions
from src.verifier import reward, extract_answer
from src.data import load_gsm8k_splits, build_prompt

LOGGER = logging.getLogger(__name__)


def _gather(policy, sae, layer, examples, gen, K):
    S, R, lengths, parses = [], [], [], []
    for ex in examples:
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        comps = sample_completions(policy, pid, K, gen)
        for c in comps:
            text = policy.tokenizer.decode(c, skip_special_tokens=True)
            S.append(collect_feature_scores(policy, sae, layer, pid, c).float().cpu().numpy())
            R.append(reward(text, ex["gold"]))
            lengths.append(len(c)); parses.append(1.0 if extract_answer(text) is not None else 0.0)
    return np.stack(S), np.array(R), np.array(lengths, float), np.array(parses)


def _corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    fd, gen = cfg["FeatureDiscoveryConfig"], cfg["GenConfig"]
    policy = load_policy(cfg, "cuda")
    splits = load_gsm8k_splits(cfg)
    feat_ex = splits["feat"]
    cut = int(len(feat_ex) * fd["val_frac"])
    disc_ex, val_ex = feat_ex[:cut], feat_ex[cut:]

    best = None
    for layer in fd["candidate_layers"]:
        sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
        Sd, Rd, Ld, Pd = _gather(policy, sae, layer, disc_ex, gen, fd["K"])
        Sv, Rv, _, _ = _gather(policy, sae, layer, val_ex, gen, fd["K"])
        activity = (Sd > 0).mean(axis=0)
        rho_d = np.array([_corr(Sd[:, m], Rd) if activity[m] >= fd["min_activity"] else 0.0
                          for m in range(Sd.shape[1])])
        order = np.argsort(rho_d)
        neg_ids = [int(m) for m in order[:fd["top_k_each_sign"]]]
        pos_ids = [int(m) for m in order[::-1][:fd["top_k_each_sign"]]]

        def entry(m):
            return {"feature_id": m, "rho_discovery": round(float(rho_d[m]), 4),
                    "rho_val": round(_corr(Sv[:, m], Rv), 4), "activity": round(float(activity[m]), 4)}
        pos, neg = [entry(m) for m in pos_ids], [entry(m) for m in neg_ids]
        # keep only features whose sign holds on validation
        pos = [e for e in pos if e["rho_val"] > 0]; neg = [e for e in neg if e["rho_val"] < 0]
        strength = sum(abs(e["rho_val"]) for e in pos + neg)
        LOGGER.info(f"layer {layer}: kept {len(pos)}+/{len(neg)}- val-strength={strength:.3f}")

        rng = np.random.default_rng(cfg["ModelConfig"]["seed"])
        active_lowcorr = [m for m in range(Sd.shape[1])
                          if activity[m] >= fd["min_activity"] and abs(rho_d[m]) < 0.02]
        ctrl_ids = rng.choice(active_lowcorr, size=min(fd["n_controls"], len(active_lowcorr)),
                              replace=False).tolist() if active_lowcorr else []
        controls = [entry(int(m)) for m in ctrl_ids]
        nuisance = {"reward_vs_length": round(_corr(Rd, Ld), 4),
                    "reward_vs_parse": round(_corr(Rd, Pd), 4)}
        cand = {"layer_L": layer, "positive": pos, "negative": neg,
                "controls": controls, "nuisance": nuisance, "_strength": strength}
        if best is None or strength > best["_strength"]:
            best = cand

    best.pop("_strength")
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(out / "features.json", "w"), indent=2)
    LOGGER.info(f"chose layer_L={best['layer_L']} -> {out}/features.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke discovery**

Run: `cd llm-activation-experiment && uv run python -m experiments.run_feature_discovery --config experiments/configs/config_smoke.yaml`
Expected: logs per-layer kept-feature counts and writes `results/smoke_L12/features.json` with non-empty `positive`/`negative` lists (smoke K is tiny so correlations are noisy — just confirm structure and that `rho_val` fields are populated).

- [ ] **Step 3 (optional): Commit** `git add experiments/run_feature_discovery.py && git commit -m "feat(phase1): feature discovery with val split + controls + nuisance"`

---

## Task 11: Phase 2 — GRPO training with frozen-trait assertion (`experiments/run_grpo.py`)

**Files:**
- Create: `llm-activation-experiment/experiments/run_grpo.py`

**Interfaces:**
- Consumes: `load_policy`, `load_sae`, `collect_feature_scores`, `grpo_step`, `features.json` from Phase 1.
- Produces: `results/<name>/checkpoints/step_XXXX/` (LoRA adapters), `results/<name>/train_metrics.jsonl`.

- [ ] **Step 1: Implement the script (with the frozen-trait guard)**

```python
import argparse, json, logging
from pathlib import Path
import torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.data import load_gsm8k_splits, build_prompt
from src.grpo import grpo_step

LOGGER = logging.getLogger(__name__)


def _assert_frozen(policy, sae, layer, pid, cid, before):
    after = collect_feature_scores(policy, sae, layer, pid, cid)
    drift = (after - before).abs().max().item()
    assert drift < 1e-4, f"TRAIT NOT FROZEN: max |Δs|={drift} (LoRA leaked onto layer<=L?)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    grpo_cfg, gen = cfg["GRPOConfig"], cfg["GenConfig"]
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    layer = json.load(open(out / "features.json"))["layer_L"]

    policy = load_policy(cfg, "cuda")
    policy.attach_lora(layer, grpo_cfg["lora_rank"])
    sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
    opt = torch.optim.Adam([p for p in policy.model.parameters() if p.requires_grad], lr=grpo_cfg["lr"])
    splits = load_gsm8k_splits(cfg)
    train = splits["train"]

    # fixed probe (prompt, completion) for the frozen-trait assertion
    probe_ex = train[0]
    probe_pid = policy.tokenizer(build_prompt(probe_ex["question"], n_shots=gen["n_shots"]),
                                 return_tensors="pt").input_ids[0]
    probe_cid = policy.tokenizer("Answer: 42", return_tensors="pt").input_ids[0]
    probe_before = collect_feature_scores(policy, sae, layer, probe_pid, probe_cid)

    metrics_path = out / "train_metrics.jsonl"
    rng = torch.Generator().manual_seed(grpo_cfg["seed"])
    with open(metrics_path, "w") as mf:
        policy.save_lora(out / "checkpoints" / "step_0000")
        for step in range(grpo_cfg["steps"]):
            idx = torch.randint(len(train), (grpo_cfg["batch_size"],), generator=rng).tolist()
            batch = [(train[i], policy.tokenizer(build_prompt(train[i]["question"], n_shots=gen["n_shots"]),
                     return_tensors="pt").input_ids[0]) for i in idx]
            m = grpo_step(policy, opt, batch, gen, cfg)
            _assert_frozen(policy, sae, layer, probe_pid, probe_cid, probe_before)
            m["step"] = step + 1
            mf.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}) + "\n")
            mf.flush()
            if (step + 1) % grpo_cfg["checkpoint_every"] == 0:
                policy.save_lora(out / "checkpoints" / f"step_{step + 1:04d}")
            LOGGER.info(f"step {step + 1}/{grpo_cfg['steps']} reward={m['mean_reward']:.3f} kl={m['kl']:.4f}")
    LOGGER.info(f"done -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke training**

Run: `cd llm-activation-experiment && uv run python -m experiments.run_grpo --config experiments/configs/config_smoke.yaml`
Expected: 8 steps log `reward=... kl=...`, the frozen-trait assertion passes every step, and `results/smoke_L12/checkpoints/step_0000..step_0008/` + `train_metrics.jsonl` are written. **If the assertion fires, the LoRA boundary is wrong — revisit Task 2/Task 5.**

- [ ] **Step 3 (optional): Commit** `git add experiments/run_grpo.py && git commit -m "feat(phase2): GRPO training + frozen-trait assertion"`

---

## Task 12: Phase 3 — Price-equation evaluation (`experiments/run_price_eval.py`)

**Files:**
- Create: `llm-activation-experiment/experiments/run_price_eval.py`

**Interfaces:**
- Consumes: checkpoints + `features.json`; `sequence_logprob`, `collect_feature_scores`, `sample_completions`.
- Produces: `results/<name>/price_eval.jsonl` (per feature, per step, per N: `direct_drift`, `price`, `cov`, `mean_omega`, `omega_var`, `omega_max`, `ess`).

- [ ] **Step 1: Implement the script**

```python
import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.logprob import sequence_logprob
from src.grpo import sample_completions
from src.data import load_gsm8k_splits, build_prompt

LOGGER = logging.getLogger(__name__)


def _feature_ids(features):
    ids = [e["feature_id"] for e in features["positive"] + features["negative"]]
    ctrl = [e["feature_id"] for e in features["controls"]]
    return ids, ctrl


def _draw(policy, examples, gen, n, rng):
    # x ~ eval prompts, a ~ pi_t(.|x): one completion per draw.
    draws = []
    for _ in range(n):
        ex = examples[int(rng.integers(len(examples)))]
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        cid = sample_completions(policy, pid, 1, gen)[0]
        draws.append((pid, cid))
    return draws


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    pe, gen = cfg["PriceEvalConfig"], cfg["GenConfig"]
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    features = json.load(open(out / "features.json"))
    layer = features["layer_L"]
    feat_ids, ctrl_ids = _feature_ids(features)
    all_ids = torch.tensor(feat_ids + ctrl_ids)

    policy = load_policy(cfg, "cuda")
    policy.attach_lora(layer, cfg["GRPOConfig"]["lora_rank"])
    sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
    eval_ex = load_gsm8k_splits(cfg)["eval"][: pe["eval_prompts"]]

    ckpts = sorted((out / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    rng = np.random.default_rng(cfg["ModelConfig"]["seed"] + 777)
    N_max = max(pe["price_samples"] + [pe["direct_samples"]])

    with open(out / "price_eval.jsonl", "w") as f:
        for t in range(len(ckpts) - 1):
            policy.load_lora(ckpts[t])
            # direct T_t (high budget) + price draws from pi_t
            price_draws = _draw(policy, eval_ex, gen, N_max, rng)
            s_price = torch.stack([collect_feature_scores(policy, sae, layer, p_, c)[all_ids]
                                   for p_, c in price_draws])              # (N_max, F)
            logp_t = torch.tensor([sequence_logprob(policy, p_, c) for p_, c in price_draws])
            direct_draws = _draw(policy, eval_ex, gen, pe["direct_samples"], rng)
            s_dir_t = torch.stack([collect_feature_scores(policy, sae, layer, p_, c)[all_ids]
                                   for p_, c in direct_draws]).mean(0)     # (F,)

            policy.load_lora(ckpts[t + 1])
            logp_tp1 = torch.tensor([sequence_logprob(policy, p_, c) for p_, c in price_draws])
            direct_draws2 = _draw(policy, eval_ex, gen, pe["direct_samples"], rng)
            s_dir_tp1 = torch.stack([collect_feature_scores(policy, sae, layer, p_, c)[all_ids]
                                     for p_, c in direct_draws2]).mean(0)
            direct_drift = (s_dir_tp1 - s_dir_t)                            # (F,)

            omega_full = torch.exp(logp_tp1 - logp_t)                       # (N_max,)
            for N in pe["price_samples"]:
                w, s = omega_full[:N], s_price[:N]
                mean_w = w.mean()
                price = (w[:, None] * s).mean(0) - s.mean(0)                # (F,)
                cov = (w[:, None] * s).mean(0) - mean_w * s.mean(0)
                ess = float((w.sum() ** 2) / (w ** 2).sum())
                for fi, fid in enumerate(feat_ids + ctrl_ids):
                    f.write(json.dumps({
                        "step": t, "feature_id": int(fid),
                        "is_control": fid in ctrl_ids, "N": N,
                        "direct_drift": round(float(direct_drift[fi]), 4),
                        "price": round(float(price[fi]), 4),
                        "cov": round(float(cov[fi]), 4),
                        "mean_omega": round(float(mean_w), 4),
                        "omega_var": round(float(w.var()), 4),
                        "omega_max": round(float(w.max()), 4),
                        "ess": round(ess, 2),
                    }) + "\n")
            LOGGER.info(f"step {t}->{t+1} mean_omega={float(omega_full.mean()):.3f} "
                        f"ess@max={float((omega_full.sum()**2)/(omega_full**2).sum()):.1f}")
    LOGGER.info(f"done -> {out}/price_eval.jsonl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke Price eval**

Run: `cd llm-activation-experiment && uv run python -m experiments.run_price_eval --config experiments/configs/config_smoke.yaml`
Expected: logs per-step `mean_omega≈1.0` and an `ess` value, writes `results/smoke_L12/price_eval.jsonl` with rows carrying `direct_drift`, `price`, `cov`, `ess`. Sanity: `mean_omega` should sit near 1.

- [ ] **Step 3 (optional): Commit** `git add experiments/run_price_eval.py && git commit -m "feat(phase3): Price-equation evaluation + omega diagnostics"`

---

## Task 13: Plots (`src/plotting.py`) + wire into Phase 3

**Files:**
- Create: `llm-activation-experiment/src/plotting.py`
- Modify: `llm-activation-experiment/experiments/run_price_eval.py` (append a plotting call after the JSONL is written)

**Interfaces:**
- Consumes: `results/<name>/price_eval.jsonl`.
- Produces: `plot_from_jsonl(jsonl_path, out_dir)` writing `expected_trait`, `price_check`, `price_convergence`, `grid_price`, `omega_diagnostics` as `.png`+`.pdf`.

- [ ] **Step 1: Implement `plotting.py`** (serif style copied from `simple-theory/run_experiment.py`)

```python
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(jsonl_path):
    rows = [json.loads(l) for l in open(jsonl_path)]
    return rows


def plot_from_jsonl(jsonl_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(jsonl_path)
    N_max = max(r["N"] for r in rows)
    feats = sorted({r["feature_id"] for r in rows})
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    # price_check: cumulative observed (direct) vs predicted (price) at N_max, per feature grid
    n = len(feats); cols = min(5, n); rows_g = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g), squeeze=False)
    for ax, fid in zip(axes.flat, feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["N"] == N_max]
        fr.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in fr]
        obs = np.cumsum([r["direct_drift"] for r in fr])
        pred = np.cumsum([r["price"] for r in fr])
        ctrl = fr[0]["is_control"] if fr else False
        ax.plot(steps, obs, color=colors[0], marker="o", ms=3, label="observed ΔT")
        ax.plot(steps, pred, color=colors[1], marker="s", ms=3, label="Price")
        ax.set_title(f"feat {fid}{' (ctrl)' if ctrl else ''}", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("GRPO step t"); fig.supylabel("cumulative trait change")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"grid_price.{ext}", dpi=150)
    plt.close(fig)

    # price_convergence: final-step price vs N, one line per feature, dashed = direct
    fig, ax = plt.subplots(figsize=(5, 4))
    last_step = max(r["step"] for r in rows)
    for i, fid in enumerate(feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["step"] == last_step]
        fr.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in fr]
        ax.plot(Ns, [r["price"] for r in fr], marker="o", ms=3, color=colors[i % len(colors)])
        ax.axhline(fr[-1]["direct_drift"], ls="--", lw=0.8, color=colors[i % len(colors)])
    ax.set_xscale("log"); ax.set_xlabel("Price sample budget N"); ax.set_ylabel("final-step estimate")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"price_convergence.{ext}", dpi=150)
    plt.close(fig)

    # omega_diagnostics: mean_omega and ess vs step (at N_max)
    fr = [r for r in rows if r["N"] == N_max and r["feature_id"] == feats[0]]
    fr.sort(key=lambda r: r["step"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    a1.plot([r["step"] for r in fr], [r["mean_omega"] for r in fr], color=colors[0], marker="o", ms=3)
    a1.axhline(1.0, ls="--", lw=0.8, color="grey"); a1.set_ylabel(r"$\bar\omega$")
    a2.plot([r["step"] for r in fr], [r["ess"] for r in fr], color=colors[2], marker="o", ms=3)
    a2.set_ylabel("ESS"); a2.set_xlabel("GRPO step t")
    a2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"omega_diagnostics.{ext}", dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Wire plotting into Phase 3** — append to `run_price_eval.py` `main()` after the JSONL loop:

```python
    from src.plotting import plot_from_jsonl
    plot_from_jsonl(out / "price_eval.jsonl", out / "plots")
    LOGGER.info(f"saved plots to {out}/plots")
```

- [ ] **Step 3: Re-run Phase 3 smoke and check plots exist**

Run: `cd llm-activation-experiment && uv run python -m experiments.run_price_eval --config experiments/configs/config_smoke.yaml && ls results/smoke_L12/plots`
Expected: lists `grid_price.png/pdf`, `price_convergence.png/pdf`, `omega_diagnostics.png/pdf`.

- [ ] **Step 4 (optional): Commit** `git add src/plotting.py experiments/run_price_eval.py && git commit -m "feat(plots): price-check grid, convergence, omega diagnostics"`

---

## Task 14: README + full end-to-end smoke pass

**Files:**
- Create: `llm-activation-experiment/README.md`

**Interfaces:**
- Produces: run documentation (experimental-principles requires a one-line description + exact command per experiment).

- [ ] **Step 1: Write `README.md`**

```markdown
# LLM Activation Experiment (Part 3)

Tests the sampled Price-equation prediction for a frozen Qwen-Scope SAE trait
in Qwen3.5-2B-Base under GRPO on GSM8K: does `mean(omega*s) - mean(s)` recover
the direct trait drift `T_{t+1}-T_t`? See `experiment-spec.md`.

## Pipeline (run in order)

```bash
# Phase 0 — capability gate (inspect success rate before continuing)
uv run python -m experiments.run_capability_check --config experiments/configs/config.yaml
# Phase 1 — pick frozen SAE trait features (+ controls) and layer L
uv run python -m experiments.run_feature_discovery --config experiments/configs/config.yaml
# Phase 2 — GRPO with LoRA strictly downstream of layer L (frozen trait)
uv run python -m experiments.run_grpo --config experiments/configs/config.yaml
# Phase 3 — Price-equation evaluation + plots
uv run python -m experiments.run_price_eval --config experiments/configs/config.yaml
```

Smoke test (tiny, one GPU, minutes): swap `config.yaml` → `config_smoke.yaml`.
Outputs under `results/<name>/`.
```

- [ ] **Step 2: Full smoke pass, all four phases in sequence**

Run:
```bash
cd llm-activation-experiment
uv run python -m experiments.run_capability_check --config experiments/configs/config_smoke.yaml && \
uv run python -m experiments.run_feature_discovery --config experiments/configs/config_smoke.yaml && \
uv run python -m experiments.run_grpo --config experiments/configs/config_smoke.yaml && \
uv run python -m experiments.run_price_eval --config experiments/configs/config_smoke.yaml
```
Expected: all four complete without error; `results/smoke_L12/` contains `capability_report.json`, `features.json`, `checkpoints/`, `train_metrics.jsonl`, `price_eval.jsonl`, `plots/`. The frozen-trait assertion passed throughout and `mean_omega≈1`.

- [ ] **Step 3 (optional): Commit** `git add llm-activation-experiment/README.md && git commit -m "docs: run instructions + full smoke pass"`

---

## Self-Review notes

- **Spec coverage:** Phase 0 (Task 9), feature discovery incl. val split / controls / nuisance (Task 10), frozen LoRA + assertion (Tasks 5, 11), teacher-forced trait score max-over-tokens (Task 6), exact-span ω with EOS/truncation handling (Task 7), k3-KL to π_0 (Task 8), high-budget direct vs Price sweep + ESS/ω diagnostics (Task 12), plots incl. controls overlay + convergence + ω diagnostics (Task 13). Pre-implementation hook verification is Task 2.
- **Deferred vs spec:** `nuisance_table` is emitted as JSON fields in `features.json` (Task 10) rather than a rendered table plot — the numbers are captured; rendering can be added if wanted. `expected_trait` and per-feature `price_check` single-panel plots are folded into the `grid_price` small-multiples panel to avoid clutter.
- **Known adaptation:** exact Qwen-Scope filenames/tensor keys and the precise hook point are pinned in Task 2 Step 1 and flow into `sae.py`/`model.py`; if the SAE turns out to read post-attn rather than post-block, adjust `SAE.hook_name` and the LoRA start index together (they must stay consistent).
