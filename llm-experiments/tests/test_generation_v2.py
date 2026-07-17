import torch
import pytest
from collections.abc import Mapping

from src.generation import generate_completions, stop_sequence_suffix
from src.prompts import encode_prompt_for_generation


class ChatTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return [10, 11, 12]

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) for ch in text]


class RawTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        return [1 if add_special_tokens else 2] + [ord(ch) for ch in text]


class DictChatTokenizer(RawTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": [21, 22, 23], "attention_mask": [1, 1, 1]}


class MappingChatResult(Mapping):
    def __init__(self):
        self.data = {"input_ids": [31, 32, 33], "attention_mask": [1, 1, 1]}

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class MappingChatTokenizer(RawTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return MappingChatResult()


class StopTokenizer(RawTokenizer):
    def __init__(self):
        self.id_to_text = {
            1: "<",
            2: "/",
            3: "answer",
            4: ">",
            5: " extra",
        }

    def encode(self, text, add_special_tokens=True):
        if text == "</answer>":
            return [1, 2, 3, 4]
        return [42]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.id_to_text.get(int(token_id), f"#{int(token_id)}") for token_id in ids)


class ScriptedModel(torch.nn.Module):
    def __init__(self, scripted_tokens):
        super().__init__()
        self.scripted_tokens = list(scripted_tokens)
        self.generation_config = type("GenerationConfig", (), {})()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None):
        step = input_ids.shape[1] - 1
        next_token = self.scripted_tokens[min(step, len(self.scripted_tokens) - 1)]
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 128), -1000.0, device=input_ids.device)
        logits[:, -1, next_token] = 1000.0
        return type("Output", (), {"logits": logits})


def test_encode_prompt_for_generation_uses_chat_template_when_available():
    tokenizer = ChatTokenizer()

    ids = encode_prompt_for_generation(tokenizer, "hello", use_chat_template=True)

    assert ids == [10, 11, 12]
    assert tokenizer.chat_calls == [([{"role": "user", "content": "hello"}], True, True)]


def test_encode_prompt_for_generation_falls_back_to_raw_encoding():
    tokenizer = RawTokenizer()

    assert encode_prompt_for_generation(tokenizer, "hi", use_chat_template=True) == [1, ord("h"), ord("i")]


def test_encode_prompt_for_generation_extracts_input_ids_from_chat_template_dict():
    tokenizer = DictChatTokenizer()

    assert encode_prompt_for_generation(tokenizer, "hi", use_chat_template=True) == [21, 22, 23]


def test_encode_prompt_for_generation_extracts_input_ids_from_mapping_chat_template_result():
    tokenizer = MappingChatTokenizer()

    assert encode_prompt_for_generation(tokenizer, "hi", use_chat_template=True) == [31, 32, 33]


def test_stop_sequence_suffix_detection_for_answer_tag():
    assert stop_sequence_suffix([7, 1, 2, 3, 4], [1, 2, 3, 4])
    assert not stop_sequence_suffix([1, 2, 3], [1, 2, 3, 4])


def test_generation_stops_at_first_complete_answer_stop_sequence():
    tokenizer = StopTokenizer()
    model = ScriptedModel([1, 2, 3, 4, 5])

    completions = generate_completions(
        model,
        tokenizer,
        "prompt",
        {"do_sample": True, "max_new_tokens": 8, "temperature": 1.0, "stop_sequence": "</answer>", "use_chat_template": False},
        num_return_sequences=1,
        device="cpu",
    )

    completion = completions[0]
    assert completion.completion_ids == [1, 2, 3, 4]
    assert completion.completion_text == "</answer>"
    assert completion.stop_reason == "answer_stop"
    assert completion.stopped_on_answer_tag is True


def test_generation_rejects_temperature_that_would_desync_logprobs():
    tokenizer = StopTokenizer()
    model = ScriptedModel([1])

    with pytest.raises(NotImplementedError, match="temperature=1.0"):
        generate_completions(
            model,
            tokenizer,
            "prompt",
            {"do_sample": True, "max_new_tokens": 1, "temperature": 0.7, "stop_sequence": "</answer>", "use_chat_template": False},
            num_return_sequences=1,
            device="cpu",
        )


def test_generation_rejects_greedy_decoding_that_would_desync_logprobs():
    tokenizer = StopTokenizer()
    model = ScriptedModel([1])

    with pytest.raises(NotImplementedError, match="do_sample=True"):
        generate_completions(
            model,
            tokenizer,
            "prompt",
            {"do_sample": False, "max_new_tokens": 1, "temperature": 1.0, "stop_sequence": "</answer>", "use_chat_template": False},
            num_return_sequences=1,
            device="cpu",
        )
