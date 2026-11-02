import pytest
import torch

from pcl.evaluate import evaluate


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2] if messages[1]['content'] == 'short' else [1, 2, 3]

    def batch_decode(self, tokens, **kwargs):
        return ['FINAL: q=1 kPa'] * len(tokens)


class FakeModel(torch.nn.Module):
    def __init__(self, terminate=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.terminate = terminate

    def generate(self, **kwargs):
        assert kwargs['do_sample'] is False
        assert kwargs['num_beams'] == 1
        assert kwargs['input_ids'].tolist() == [[0, 1, 2], [1, 2, 3]]
        assert kwargs['attention_mask'].tolist() == [[0, 1, 1], [1, 1, 1]]
        tail = [5, 9] if self.terminate else [5, 5]
        return torch.cat([kwargs['input_ids'], torch.tensor([tail, tail])], dim=1)


def records(split='dev'):
    return [{'id': str(i), 'prompt': p, 'group': str(i), 'skill': 'B', 'family': 'pressure', 'split': split,
             'reference': {'q': {'value': 1000, 'unit': 'Pa', 'atol': 0.001}}}
            for i, p in enumerate(['short', 'long'])]


def test_greedy_left_padding_and_truncation_failure():
    result = evaluate(FakeModel(), FakeTokenizer(), records(), batch_size=2, max_new_tokens=2)
    assert all(row['correct'] for row in result)
    truncated = evaluate(FakeModel(False), FakeTokenizer(), records(), batch_size=2, max_new_tokens=2)
    assert not any(row['correct'] for row in truncated)


def test_evaluator_rejects_test_without_authorization():
    with pytest.raises(ValueError, match='locked'):
        evaluate(FakeModel(), FakeTokenizer(), records('test'))
