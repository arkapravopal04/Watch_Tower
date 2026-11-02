import pytest

from pcl.tokenization import collate, encode_record


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3, 4, 5, 6] if messages[-1]['role'] == 'assistant' else [1, 2, 3]


def test_answer_mask_and_no_truncation():
    record = {'id': 'x', 'prompt': 'p', 'response': 'r', 'skill': 'A', 'split': 'train', 'group': 'g'}
    encoded = encode_record(Tokenizer(), record, 6)
    assert encoded['labels'] == [-100, -100, -100, 4, 5, 6]
    with pytest.raises(ValueError, match='truncation'):
        encode_record(Tokenizer(), record, 5)
    short = {**encoded, 'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, -100, 4]}
    batch = collate([encoded, short], 0)
    assert batch['labels'][1].tolist() == [-100, -100, -100, 4, -100, -100]
    assert batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 0, 0]
