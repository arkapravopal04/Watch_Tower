import copy

import pytest
import torch

from pcl.protected_dora import attach_adapters
from pcl.toy import TinyCausalLM, toy_records
from pcl.train import TrainConfig, Trainer, seed_everything


def test_resume_matches_uninterrupted_with_dropout_and_partial_batch(tmp_path):
    seed_everything(2)
    model = TinyCausalLM()
    attach_adapters(model, rank=3)
    initial = copy.deepcopy(model)
    config = TrainConfig(epochs=2, microbatch=2, accumulation=2, learning_rate=0.01)
    records = toy_records(11)
    seed_everything(20)
    full = Trainer(model, records, config, pad_id=0, provenance={'base': 'toy-v1'})
    full.run()
    seed_everything(20)
    interrupted = Trainer(copy.deepcopy(initial), records, config, pad_id=0, provenance={'base': 'toy-v1'})
    checkpoint = tmp_path / 'resume.pt'
    interrupted.run(stop_after_updates=2, checkpoint_path=checkpoint)
    seed_everything(888)
    resumed = Trainer(initial, records, config, pad_id=0, provenance={'base': 'toy-v1'})
    resumed.resume(checkpoint)
    resumed.run()
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    assert full.history == resumed.history
    assert resumed.state['resume_count'] == 1
    assert full.state['answer_tokens'] == resumed.state['answer_tokens']


def test_resume_rejects_changed_provenance(tmp_path):
    model = TinyCausalLM()
    attach_adapters(model)
    trainer = Trainer(model, toy_records(), TrainConfig(), pad_id=0, provenance={'base': 'x'})
    path = tmp_path / 'checkpoint.pt'
    trainer.save(path)
    trainer.provenance = {'base': 'y'}
    with pytest.raises(ValueError, match='provenance'):
        trainer.resume(path)
