import copy

import pytest
import torch

from pcl.checkpoints import load_checkpoint, restore_adapters
from pcl.protected_dora import attach_adapters, adapter_modules
from pcl.toy import TinyCausalLM, toy_records
from pcl.train import Trainer, TrainConfig


def test_restore_rejects_wrong_base_and_invalid_shapes_before_attachment(tmp_path):
    base = TinyCausalLM()
    model = copy.deepcopy(base)
    attach_adapters(model, rank=2)
    trainer = Trainer(model, toy_records(), TrainConfig(), pad_id=0, provenance={'test': 'restore'})
    trainer.save(tmp_path / 'cp.pt')
    cp = load_checkpoint(tmp_path / 'cp.pt')
    wrong = TinyCausalLM()
    with pytest.raises(ValueError, match='base hash'):
        restore_adapters(wrong, cp)
    assert not adapter_modules(wrong)
    cp['layers']['v_proj']['magnitude'] = torch.ones(1)
    with pytest.raises(ValueError, match='shape'):
        restore_adapters(base, cp)
    assert not adapter_modules(base)
