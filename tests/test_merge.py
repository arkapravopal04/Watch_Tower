import copy

import pytest
import torch

from pcl.hashing import model_hashes
from pcl.merge import export_adapters, merge_export, preview_merge
from pcl.protected_dora import adapter_modules, attach_adapters


def trained_pair(dtype=torch.float32):
    torch.manual_seed(15)
    base = torch.nn.ModuleDict({'q_proj': torch.nn.Linear(9, 5, dtype=dtype),
                                'v_proj': torch.nn.Linear(9, 4, dtype=dtype)})
    live = copy.deepcopy(base)
    q = torch.linalg.qr(torch.randn(9, 3)).Q
    attach_adapters(live, rank=2, bases={'q_proj': q, 'v_proj': q})
    with torch.no_grad():
        for layer in adapter_modules(live).values():
            layer.B.normal_(std=0.1)
            layer.magnitude.mul_(1.2)
    return base, live


def test_preview_is_nonmutating_and_wrong_or_repeated_merge_rejected():
    base, live = trained_pair()
    artifact = export_adapters(live)
    initial = model_hashes(base)
    report = preview_merge(base, artifact)
    assert report and model_hashes(base) == initial
    wrong = copy.deepcopy(base)
    with torch.no_grad():
        wrong['q_proj'].bias.add_(1)
    with pytest.raises(ValueError, match='base hash'):
        merge_export(wrong, artifact)
    merge_export(base, artifact)
    once = model_hashes(base)
    with pytest.raises(ValueError, match='already merged'):
        merge_export(base, artifact)
    assert model_hashes(base) == once


def test_overflow_in_last_layer_does_not_modify_first_layer():
    base, live = trained_pair(torch.float16)
    artifact = export_adapters(live)
    artifact['layers']['v_proj']['scale'] = 1e40
    initial = model_hashes(base)
    with pytest.raises(ValueError, match='overflow'):
        merge_export(base, artifact)
    assert model_hashes(base) == initial


def test_live_failure_keeps_wrappers_and_base():
    from pcl.merge import merge_live
    base, live = trained_pair(torch.float16)
    with torch.no_grad():
        live['v_proj'].magnitude.fill_(1e10)
    before = model_hashes(live)
    with pytest.raises(ValueError):
        merge_live(live)
    assert model_hashes(live) == before
    assert len(adapter_modules(live)) == 2






def test_fp16_merge_reports_actual_rounding_residual():
    torch.manual_seed(8)
    model = torch.nn.ModuleDict({'q_proj': torch.nn.Linear(16, 8).half()})
    plain = copy.deepcopy(model)
    q = torch.linalg.qr(torch.randn(16, 4)).Q
    attach_adapters(model, rank=2, bases={'q_proj': q})
    with torch.no_grad():
        model['q_proj'].B.normal_(std=0.01)
        model['q_proj'].magnitude.mul_(1.01)
    result = merge_export(plain, export_adapters(model))['q_proj']
    assert result['intended_residual'] < 1e-5
    assert result['rounding_error_norm'] > 0
    assert result['stored_residual'] > result['intended_residual']
