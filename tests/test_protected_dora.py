import copy

import pytest
import torch

from pcl.hashing import model_hashes
from pcl.merge import EXPORT_FORMAT, merge_export, projection_residual
from pcl.protected_dora import ProtectedDoRALinear, attach_adapters, assert_only_adapters_trainable


def setup_layer(k=3, dtype=torch.float32):
    torch.manual_seed(1)
    base = torch.nn.Linear(12, 7, dtype=dtype)
    q = torch.linalg.qr(torch.randn(12, k)).Q
    return ProtectedDoRALinear(base, q, rank=3, alpha=6)


def test_identity_and_gradient_flow():
    layer = setup_layer()
    x = torch.randn(3, 5, 12)
    initial = layer.base.weight.detach().clone()
    assert torch.equal(layer(x), layer.base(x))
    optimizer = torch.optim.AdamW([layer.A, layer.B, layer.magnitude], lr=0.01)
    layer(x).square().mean().backward()
    assert layer.B.grad.norm() > 0
    assert layer.magnitude.grad.norm() > 0
    assert layer.A.grad.norm() == 0
    optimizer.step()
    optimizer.zero_grad()
    layer(x).square().mean().backward()
    assert layer.A.grad.norm() > 0
    assert torch.equal(initial, layer.base.weight)
    assert layer.base.weight.grad is None




def test_complete_update_protected_after_adam_and_magnitude_changes():
    layer = setup_layer()
    optimizer = torch.optim.AdamW([layer.A, layer.B, layer.magnitude], lr=0.05)
    for _ in range(4):
        optimizer.zero_grad()
        layer(torch.randn(5, 12)).square().mean().backward()
        optimizer.step()
    with torch.no_grad():
        layer.magnitude.mul_(1.5)  # would break protection if only A were projected
    x = torch.randn(5, 3) @ layer.Q.T
    assert torch.allclose(layer(x), layer.base(x), atol=2e-6, rtol=1e-5)
    container = torch.nn.ModuleDict({'projection': copy.deepcopy(layer.base)})
    original = container['projection'].weight.detach().clone()
    report = merge_export(container, {'format': EXPORT_FORMAT, 'base_hashes': model_hashes(container),
                                      'layers': {'projection': layer.export()}})
    assert report['projection']['intended_residual'] < 1e-6
    assert projection_residual(container['projection'].weight - original, layer.Q) < 1e-6
    x = torch.randn(5, 12)
    assert torch.allclose(layer(x), container['projection'](x), atol=2e-6, rtol=1e-5)


def test_freeze_all_other_weights_and_precision_assertion():
    model = torch.nn.ModuleDict({'q_proj': torch.nn.Linear(12, 7), 'head': torch.nn.Linear(7, 2)})
    attach_adapters(model)
    assert_only_adapters_trainable(model)
    assert not model['head'].weight.requires_grad
    with pytest.raises(ValueError, match='previous stage'):
        attach_adapters(model)
    model.half()
    with pytest.raises(RuntimeError, match='FP32'):
        assert_only_adapters_trainable(model)


def test_invalid_basis():
    with pytest.raises(ValueError, match='orthonormal'):
        ProtectedDoRALinear(torch.nn.Linear(12, 7), torch.ones(12, 2))


def test_fp16_base_uses_fp32_trainables():
    layer = setup_layer(dtype=torch.float16)
    with torch.no_grad():
        layer.B.normal_(std=0.01)
    output = layer(torch.randn(4, 12).half())
    output.float().sum().backward()
    assert output.dtype == torch.float16
    assert layer.A.grad.dtype == torch.float32
    assert torch.isfinite(layer.magnitude.grad).all()
