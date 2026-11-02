"""Independent dense equations, gradients, and an optional pinned PEFT check."""

import copy

import pytest
import torch
import torch.nn.functional as F

from pcl.protected_dora import ProtectedDoRALinear
from pcl.merge import merge_live


@pytest.mark.parametrize('k', [0, 3, 9])
def test_dense_full_update_equation_and_detached_norm_gradients(k):
    torch.manual_seed(43)
    q = torch.linalg.qr(torch.randn(9, k)).Q
    layer = ProtectedDoRALinear(torch.nn.Linear(9, 5), q, rank=3, alpha=4.5)
    with torch.no_grad():
        layer.B.normal_(std=0.2)
        layer.magnitude.mul_(1.3)
    x = torch.randn(2, 4, 9, requires_grad=True)
    a = layer.A.detach().clone().requires_grad_()
    b = layer.B.detach().clone().requires_grad_()
    m = layer.magnitude.detach().clone().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    weight = layer.base.weight.detach()
    direction = weight + layer.scale * b @ a
    dora = m[:, None] * direction / direction.norm(dim=1, keepdim=True).detach()
    delta = (dora - weight) @ (torch.eye(9) - q @ q.T)
    expected = F.linear(xr, weight + delta, layer.base.bias)
    actual = layer(x)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
    coefficient = torch.randn_like(actual)
    (actual * coefficient).sum().backward()
    (expected * coefficient).sum().backward()
    for left, right in ((layer.A, a), (layer.B, b), (layer.magnitude, m), (x, xr)):
        assert torch.allclose(left.grad, right.grad, atol=3e-6, rtol=2e-5)


def test_q_is_owned_and_train_eval_never_merge():
    q = torch.eye(8)[:, :2]
    layer = ProtectedDoRALinear(torch.nn.Linear(8, 4), q, rank=2)
    weight = layer.base.weight.detach().clone()
    saved_q = layer.Q.clone()
    q.zero_()
    assert torch.equal(saved_q, layer.Q)
    with torch.no_grad():
        layer.B.normal_()
        layer.magnitude.mul_(1.1)
    layer.eval()
    layer(torch.randn(2, 8))
    layer.train()
    layer(torch.randn(2, 8))
    layer.export()
    assert torch.equal(weight, layer.base.weight)


@pytest.mark.parametrize('bias', [True, False])
def test_matches_pinned_peft_forward_gradients_optimizer_and_merge(bias):
    peft = pytest.importorskip('peft')
    if peft.__version__ != '0.18.0':
        # Kaggle images may preinstall another PEFT; parity is only claimed for the pin.
        pytest.skip(f'reference suite requires PEFT 0.18.0, found {peft.__version__}; install ".[reference]"')
    from peft.tuners.lora.layer import Linear
    torch.manual_seed(26)
    base = torch.nn.Linear(9, 5, bias=bias).requires_grad_(False)
    ours = ProtectedDoRALinear(copy.deepcopy(base), rank=3, alpha=6)
    reference = Linear(copy.deepcopy(base), adapter_name='default', r=3, lora_alpha=6,
                       lora_dropout=0, use_dora=True)
    reference_params = [reference.lora_A['default'].weight, reference.lora_B['default'].weight,
                        reference.lora_magnitude_vector['default'].weight]
    our_params = [ours.A, ours.B, ours.magnitude]
    with torch.no_grad():
        ours.B.normal_(std=0.1)
        ours.magnitude.mul_(1.2)
        for left, right in zip(our_params, reference_params):
            right.copy_(left)
    opt = torch.optim.AdamW(our_params, lr=0.001, weight_decay=0.01)
    ref_opt = torch.optim.AdamW(reference_params, lr=0.001, weight_decay=0.01)
    for _ in range(3):
        x = torch.randn(2, 3, 9, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        output, expected = ours(x), reference(xr)
        assert torch.allclose(output, expected, atol=2e-6, rtol=1e-5)
        opt.zero_grad()
        ref_opt.zero_grad()
        coefficient = torch.randn_like(output)
        (output * coefficient).sum().backward()
        (expected * coefficient).sum().backward()
        for left, right in zip(our_params + [x], reference_params + [xr]):
            assert torch.allclose(left.grad, right.grad, atol=3e-6, rtol=2e-5)
        opt.step()
        ref_opt.step()
    container = torch.nn.ModuleDict({'q_proj': ours})
    merge_live(container)
    reference.merge(safe_merge=True)
    assert torch.allclose(container['q_proj'].weight, reference.base_layer.weight, atol=2e-6, rtol=1e-5)
    if bias:
        assert torch.equal(container['q_proj'].bias, reference.base_layer.bias)
