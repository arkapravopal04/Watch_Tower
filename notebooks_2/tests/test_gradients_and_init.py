"""Gradient capture, the SVD initializer, and the Lean gradient formulas against PyTorch's DoRA."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('transformers')

from pcl.protected_dora import adapter_modules, attach_adapters  # noqa: E402
from pcl.protection import random_basis_like  # noqa: E402
from pcl.tokenization import collate  # noqa: E402
from pcl.train import seed_everything  # noqa: E402
from pclx.gradients import answer_tokens, capture_gradients  # noqa: E402
from pclx.spectrum import analyze_layer, analyze_model, go_no_go, summarize  # noqa: E402
from pclx.svd_init import apply_svd_init, effective  # noqa: E402


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=256,
                         pad_token_id=0, eos_token_id=1)
    return Qwen3ForCausalLM(config).float().eval()


def records(count=6, seed=3):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(count):
        ids = torch.randint(2, 64, (8 + i % 3,), generator=generator).tolist()
        rows.append({'id': str(i), 'skill': 'B', 'group': str(i), 'split': 'train', 'input_ids': ids,
                     'labels': [-100] * 4 + ids[4:]})
    return rows


def bases_for(model, k=4):
    targets = {n: m for n, m in model.named_modules() if n.rsplit('.', 1)[-1] in ('q_proj', 'v_proj')}
    return {n: random_basis_like(torch.linalg.qr(torch.randn(m.in_features, k))[0], i) for i, (n, m) in enumerate(targets.items())}


def loss_of(model, rows):
    total = answer_tokens(rows)
    loss = 0
    for row in rows:
        batch = collate([row], 0)
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
        loss = loss + F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), batch['labels'][:, 1:].reshape(-1),
                                      ignore_index=-100, reduction='sum')
    return loss / total


def test_hook_gradient_equals_autograd_weight_gradient():
    model, rows = tiny_model(), records()
    bases = bases_for(model)
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases)
    hooked, info = capture_gradients(model, rows, pad_id=0, loss_scale=8.0)
    assert info['examples'] == len(rows)
    for layer in adapter_modules(model).values():
        layer.base.weight.requires_grad_(True)
    model.zero_grad()
    loss_of(model, rows).backward()
    for name, layer in adapter_modules(model).items():
        assert torch.allclose(hooked[name], layer.base.weight.grad, atol=1e-5, rtol=1e-4), name


def test_hook_gradient_is_unchanged_by_gradient_checkpointing():
    """NB7 captures with pcl's non-reentrant checkpointing on; recomputation must not double-count."""
    rows = records()
    results = []
    for checkpointing in (False, True):
        model = tiny_model()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        seed_everything(0)
        attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
        results.append(capture_gradients(model, rows, pad_id=0, loss_scale=8.0)[0])
    for name in results[0]:
        assert torch.allclose(results[0][name], results[1][name], atol=1e-6, rtol=1e-5), name


def test_overflowing_layers_are_retried_without_double_counting():
    model, rows = tiny_model(), records(2)
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
    reference, _ = capture_gradients(model, rows, pad_id=0, loss_scale=1.0)
    # An absurd starting scale overflows FP32 too, so every layer is retried at smaller scales.
    retried, info = capture_gradients(model, rows, pad_id=0, loss_scale=1e39, min_loss_scale=1e-6)
    assert info['loss_scale_retries'] > 0
    for name in reference:
        assert torch.allclose(reference[name], retried[name], atol=1e-6, rtol=1e-4), name


def test_gram_singular_vectors_match_svd():
    from pclx.spectrum import right_singular
    torch.manual_seed(2)
    m = torch.randn(40, 12) @ torch.diag(torch.linspace(3, 0.1, 12))
    values, vectors = right_singular(m, 4)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    assert torch.allclose(values, s, atol=1e-5)
    assert torch.allclose((vectors.T @ vh[:4].T).abs(), torch.eye(4), atol=1e-4)  # same vectors up to sign


def test_lean_gradient_formulas_match_pytorch_dora():
    """gradB = s·G·P·Aᵀ and gradA = 0 at initialization (FirstStep.lean), for the real ProtectedDoRALinear."""
    model, rows = tiny_model(), records()
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
    gradients, _ = capture_gradients(model, rows, pad_id=0, loss_scale=1.0)
    model.zero_grad()
    loss_of(model, rows).backward()
    for name, layer in adapter_modules(model).items():
        projector = torch.eye(layer.Q.shape[0]) - layer.Q @ layer.Q.T
        expected = layer.scale * gradients[name] @ projector @ layer.A.detach().T
        assert torch.allclose(layer.B.grad, expected, atol=1e-6, rtol=1e-4), name
        assert float(layer.A.grad.abs().max()) == 0.0


def test_svd_init_is_identity_scaled_and_inside_the_complement():
    model, rows = tiny_model(), records()
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
    kaiming = {n: l.A.detach().clone() for n, l in adapter_modules(model).items()}
    gradients, _ = capture_gradients(model, rows, pad_id=0)
    layers, bases = analyze_model(model, gradients, rank=2)
    batch = collate(rows[:2], 0)
    before = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    report = apply_svd_init(model, bases, 'projected')
    after = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    assert torch.equal(before, after)  # B = 0: the initialization changes nothing in the forward pass
    for name, layer in adapter_modules(model).items():
        c = float(torch.linalg.matrix_norm(effective(kaiming[name], layer.Q), ord=2))
        assert report[name]['scale'] == pytest.approx(c, rel=1e-6)
        assert torch.allclose(layer.A @ layer.A.T, c ** 2 * torch.eye(2), atol=1e-5)
        assert report[name]['leak_into_q'] < 1e-5
        # Lean's svdInit_minimizes_first_order_opNorm, numerically: SVD beats this seed's Kaiming draw (SGD).
        first = layers[name]['first_order_projected']['sgd']
        assert first['svd'] >= first['kaiming'] * (1 - 1e-6)
    summary = summarize(layers)
    assert summary['first_order_projected_sgd']['ratio'] >= 1 - 1e-6
    assert set(go_no_go({0: summary})) == {'run_svd_arms', 'per_seed', 'rule'}


def test_svd_initialized_first_step_is_the_eckart_young_truncation():
    """FirstStep.lean, end to end on the real ProtectedDoRALinear: with A = c·V_rᵀ built from the
    captured G, autograd's B gradient is s·G·P·Aᵀ, and one plain step on B changes the weight by
    exactly -(η s² c²)·H V_r V_rᵀ (`svdInit_firstStep_eq_truncate`), the best rank-r approximation
    of -(η s² c²)·H (`svdInit_firstStep_best`), with H = G (I - Q Qᵀ) truncated by torch's own SVD."""
    model, rows = tiny_model(), records()
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
    gradients, _ = capture_gradients(model, rows, pad_id=0, loss_scale=1.0)
    _, bases = analyze_model(model, gradients, rank=2)
    apply_svd_init(model, bases, 'projected')
    model.zero_grad()
    loss_of(model, rows).backward()
    eta = 1e-2
    for name, layer in adapter_modules(model).items():
        q, a, s = layer.Q.double(), layer.A.detach().double(), layer.scale
        projector = torch.eye(q.shape[0], dtype=torch.float64) - q @ q.T
        g = gradients[name].double()
        h = g @ projector
        assert torch.allclose(layer.B.grad.double(), s * g @ projector @ a.T, atol=1e-6, rtol=1e-4), name
        delta_w = s * (-eta * layer.B.grad.double()) @ a @ projector       # gains held at 1 (B = 0)
        u, sigma, vh = torch.linalg.svd(h, full_matrices=False)
        truncation = u[:, :2] @ torch.diag(sigma[:2]) @ vh[:2]
        c = bases[name]['scale_projected']
        expected = -(eta * s ** 2 * c ** 2) * truncation
        assert torch.allclose(delta_w, expected, atol=1e-7, rtol=1e-3), name
        # Its distance to the dense step is the Eckart–Young minimum over rank-2 matrices, Σ_{i>2} σᵢ².
        dense = -(eta * s ** 2 * c ** 2) * h
        assert float((dense - delta_w).norm() ** 2) == pytest.approx(
            float((eta * s ** 2 * c ** 2) ** 2 * sigma[2:].pow(2).sum()), rel=1e-3)


def test_first_order_prediction_matches_an_actual_sgd_step():
    """The SGD first-order quantity s²‖H Aᵀ‖² equals ⟨G, ΔW⟩/(-η) for a real step on B."""
    model, rows = tiny_model(), records()
    seed_everything(0)
    attach_adapters(model, rank=2, alpha=4, bases=bases_for(model))
    gradients, _ = capture_gradients(model, rows, pad_id=0, loss_scale=1.0)
    layers, _ = analyze_model(model, gradients, rank=2)
    model.zero_grad()
    loss_of(model, rows).backward()
    eta = 1e-3
    predicted = sum(d['first_order_projected']['sgd']['kaiming'] for d in layers.values())
    actual = 0.0
    for name, layer in adapter_modules(model).items():
        delta_b = -eta * layer.B.grad
        projector = torch.eye(layer.Q.shape[0]) - layer.Q @ layer.Q.T
        delta_w = layer.scale * delta_b @ layer.A.detach() @ projector
        actual += float((gradients[name] * delta_w).sum())
    assert actual == pytest.approx(-eta * predicted, rel=1e-4)


def test_analyze_layer_on_a_known_low_rank_gradient():
    torch.manual_seed(1)
    width, out, k = 12, 10, 3
    q = torch.linalg.qr(torch.randn(width, k))[0]
    p = torch.eye(width) - q @ q.T
    g = (torch.randn(out, 2) @ torch.randn(2, width)) @ p  # rank 2, already in the complement
    diagnostics, bases = analyze_layer(g, q, torch.randn(4, width), scale=2.0, rank=4)
    assert diagnostics['in_q'] < 1e-10
    assert diagnostics['top_projected']['8'] == pytest.approx(1.0)
    assert diagnostics['zero_singular_in_basis'] == 2  # rank deficiency is reported, not hidden
    assert torch.allclose(bases['projected'].T @ bases['projected'], torch.eye(4), atol=1e-5)
