"""Gradient-spectrum diagnostics at M_A, and the SVD bases the SVD-initialized arms use.

Per adapted layer, with G = dL_B/dW at M_A, P = I - Q Qᵀ and H = G P:
* `in_q`: share of the B gradient's energy inside A's protected input space,
  ‖G Q‖² / ‖G‖², i.e. how directly learning B pushes on what protection freezes;
* `top_r`: share of ‖H‖² captured by the top r singular directions, against the
  r / (in - k) a uniformly random r-dimensional subspace captures on average;
* `first_order`: the predicted first-step loss decrease for the seed's own
  Kaiming A and for the SVD initializer, for plain SGD (∝ s²‖H Aᵀ‖_F², the
  quantity in the Lean theorems) and for AdamW's sign-like first step
  (∝ s ‖H Aᵀ‖₁, entrywise). Summed over layers, their ratio is the predicted
  first-step speed-up. The magnitude path is the same for both initializers.
"""

import math
from pathlib import Path

import torch

from pcl.protected_dora import adapter_modules

from .svd_init import effective

RANKS = (1, 8, 32, 128)


def _top_fractions(energy: torch.Tensor, total: float) -> dict:
    return {str(r): float(energy[:r].sum() / total) if total > 0 else 0.0 for r in RANKS}


def right_singular(m: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All singular values (descending) and the top-`rank` right singular vectors of m.

    Eigendecomposition of the in×in Gram matrix mᵀm in FP64: the same values and
    vectors as an SVD of the out×in matrix (the top ones are what is used), and
    much faster than a full SVD on a T4.
    """
    m64 = m.double()
    values, vectors = torch.linalg.eigh(m64.T @ m64)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    return values.sqrt().float(), vectors[:, :rank].float().contiguous()


@torch.no_grad()
def analyze_layer(g: torch.Tensor, q: torch.Tensor, a: torch.Tensor, scale: float, rank: int) -> tuple[dict, dict]:
    """Diagnostics and SVD bases for one layer (on g's device)."""
    g, q, a = g.float(), q.to(g).float(), a.to(g).float()
    h = g - (g @ q) @ q.T if q.shape[1] else g
    total, projected_total = float(g.pow(2).sum()), float(h.pow(2).sum())
    s_h, v_proj = right_singular(h, rank)
    s_g, v_unproj = right_singular(g, rank)
    a_eff = effective(a, q)
    c_proj = float(torch.linalg.matrix_norm(a_eff, ord=2))
    c_unproj = float(torch.linalg.matrix_norm(a, ord=2))
    complement = g.shape[1] - q.shape[1]

    def first_order(matrix, a_kaiming, v, c):
        kaiming, svd = matrix @ a_kaiming.T, c * (matrix @ v)
        return {'sgd': {'kaiming': scale ** 2 * float(kaiming.pow(2).sum()), 'svd': scale ** 2 * float(svd.pow(2).sum())},
                'adam': {'kaiming': scale * float(kaiming.abs().sum()), 'svd': scale * float(svd.abs().sum())}}
    diagnostics = {
        'shape': list(g.shape), 'k': q.shape[1], 'grad_energy': total,
        'in_q': float((g @ q).pow(2).sum() / total) if total > 0 and q.shape[1] else 0.0,
        'top_projected': _top_fractions(s_h.pow(2), projected_total),
        'top_unprojected': _top_fractions(s_g.pow(2), total),
        'random_baseline': {str(r): min(1.0, r / complement) if complement > 0 else 1.0 for r in RANKS},
        'effective_rank_projected': float(projected_total ** 2 / s_h.pow(4).sum()) if projected_total > 0 else 0.0,
        'singular_values_projected': s_h[:64].tolist(),
        'scales': {'projected': c_proj, 'unprojected': c_unproj},
        'first_order_projected': first_order(h, a_eff, v_proj, c_proj),
        'first_order_unprojected': first_order(g, a, v_unproj, c_unproj),
        # Gram eigenvalues put numerical zeros near 1e-8 of the top singular value; below 1e-5 of it is a zero direction.
        'zero_singular_in_basis': int((s_h[:rank] <= 1e-5 * max(float(s_h[0]), 1e-30)).sum()),
    }
    return diagnostics, {'projected': v_proj.cpu(), 'unprojected': v_unproj.cpu(),
                         'scale_projected': c_proj, 'scale_unprojected': c_unproj}


def first_update(records: list[dict], memory: list[dict], config, order: str) -> list[dict]:
    """The examples of a run's first optimizer update, built exactly as `pcl train` / `pclx` build the order.

    'b' is the B-only order (protected, sequential); 'replay' is the replay arms' token-matched mix
    with Stage-A memory. The Lean results concern one plain step on the loss whose gradient is G, so G
    is taken on exactly these examples, normalized like the Trainer (by this update's answer tokens).
    """
    from pcl.train import make_order, replay_order
    size = config.microbatch * config.accumulation
    if order == 'b':
        return [records[i] for i in make_order(records, config)[:size]]
    if order == 'replay':
        matched = sum(len(records[i]['input_ids']) for i in make_order(records, config))
        mixed, indices = replay_order(records, memory, token_budget=matched, seed=config.seed)
        return [mixed[i] for i in indices[:size]]
    raise ValueError(f'unknown training order {order!r}')


def analyze_model(model, gradients: dict, rank: int) -> tuple[dict, dict]:
    layers, bases = {}, {}
    for name, layer in adapter_modules(model).items():
        layers[name], bases[name] = analyze_layer(gradients[name], layer.Q, layer.A.detach(), layer.scale, rank)
    return layers, bases


def summarize(layers: dict) -> dict:
    """Medians over layers and first-order totals (summed over layers, as the loss sees them)."""
    def median(values):
        values = sorted(values)
        n = len(values)
        return float((values[(n - 1) // 2] + values[n // 2]) / 2) if n else math.nan
    out = {'layers': len(layers),
           'median_in_q': median([d['in_q'] for d in layers.values()]),
           'median_top_projected': {r: median([d['top_projected'][r] for d in layers.values()]) for r in map(str, RANKS)},
           'median_top_unprojected': {r: median([d['top_unprojected'][r] for d in layers.values()]) for r in map(str, RANKS)},
           'median_random_baseline': {r: median([d['random_baseline'][r] for d in layers.values()]) for r in map(str, RANKS)},
           'median_effective_rank_projected': median([d['effective_rank_projected'] for d in layers.values()])}
    for kind in ('projected', 'unprojected'):
        for optimizer in ('sgd', 'adam'):
            k = sum(d[f'first_order_{kind}'][optimizer]['kaiming'] for d in layers.values())
            s = sum(d[f'first_order_{kind}'][optimizer]['svd'] for d in layers.values())
            out[f'first_order_{kind}_{optimizer}'] = {'kaiming': k, 'svd': s, 'ratio': s / k if k > 0 else math.nan}
    return out


def markdown(seed: int, summary: dict, info: dict) -> str:
    rows = [f'## Seed {seed}', '',
            f"Gradient sample: the {info['examples']} B-training examples ({info['answer_tokens']} answer tokens) "
            f"of the first update of this seed's training order, the step the SVD initializer is built for.", '',
            f"- B-gradient energy inside A's protected subspace (median over {summary['layers']} layers): "
            f"**{100 * summary['median_in_q']:.1f}%**",
            f"- Median effective rank of the projected gradient: {summary['median_effective_rank_projected']:.1f}", '',
            '| top r | projected gradient | unprojected gradient | random r-dim subspace |', '|---|---|---|---|']
    for r in map(str, RANKS):
        rows.append(f"| {r} | {100 * summary['median_top_projected'][r]:.1f}% | {100 * summary['median_top_unprojected'][r]:.1f}% | "
                    f"{100 * summary['median_random_baseline'][r]:.2f}% |")
    rows += ['', '| predicted first-step decrease, SVD ÷ Kaiming | plain SGD | AdamW (sign-like first step) |', '|---|---|---|']
    for kind in ('projected', 'unprojected'):
        rows.append(f"| {kind} (arms: {'protected*_svd' if kind == 'projected' else 'sequential_svd'}) | "
                    f"{summary[f'first_order_{kind}_sgd']['ratio']:.2f}× | {summary[f'first_order_{kind}_adam']['ratio']:.2f}× |")
    return '\n'.join(rows) + '\n'


def go_no_go(summaries: dict, *, min_capture_ratio: float = 10.0, min_adam_speedup: float = 1.2) -> dict:
    """The rule fixed before any SVD-initialized run: run the SVD arms only if the premise holds.

    Premise: the projected B gradient is concentrated (median top-8 share at least
    `min_capture_ratio` × the random baseline) and the predicted AdamW first-step
    speed-up is at least `min_adam_speedup` in every seed analyzed.
    """
    checks = {}
    for seed, s in summaries.items():
        capture = s['median_top_projected']['8'] / max(s['median_random_baseline']['8'], 1e-30)
        checks[seed] = {'capture_ratio_top8': capture, 'adam_speedup': s['first_order_projected_adam']['ratio'],
                        'passed': capture >= min_capture_ratio and s['first_order_projected_adam']['ratio'] >= min_adam_speedup}
    return {'run_svd_arms': bool(checks) and all(c['passed'] for c in checks.values()), 'per_seed': checks,
            'rule': {'min_capture_ratio': min_capture_ratio, 'min_adam_speedup': min_adam_speedup}}


def write_bases(path, *, bases: dict, model_hashes: dict, protection_hash: str, example_ids: list, rank: int, info: dict):
    from pcl.io import atomic_write
    artifact = {'format': 'pclx-svd-bases-v1', 'model_hashes': model_hashes, 'protection_hash': protection_hash,
                'example_ids': example_ids, 'rank': rank, 'bases': bases, 'gradient': info}
    atomic_write(Path(path), lambda temporary: torch.save(artifact, temporary))
