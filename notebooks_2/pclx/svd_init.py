"""SVD initialization of DoRA's A factor, matched to the Kaiming draw's spectral norm.

For each adapted layer, A = c Vᵣᵀ, where Vᵣ holds the top-r right singular
vectors of the B gradient (projected by P = I - Q Qᵀ for protected arms) and c
is the largest singular value of the layer's own Kaiming draw as the forward
pass sees it (A P when protected). This is the comparison the Lean results
cover: among all rank-r A with spectral norm at most c, A = c Vᵣᵀ gives the most
negative first-order loss change of a plain gradient step on B
(`svdInit_minimizes_first_order_opNorm`). The Kaiming draw is consumed first, so
the random-number stream (and everything after it) matches the paired run.
"""

import torch

from pcl.protected_dora import adapter_modules


def effective(a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """A as the protected forward pass uses it: A (I - Q Qᵀ)."""
    return a - (a @ q) @ q.T if q.shape[1] else a


def matched_scale(a: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(effective(a.float(), q.float()), ord=2))


@torch.no_grad()
def apply_svd_init(model, bases: dict, kind: str) -> dict:
    """Overwrite each layer's (Kaiming) A with c Vᵣᵀ; returns per-layer scales and checks."""
    if kind not in ('projected', 'unprojected'):
        raise ValueError('kind must be projected or unprojected')
    modules = adapter_modules(model)
    if set(modules) != set(bases):
        raise ValueError('SVD bases do not cover exactly the adapted layers')
    report = {}
    for name, layer in modules.items():
        v = bases[name][kind].to(device=layer.A.device, dtype=torch.float32)
        rank, width = layer.A.shape
        if v.shape != (width, rank):
            raise ValueError(f'{name}: basis shape {tuple(v.shape)} != ({width}, {rank})')
        gram = v.T @ v
        if not torch.allclose(gram, torch.eye(rank, device=v.device), atol=1e-4):
            raise ValueError(f'{name}: SVD basis columns are not orthonormal')
        if float(layer.B.abs().max()) != 0:
            raise ValueError(f'{name}: SVD initialization must happen at B = 0')
        if kind == 'unprojected' and layer.Q.shape[1]:
            raise ValueError(f'{name}: unprojected SVD bases belong to an unprotected run')
        scale = matched_scale(layer.A, layer.Q)
        # c is this run's own Kaiming draw; it must be the draw NB7 analyzed, or the pairing is broken.
        if abs(scale - bases[name][f'scale_{kind}']) > 1e-5 * bases[name][f'scale_{kind}']:
            raise ValueError(f'{name}: Kaiming draw differs from the one the SVD bases were built with')
        layer.A.copy_(scale * v.T)
        report[name] = {'scale': scale,
                        'leak_into_q': float(torch.linalg.vector_norm(layer.A @ layer.Q)) if layer.Q.shape[1] else 0.0}
    return report
