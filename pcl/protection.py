"""Uncentered second moments -> numerically checked protected bases."""

import torch

from .protected_dora import validate_basis


@torch.no_grad()
def build_basis(moment: torch.Tensor, *, target_energy: float = 0.95, max_k: int = 128,
                rank_tolerance: float = 1e-7, negative_tolerance: float = 1e-6) -> tuple[torch.Tensor, dict]:
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or not torch.isfinite(moment).all():
        raise ValueError("second moment must be square and finite")
    if not 0 < target_energy <= 1 or max_k < 1 or not 0 <= rank_tolerance < 1 or negative_tolerance < 0:
        raise ValueError("invalid basis selection settings")
    c = moment.detach().cpu().double()
    c = (c + c.T) * 0.5
    values, vectors = torch.linalg.eigh(c)
    largest = values[-1].item()
    if largest <= 0 or values[0].item() < -negative_tolerance * largest:
        raise ValueError("no positive energy, or materially negative eigenvalues")
    smallest = values[0].item()
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    energy = values.cumsum(0) / values.sum()
    target_k = min(int((energy < target_energy).sum()) + 1, len(values))
    numerical_rank = int((values > rank_tolerance * values[0]).sum())
    k = min(target_k, max_k, numerical_rank)
    if k < 1:
        raise ValueError("no reliable directions")
    q = vectors[:, :k].float()
    validate_basis(q, c.shape[0])
    actual_energy = float(torch.trace(q.double().T @ c @ q.double()) / torch.trace(c))
    return q, {"k": k, "target_k": target_k, "numerical_rank": numerical_rank,
               "target_energy": target_energy, "captured_energy": actual_energy,
               "target_met": actual_energy >= target_energy - 1e-7,
               "minimum_eigenvalue": smallest, "maximum_eigenvalue": largest,
               "remaining_dimensions": c.shape[0] - k}


def random_basis_like(q: torch.Tensor, seed: int) -> torch.Tensor:
    validate_basis(q, q.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(q.shape, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(values, mode="reduced").Q.float().to(q.device)

