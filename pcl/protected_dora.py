"""Ordinary DoRA and a hard input-projected DoRA variant.

For PyTorch weights shaped (out, in), normalize each output row. Let
D = diag(m / ||W + sBA||row) (W + sBA). The protected weight is
W + (D - W)(I - QQ.T), NOT just DoRA with projected low-rank factors.
The row norm is detached, following DoRA's reduced-overhead formulation.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


def validate_basis(q: torch.Tensor, width: int, atol: float = 1e-5) -> None:
    if q.ndim != 2 or q.shape[0] != width or q.shape[1] > width:
        raise ValueError("Q must have shape (input width, k), with k <= input width")
    if not torch.isfinite(q).all():
        raise ValueError("Q contains nonfinite values")
    with torch.autocast(device_type=q.device.type, enabled=False):
        gram = q.float().T @ q.float()
        if not torch.allclose(gram, torch.eye(q.shape[1], device=q.device, dtype=torch.float32), atol=atol, rtol=0):
            raise ValueError("Q columns are not orthonormal")


class ProtectedDoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, q: torch.Tensor | None = None,
                 rank: int = 8, alpha: float = 16, norm_chunk_size: int = 256):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or not math.isfinite(alpha) or alpha <= 0 or norm_chunk_size < 1:
            raise ValueError("require Linear base, positive rank, alpha, and norm chunk size")
        if base.weight.dtype not in (torch.float16, torch.float32):
            raise ValueError('DoRA base must be FP16 or FP32, cast before wrapping')
        self.base = base.requires_grad_(False)
        self.norm_chunk_size = norm_chunk_size
        device = base.weight.device
        if q is None:
            q = torch.empty(base.in_features, 0, device=device)
        # Own the protection state; later edits to the caller's Q must not alter it.
        q = q.detach().to(device=device, dtype=torch.float32).clone()
        validate_basis(q, base.in_features)
        self.register_buffer("Q", q)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.magnitude = nn.Parameter(self.direction_norm())

    def check_precision(self) -> None:
        if any(t.dtype != torch.float32 for t in (self.A, self.B, self.Q, self.magnitude)):
            raise RuntimeError("DoRA factors, magnitudes and Q must stay FP32; cast the base before wrapping")

    @torch.no_grad()
    def direction_norm(self) -> torch.Tensor:
        # Chunking bounds temporary dense storage and avoids cancellation in a
        # Gram-matrix norm expansion. Do not keep dense adapted weights in autograd.
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            norms = []
            for start in range(0, self.base.out_features, self.norm_chunk_size):
                stop = start + self.norm_chunk_size
                direction = self.base.weight[start:stop].float() + (self.B[start:stop] @ self.A) * self.scale
                norms.append(torch.linalg.vector_norm(direction, dim=1))
            result = torch.cat(norms)
            if not torch.isfinite(result).all() or (result <= 1e-12).any():
                raise ValueError("DoRA direction has a zero or nonfinite row norm")
            return result

    def effective_a(self) -> torch.Tensor:
        self.check_precision()
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            return self.A - (self.A @ self.Q) @ self.Q.T if self.Q.shape[1] else self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check_precision()
        output = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            residual_x = xf - (xf @ self.Q) @ self.Q.T if self.Q.shape[1] else xf
            gain = self.magnitude / self.direction_norm()
            # Both the magnitude update and the directional update see only
            # the residual input. Bias stays frozen and is never scaled.
            base_residual = F.linear(residual_x, self.base.weight.float())
            low_rank = ((residual_x @ self.A.T) @ self.B.T) * self.scale
            delta = (gain - 1) * base_residual + gain * low_rank
        return output + delta.to(output.dtype)

    @torch.no_grad()
    def export(self) -> dict:
        self.check_precision()
        if not all(torch.isfinite(t).all() for t in (self.A, self.B, self.magnitude)):
            raise ValueError('cannot export nonfinite DoRA parameters')
        validate_basis(self.Q, self.base.in_features)
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            gain = self.magnitude / self.direction_norm()
        return {"A_eff": self.effective_a().detach().cpu().clone(),
                "B": self.B.detach().cpu().clone(), "scale": self.scale,
                "row_gain": gain.cpu().clone(), "Q": self.Q.detach().cpu().clone()}


def adapter_modules(model: nn.Module) -> dict[str, ProtectedDoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, ProtectedDoRALinear)}


def attach_adapters(model: nn.Module, *, rank: int = 8, alpha: float = 16,
                    bases: dict[str, torch.Tensor] | None = None,
                    targets: tuple[str, ...] = ("q_proj", "v_proj")) -> list[str]:
    if adapter_modules(model):
        raise ValueError("merge the previous stage before attaching a fresh adapter")
    layers = {name: module for name, module in model.named_modules()
              if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)}
    if not layers:
        raise ValueError("no target Linear modules found")
    if bases is not None and bases.keys() != layers.keys():
        raise ValueError("protection layer names must exactly match target layers")
    if bases is not None:
        for name, layer in layers.items():
            validate_basis(bases[name], layer.in_features)
    model.requires_grad_(False)
    for name, layer in layers.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, ProtectedDoRALinear(layer, None if bases is None else bases[name], rank, alpha))
    assert_only_adapters_trainable(model)
    return list(layers)


def assert_only_adapters_trainable(model: nn.Module) -> None:
    modules = adapter_modules(model)
    if not modules:
        raise ValueError("model has no adapters")
    expected = {id(p) for module in modules.values() for p in (module.A, module.B, module.magnitude)}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected:
        raise ValueError("exactly DoRA's A, B, and magnitude parameters must be trainable")
    for module in modules.values():
        module.check_precision()
