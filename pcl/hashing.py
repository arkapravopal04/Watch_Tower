"""Stable tensor identities, including the plain base beneath live adapters."""

import hashlib

import torch


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{list(value.shape)}:".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {name: tensor_hash(value) for name, value in model.state_dict().items()}


def base_model_hashes(model: torch.nn.Module) -> dict[str, str]:
    from .protected_dora import adapter_modules
    wrappers = adapter_modules(model)
    result = {}
    for name, value in model.state_dict().items():
        mapped = name
        for prefix in wrappers:
            if name in {f'{prefix}.{key}' for key in ('A', 'B', 'Q', 'magnitude')}:
                mapped = None
                break
            if name.startswith(prefix + '.base.'):
                mapped = prefix + '.' + name[len(prefix + '.base.'):]
                break
        if mapped is not None:
            result[mapped] = tensor_hash(value)
    return result
