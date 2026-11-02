"""Explicit, base-bound DoRA merges with a no-mutation preflight.

Only merge_export/merge_live change weights. Export, preview, train(), eval(),
and checkpoint restore never merge. In-memory merging is deliberately in-place;
the workflow loads a disposable base and publishes a separate verified patch.
"""

import math

import torch

from .hashing import base_model_hashes
from .protected_dora import adapter_modules, validate_basis

EXPORT_FORMAT = 'pcl-dora-effective-v2'


def projection_residual(update: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(update @ q) / torch.linalg.vector_norm(update).clamp_min(1e-30))


def export_adapters(model: torch.nn.Module) -> dict:
    modules = adapter_modules(model)
    if not modules:
        raise ValueError('no adapters to export')
    return {'format': EXPORT_FORMAT, 'base_hashes': base_model_hashes(model),
            'layers': {name: layer.export() for name, layer in modules.items()}}


def _validate_export(layer, factors):
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError('DoRA export target must be a plain Linear layer')
    if layer.weight.dtype not in (torch.float16, torch.float32):
        raise ValueError('merge storage must use the declared FP16 or FP32 protocol')
    if set(factors) != {'A_eff', 'B', 'Q', 'scale', 'row_gain'}:
        raise ValueError('invalid effective DoRA factor fields')
    a, b, q, gain = (factors[key] for key in ('A_eff', 'B', 'Q', 'row_gain'))
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not torch.isfinite(t).all()
           for t in (a, b, q, gain)):
        raise ValueError('exported factors must be finite FP32 tensors')
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or a.shape[1] != layer.in_features
            or b.shape != (layer.out_features, a.shape[0]) or gain.shape != (layer.out_features,)):
        raise ValueError('exported factors do not match the base layer')
    validate_basis(q, layer.in_features)
    if not math.isfinite(factors['scale']) or factors['scale'] <= 0:
        raise ValueError('export scale must be finite and positive')


def _targets(model, artifact):
    if artifact.get('format') != EXPORT_FORMAT or not artifact.get('layers'):
        raise ValueError('require a base-bound effective DoRA export')
    if not artifact.get('base_hashes') or base_model_hashes(model) != artifact['base_hashes']:
        raise ValueError('merge base hash mismatch: wrong base or adapter already merged')
    wrappers = adapter_modules(model)
    if wrappers and wrappers.keys() != artifact['layers'].keys():
        raise ValueError('export does not match the live adapter targets')
    targets = {}
    seen_weights = set()
    for name, factors in artifact['layers'].items():
        try:
            layer = wrappers[name].base if wrappers else model.get_submodule(name)
        except (KeyError, AttributeError) as error:
            raise ValueError(f'missing merge layer: {name}') from error
        _validate_export(layer, factors)
        storage = (str(layer.weight.device), layer.weight.untyped_storage().data_ptr())
        if storage in seen_weights:
            raise ValueError('shared target weights require an explicitly supported merge protocol')
        seen_weights.add(storage)
        aliases = [n for n, p in model.named_parameters(remove_duplicate=False)
                   if p.untyped_storage().data_ptr() == storage[1] and str(p.device) == storage[0]]
        if len(aliases) != 1:
            raise ValueError('tied target weights are not supported')
        targets[name] = layer
    return targets


def _merged_weight(layer, factors):
    device = layer.weight.device
    with torch.autocast(device_type=device.type, enabled=False):
        a, b, q = (factors[key].to(device=device) for key in ('A_eff', 'B', 'Q'))
        original = layer.weight.float()
        gain = factors['row_gain'].to(device=device).unsqueeze(1)
        residual_weight = original - (original @ q) @ q.T if q.shape[1] else original
        intended = (gain - 1) * residual_weight + gain * (b @ a) * factors['scale']
        merged = (original + intended).to(layer.weight.dtype)
        if not torch.isfinite(intended).all() or not torch.isfinite(merged).all():
            raise ValueError('merge overflow or nonfinite update')
        actual = merged.float() - original
        report = {'intended_residual': projection_residual(intended, q),
                  'stored_residual': projection_residual(actual, q),
                  'rounding_error_norm': float(torch.linalg.vector_norm(actual - intended)),
                  'update_norm': float(torch.linalg.vector_norm(intended))}
        if not all(math.isfinite(value) for value in report.values()):
            raise ValueError('nonfinite merge diagnostics')
    return merged, report


@torch.no_grad()
def preview_merge(model: torch.nn.Module, artifact: dict) -> dict[str, dict]:
    """Validate every target and rounding result without changing any weight."""
    targets = _targets(model, artifact)
    return {name: _merged_weight(layer, artifact['layers'][name])[1] for name, layer in targets.items()}


@torch.no_grad()
def merge_export(model: torch.nn.Module, artifact: dict) -> dict[str, dict]:
    """Explicitly mutate a disposable plain base, after validating all layers.

    Preflight prevents numerical/shape errors from leaving a partial merge.
    Process interruption/OOM is not an in-memory transaction; reload the base
    in that case. The disk workflow publishes only a fully verified model.
    """
    if adapter_modules(model):
        raise ValueError('use merge_live for a model with live adapters')
    diagnostics = preview_merge(model, artifact)
    for name, factors in artifact['layers'].items():
        layer = model.get_submodule(name)
        layer.weight.copy_(_merged_weight(layer, factors)[0])
    return diagnostics


@torch.no_grad()
def merge_live(model: torch.nn.Module) -> dict:
    artifact = export_adapters(model)
    diagnostics = preview_merge(model, artifact)
    wrappers = adapter_modules(model)
    for name, wrapper in wrappers.items():
        wrapper.base.weight.copy_(_merged_weight(wrapper.base, artifact['layers'][name])[0])
    for name, wrapper in wrappers.items():
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, wrapper.base)
    return diagnostics


@torch.no_grad()
def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> dict:
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError('logit tolerances must be finite and nonnegative')
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError('logit tensors must have matching nonempty shapes')
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    error = (reference.float() - candidate.float()).abs()
    return {'passed': finite and bool(torch.allclose(reference.float(), candidate.float(), atol=atol, rtol=rtol)),
            'finite': finite, 'max_abs_error': float(error.max()) if finite else None,
            'changed_argmax': int((reference.argmax(-1) != candidate.argmax(-1)).sum())}
