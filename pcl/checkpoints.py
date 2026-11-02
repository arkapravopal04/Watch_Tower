"""Load a selected raw checkpoint without merging or resuming its optimizer."""

import math

import torch

from .hashing import base_model_hashes
from .protected_dora import ProtectedDoRALinear, adapter_modules, assert_only_adapters_trainable, validate_basis

CHECKPOINT_FORMAT = 'pcl-dora-resume-v2'


def validate_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    if checkpoint.get('format') != CHECKPOINT_FORMAT or not checkpoint.get('layers'):
        raise ValueError('expected a nonempty raw DoRA checkpoint, not an effective export')
    if base_model_hashes(model) != checkpoint['base_hashes']:
        raise ValueError('checkpoint base hash mismatch')
    wrappers = adapter_modules(model)
    if wrappers and wrappers.keys() != checkpoint['layers'].keys():
        raise ValueError('checkpoint target layers mismatch')
    for name, saved in checkpoint['layers'].items():
        try:
            layer = wrappers[name].base if wrappers else model.get_submodule(name)
        except (AttributeError, KeyError) as error:
            raise ValueError(f'missing checkpoint layer: {name}') from error
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError('checkpoint targets must be Linear layers')
        a, b, m, q = (saved[key] for key in ('A', 'B', 'magnitude', 'Q'))
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (a, b, m, q)):
            raise ValueError('checkpoint contains nonfinite or non-FP32 adapter parameters')
        if (a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != layer.in_features
                or b.shape != (layer.out_features, a.shape[0]) or m.shape != (layer.out_features,)):
            raise ValueError('checkpoint adapter shape mismatch')
        validate_basis(q, layer.in_features)
        if not math.isfinite(saved['scale']) or saved['scale'] <= 0 or saved['norm_chunk_size'] < 1:
            raise ValueError('invalid checkpoint DoRA settings')


def restore_adapters(model: torch.nn.Module, checkpoint: dict) -> None:
    """Attach the selected raw A/B/m/Q to its exact plain base. Never merge."""
    if adapter_modules(model):
        raise ValueError('restore requires a plain base; use Trainer.resume for an existing training model')
    validate_checkpoint(checkpoint, model)
    # Construct wrappers off-model before installing them.
    wrappers = {}
    for name, saved in checkpoint['layers'].items():
        rank = saved['A'].shape[0]
        wrapper = ProtectedDoRALinear(model.get_submodule(name), saved['Q'], rank=rank,
                                      alpha=rank * saved['scale'], norm_chunk_size=saved['norm_chunk_size'])
        wrapper.scale = saved['scale']
        with torch.no_grad():
            for key in ('A', 'B', 'magnitude'):
                getattr(wrapper, key).copy_(saved[key])
        wrapper.direction_norm()  # reject undefined DoRA directions before installing
        wrappers[name] = wrapper
    model.requires_grad_(False)
    for name, wrapper in wrappers.items():
        parent_name, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent_name) if parent_name else model, leaf, wrapper)
    assert_only_adapters_trainable(model)


def load_checkpoint(path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=True)
