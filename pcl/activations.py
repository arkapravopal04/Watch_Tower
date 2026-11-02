"""Balanced token sampling and streaming uncentered covariance capture."""

import random
from collections import defaultdict

import torch


def sample_positions(records: list[dict], lengths: list[int], max_tokens: int,
                     seed: int) -> dict[str, list[int]]:
    if not records or len(records) != len(lengths) or max_tokens < 1 or any(n < 1 for n in lengths):
        raise ValueError("invalid memory lengths or token budget")
    if any(record["split"] != "train" for record in records):
        raise ValueError("protection memory must come from the training split")
    if len({r['id'] for r in records}) != len(records):
        raise ValueError("duplicate memory records")
    groups = defaultdict(list)
    rng = random.Random(seed)
    for record, length in zip(records, lengths):
        groups[record["group"]].extend((record["id"], position) for position in range(length))
    for positions in groups.values():
        rng.shuffle(positions)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = defaultdict(list)
    remaining = min(max_tokens, sum(lengths))
    while remaining:
        for key in keys:
            if groups[key] and remaining:
                item, position = groups[key].pop()
                selected[item].append(position)
                remaining -= 1
    return {key: sorted(value) for key, value in selected.items()}


class ActivationCollector:
    """Use in eval mode without checkpointing; set a mask before every forward.

    Separate q/v streams are deliberately retained until equality has been
    established for the chosen architecture. Statistics live on CPU by default.
    """
    def __init__(self, modules: dict[str, torch.nn.Module]):
        if not modules:
            raise ValueError("no activation streams selected")
        self.modules = modules
        self.device = "cpu"
        self.moments = {}
        self.counts = defaultdict(int)
        self.mask = None
        self.handles = []

    def select(self, mask: torch.Tensor) -> None:
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("selection must be a boolean batch-by-token mask")
        self.mask = mask

    def _hook(self, name):
        def capture(module, inputs):
            x = inputs[0].detach()
            if self.mask is None or x.shape[:-1] != self.mask.shape:
                raise ValueError("activation shape does not match the selected token mask")
            with torch.autocast(device_type=x.device.type, enabled=False):
                selected = x[self.mask.to(x.device)].float()
                if not torch.isfinite(selected).all():
                    raise ValueError("nonfinite captured activations")
                if name not in self.moments:
                    self.moments[name] = torch.zeros(x.shape[-1], x.shape[-1], device=self.device, dtype=torch.float32)
                # Only the sampled rows move to CPU, not a full d-by-d matrix each batch.
                selected = selected.to(self.device)
                with torch.autocast(device_type=selected.device.type, enabled=False):
                    self.moments[name].addmm_(selected.T, selected)
                self.counts[name] += len(selected)
        return capture

    def __enter__(self):
        self.handles = [module.register_forward_pre_hook(self._hook(name)) for name, module in self.modules.items()]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_moments(model, encoded: list[dict], selected: dict[str, list[int]]) -> tuple[dict, dict]:
    """Teacher-force each record and accumulate q/v input second moments at the selected positions."""
    modules = {name: module for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in ("q_proj", "v_proj") and isinstance(module, torch.nn.Linear)}
    model.eval()
    device = next(model.parameters()).device
    with ActivationCollector(modules) as collector, torch.inference_mode():
        for record in encoded:
            positions = selected.get(record["id"], [])
            if not positions:
                continue
            ids = torch.tensor([record["input_ids"]], device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[0, positions] = True
            collector.select(mask)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return collector.moments, collector.counts
