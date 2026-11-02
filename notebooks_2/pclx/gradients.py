"""The B-task gradient with respect to every adapted q/v weight, without parameter gradients.

For a Linear layer y = x Wᵀ, dL/dW = Σ_tokens δᵀ x with δ = dL/dy. Forward hooks
keep each wrapper's input and register a hook on its output gradient, so the
FP16 base weights never need `requires_grad`. Each layer's contribution goes
straight into its FP32 sum (no second full copy of all gradients), which keeps
the peak on a 15 GB T4 at about the model plus one gradient set.

The loss is the training loss: answer-token cross-entropy normalized by all
answer tokens of the sample, scaled like GradScaler so FP16 backward does not
underflow. A layer whose gradient overflows in FP16 is skipped and retried at a
16x smaller scale; layers already added for that example are not added again.
This works with gradient checkpointing (pcl's non-reentrant mode): hooks that
fire again during recomputation register gradient hooks on tensors that never
receive a gradient, so each layer is still counted exactly once per example.
"""

import torch
import torch.nn.functional as F

from pcl.protected_dora import adapter_modules
from pcl.tokenization import collate


def answer_tokens(records: list[dict]) -> int:
    return sum(sum(label != -100 for label in record['labels'][1:]) for record in records)


def capture_gradients(model, records: list[dict], *, pad_id: int, loss_scale: float | None = None,
                      min_loss_scale: float = 1.0, store_device=None) -> tuple[dict, dict]:
    """Return ({layer: dL/dW in FP32}, info) for the adapted layers of `model`.

    The adapters must be attached at their identity initialization (B = 0, unit
    gains), so the forward pass is exactly the base model's. The default loss
    scale matches training's first update per token (GradScaler starts at 2^16
    for one 16-example update), with automatic back-off on overflow. The model is
    run in train mode, as in training (Qwen3 has no dropout), so gradient
    checkpointing applies if the caller enabled it.
    """
    modules = adapter_modules(model)
    if not modules:
        raise ValueError('attach the adapters (identity initialization) before capturing gradients')
    if any(float(layer.B.detach().abs().max()) != 0 for layer in modules.values()):
        raise ValueError('capture the gradient at B = 0, where the adapted model equals its base')
    total = answer_tokens(records)
    if not total:
        raise ValueError('no supervised answer tokens in the gradient sample')
    device = next(model.parameters()).device
    store_device = store_device or device
    sums = {name: torch.zeros(layer.base.out_features, layer.base.in_features, dtype=torch.float32,
                              device=store_device) for name, layer in modules.items()}
    state = {'scale': None, 'added': set(), 'overflow': set(), 'fired': {}}

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach()

            def on_grad(grad, name=name, x=x):
                state['fired'][name] = state['fired'].get(name, 0) + 1
                if name in state['added']:
                    return
                with torch.autocast(device_type=grad.device.type, enabled=False):
                    g = grad.reshape(-1, grad.shape[-1]).float()
                    if not torch.isfinite(g).all():
                        state['overflow'].add(name)
                        return
                    sums[name].add_((g.T @ x.reshape(-1, x.shape[-1]).float()).to(store_device), alpha=1 / state['scale'])
                state['added'].add(name)
            if not output.requires_grad:
                raise RuntimeError(f'{name}: adapted output does not require grad; is B trainable?')
            output.register_hook(on_grad)
        return hook

    handles = [layer.register_forward_hook(make_hook(name)) for name, layer in modules.items()]
    autocast = device.type == 'cuda' and next(model.parameters()).dtype == torch.float16
    was_training = model.training
    model.train()
    scale = loss_scale if loss_scale is not None else 2.0 ** 16 * max(1.0, len(records) / 16)
    retries = 0
    try:
        for record in records:
            state['added'].clear()
            while True:
                state['overflow'].clear()
                state['fired'].clear()
                state['scale'] = scale
                model.zero_grad(set_to_none=True)
                batch = collate([record], pad_id, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
                    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                   use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       batch['labels'][:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
                (loss * (scale / total)).backward()
                del logits, loss
                if any(count != 1 for count in state['fired'].values()) or len(state['fired']) != len(modules):
                    raise RuntimeError(f'each adapted layer must receive exactly one gradient per example: {state["fired"]}')
                if len(state['added']) == len(modules):
                    break
                if scale / 16 < min_loss_scale:
                    raise FloatingPointError('nonfinite gradient even at the minimum loss scale')
                scale, retries = scale / 16, retries + 1
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return sums, {'examples': len(records), 'answer_tokens': total, 'final_loss_scale': scale,
                  'loss_scale_retries': retries, 'loss': 'answer-token CE normalized by all answer tokens'}
