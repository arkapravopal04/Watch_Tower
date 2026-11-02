"""Single-device answer-token-normalized training with exact boundary resumes.

No Trainer dependency; the same loop serves every adapter method. Checkpoints
are trusted local torch files and contain raw factors, never exported A_eff.
"""

import random
import time
import hashlib
import json
import math
import shutil
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .io import atomic_write
from .checkpoints import CHECKPOINT_FORMAT, validate_checkpoint
from .hashing import base_model_hashes
from .protected_dora import adapter_modules, assert_only_adapters_trainable
from .tokenization import collate

PROGRESS_SECONDS = 300


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 2
    microbatch: int = 1
    accumulation: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    token_budget: int | None = None
    fp16: bool = False

    def __post_init__(self):
        if min(self.epochs, self.microbatch, self.accumulation) < 1:
            raise ValueError("epochs and batch sizes must be positive")
        if (not all(math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.max_grad_norm))
                or self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0):
            raise ValueError("invalid optimizer settings")
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token budget must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_order(records: list[dict], config: TrainConfig) -> list[int]:
    """Fixed shuffled epochs, stopped before a complete example exceeds budget."""
    rng = random.Random(config.seed)
    order, tokens = [], 0
    for _ in range(config.epochs):
        indices = list(range(len(records)))
        rng.shuffle(indices)
        for index in indices:
            length = len(records[index]["input_ids"])
            if config.token_budget is not None and tokens + length > config.token_budget:
                return order
            order.append(index)
            tokens += length
    return order


def replay_order(new: list[dict], old: list[dict], *, token_budget: int,
                 old_fraction: float = 0.1, seed: int = 0) -> tuple[list[dict], list[int]]:
    """Mix full examples by token deficit; report actual share (rounding is inevitable)."""
    if not new or not old or not 0 < old_fraction < 1 or token_budget < 1:
        raise ValueError("replay requires both pools, a fraction in (0,1), and a token budget")
    if any(record["split"] != "train" for record in new + old):
        raise ValueError("replay must use training data only")
    rng = random.Random(seed)
    records = new + old
    pools = [list(range(len(new))), list(range(len(new), len(records)))]
    queues = [[], []]
    order, total, old_tokens = [], 0, 0
    while total < token_budget:
        source = int(old_tokens < old_fraction * total)
        if not queues[source]:
            queues[source] = pools[source].copy()
            rng.shuffle(queues[source])
        index = queues[source].pop()
        length = len(records[index]["input_ids"])
        if total + length > token_budget:
            break
        order.append(index)
        total += length
        old_tokens += source * length
    return records, order


class Trainer:
    def __init__(self, model, records: list[dict], config: TrainConfig, *, pad_id: int,
                 provenance: dict, order: list[int] | None = None):
        assert_only_adapters_trainable(model)
        if not records or any(record["split"] != "train" for record in records):
            raise ValueError("training requires nonempty training-only records")
        if not provenance:
            raise ValueError("record base/data/config provenance before training")
        self.model, self.records, self.config = model, records, config
        self.pad_id, self.provenance = pad_id, provenance
        self.base_hashes = base_model_hashes(model)
        if provenance.get('base_tensor_hashes', self.base_hashes) != self.base_hashes:
            raise ValueError('training provenance does not match the frozen base')
        self.data_digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        self.device = next(model.parameters()).device
        if config.fp16 and self.device.type != "cuda":
            raise ValueError("FP16 training is supported only on CUDA")
        self.order = make_order(records, config) if order is None else list(order)
        if not self.order or any(i < 0 or i >= len(records) for i in self.order):
            raise ValueError("invalid or empty training order")
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)
        self.state = {"cursor": 0, "updates": 0, "attempted_updates": 0, "skipped_updates": 0,
                      "nonpadding_tokens": 0, "answer_tokens": 0, "tokens_by_skill": {}, "resume_count": 0}
        self.history = []
        # Dev scores per selectable checkpoint; persisted so selection survives resumes.
        self.evaluations = []

    def save(self, path) -> None:
        layers = {name: {"A": layer.A.detach().cpu(), "B": layer.B.detach().cpu(),
                          "Q": layer.Q.detach().cpu(), "magnitude": layer.magnitude.detach().cpu(),
                          "scale": layer.scale, "norm_chunk_size": layer.norm_chunk_size}
                  for name, layer in adapter_modules(self.model).items()}
        assert_only_adapters_trainable(self.model)
        if base_model_hashes(self.model) != self.base_hashes:
            raise ValueError('frozen base changed during training; refusing to save a misleading checkpoint')
        checkpoint = {"format": CHECKPOINT_FORMAT, "layers": layers, "base_hashes": self.base_hashes,
                      "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                      "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                      "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                      "config": asdict(self.config), "provenance": self.provenance,
                      "order": self.order, "state": self.state, "history": self.history,
                      "evaluations": self.evaluations,
                      "data_digest": self.data_digest,
                      "record_ids": [record['id'] for record in self.records]}
        atomic_write(path, lambda temporary: torch.save(checkpoint, temporary))

    def resume(self, path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(self.config):
            raise ValueError("resume configuration mismatch")
        if checkpoint["provenance"] != self.provenance or checkpoint["order"] != self.order:
            raise ValueError("resume provenance or data-order mismatch")
        if checkpoint["record_ids"] != [record['id'] for record in self.records] or checkpoint['data_digest'] != self.data_digest:
            raise ValueError("resume dataset mismatch")
        validate_checkpoint(checkpoint, self.model)
        if checkpoint['rng_cuda'] and (self.device.type != 'cuda' or len(checkpoint['rng_cuda']) != torch.cuda.device_count()):
            raise ValueError('resume CUDA topology mismatch')
        modules = adapter_modules(self.model)
        if modules.keys() != checkpoint['layers'].keys():
            raise ValueError("resume target layers mismatch")
        for name, layer in modules.items():
            saved = checkpoint['layers'][name]
            if (layer.scale != saved['scale'] or not torch.equal(layer.Q.cpu(), saved['Q'])
                    or layer.norm_chunk_size != saved['norm_chunk_size']
                    or layer.A.shape != saved['A'].shape):
                raise ValueError('resume protection, rank, scale or norm settings mismatch')
        with torch.no_grad():
            for name, layer in modules.items():
                saved = checkpoint['layers'][name]
                for key in ("A", "B", "magnitude"):
                    getattr(layer, key).copy_(saved[key])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.state, self.history = checkpoint["state"], checkpoint["history"]
        self.evaluations = checkpoint["evaluations"]
        self.state["resume_count"] += 1
        torch.set_rng_state(checkpoint["rng_cpu"])
        random.setstate(checkpoint["rng_python"])
        if checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        assert_only_adapters_trainable(self.model)

    def run(self, *, checkpoint_path=None, checkpoint_seconds: float = 1800,
            stop_after_updates: int | None = None, time_guard=None, on_checkpoint=None,
            checkpoint_dir=None, checkpoint_updates: int | None = None) -> dict:
        """Stop/checkpoint only at optimizer boundaries.

        Selectable checkpoints (callback for dev scoring, then a copy in
        checkpoint_dir) happen every `checkpoint_updates` and at completion,
        so their update numbers never depend on wall-clock time. Timed saves
        only refresh the resume file. Without a cadence, timed saves are also
        selectable boundaries.
        """
        self.model.train()
        if checkpoint_seconds <= 0 or (checkpoint_updates is not None and checkpoint_updates < 1):
            raise ValueError('checkpoint intervals must be positive')
        if checkpoint_dir and not checkpoint_path:
            raise ValueError('selectable checkpoints are copies of the resume file; pass checkpoint_path too')
        selectable_done = {'update': None}

        def selectable_boundary():
            if on_checkpoint:
                on_checkpoint(self)
                self.model.train()
            if checkpoint_path:
                self.save(checkpoint_path)
            if checkpoint_dir:
                directory = Path(checkpoint_dir)
                path = directory / f"update-{self.state['attempted_updates']:08d}.pt"
                # Never replace a selectable historical checkpoint on resume.
                if not path.exists():
                    atomic_write(path, lambda temporary: shutil.copyfile(checkpoint_path, temporary))
            selectable_done['update'] = self.state['attempted_updates']
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        last_save = time.monotonic()
        update_size = self.config.microbatch * self.config.accumulation
        total_updates = math.ceil(len(self.order) / update_size)
        last_report = started = time.monotonic()
        start_update = self.state['attempted_updates']
        while self.state["cursor"] < len(self.order):
            if (stop_after_updates is not None and self.state["attempted_updates"] >= stop_after_updates) or (time_guard and time_guard.should_stop()):
                break
            indices = self.order[self.state["cursor"]:self.state["cursor"] + update_size]
            records = [self.records[index] for index in indices]
            answer_count = sum(sum(label != -100 for label in record['labels'][1:]) for record in records)
            if not answer_count:
                raise ValueError("batch has no supervised next-token targets")
            self.optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for start in range(0, len(records), self.config.microbatch):
                batch = collate(records[start:start + self.config.microbatch], self.pad_id, self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.config.fp16):
                    logits = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                # Accumulated loss is normalized over ALL answer tokens in this update.
                loss_sum = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                           batch['labels'][:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("nonfinite training loss")
                self.scaler.scale(loss_sum / answer_count).backward()
                loss_total += float(loss_sum.detach())
            self.scaler.unscale_(self.optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in self.parameters)
            if not finite and not self.config.fp16:
                raise FloatingPointError("nonfinite adapter gradients")
            if finite:
                grad_norms = {key: sum(float(getattr(layer, key).grad.norm())
                                      for layer in adapter_modules(self.model).values()
                                      if getattr(layer, key).grad is not None)
                              for key in ('A', 'B', 'magnitude')}
                if self.state['updates'] == 0 and grad_norms['B'] == 0:
                    raise RuntimeError("first update has no B gradient; inspect the checkpointing path")
                torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm, error_if_nonfinite=True)
            else:
                grad_norms = None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.state['attempted_updates'] += 1
            self.state['updates' if finite else 'skipped_updates'] += 1
            self.state['cursor'] += len(indices)
            self.state['answer_tokens'] += answer_count
            for record in records:
                n = len(record['input_ids'])
                self.state['nonpadding_tokens'] += n
                counts = self.state['tokens_by_skill']
                counts[record['skill']] = counts.get(record['skill'], 0) + n
            self.history.append({"update": self.state['attempted_updates'], "loss": loss_total / answer_count,
                                 "gradients_finite": bool(finite), "gradient_norms": grad_norms,
                                 "scaler": self.scaler.get_scale()})
            if time.monotonic() - last_report >= PROGRESS_SECONDS:
                done = self.state['attempted_updates']
                rate = (done - start_update) / (time.monotonic() - started)
                print(f"[train] update {done}/{total_updates}, loss {self.history[-1]['loss']:.3f}, "
                      f"about {(total_updates - done) / rate:.0f} s of training left", flush=True)
                last_report = time.monotonic()
            timed = time.monotonic() - last_save >= checkpoint_seconds
            on_cadence = bool(checkpoint_updates) and self.state['attempted_updates'] % checkpoint_updates == 0
            if (checkpoint_path or checkpoint_dir or on_checkpoint) and (on_cadence or (timed and not checkpoint_updates)):
                selectable_boundary()
                last_save = time.monotonic()
            elif timed and checkpoint_path:
                self.save(checkpoint_path)
                last_save = time.monotonic()
        complete = self.state["cursor"] >= len(self.order)
        if complete and selectable_done['update'] != self.state['attempted_updates']:
            selectable_boundary()
        elif checkpoint_path:
            self.save(checkpoint_path)
        return dict(self.state)
