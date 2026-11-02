"""Training driver for the follow-up arms: `pcl train` plus SVD init and a sparse dev schedule.

Everything that defines a run is pcl's own code: the model loader, encoding,
data order (`make_order` / `replay_order`), adapter attachment under the same
seed, the Trainer (optimizer, GradScaler, clipping, exact resume) and the dev
scorer. What changes, and why each change keeps runs paired with the main study:

* `init`: after the seeded Kaiming draw (so the random stream is unchanged),
  A may be replaced by the spectral-norm-matched SVD initializer.
* Dev sets are scored only at `score_at` (plus the final update) instead of
  every 35 updates. Only scored updates get selectable checkpoint copies. The
  main study scored the same updates, so selection on this grid is comparable.
* Update 0 is the same model for every Stage-B run of a seed (B = 0 is the
  identity), so its dev predictions are copied from a main-study run after
  checking the model hashes, dev-file hashes and prediction hashes.
* Gradient checkpointing (pure recomputation) is switched off only if a probe
  on the longest training example shows the step fits in memory.
"""

import dataclasses
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from pcl.io import atomic_write, file_hash, read_json, read_jsonl, write_json, write_jsonl

from . import arms as arm_specs

NEXT_SOURCE = Path(__file__).resolve().parent


def _source_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))}


def find_update0(candidates, *, model_hashes: dict, dev_hashes: dict) -> dict | None:
    """A main-study run of the same M_A whose update-0 dev predictions can be reused."""
    for run in map(Path, candidates):
        manifest_path = run / 'run_manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest['provenance'].get('base_tensor_hashes') != model_hashes:
            continue
        if any(manifest.get('dev_sets', {}).get(name) != digest for name, digest in dev_hashes.items()):
            continue
        entry = next((e for e in manifest.get('evaluations', []) if e['update'] == 0), None)
        if not entry or set(entry['sets']) != set(dev_hashes):
            continue
        files = {name: run / meta['predictions'] for name, meta in entry['sets'].items()}
        if all(path.is_file() and file_hash(path) == entry['sets'][name]['predictions_hash'] for name, path in files.items()):
            return {'run': str(run), 'entry': entry, 'files': files}
    return None


def _probe_memory(model, records, pad_id, fraction: float) -> dict:
    """Peak memory of one forward/backward on the longest example without checkpointing."""
    from pcl.tokenization import collate
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        return {'checkpointing': True, 'reason': 'no CUDA device'}
    longest = max(records, key=lambda r: len(r['input_ids']))
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        batch = collate([longest], pad_id, next(model.parameters()).device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch['labels'][:, 1:].reshape(-1),
                               ignore_index=-100, reduction='sum')
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    except torch.cuda.OutOfMemoryError:
        peak = None
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    # The optimizer state for the adapters (two FP32 moments) is small; keep a margin for it and fragmentation.
    fits = peak is not None and peak < fraction * total
    return {'checkpointing': not fits, 'probe_peak_bytes': peak, 'device_bytes': total, 'fraction': fraction,
            'longest_tokens': len(longest['input_ids'])}


def train_next(*, model_config, train_config, data, out, arm, seed, dev: dict, score_at, protection=None,
               memory=None, svd_bases=None, reuse_update0=(), resume=None, session_seconds=None,
               reserve_seconds=1800, eval_batch_size=1, checkpointing='auto', memory_fraction=0.82,
               stop_after_updates=None) -> dict:
    from pcl.cli import _load, _yaml
    from pcl.evaluate import evaluate, summarize
    from pcl.hashing import model_hashes
    from pcl.kaggle import TimeGuard, environment
    from pcl.merge import export_adapters
    from pcl.protected_dora import attach_adapters
    from pcl.registry import identity
    from pcl.tokenization import encode_record
    from pcl.train import TrainConfig, Trainer, make_order, replay_order, seed_everything

    spec = arm_specs.arm(arm) if arm in arm_specs.ARMS else {'method': arm, 'init': 'kaiming'}
    method, init = spec['method'], spec['init']
    model, tokenizer, snapshot, model_config = _load(SimpleNamespace(model_config=model_config))
    raw = _yaml(train_config)
    if raw.pop('adapter') != 'dora':
        raise ValueError('this study uses DoRA')
    rank, alpha = raw.pop('rank'), raw.pop('alpha')
    raw['seed'] = seed
    config = TrainConfig(**raw)
    if config.fp16 != (model_config['precision'] == 'fp16'):
        raise ValueError('training and model precision must agree')
    output = Path(out)
    if (output / 'run_manifest.json').exists() and not resume:
        raise FileExistsError('use a fresh run directory, or explicitly resume')
    base_hashes = model_hashes(model)
    records = [encode_record(tokenizer, r, model_config['max_length']) for r in read_jsonl(data)]
    bases, protection_hash = None, None
    if method in ('protected', 'random_q', 'protected_replay'):
        if not protection:
            raise ValueError(f'{arm} needs a protection basis from its exact Stage-A model')
        artifact = torch.load(protection, map_location='cpu', weights_only=True)
        if artifact['model_hashes'] != base_hashes:
            raise ValueError('protection was collected from a different model')
        bases, protection_hash = artifact['bases'], file_hash(protection)
    seed_everything(config.seed)  # the same raw-factor (Kaiming) draw as every paired main-study run
    attach_adapters(model, rank=rank, alpha=alpha, bases=bases)
    init_report, svd_hash, svd_example_ids = None, None, None
    if init.startswith('svd'):
        if not svd_bases:
            raise ValueError(f'{arm} needs the SVD bases written by `pclx spectrum` for this seed')
        artifact = torch.load(svd_bases, map_location='cpu', weights_only=True)
        if artifact['model_hashes'] != base_hashes:
            raise ValueError('SVD bases were computed on a different model')
        if init == 'svd_projected' and artifact['protection_hash'] != protection_hash:
            raise ValueError('projected SVD bases were computed with a different protection basis')
        svd_example_ids = artifact['example_ids']
        from .svd_init import apply_svd_init
        init_report = apply_svd_init(model, artifact['bases'], init.removeprefix('svd_'))
        svd_hash = file_hash(svd_bases)
    order = None
    if method in ('replay', 'protected_replay'):
        if not memory or config.token_budget is None:
            raise ValueError('replay requires the Stage-A memory and a fixed total token budget')
        pool = [encode_record(tokenizer, r, model_config['max_length']) for r in read_jsonl(memory)]
        matched = sum(len(records[i]['input_ids']) for i in make_order(records, config))
        records, order = replay_order(records, pool, token_budget=matched, seed=config.seed)
    probe = {'checkpointing': True, 'reason': 'requested'} if checkpointing == 'on' else (
        _probe_memory(model, records, tokenizer.pad_token_id, memory_fraction) if checkpointing == 'auto'
        else {'checkpointing': False, 'reason': 'requested'})
    if probe['checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    session_environment = environment()
    provenance = {'base_tensor_hashes': base_hashes, 'snapshot': identity(snapshot), 'data_hash': file_hash(data),
                  'protection_hash': protection_hash, 'memory_hash': file_hash(memory) if memory else None,
                  'method': method, 'adapter': 'dora', 'rank': rank, 'alpha': alpha, 'model_config': model_config,
                  'environment': {k: session_environment[k] for k in ('python', 'packages', 'cuda', 'gpus')},
                  'arm': arm, 'init': init, 'svd_bases_hash': svd_hash, 'gradient_checkpointing': probe['checkpointing']}
    provenance['source_hashes'] = _source_hashes(Path(__import__('pcl').__file__).parent)
    provenance['next_source_hashes'] = _source_hashes(NEXT_SOURCE)
    session_next_hashes = provenance['next_source_hashes']
    if resume:
        # A driver-only fix must not orphan a saved run. pcl's hashes and everything else stay strictly checked.
        provenance['next_source_hashes'] = torch.load(resume, map_location='cpu', weights_only=True)['provenance']['next_source_hashes']
    trainer = Trainer(model, records, config, pad_id=tokenizer.pad_token_id, provenance=provenance, order=order)
    update_size = config.microbatch * config.accumulation
    if svd_example_ids is not None:
        first = [trainer.records[i]['id'] for i in trainer.order[:update_size]]
        if first != svd_example_ids:
            raise ValueError('the SVD bases were not built from this run\'s first update; '
                             'the Lean first-step results would not apply')
    if resume:
        trainer.resume(resume)
    dev_sets = {name: read_jsonl(path) for name, path in dev.items()}
    dev_hashes = {name: file_hash(path) for name, path in dev.items()}
    if any(r['split'] != 'dev' for rows in dev_sets.values() for r in rows):
        raise ValueError('checkpoint scoring uses development data only')
    total_updates = math.ceil(len(trainer.order) / update_size)
    points = sorted({p for p in score_at if 0 < p < total_updates} | {total_updates})
    timings = {'scoring_seconds': []}
    reused = None

    def scored(update):
        return any(entry['update'] == update for entry in trainer.evaluations)

    def score():
        update = trainer.state['attempted_updates']
        started = time.monotonic()
        entry = {'update': update, 'sets': {}}
        for name, rows in dev_sets.items():
            predictions = evaluate(model, tokenizer, rows, batch_size=eval_batch_size,
                                   max_new_tokens=model_config['max_new_tokens'], max_context=model_config['max_context'],
                                   label=f'dev {name} @ update {update}')
            relative = f'dev/update-{update:08d}/{name}.jsonl'
            write_jsonl(output / relative, predictions)
            entry['sets'][name] = {**summarize(predictions), 'predictions': relative,
                                   'predictions_hash': file_hash(output / relative)}
        trainer.evaluations.append(entry)
        model.train()
        timings['scoring_seconds'].append(time.monotonic() - started)
        print(f'[dev] update {update}: ' + ', '.join(f"{n} {v['accuracy']:.3f}" for n, v in entry['sets'].items()), flush=True)

    def selectable():
        trainer.save(output / 'resume.pt')
        target = output / 'checkpoints' / f"update-{trainer.state['attempted_updates']:08d}.pt"
        if not target.exists():
            atomic_write(target, lambda temporary: shutil.copyfile(output / 'resume.pt', temporary))

    if trainer.state['attempted_updates'] == 0 and not scored(0):
        found = find_update0(reuse_update0, model_hashes=base_hashes, dev_hashes=dev_hashes)
        if found:
            entry = {'update': 0, 'sets': {}}
            for name, source in found['files'].items():
                relative = f'dev/update-00000000/{name}.jsonl'
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output / relative)
                entry['sets'][name] = {**found['entry']['sets'][name], 'predictions': relative}
            trainer.evaluations.append(entry)
            reused = found['run']
            print(f'[dev] update 0 reused from {reused}', flush=True)
        else:
            score()
    guard = TimeGuard(session_seconds, reserve_seconds) if session_seconds else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokens_before, started, train_seconds = trainer.state['nonpadding_tokens'], time.perf_counter(), 0.0
    for point in points:
        if stop_after_updates is not None:
            point = min(point, stop_after_updates)
        if trainer.state['attempted_updates'] < point:
            segment = time.perf_counter()
            trainer.run(checkpoint_path=output / 'resume.pt', checkpoint_seconds=1800, stop_after_updates=point,
                        time_guard=guard)
            train_seconds += time.perf_counter() - segment
        if trainer.state['attempted_updates'] < point:
            break  # the session clock ran out; the next session resumes here
        if trainer.state['attempted_updates'] > point:
            continue  # an earlier session already passed this point
        if point in points and not scored(point):
            estimate = max(timings['scoring_seconds'] or [0])
            if guard and guard.should_stop() and estimate:
                break  # not enough time left to score; the resumed run scores this update first
            score()
            selectable()
        if stop_after_updates is not None and trainer.state['attempted_updates'] >= stop_after_updates:
            break
    elapsed = time.perf_counter() - started
    state = dict(trainer.state)
    if not (output / 'resume.pt').exists() or trainer.state['attempted_updates'] not in [e['update'] for e in trainer.evaluations]:
        trainer.save(output / 'resume.pt')
    metrics = {'session_wall_seconds': elapsed, 'training_wall_seconds': train_seconds,
               'training_tokens_per_second': (state['nonpadding_tokens'] - tokens_before) / max(train_seconds, 1e-9),
               'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
               'scoring_seconds': timings['scoring_seconds']}
    atomic_write(output / 'adapter.pt', lambda temporary: torch.save(export_adapters(model), temporary))
    complete = state['cursor'] == len(trainer.order) and scored(total_updates)
    manifest = {'provenance': provenance, 'train_config': dataclasses.asdict(config), 'state': state,
                'complete': complete, 'adapter_hash': file_hash(output / 'adapter.pt'),
                'checkpoint_hash': file_hash(output / 'resume.pt'), 'history': trainer.history, 'metrics': metrics,
                'session_environment': session_environment, 'evaluations': trainer.evaluations, 'dev_sets': dev_hashes,
                'next': {'arm': arm, 'init': init, 'score_at': points, 'update0_reused_from': reused,
                         'session_next_source_hashes': session_next_hashes,
                         'init_report': init_report, 'memory_probe': probe}}
    write_json(output / 'run_manifest.json', manifest)
    return {'out': str(output), 'state': state, 'complete': complete}
