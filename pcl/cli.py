"""Thin command-line entry points; experiment logic lives in package modules."""

import argparse
import dataclasses
from pathlib import Path

import yaml

from .io import atomic_write, file_hash, read_json, read_jsonl, write_json, write_jsonl

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def _yaml(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _load(args):
    from .model_setup import load_model
    config = _yaml(args.model_config)
    model, tokenizer, manifest = load_model(config['path'], manifest_path=config['snapshot_manifest'],
                                            precision=config['precision'], device=config['device'],
                                            backend=config['backend'], patch=config.get('patch'))
    return model, tokenizer, manifest, config


def _named_paths(values) -> dict:
    result = {}
    for value in values or []:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in result:
            raise ValueError(f'expected unique NAME=PATH, got {value!r}')
        result[name] = path
    return result


def generate_data(args):
    from .data.generator import generate
    from .data.grader import VERSION
    from .data.splits import stratified_subset
    from .tokenization import encode_record
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("dataset output must be empty; never replace a frozen data manifest")
    sizes = {'train': args.train, 'dev': args.dev, 'test': args.test, 'test_hard': args.test_hard}
    records = generate(args.skill, sizes, args.seed, allow_test=args.seal_test)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        for record in records:  # raises on any over-length example; nothing is truncated
            record['token_length'] = len(encode_record(tokenizer, record, args.max_length)['input_ids'])
    files = {}
    for split in sizes:
        selected = [record for record in records if record['split'] == split]
        if selected:
            write_jsonl(output / f'{split}.jsonl', selected)
            files[split] = file_hash(output / f'{split}.jsonl')
    for source, name, count in [('train', 'memory', args.memory), ('dev', 'dev200', args.dev_subset)]:
        pool = [record for record in records if record['split'] == source]
        if pool and count:
            write_jsonl(output / f'{name}.jsonl', stratified_subset(pool, min(count, len(pool)), args.seed + 1))
            files[name] = file_hash(output / f'{name}.jsonl')
    manifest = {'skill': args.skill, 'seed': args.seed, 'files': files, 'grader_version': VERSION,
                'split_unit': 'physical scenario; two related wordings per group; equation families shared',
                'groups': {split: len({r['group'] for r in records if r['split'] == split}) for split in sizes},
                'token_lengths_verified': bool(args.tokenizer), 'max_length': args.max_length}
    if args.expect_manifest:
        expected = read_json(args.expect_manifest)['files']
        mismatched = [name for name in ('train', 'dev', 'dev200', 'memory')
                      if name in expected and files.get(name) != expected[name]]
        if mismatched:
            raise ValueError(f'sealed bank does not reproduce the development files {mismatched}; '
                             'use the same sizes, seed and tokenizer')
        manifest['development_bank_verified'] = True
    write_json(output / 'split_manifest.json', manifest)
    return {'out': str(output), 'counts': sizes, 'token_lengths_verified': bool(args.tokenizer)}


def train(args):
    import time
    import torch
    from .kaggle import TimeGuard, environment
    from .hashing import model_hashes
    from .merge import export_adapters
    from .protected_dora import attach_adapters
    from .protection import random_basis_like
    from .registry import identity
    from .tokenization import encode_record
    from .train import TrainConfig, Trainer, make_order, replay_order, seed_everything
    model, tokenizer, snapshot, model_config = _load(args)
    train_config = _yaml(args.train_config)
    if train_config.pop('adapter') != 'dora':
        raise ValueError("this study uses DoRA")
    rank, alpha = train_config.pop('rank'), train_config.pop('alpha')
    if args.seed is not None:
        train_config['seed'] = args.seed
    config = TrainConfig(**train_config)
    if config.fp16 != (model_config['precision'] == 'fp16'):
        raise ValueError("training and model precision must agree")
    output = Path(args.out)
    if (output / 'run_manifest.json').exists() and not args.resume:
        raise FileExistsError("use a fresh run directory, or explicitly resume")
    base_hashes = model_hashes(model)
    records = [encode_record(tokenizer, record, model_config['max_length']) for record in read_jsonl(args.data)]
    bases = None
    protection_hash = None
    if args.method in ('protected', 'random_q', 'protected_replay'):
        if not args.protection:
            raise ValueError("selected method requires protection from its exact Stage-A model")
        protection = torch.load(args.protection, map_location='cpu', weights_only=True)
        if protection['model_hashes'] != base_hashes:
            raise ValueError("protection was collected from a different model")
        bases = protection['bases']
        protection_hash = file_hash(args.protection)
        if args.method == 'random_q':
            bases = {name: random_basis_like(q, config.seed + i + 100000) for i, (name, q) in enumerate(sorted(bases.items()))}
    seed_everything(config.seed)  # same raw-factor initialization in every paired method
    attach_adapters(model, rank=rank, alpha=alpha, bases=bases)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    order = None
    if args.method in ('replay', 'protected_replay'):
        if not args.memory or config.token_budget is None:
            raise ValueError("replay requires the Stage-A memory and a fixed total token budget")
        memory = [encode_record(tokenizer, record, model_config['max_length']) for record in read_jsonl(args.memory)]
        # Match the paired non-replay run's total tokens: whichever of epochs or
        # token_budget binds there also binds here, so replay gets no extra compute.
        matched_budget = sum(len(records[i]['input_ids']) for i in make_order(records, config))
        records, order = replay_order(records, memory, token_budget=matched_budget, seed=config.seed)
    session_environment = environment()
    # Resume must match the software/accelerator stack, but not host details
    # (kernel release, free disk) that legitimately vary between Kaggle sessions.
    provenance = {'base_tensor_hashes': base_hashes, 'snapshot': identity(snapshot),
                  'data_hash': file_hash(args.data), 'protection_hash': protection_hash,
                  'memory_hash': file_hash(args.memory) if args.memory else None,
                  'method': args.method, 'adapter': 'dora', 'rank': rank, 'alpha': alpha,
                  'model_config': model_config,
                  'environment': {key: session_environment[key] for key in ('python', 'packages', 'cuda', 'gpus')}}
    source_root = Path(__file__).parent
    provenance['source_hashes'] = {p.relative_to(source_root).as_posix(): file_hash(p) for p in sorted(source_root.rglob('*.py'))}
    trainer = Trainer(model, records, config, pad_id=tokenizer.pad_token_id, provenance=provenance, order=order)
    if args.resume:
        trainer.resume(args.resume)
    dev_sets = {name: read_jsonl(path) for name, path in _named_paths(args.dev).items()}
    if any(record['split'] != 'dev' for rows in dev_sets.values() for record in rows):
        raise ValueError('checkpoint scoring uses development data only')

    def score(current):
        from .evaluate import evaluate, summarize
        update = current.state['attempted_updates']
        if any(entry['update'] == update for entry in current.evaluations):
            return
        entry = {'update': update, 'sets': {}}
        for name, rows in dev_sets.items():
            predictions = evaluate(model, tokenizer, rows, batch_size=args.eval_batch_size,
                                   max_new_tokens=model_config['max_new_tokens'], max_context=model_config['max_context'],
                                   label=f'dev {name} @ update {update}')
            relative = f'dev/update-{update:08d}/{name}.jsonl'
            write_jsonl(output / relative, predictions)
            entry['sets'][name] = {**summarize(predictions), 'predictions': relative,
                                   'predictions_hash': file_hash(output / relative)}
        current.evaluations.append(entry)
        print(f'[dev] update {update}: ' + ', '.join(f"{n} {v['accuracy']:.3f}" for n, v in entry['sets'].items()),
              flush=True)

    if dev_sets and trainer.state['attempted_updates'] == 0:
        score(trainer)  # identity adapters: exactly the starting model, the selection reference
    guard = TimeGuard(args.session_seconds, args.reserve_seconds) if args.session_seconds else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    tokens_before = trainer.state['nonpadding_tokens']
    started = time.perf_counter()
    state = trainer.run(checkpoint_path=output / 'resume.pt', checkpoint_seconds=args.checkpoint_seconds,
                        time_guard=guard, stop_after_updates=args.stop_after_updates,
                        checkpoint_dir=output / 'checkpoints', checkpoint_updates=args.checkpoint_updates,
                        on_checkpoint=score if dev_sets else None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {'training_wall_seconds': elapsed,
               'training_tokens_per_second': (state['nonpadding_tokens'] - tokens_before) / max(elapsed, 1e-9),
               'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
               'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
    atomic_write(output / 'adapter.pt', lambda temporary: torch.save(export_adapters(model), temporary))
    manifest = {'provenance': provenance, 'train_config': dataclasses.asdict(config), 'state': state,
                'complete': state['cursor'] == len(trainer.order),
                'adapter_hash': file_hash(output / 'adapter.pt'), 'checkpoint_hash': file_hash(output / 'resume.pt'),
                'history': trainer.history, 'metrics': metrics, 'session_environment': session_environment,
                'evaluations': trainer.evaluations, 'dev_sets': {n: file_hash(p) for n, p in _named_paths(args.dev).items()}}
    write_json(output / 'run_manifest.json', manifest)
    return {'out': str(output), 'state': state, 'complete': manifest['complete']}


def protect(args):
    import torch
    from .activations import capture_moments, sample_positions
    from .hashing import model_hashes
    from .protection import build_basis
    from .tokenization import encode_record
    model, tokenizer, snapshot, model_config = _load(args)
    records = read_jsonl(args.memory)
    encoded = [encode_record(tokenizer, record, model_config['max_length']) for record in records]
    settings = _yaml(args.protection_config)
    selected = sample_positions(records, [len(record['input_ids']) for record in encoded],
                                settings.pop('max_tokens'), settings.pop('memory_seed'))
    moments, counts = capture_moments(model, encoded, selected)
    bases, diagnostics = {}, {}
    for name in sorted(moments):
        bases[name], diagnostics[name] = build_basis(moments.pop(name), **settings)
        diagnostics[name]['tokens'] = counts[name]
    artifact = {'format': 'pcl-protection-v1', 'model_hashes': model_hashes(model), 'bases': bases,
                'diagnostics': diagnostics, 'memory_hash': file_hash(args.memory), 'positions': selected,
                'config': _yaml(args.protection_config), 'streams': 'separate q/v; no unverified sharing'}
    atomic_write(args.out, lambda temporary: torch.save(artifact, temporary))
    write_json(str(args.out) + '.json', {key: value for key, value in artifact.items() if key != 'bases'})
    return {'out': args.out, 'layers': len(bases), 'diagnostics': diagnostics}


def merge(args):
    from .checkpoints import load_checkpoint
    from .merge_workflow import verified_merge
    # No made-up universal precision thresholds: limits come from the pilot's study.yaml.
    limits = _yaml(args.settings)['merge']
    missing = [key for key in ('atol', 'rtol', 'max_stored_residual') if limits.get(key) is None]
    if missing:
        raise ValueError(f'the settings file lacks pilot-fixed merge limits {missing}')
    selection = read_json(args.selection)
    if file_hash(selection['checkpoint']) != selection['checkpoint_hash']:
        raise ValueError('selected checkpoint changed after selection')
    model, tokenizer, snapshot, config = _load(args)
    records = [record for path in args.dev for record in read_jsonl(path)]
    return verified_merge(model, tokenizer, snapshot, config, load_checkpoint(selection['checkpoint']), records,
                          checkpoint_hash=selection['checkpoint_hash'], data_hashes={str(p): file_hash(p) for p in args.dev},
                          out=args.out, probes=limits.get('probes', 8), batch_size=args.batch_size,
                          atol=limits['atol'], rtol=limits['rtol'], max_stored_residual=limits['max_stored_residual'],
                          max_intended_residual=limits.get('max_intended_residual', 1e-5),
                          max_changed_answers=limits.get('max_changed_answers', 0),
                          max_accuracy_shift_pp=limits.get('max_accuracy_shift_pp'),
                          include_files=_named_paths(args.include))


def download(args):
    import shutil
    from huggingface_hub import HfApi, snapshot_download
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('download into a new, empty directory')
    pinned = _yaml(CONFIGS / 'model.yaml')
    model_id, revision = pinned['model_id'], pinned['revision']
    resolved = HfApi().model_info(model_id, revision=revision).sha
    if resolved != revision:
        raise ValueError(f'{revision} resolved to {resolved}; refusing an inexact revision')
    snapshot_download(repo_id=model_id, revision=revision, local_dir=output)
    shutil.rmtree(output / '.cache', ignore_errors=True)  # download bookkeeping, not model files
    return snapshot_manifest(output, revision)


def snapshot_manifest(path, revision) -> dict:
    from .model_setup import verify_snapshot
    root = Path(path).resolve()
    target = root / 'snapshot_manifest.json'
    if target.exists():
        raise FileExistsError("snapshot is already pinned")
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError("provide the exact Hugging Face snapshot commit")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
    if 'config.json' not in files:
        raise ValueError("not a complete local model snapshot")
    write_json(target, {'revision': revision, 'files': files})
    verify_snapshot(root, target)
    return {'manifest': str(target), 'files': len(files), 'revision': revision}


def audit(args):
    from .data.audit import audit_directories
    summary = audit_directories(args.data, args.out, sample=args.sample)
    if not summary['passed']:
        raise SystemExit('audit failed:\n' + '\n'.join(summary['problems'][:50]))
    return {key: summary[key] for key in ('records', 'passed', 'groups', 'by_skill_split', 'token_length')}


def pilot(args):
    from .pilot import run_pilot
    return run_pilot(model_config=args.model_config, data_root=args.data_root, out=args.out,
                     train_config=args.train_config, protection_config=args.protection_config, probes=args.probes,
                     train_updates=args.train_updates, eval_items=args.eval_items, batch_sizes=tuple(args.batch_sizes),
                     merge_items=args.merge_items, capture_tokens=args.capture_tokens)


def select(args):
    from .study import select_checkpoint
    constraints = []
    for value in args.constraint:
        name, margin, *reference = value.split(':', 2)
        constraints.append((name, float(margin), reference[0] if reference else None))
    result = select_checkpoint(args.runs, primary=args.maximize, constraints=constraints, out=args.out)
    return {key: result[key] for key in ('checkpoint', 'update', 'eligible')}


def _stage_kwargs(args):
    return dict(model_config=args.model_config, settings=args.settings, train_config=args.train_config,
                data_root=args.data_root, seed=args.seed, out=args.out, previous=args.previous,
                session_seconds=args.session_seconds, reserve_seconds=args.reserve_seconds)


def stage_a(args):
    from .pipeline import stage_a as run
    return run(**_stage_kwargs(args))


def stage_b(args):
    from .pipeline import stage_b as run
    return run(**_stage_kwargs(args), stage_a_dir=args.stage_a, methods=args.methods)


def stage_joint(args):
    from .pipeline import stage_joint as run
    return run(**_stage_kwargs(args))


def register(args):
    from .study import register_study
    return register_study(args.spec, args.registry)


def evaluate_locked(args):
    from .study import evaluate_locked as run
    return run(args.registry, args.artifact_id, args.model_config, test_root=args.test_root,
               search_roots=args.search, batch_size=args.batch_size, session_seconds=args.session_seconds,
               reserve_seconds=args.reserve_seconds)


def decide(args):
    from .study import decide as run
    result = run(args.registry, smoke_report=args.smoke_report, search_roots=args.search)
    return {key: result[key] for key in ('candidate', 'accepted', 'gates', 'numerical_checks')}


def report(args):
    from .study import report as run
    return run(args.registry, args.out)


def check_env(args):
    from .kaggle import environment
    result = environment()
    write_json(args.out, result)
    return result


def smoke(args):
    from .smoke import run_smoke
    return run_smoke(args.out)


def parser():
    root = argparse.ArgumentParser(description='Protected continual learning with DoRA')
    commands = root.add_subparsers(dest='command', required=True)
    p = commands.add_parser('generate')
    p.add_argument('--skill', choices=['A', 'B', 'general'], required=True)
    p.add_argument('--out', required=True)
    for name, default in [('train', 2000), ('dev', 500), ('test', 0), ('test-hard', 0), ('memory', 256),
                          ('dev-subset', 200), ('seed', 0), ('max-length', 512)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--seal-test', action='store_true')
    p.add_argument('--tokenizer')
    p.add_argument('--expect-manifest', help="development bank's split_manifest.json; sealing must reproduce it")
    p.set_defaults(func=generate_data)
    p = commands.add_parser('audit', help='independent re-derivation of every reference answer')
    p.add_argument('--data', nargs='+', required=True, help='generated data directories')
    p.add_argument('--out', required=True)
    p.add_argument('--sample', type=int, default=100)
    p.set_defaults(func=audit)
    p = commands.add_parser('train')
    p.add_argument('--model-config', required=True)
    p.add_argument('--train-config', default=str(CONFIGS / 'train.yaml'))
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['sequential', 'protected', 'replay', 'random_q', 'protected_replay', 'joint'],
                   default='sequential')
    p.add_argument('--protection')
    p.add_argument('--memory')
    p.add_argument('--seed', type=int)
    p.add_argument('--resume')
    p.add_argument('--stop-after-updates', type=int, help='end early, as if the session ran out (tests)')
    p.add_argument('--checkpoint-seconds', type=float, default=1800)
    p.add_argument('--checkpoint-updates', type=int)
    p.add_argument('--session-seconds', type=float)
    p.add_argument('--reserve-seconds', type=float, default=900)
    p.add_argument('--dev', action='append', metavar='NAME=PATH',
                   help='score this dev set at every selectable checkpoint (repeatable)')
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.set_defaults(func=train)
    p = commands.add_parser('select', help='dev-only checkpoint selection')
    p.add_argument('--runs', nargs='+', required=True, help='every session output directory of one training run')
    p.add_argument('--maximize', action='append', required=True, metavar='DEV_SET')
    p.add_argument('--constraint', action='append', default=[], metavar='SET:MARGIN_PP[:REFERENCE_PREDICTIONS]')
    p.add_argument('--out', required=True)
    p.set_defaults(func=select)
    p = commands.add_parser('merge', help='verified merge of a selected checkpoint into a weight patch')
    p.add_argument('--model-config', required=True)
    p.add_argument('--selection', required=True, help='selection.json written by `pcl select`')
    p.add_argument('--settings', required=True, help="the pilot's study.yaml (merge limits)")
    p.add_argument('--dev', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--include', action='append', metavar='NAME=PATH',
                   help='bundle a provenance file (e.g. selection.json) into the published patch')
    p.set_defaults(func=merge)
    p = commands.add_parser('protect')
    p.add_argument('--model-config', required=True)
    p.add_argument('--protection-config', default=str(CONFIGS / 'protection.yaml'))
    p.add_argument('--memory', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=protect)
    p = commands.add_parser('check-env')
    p.add_argument('--out', required=True)
    p.set_defaults(func=check_env)
    p = commands.add_parser('smoke')
    p.add_argument('--out', required=True)
    p.set_defaults(func=smoke)
    p = commands.add_parser('download', help='fetch the pinned model revision from configs/model.yaml (needs internet)')
    p.add_argument('--out', required=True)
    p.set_defaults(func=download)
    p = commands.add_parser('pilot', help='GPU precision/resource pilot; writes study.yaml')
    p.add_argument('--model-config', required=True)
    p.add_argument('--data-root', required=True, help='directory with A/, B/ and general/ banks')
    p.add_argument('--out', required=True)
    p.add_argument('--train-config', default=str(CONFIGS / 'train.yaml'))
    p.add_argument('--protection-config', default=str(CONFIGS / 'protection.yaml'))
    p.add_argument('--probes', type=int, default=6)
    p.add_argument('--train-updates', type=int, default=8)
    p.add_argument('--eval-items', type=int, default=48)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 16, 32])
    p.add_argument('--merge-items', type=int, default=64)
    p.add_argument('--capture-tokens', type=int, default=2048)
    p.set_defaults(func=pilot)
    for name, function in (('stage-a', stage_a), ('stage-b', stage_b), ('stage-joint', stage_joint)):
        p = commands.add_parser(name, help=f'{name}: train, dev-select, verified merge' + (', protect' if name == 'stage-a' else ''))
        p.add_argument('--model-config', required=True, help='M0 model config')
        p.add_argument('--settings', required=True, help="the pilot's study.yaml")
        p.add_argument('--train-config', default=str(CONFIGS / 'train.yaml'))
        p.add_argument('--data-root', required=True)
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--previous', nargs='*', default=[], help='earlier session outputs of this same stage')
        p.add_argument('--session-seconds', type=float, help='session time remaining now; all steps share it')
        p.add_argument('--reserve-seconds', type=float, default=3600)
        if name == 'stage-b':
            p.add_argument('--stage-a', required=True, help='the stage-a output for this seed')
            p.add_argument('--methods', nargs='+', default=['sequential', 'protected', 'random_q', 'replay'])
        p.set_defaults(func=function)
    p = commands.add_parser('register', help='freeze artifacts, test files, gates and the selection rule')
    p.add_argument('--spec', required=True)
    p.add_argument('--registry', required=True)
    p.set_defaults(func=register)
    p = commands.add_parser('evaluate-locked', help='score registered artifacts on every registered test file, once')
    p.add_argument('--registry', required=True)
    p.add_argument('--artifact-id', required=True, nargs='+')
    p.add_argument('--model-config', required=True, help='M0 model config; the patch is found by its hashes')
    p.add_argument('--session-seconds', type=float, help='session time remaining now')
    p.add_argument('--reserve-seconds', type=float, default=600)
    p.add_argument('--test-root', required=True)
    p.add_argument('--search', nargs='+', default=['/kaggle/input'], help='where to look for stage patches')
    p.add_argument('--batch-size', type=int, default=1)
    p.set_defaults(func=evaluate_locked)
    p = commands.add_parser('decide', help='apply the registered gates to the primary candidate')
    p.add_argument('--registry', required=True)
    p.add_argument('--smoke-report', required=True, help='smoke_report.json from the environment check')
    p.add_argument('--search', nargs='+', default=['/kaggle/input'])
    p.set_defaults(func=decide)
    p = commands.add_parser('report', help='tables and contrasts for every registered artifact')
    p.add_argument('--registry', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=report)
    return root


def invoke(argv):
    """Run one command in-process and release its model memory afterwards."""
    import gc
    args = parser().parse_args([str(value) for value in argv])
    try:
        return args.func(args)
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main(argv=None):
    import json
    args = parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, allow_nan=False, default=str))


if __name__ == '__main__':
    main()
