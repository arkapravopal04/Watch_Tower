"""Command-line entry points: `python -m pclx <command> --help`."""

import argparse
import gc
import json
from pathlib import Path

from pcl.io import file_hash, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def _found(args):
    from .plan import discover_next
    return discover_next(args.search, exclude=args.exclude)


def spectrum(args):
    """NB7 work: 'spectrum:SEED' (gradient spectrum + SVD bases) and 'k256:SEED' (the k<=256 basis), in order."""
    results = {}
    for task in args.tasks:
        kind, seed = task.split(':')
        out = Path(args.out) / f'seed{int(seed)}'
        if kind == 'spectrum':
            results[task] = _spectrum_one(args, int(seed), out)
        elif kind == 'k256':
            results[task] = _k256_one(args, int(seed), out)
        else:
            raise ValueError(f'unknown NB7 task {task!r}')
        gc.collect()
    return results


def _model_a(args, seed, out):
    from pcl.kaggle import stage_a_for, write_model_config
    stage_a = stage_a_for(_found(args)['main'], seed)
    out.mkdir(parents=True, exist_ok=True)
    return stage_a, write_model_config(args.model_config, out / f'model_a_seed{seed}.yaml', patch=stage_a / 'merged_a')


def _spectrum_one(args, seed, out):
    import torch
    from types import SimpleNamespace
    from pcl.cli import _load, _yaml
    from pcl.hashing import model_hashes
    from pcl.protected_dora import attach_adapters
    from pcl.tokenization import encode_record
    from pcl.train import TrainConfig, seed_everything
    from .gradients import capture_gradients
    from .spectrum import analyze_model, first_update, markdown, summarize, write_bases
    from .plan import SPECTRUM_MARKER, SVD_BASES
    stage_a, model_a = _model_a(args, seed, out)
    if not all((out / name).exists() for name in SVD_BASES.values()):
        model, tokenizer, _, model_config = _load(SimpleNamespace(model_config=model_a))
        hashes = model_hashes(model)
        raw = _yaml(args.train_config)
        raw.pop('adapter')
        rank, alpha = raw.pop('rank'), raw.pop('alpha')
        raw['seed'] = seed
        config = TrainConfig(**raw)
        records = [encode_record(tokenizer, r, model_config['max_length'])
                   for r in read_jsonl(Path(args.data_root) / 'B' / 'train.jsonl')]
        memory = [encode_record(tokenizer, r, model_config['max_length'])
                  for r in read_jsonl(Path(args.data_root) / 'A' / 'memory.jsonl')]
        protection_path = stage_a / 'protection_a.pt'
        protection = torch.load(protection_path, map_location='cpu', weights_only=True)
        if protection['model_hashes'] != hashes:
            raise ValueError('protection basis belongs to a different model')
        seed_everything(config.seed)  # the exact Kaiming draw every paired run starts from
        attach_adapters(model, rank=rank, alpha=alpha, bases=protection['bases'])
        # As in training: pcl's DoRA forward keeps FP32 weight copies for backward, so without
        # recomputation the activations would not fit next to the gradient sums on a T4.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # G is the gradient of exactly the loss the run's first update steps on (the Lean premise):
        # its 16 examples, normalized by their answer tokens. The replay arms' first update mixes in
        # Stage-A memory, so they get their own G. The B-only G also gives the diagnostics.
        for order, name in SVD_BASES.items():
            sample = first_update(records, memory, config, order)
            gradients, info = capture_gradients(model, sample, pad_id=tokenizer.pad_token_id)
            info.update(order=order, updates=1)
            layers, bases = analyze_model(model, gradients, rank)
            del gradients
            write_bases(out / name, bases=bases, model_hashes=hashes, protection_hash=file_hash(protection_path),
                        example_ids=[r['id'] for r in sample], rank=rank, info=info)
            if order == 'b':
                summary = summarize(layers)
                write_json(out / 'spectrum.json', {'seed': seed, 'summary': summary, 'gradient': info, 'layers': layers})
                (out / 'spectrum.md').write_text(markdown(seed, summary, info), encoding='utf-8')
            del layers, bases
            gc.collect()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = read_json(out / 'spectrum.json')
    write_json(out / SPECTRUM_MARKER, {'seed': seed, 'status': 'complete', 'summary': result['summary'],
                                       'stage_a': str(stage_a),
                                       'svd_bases': {o: file_hash(out / n) for o, n in SVD_BASES.items()}})
    return result['summary']


def _k256_one(args, seed, out):
    """The tier-3 basis with pcl's own capture: the same tokens and moments, so its first 128 directions are Q's."""
    from pcl.cli import invoke
    from .plan import K256_MARKER
    _, model_a = _model_a(args, seed, out)
    if not (out / 'protection_k256.pt').exists():
        invoke(['protect', '--model-config', model_a, '--protection-config', ROOT / 'configs' / 'protection_k256.yaml',
                '--memory', Path(args.data_root) / 'A' / 'memory.jsonl', '--out', out / 'protection_k256.pt'])
    diagnostics = read_json(str(out / 'protection_k256.pt') + '.json')['diagnostics']
    ks = sorted(d['k'] for d in diagnostics.values())
    energies = sorted(d['captured_energy'] for d in diagnostics.values())
    summary = {'seed': seed, 'status': 'complete', 'protection_k256': file_hash(out / 'protection_k256.pt'),
               'median_k': ks[len(ks) // 2], 'layers_at_cap': sum(k == 256 for k in ks),
               'median_captured_energy': energies[len(energies) // 2]}
    write_json(out / K256_MARKER, summary)
    return summary


def spectrum_report(args):
    from .items import next_settings
    from .spectrum import go_no_go
    found = _found(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(found['spectra'])
    summaries = {seed: read_json(found['spectra'][seed] / 'spectrum.json')['summary'] for seed in seeds}
    decision = go_no_go(summaries, **next_settings()['go_no_go'])
    text = ['# Gradient spectrum at M_A (skill-B gradient, every adapted q/v layer)', '']
    text += [(found['spectra'][seed] / 'spectrum.md').read_text(encoding='utf-8') for seed in seeds]
    text += ['## Decision rule (fixed before any SVD-initialized run)', '',
             f"Run the SVD arms only if, in every seed, the median top-8 share of the projected gradient is at least "
             f"{decision['rule']['min_capture_ratio']:.0f}× a random 8-dim subspace's and the predicted AdamW first-step "
             f"speed-up is at least {decision['rule']['min_adam_speedup']:.1f}×.", '',
             '| seed | top-8 share ÷ random | AdamW speed-up | passed |', '|---|---|---|---|']
    text += [f"| {seed} | {c['capture_ratio_top8']:.0f}× | {c['adam_speedup']:.2f}× | {c['passed']} |"
             for seed, c in decision['per_seed'].items()]
    text += ['', f"**Run the SVD-initialized arms: {decision['run_svd_arms']}**", '']
    (out / 'spectrum_report.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(out / 'go_no_go.json', decision)
    return decision


def queue(args):
    from .items import run_queue
    found = _found(args)
    return run_queue(found, args.items, session_seconds=args.session_seconds, out=args.out,
                     model_config=args.model_config, settings=args.settings, data_root=args.data_root,
                     train_config=args.train_config)


def report(args):
    from .report import build_report
    return build_report(_found(args), args.out)


def parser():
    from pcl.kaggle import CONFIGS
    root = argparse.ArgumentParser(description='Follow-up experiments on the protected-DoRA study')
    commands = root.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--search', nargs='+', required=True, help='roots holding the attached inputs (and outputs)')
        p.add_argument('--exclude', help='the code dataset, never searched')
    p = commands.add_parser('spectrum', help="NB7 tasks in order: 'spectrum:SEED' and/or 'k256:SEED'")
    common(p)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--model-config', required=True, help='M0 model config')
    p.add_argument('--data-root', required=True)
    p.add_argument('--train-config', default=str(CONFIGS / 'train.yaml'))
    p.add_argument('--out', required=True, help='each seed writes OUT/seed<N>')
    p.set_defaults(func=spectrum)
    p = commands.add_parser('spectrum-report', help='combine seeds and apply the go/no-go rule')
    common(p)
    p.add_argument('--out', required=True)
    p.set_defaults(func=spectrum_report)
    p = commands.add_parser('queue', help="run 'arm:seed' work items in order until the session ends")
    common(p)
    p.add_argument('--items', nargs='+', required=True)
    p.add_argument('--model-config', required=True, help='M0 model config')
    p.add_argument('--settings', required=True, help="the pilot's study.yaml (with amendments)")
    p.add_argument('--data-root', required=True)
    p.add_argument('--train-config', default=str(CONFIGS / 'train.yaml'))
    p.add_argument('--session-seconds', type=float)
    p.add_argument('--out', required=True)
    p.set_defaults(func=queue)
    p = commands.add_parser('report', help='tables, paired contrasts and figures for every arm found')
    common(p)
    p.add_argument('--out', required=True)
    p.set_defaults(func=report)
    return root


def invoke(argv):
    args = parser().parse_args([str(a) for a in argv])
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
    args = parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, allow_nan=False, default=str))
