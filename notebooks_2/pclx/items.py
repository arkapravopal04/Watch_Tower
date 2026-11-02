"""One work item = one arm on one seed: train (resuming), select on dev, record. No merge.

The follow-up arms are exploratory comparisons against the main study's
verified runs, not promotion candidates, so they skip the verified merge (about
an hour of answer generation per run). Each run still saves its exported
adapter and every selectable checkpoint, so a winning arm can be merged and
put through a locked study later.
"""

import shutil
import time
from pathlib import Path

import yaml

from pcl.io import read_json, write_json
from pcl.kaggle import CONFIGS, stage_a_for, write_model_config

from . import arms as arm_specs
from .plan import ITEM_MARKER, REPLICATION_MARKER, SVD_BASES, main_runs
from .runs import train_next

NEXT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'next.yaml'


def release_gpu() -> None:
    """Free the previous run's model before the next one loads (two 8 GB models do not fit on a T4)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def next_settings() -> dict:
    """configs/next.yaml; PCLX_NEXT_CONFIG points elsewhere only for the miniature rehearsal."""
    import os
    return yaml.safe_load(Path(os.environ.get('PCLX_NEXT_CONFIG', NEXT_CONFIG)).read_text(encoding='utf-8'))


def _dev(data_root) -> dict:
    data = Path(data_root)
    return {'B': data / 'B' / 'dev200.jsonl', 'A': data / 'A' / 'dev200.jsonl', 'general': data / 'general' / 'dev200.jsonl'}


def run_item(found, *, arm, seed, model_config, settings, data_root, out, session_seconds=None,
             train_config=None, reserve_seconds=None) -> dict:
    spec, cfg = arm_specs.arm(arm), next_settings()
    study = yaml.safe_load(Path(settings).read_text(encoding='utf-8'))
    out, data = Path(out), Path(data_root)
    out.mkdir(parents=True, exist_ok=True)
    previous = [entry['dir'] for entry in found['items'].get((arm, seed), [])]
    for entry in found['items'].get((arm, seed), []):
        if entry.get('status') == 'complete':
            summary = {**{k: v for k, v in entry.items() if k != 'dir'}, 'adopted_from': str(entry['dir'])}
            write_json(out / ITEM_MARKER, summary)
            return summary
    stage_a = stage_a_for(found['main'], seed)
    model_a = write_model_config(model_config, out / 'model_a.yaml', patch=stage_a / 'merged_a')
    spectrum, k256 = found['spectra'].get(seed), found['k256'].get(seed)
    protection = {'a': stage_a / 'protection_a.pt', 'k256': k256 / 'protection_k256.pt' if k256 else None,
                  None: None}[spec['protection']]
    order = 'replay' if 'replay' in spec['method'] else 'b'
    svd_bases = spectrum / SVD_BASES[order] if spectrum and spec['init'].startswith('svd') else None
    previous_runs = [Path(p) / 'train' for p in previous if (Path(p) / 'train' / 'run_manifest.json').is_file()]
    resume = None
    if previous_runs:
        latest = max(previous_runs, key=lambda p: read_json(p / 'run_manifest.json')['state']['attempted_updates'])
        resume = latest / 'resume.pt'
    result = train_next(model_config=model_a, train_config=train_config or CONFIGS / 'train.yaml',
                        data=data / 'B' / 'train.jsonl', out=out / 'train', arm=arm, seed=seed, dev=_dev(data),
                        score_at=cfg['score_at'], protection=protection,
                        memory=data / 'A' / 'memory.jsonl' if 'replay' in spec['method'] else None,
                        svd_bases=svd_bases, reuse_update0=main_runs(found, seed), resume=resume,
                        session_seconds=session_seconds, reserve_seconds=reserve_seconds or cfg['reserve_seconds'],
                        eval_batch_size=study['eval_batch_size'], checkpointing=cfg['gradient_checkpointing'])
    release_gpu()
    summary = {'arm': arm, 'seed': seed, 'status': 'incomplete', **{k: spec[k] for k in ('method', 'init', 'protection', 'tier')}}
    if result['complete']:
        from pcl.study import select_checkpoint
        constraints = [('A', study['selection']['retain_pp'], None),
                       ('general', study['selection']['general_pp'], str(stage_a / 'references' / 'm0_general_dev200.jsonl'))]
        selection = select_checkpoint([out / 'train', *previous_runs], primary=['B'], constraints=constraints,
                                      out=out / 'selection.json')
        summary.update(status='complete', update=selection['update'], eligible=selection['eligible'])
    write_json(out / ITEM_MARKER, summary)
    return summary


def replicate(found, *, seed, model_config, settings, data_root, out, updates=None, train_config=None) -> dict:
    """Re-run the first updates of the main study's protected run through this driver and compare losses.

    Paired comparisons with reused baselines are only valid if this driver
    reproduces pcl's training. Both checkpointing modes are checked, because the
    arms may run without checkpointing when it fits.
    """
    cfg = next_settings()
    updates = updates or cfg['replication_updates']
    study = yaml.safe_load(Path(settings).read_text(encoding='utf-8'))
    out, data = Path(out), Path(data_root)
    stage_a = stage_a_for(found['main'], seed)
    baseline = max(main_runs(found, seed, 'protected'), key=lambda p: read_json(p / 'run_manifest.json')['state']['attempted_updates'],
                   default=None)
    if baseline is None:
        raise FileNotFoundError(f'attach the main-study NB4 output with the seed-{seed} protected run')
    reference = [h['loss'] for h in read_json(baseline / 'run_manifest.json')['history'][:updates]]
    model_a = write_model_config(model_config, out / 'model_a.yaml', patch=stage_a / 'merged_a')
    results = {}
    for mode in ('auto', 'on'):
        if mode == 'on' and results['auto']['checkpointing']:
            results['on'] = {**results['auto'], 'same_run_as': 'auto'}  # auto kept checkpointing on: identical run
            continue
        run = out / f'checkpointing_{mode}'
        if run.exists():
            shutil.rmtree(run)
        train_next(model_config=model_a, train_config=train_config or CONFIGS / 'train.yaml',
                   data=data / 'B' / 'train.jsonl', out=run, arm='protected', seed=seed, dev=_dev(data), score_at=[],
                   protection=stage_a / 'protection_a.pt', reuse_update0=main_runs(found, seed),
                   eval_batch_size=study['eval_batch_size'], checkpointing=mode, stop_after_updates=updates)
        release_gpu()
        manifest = read_json(run / 'run_manifest.json')
        losses = [h['loss'] for h in manifest['history'][:updates]]
        worst = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(losses, reference))
        results[mode] = {'losses': losses, 'max_relative_difference': worst,
                         'checkpointing': manifest['next']['memory_probe']['checkpointing']}
    tolerance = cfg['replication_tolerance']
    summary = {'seed': seed, 'status': 'complete', 'updates': updates, 'baseline_run': str(baseline),
               'reference_losses': reference, 'modes': results, 'tolerance': tolerance,
               'passed': all(r['max_relative_difference'] <= tolerance for r in results.values()),
               'note': 'the arms use mode "auto"; if only "on" passes, set gradient_checkpointing: on in configs/next.yaml'}
    write_json(out / REPLICATION_MARKER, summary)
    return summary


def run_queue(found, items, *, session_seconds, out, **kwargs) -> list[dict]:
    """Run 'arm:seed' items in order until the session clock is spent."""
    deadline = time.monotonic() + session_seconds if session_seconds else None
    results = []
    for item in items:
        name, seed = item.rsplit(':', 1)
        seed = int(seed)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining < next_settings()['min_item_seconds']:
            results.append({'item': item, 'status': 'not started (session time)'})
            continue
        if name == 'replicate':
            summary = replicate(found, seed=seed, out=Path(out) / 'replication', **kwargs)
        else:
            summary = run_item(found, arm=name, seed=seed, out=Path(out) / name / f'seed{seed}',
                               session_seconds=remaining, **kwargs)
        release_gpu()
        results.append({'item': item, **{k: v for k, v in summary.items() if k in ('status', 'update', 'eligible', 'passed')}})
        print('[queue]', results[-1], flush=True)
        if summary.get('status') != 'complete':
            break
    return results
