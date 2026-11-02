"""One command per Kaggle notebook cell: train -> dev selection -> verified merge.

Every step writes into a fresh output directory and is skipped when its
result already exists, so rerunning a cell after an error is safe. When a
session ends mid-training, rerun the stage in a new session with the old
output attached and passed as --previous; training resumes exactly, and
anything an earlier session already published is reused.
"""

import shutil
import time
from pathlib import Path

import yaml

from .io import read_json, read_jsonl, write_json, write_jsonl
from .kaggle import CONFIGS, write_model_config

# Time kept back from training for dev selection, the verified merge and Q capture.
PIPELINE_RESERVE_SECONDS = 3600
METHODS = ('sequential', 'protected', 'random_q', 'replay', 'protected_replay')


class _Stage:
    """Shared setup: output directory, the pilot's settings, one session clock for every step."""

    def __init__(self, settings, out, session_seconds):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.settings_path = settings
        self.settings = yaml.safe_load(Path(settings).read_text(encoding='utf-8'))
        self.deadline = time.monotonic() + session_seconds if session_seconds else None

    def finish(self, **result) -> dict:
        write_json(self.out / 'stage_summary.json', result)
        return result


def _invoke(argv):
    from .cli import invoke
    print('pcl', ' '.join(map(str, argv)), flush=True)
    return invoke([str(a) for a in argv])


def _train_and_select(stage, *, model_config, train_config, data, out, dev, primary, constraints, previous=(),
                      method='sequential', protection=None, memory=None, seed=None,
                      reserve_seconds=PIPELINE_RESERVE_SECONDS) -> dict | None:
    run = out / 'train'
    previous_runs = [Path(p) for p in previous if (Path(p) / 'run_manifest.json').is_file()]
    done = run / 'run_manifest.json'
    if not done.exists():
        remaining = None if stage.deadline is None else stage.deadline - time.monotonic()
        if remaining is not None and remaining <= reserve_seconds + 60:
            return None
        argv = ['train', '--model-config', model_config, '--train-config', train_config, '--data', data,
                '--out', run, '--method', method, '--checkpoint-updates', stage.settings['dev_every_updates'],
                '--eval-batch-size', stage.settings['eval_batch_size'], '--seed', seed]
        for name, path in dev.items():
            argv += ['--dev', f'{name}={path}']
        if protection:
            argv += ['--protection', protection]
        if memory:
            argv += ['--memory', memory]
        if remaining is not None:
            argv += ['--session-seconds', remaining, '--reserve-seconds', reserve_seconds]
        if previous_runs:
            latest = max(previous_runs, key=lambda p: read_json(p / 'run_manifest.json')['state']['attempted_updates'])
            argv += ['--resume', latest / 'resume.pt']
        _invoke(argv)
    if not read_json(done)['complete']:
        return None
    selection = out / 'selection.json'
    if not selection.exists():
        _invoke(['select', '--runs', run, *previous_runs, '--out', selection,
                 *sum((['--maximize', name] for name in primary), []),
                 *sum((['--constraint', f'{n}:{m}' + (f':{r}' if r else '')] for n, m, r in constraints), [])])
    return read_json(selection)


def _merge(stage, *, model_config, out, dev_files, run_manifest) -> dict:
    """Verified merge of out/selection.json into out/<name>, bundling the provenance files."""
    target, selection = out, out.parent / 'selection.json'
    if (target / 'patch_manifest.json').exists():
        return read_json(target / 'merge_validation.json')
    report = _invoke(['merge', '--model-config', model_config, '--selection', selection, '--dev', *dev_files,
                      '--out', target, '--settings', stage.settings_path, '--batch-size', stage.settings['eval_batch_size'],
                      '--include', f'selection.json={selection}', '--include', f'run_manifest.json={run_manifest}'])
    if not report.get('passed'):
        raise RuntimeError(f'verified merge failed; inspect {target}.merge_report.json before changing anything')
    return report


def _adopt(previous, out, names) -> bool:
    """Copy a verified patch (and its siblings) that an earlier session already published."""
    for root in map(Path, previous):
        patch = root / names[0] / 'patch_manifest.json'
        if patch.is_file() and read_json(patch).get('merge_status') == 'verified':
            for name in names:
                source, target = root / name, out / name
                if source.exists() and not target.exists():
                    (shutil.copytree if source.is_dir() else shutil.copyfile)(source, target)
            return True
    return False


def _latest_manifest(out, previous) -> Path:
    candidates = [out / 'train' / 'run_manifest.json'] + [Path(p) / 'run_manifest.json' for p in previous]
    return max((p for p in candidates if p.is_file()), key=lambda p: read_json(p)['state']['attempted_updates'])


def stage_a(*, model_config, settings, train_config, data_root, seed, out, previous=(), session_seconds=None,
            reserve_seconds=PIPELINE_RESERVE_SECONDS) -> dict:
    stage, data = _Stage(settings, out, session_seconds), Path(data_root)
    out = stage.out
    previous_runs = [Path(p) / 'train' for p in previous]
    dev = {'A': data / 'A' / 'dev200.jsonl', 'general': data / 'general' / 'dev200.jsonl'}
    if _adopt(previous, out, ['merged_a', 'selection.json', 'references']):
        selection = read_json(out / 'selection.json')
    else:
        selection = _train_and_select(stage, model_config=model_config, train_config=train_config,
                                      data=data / 'A' / 'train.jsonl', out=out, dev=dev, primary=['A'],
                                      constraints=[('general', stage.settings['selection']['general_pp'], None)],
                                      previous=previous_runs, seed=seed, reserve_seconds=reserve_seconds)
    if selection is None:
        return stage.finish(stage='a', status='incomplete', seed=seed)
    # Keep M0's update-0 general predictions: Stage B's general constraint is relative to M0.
    reference = out / 'references' / 'm0_general_dev200.jsonl'
    if not reference.exists():
        from .study import _find
        evaluation = next(e for e in read_json(_latest_manifest(out, previous_runs))['evaluations']
                          if e['update'] == 0)['sets']['general']
        source = _find([out / 'train', *previous_runs], evaluation['predictions'], evaluation['predictions_hash'])
        reference.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, reference)
    merge = _merge(stage, model_config=model_config, out=out / 'merged_a', dev_files=[dev['A'], dev['general']],
                   run_manifest=None if (out / 'merged_a').exists() else _latest_manifest(out, previous_runs))
    model_a = write_model_config(model_config, out / 'model_a.yaml', patch=out / 'merged_a')
    protection = out / 'protection_a.pt'
    if not protection.exists():
        _invoke(['protect', '--model-config', model_a, '--protection-config', CONFIGS / 'protection.yaml',
                 '--memory', data / 'A' / 'memory.jsonl', '--out', protection])
    return stage.finish(stage='a', status='complete', seed=seed,
                        selection={k: selection[k] for k in ('update', 'eligible')},
                        patch=str(out / 'merged_a'), protection=str(protection), merge_passed=merge['passed'])


def stage_b(*, model_config, settings, train_config, data_root, stage_a_dir, seed, methods, out, previous=(),
            session_seconds=None, reserve_seconds=PIPELINE_RESERVE_SECONDS) -> dict:
    stage, data, stage_a_dir = _Stage(settings, out, session_seconds), Path(data_root), Path(stage_a_dir)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f'unknown Stage-B methods: {sorted(unknown)}')
    model_a = write_model_config(model_config, stage.out / 'model_a.yaml', patch=stage_a_dir / 'merged_a')
    dev = {'B': data / 'B' / 'dev200.jsonl', 'A': data / 'A' / 'dev200.jsonl',
           'general': data / 'general' / 'dev200.jsonl'}
    constraints = [('A', stage.settings['selection']['retain_pp'], None),
                   ('general', stage.settings['selection']['general_pp'],
                    stage_a_dir / 'references' / 'm0_general_dev200.jsonl')]
    results = {}
    for method in methods:
        method_out = stage.out / method
        previous_runs = [Path(p) / method / 'train' for p in previous]
        if _adopt([Path(p) / method for p in previous], method_out, ['merged', 'selection.json']):
            selection = read_json(method_out / 'selection.json')
            results[method] = {'status': 'complete', 'update': selection['update'], 'eligible': selection['eligible'],
                               'patch': str(method_out / 'merged'), 'from_previous_session': True}
            continue
        selection = _train_and_select(
            stage, model_config=model_a, train_config=train_config, data=data / 'B' / 'train.jsonl', out=method_out,
            dev=dev, primary=['B'], constraints=constraints, previous=previous_runs, method=method, seed=seed,
            protection=stage_a_dir / 'protection_a.pt' if method in ('protected', 'random_q', 'protected_replay') else None,
            memory=data / 'A' / 'memory.jsonl' if 'replay' in method else None, reserve_seconds=reserve_seconds)
        if selection is None:
            results[method] = {'status': 'incomplete'}
            break  # the session clock is spent; resume in the next session
        _merge(stage, model_config=model_a, out=method_out / 'merged', dev_files=[dev['B'], dev['A']],
               run_manifest=_latest_manifest(method_out, previous_runs))
        results[method] = {'status': 'complete', 'update': selection['update'], 'eligible': selection['eligible'],
                           'patch': str(method_out / 'merged')}
    complete = all(results.get(m, {}).get('status') == 'complete' for m in methods)
    return stage.finish(stage='b', status='complete' if complete else 'incomplete', seed=seed, methods=results)


def stage_joint(*, model_config, settings, train_config, data_root, seed, out, previous=(), session_seconds=None,
                reserve_seconds=PIPELINE_RESERVE_SECONDS) -> dict:
    stage, data = _Stage(settings, out, session_seconds), Path(data_root)
    out = stage.out
    previous_runs = [Path(p) / 'train' for p in previous]
    if _adopt(previous, out, ['merged', 'selection.json']):
        selection = read_json(out / 'selection.json')
        return stage.finish(stage='joint', status='complete', seed=seed, update=selection['update'])
    joint, joint_config = out / 'joint_train.jsonl', out / 'train_joint.yaml'
    if not joint.exists():
        rows = read_jsonl(data / 'A' / 'train.jsonl') + read_jsonl(data / 'B' / 'train.jsonl')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('joint training IDs collide')
        write_jsonl(joint, rows)
        # Joint A+B from M0 gets the combined Stage-A plus Stage-B budget.
        config = yaml.safe_load(Path(train_config).read_text(encoding='utf-8'))
        config['token_budget'] *= 2
        joint_config.write_text(yaml.safe_dump(config), encoding='utf-8')
    dev = {'A': data / 'A' / 'dev200.jsonl', 'B': data / 'B' / 'dev200.jsonl',
           'general': data / 'general' / 'dev200.jsonl'}
    selection = _train_and_select(stage, model_config=model_config, train_config=joint_config, data=joint, out=out,
                                  dev=dev, primary=['A', 'B'],
                                  constraints=[('general', stage.settings['selection']['general_pp'], None)],
                                  previous=previous_runs, method='joint', seed=seed, reserve_seconds=reserve_seconds)
    if selection is None:
        return stage.finish(stage='joint', status='incomplete', seed=seed)
    _merge(stage, model_config=model_config, out=out / 'merged', dev_files=[dev['A'], dev['B']],
           run_manifest=_latest_manifest(out, previous_runs))
    return stage.finish(stage='joint', status='complete', seed=seed, update=selection['update'])
