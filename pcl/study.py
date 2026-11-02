"""Dev-only checkpoint selection and the locked-study commands.

Selection reads the dev scores recorded at every selectable checkpoint. The
locked commands only ever touch registered artifacts and registered test
files; references (M0, each seed's M_A) are registered artifacts too.
"""

from pathlib import Path

import yaml

from .io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from .model_setup import PATCH_MANIFEST
from .registry import Registry, identity
from .stats import acceptance_gates, compare_predictions, paired_bootstrap

STAGE_ROLES = ('stage_b', 'joint')


def _accuracy(rows, ids=None):
    selected = [row for row in rows if ids is None or row['id'] in ids]
    return sum(row['correct'] for row in selected) / len(selected)


def _find(runs, relative, expected_hash=None):
    for run in runs:
        path = run / relative
        if path.is_file() and (expected_hash is None or file_hash(path) == expected_hash):
            return path
    raise FileNotFoundError(f'{relative} not found in {[str(r) for r in runs]}; attach every session output of this run')


def select_checkpoint(run_dirs, *, primary, constraints=(), out=None) -> dict:
    """Maximize mean dev accuracy over `primary` sets among checkpoints meeting
    every (set, margin_pp, reference) constraint; ties go to the earliest update.

    A reference of None means the update-0 scores (the starting model).
    If no checkpoint is eligible, the best primary score is reported and the
    run is marked ineligible for promotion rather than silently omitted.
    """
    runs = [Path(d) for d in run_dirs]
    manifests = [read_json(run / 'run_manifest.json') for run in runs if (run / 'run_manifest.json').is_file()]
    if not manifests:
        raise FileNotFoundError('no run_manifest.json in the given run directories')
    latest = max(manifests, key=lambda m: m['state']['attempted_updates'])
    if not latest['complete']:
        raise ValueError('training is incomplete; resume it in the next session before selecting')
    evaluations = {entry['update']: entry for entry in latest.get('evaluations', [])}
    if 0 not in evaluations:
        raise ValueError('missing the update-0 reference scores; train with --dev')

    def predictions(update, name):
        meta = evaluations[update]['sets'][name]
        return read_jsonl(_find(runs, meta['predictions'], meta['predictions_hash']))

    references = {}
    for name, margin, reference in constraints:
        rows = read_jsonl(reference) if reference else predictions(0, name)
        references[name] = {row['id']: row for row in rows}
    table = []
    for update in sorted(u for u in evaluations if u > 0):
        try:
            checkpoint = _find(runs, f'checkpoints/update-{update:08d}.pt')
        except FileNotFoundError:
            continue
        row = {'update': update, 'checkpoint': str(checkpoint.resolve()),
               'primary': sum(evaluations[update]['sets'][name]['accuracy'] for name in primary) / len(primary),
               'constraints': {}, 'eligible': True}
        for name, margin, _ in constraints:
            current = predictions(update, name)
            ids = {r['id'] for r in current}
            if ids != set(references[name]):
                raise ValueError(f'reference for {name} does not cover the same dev items')
            delta = 100 * (_accuracy(current) - _accuracy(references[name].values()))
            row['constraints'][name] = {'delta_pp': delta, 'margin_pp': margin, 'met': delta >= margin}
            row['eligible'] &= delta >= margin
        table.append(row)
    if not table:
        raise ValueError('no scored checkpoint files found; keep checkpoints and score dev during training')
    eligible = [row for row in table if row['eligible']]
    best = max(eligible or table, key=lambda row: (row['primary'], -row['update']))
    result = {'checkpoint': best['checkpoint'], 'checkpoint_hash': file_hash(best['checkpoint']),
              'update': best['update'], 'eligible': bool(eligible),
              'rule': {'maximize': primary, 'constraints': [list(c) for c in constraints],
                       'tie_break': 'earliest update', 'reference': 'update 0 unless a reference file is given',
                       'basis': 'dev point estimates; locked intervals decide promotion'},
              'table': table, 'run_identity': identity(latest['provenance'])}
    if out:
        write_json(out, result)
    return result


def _artifact_hashes(entry) -> dict:
    if 'patch' in entry:
        manifest = read_json(Path(entry['patch']) / PATCH_MANIFEST)
        if manifest.get('merge_status') != 'verified':
            raise ValueError(f"unverified patch: {entry['patch']}")
        return manifest['tensor_hashes']
    return read_json(entry['hashes'])['tensor_hashes']


def register_study(spec_path, registry_dir) -> dict:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding='utf-8'))
    artifacts, roles = {}, {}
    for artifact_id, entry in spec['artifacts'].items():
        artifacts[artifact_id] = _artifact_hashes(entry)
        roles[artifact_id] = {key: value for key, value in entry.items() if key not in ('patch', 'hashes')}
    bases = [a for a, r in roles.items() if r.get('role') == 'base']
    if len(bases) != 1:
        raise ValueError('register exactly one base (M0) artifact')
    for artifact_id, role in roles.items():
        if role.get('role') == 'stage_b' and not any(r.get('role') == 'stage_a' and r.get('seed') == role.get('seed')
                                                    for r in roles.values()):
            raise ValueError(f'{artifact_id} needs its seed-{role.get("seed")} M_A reference registered')
    primary = spec['primary_candidate']
    if roles.get(primary, {}).get('role') != 'stage_b':
        raise ValueError('the primary candidate must be a registered Stage-B artifact')
    test_files = {}
    for name, path in spec['test_files'].items():
        if any(not row['split'].startswith('test') for row in read_jsonl(path)):
            raise ValueError(f'{path} is not a locked test file')
        test_files[name] = file_hash(path)
    if not {'A', 'B', 'general'} <= test_files.keys():
        raise ValueError('register at least the A, B and general locked test files')
    gates = yaml.safe_load(Path(spec['gates']).read_text(encoding='utf-8'))
    study = {'artifacts': artifacts, 'roles': roles, 'primary_candidate': primary, 'test_files': test_files,
             'gates': {key: float(gates[key]) for key in ('learn_pp', 'retain_pp', 'general_pp')},
             'selection_rule': spec['selection_rule'], 'notes': spec.get('notes', '')}
    study_hash = Registry(registry_dir).register(study)
    return {'registry': str(registry_dir), 'study_hash': study_hash, 'artifacts': sorted(artifacts),
            'test_files': sorted(test_files), 'primary_candidate': primary}


def find_patch(hashes: dict, search_roots) -> Path:
    target = identity(hashes)
    for root in search_roots:
        for manifest in Path(root).rglob(PATCH_MANIFEST):
            data = read_json(manifest)
            if data.get('merge_status') == 'verified' and identity(data.get('tensor_hashes', {})) == target:
                return manifest.parent
    raise FileNotFoundError('registered artifact patch not found under the search roots; attach its stage output')


def _find_test_files(study, test_root) -> dict:
    wanted = {digest: name for name, digest in study['test_files'].items()}
    found = {}
    for path in Path(test_root).rglob('*.jsonl'):
        digest = file_hash(path)
        if digest in wanted:
            found[wanted[digest]] = path
    missing = set(study['test_files']) - set(found)
    if missing:
        raise FileNotFoundError(f'registered test files not found under {test_root}: {sorted(missing)}')
    return found


def evaluate_locked(registry_dir, artifact_ids, model_config, *, test_root, search_roots=(), batch_size=1,
                    session_seconds=None, reserve_seconds=600) -> dict:
    """Score registered artifacts on every registered test file, once each.

    A file is only started when the measured scoring speed says it can finish
    before the session deadline: an interrupted locked run stays reserved.
    Completed files are skipped, so rerunning in a new session continues.
    """
    import time
    from .evaluate import evaluate, summarize
    from .hashing import model_hashes
    from .model_setup import load_model
    deadline = time.monotonic() + session_seconds if session_seconds else None
    registry = Registry(registry_dir)
    study = read_json(registry.root / 'study.json')
    unknown = [a for a in artifact_ids if a not in study['artifacts']]
    if unknown:
        raise ValueError(f'not registered: {unknown}')
    tests = _find_test_files(study, test_root)
    sizes = {name: len(read_jsonl(path)) for name, path in tests.items()}
    config = yaml.safe_load(Path(model_config).read_text(encoding='utf-8'))
    seconds_per_item, results = None, {}
    for artifact_id in artifact_ids:
        status = {}
        for path in (registry.root / 'locked').glob('*.json'):
            record = read_json(path)
            status[(record['artifact_id'], record['data_hash'])] = record['status']
        pending = {name: path for name, path in tests.items()
                   if status.get((artifact_id, study['test_files'][name])) != 'completed'}
        interrupted = [name for name in pending if status.get((artifact_id, study['test_files'][name])) == 'reserved']
        if interrupted:
            results[artifact_id] = {'error': f'{interrupted} interrupted earlier; reserved for investigation, never rerun'}
            continue
        if not pending:
            results[artifact_id] = {'already_completed': sorted(tests)}
            continue
        patch = None if study['roles'][artifact_id].get('role') == 'base' else find_patch(study['artifacts'][artifact_id], search_roots)
        model, tokenizer, _ = load_model(config['path'], manifest_path=config['snapshot_manifest'],
                                         precision=config['precision'], device=config['device'],
                                         backend=config['backend'], patch=patch)
        hashes = model_hashes(model)
        summaries = {}
        for name, path in sorted(pending.items()):
            if deadline is not None:
                remaining = deadline - time.monotonic() - reserve_seconds
                estimate = sizes[name] * (seconds_per_item or 0)
                if seconds_per_item is None and remaining < 3600 or estimate > remaining:
                    summaries[name] = 'deferred to the next session (not reserved)'
                    continue
            started = time.monotonic()
            reservation = registry.reserve_test(artifact_id, hashes, path)
            predictions = evaluate(model, tokenizer, read_jsonl(path), batch_size=batch_size,
                                   max_new_tokens=config['max_new_tokens'], max_context=config['max_context'],
                                   locked_authorized=True, label=f'locked {artifact_id} {name}')
            out = registry.root / 'predictions' / artifact_id / f'{name}.jsonl'
            write_jsonl(out, predictions)
            registry.finish_test(reservation, out)
            elapsed = (time.monotonic() - started) / sizes[name]
            seconds_per_item = max(seconds_per_item or 0, elapsed)
            summaries[name] = summarize(predictions)
        results[artifact_id] = summaries
        del model
        import gc
        gc.collect()
    deferred = any(isinstance(value, str) for summary in results.values() for value in summary.values())
    errors = any('error' in summary for summary in results.values())
    return {'complete': not deferred and not errors, 'results': results}


def _locked_predictions(registry, study) -> dict:
    names = {digest: name for name, digest in study['test_files'].items()}
    return {(record['artifact_id'], names[record['data_hash']]): read_jsonl(registry.predictions_path(record['predictions']))
            for record in registry.completed()}


def _references(study, artifact_id):
    roles = study['roles']
    base = next(a for a, r in roles.items() if r.get('role') == 'base')
    seed = roles[artifact_id].get('seed')
    stage_a = next((a for a, r in roles.items() if r.get('role') == 'stage_a' and r.get('seed') == seed), None)
    return base, stage_a


def numerical_checks(patch_dirs, smoke_report) -> dict:
    checks = {key: True for key in ('initialization', 'frozen_parameters', 'projection', 'export', 'reload', 'merge')}
    for patch in patch_dirs:
        validation = read_json(Path(patch) / 'merge_validation.json')
        run = read_json(Path(patch) / 'run_manifest.json')
        first = next((h for h in run['history'] if h['gradients_finite']), None)
        checks['initialization'] &= bool(first and first['gradient_norms']['A'] == 0 and first['gradient_norms']['B'] > 0)
        checks['frozen_parameters'] &= bool(run['complete'])
        limit = validation['limits']['max_intended_residual']
        checks['projection'] &= bool(validation['preflight_passed'] and all(
            d['intended_residual'] <= limit for d in validation['diagnostics'].values()))
        checks['export'] &= bool(validation['live_exported']['passed'])
        checks['merge'] &= bool(validation['live_merged']['passed'])
        checks['reload'] &= bool(validation['tensor_hashes_match'] and validation['merged_reloaded']['passed'])
    smoke = read_json(smoke_report)
    checks['resume'] = bool(smoke.get('passed') and smoke.get('resume_identical'))
    return checks


def decide(registry_dir, *, smoke_report, search_roots=(), repetitions=10000, seed=0) -> dict:
    registry = Registry(registry_dir)
    study = read_json(registry.root / 'study.json')
    candidate = study['primary_candidate']
    base, stage_a = _references(study, candidate)
    predictions = _locked_predictions(registry, study)

    def contrast(reference, name):
        key, ref = (candidate, name), (reference, name)
        if key not in predictions or ref not in predictions:
            raise ValueError(f'missing locked predictions for {key} or {ref}')
        return compare_predictions(predictions[key], predictions[ref], repetitions=repetitions, seed=seed)

    contrasts = {'A': contrast(stage_a, 'A'), 'B': contrast(stage_a, 'B'), 'general': contrast(base, 'general')}
    patches = [find_patch(study['artifacts'][a], search_roots) for a in (stage_a, candidate)]
    checks = numerical_checks(patches, smoke_report)
    gates = acceptance_gates(contrasts, checks, **study['gates'])
    result = {**gates, 'contrasts': contrasts, 'numerical_checks': checks,
              'references': {'A': stage_a, 'B': stage_a, 'general': base}}
    registry.record_decision(candidate, study['artifacts'][candidate], result)
    return {'candidate': candidate, **result}


def report(registry_dir, out_dir, *, repetitions=10000, seed=0) -> dict:
    registry = Registry(registry_dir)
    study = read_json(registry.root / 'study.json')
    predictions = _locked_predictions(registry, study)
    roles = study['roles']
    tests = sorted(study['test_files'])
    accuracy = {artifact: {name: {'accuracy': _accuracy(rows), 'n': len(rows),
                                  'format_failures': sum(not r['format_ok'] for r in rows)}
                           for (a, name), rows in predictions.items() if a == artifact}
                for artifact in study['artifacts']}

    def compare(left, right, name):
        if (left, name) in predictions and (right, name) in predictions:
            return compare_predictions(predictions[(left, name)], predictions[(right, name)],
                                       repetitions=repetitions, seed=seed)
        return None

    contrasts = {}
    for artifact, role in roles.items():
        if role.get('role') not in STAGE_ROLES:
            continue
        base, stage_a = _references(study, artifact)
        reference = stage_a or base
        contrasts[artifact] = {name: compare(artifact, base if name.startswith('general') else reference, name)
                               for name in tests}
    pairs = {}
    seeds = sorted({r.get('seed') for r in roles.values() if r.get('role') == 'stage_b'})
    by_method = {}
    for artifact, role in roles.items():
        if role.get('role') == 'stage_b':
            by_method.setdefault(role['method'], {})[role['seed']] = artifact
    for method in sorted(by_method):
        if method == 'sequential' or 'sequential' not in by_method:
            continue
        for s in seeds:
            if s in by_method[method] and s in by_method['sequential']:
                for name in ('A', 'B'):
                    pairs[f'{method}-vs-sequential/seed{s}/{name}'] = compare(by_method[method][s], by_method['sequential'][s], name)
    pooled = {}
    for method, members in by_method.items():
        usable = [s for s in sorted(members) if _references(study, members[s])[1]]
        if len(usable) < 2:
            continue
        for name in ('A', 'B'):
            try:
                rows = [{r['id']: r for r in predictions[(members[s], name)]} for s in usable]
                refs = [{r['id']: r for r in predictions[(_references(study, members[s])[1], name)]} for s in usable]
            except KeyError:
                continue
            ids = sorted(rows[0])
            pooled[f'{method}/{name}'] = paired_bootstrap(
                [[row[i]['correct'] for i in ids] for row in rows], [[ref[i]['correct'] for i in ids] for ref in refs],
                [rows[0][i]['group'] for i in ids], repetitions=repetitions, seed=seed, resample_seeds=True)
    decision_files = list(registry.root.glob('decision-*.json'))
    decision = read_json(decision_files[0]) if decision_files else None
    result = {'study_hash': identity(study), 'accuracy': accuracy, 'contrasts_vs_reference': contrasts,
              'paired_method_contrasts': pairs, 'pooled_seed_contrasts': pooled, 'decision': decision,
              'interpretation': 'Pooled intervals are exploratory; the gate applies only to the preregistered '
                                'primary artifact. Hard splits are reported, not gated.'}
    out = Path(out_dir)
    write_json(out / 'report.json', result)
    (out / 'report.md').write_text(_markdown(result, tests), encoding='utf-8')
    return {'report': str(out / 'report.md'), 'decision': decision and decision.get('accepted')}


def _fmt(contrast):
    if not contrast:
        return '—'
    return f"{contrast['estimate_pp']:+.1f} [{contrast['lower_pp']:+.1f}, {contrast['upper_pp']:+.1f}]"


def _markdown(result, tests) -> str:
    lines = ['# Locked study report', '', f"Study hash `{result['study_hash']}`", '']
    decision = result['decision']
    if decision:
        lines += [f"**Primary candidate `{decision['artifact_id']}`: {'ACCEPTED' if decision['accepted'] else 'REJECTED (keep M_A)'}**",
                  '', '| Gate | Passed |', '|---|---|'] + [f'| {k} | {v} |' for k, v in decision['gates'].items()] + ['']
    lines += ['## Accuracy (%)', '', '| Artifact | ' + ' | '.join(tests) + ' |', '|---' * (len(tests) + 1) + '|']
    for artifact, scores in result['accuracy'].items():
        lines.append(f'| {artifact} | ' + ' | '.join(f"{100 * scores[t]['accuracy']:.1f}" if t in scores else '—'
                                                     for t in tests) + ' |')
    lines += ['', '## Change vs reference, pp [95% group-bootstrap CI]',
              'Reference: same-seed M_A for skills, M0 for general.', '',
              '| Artifact | ' + ' | '.join(tests) + ' |', '|---' * (len(tests) + 1) + '|']
    for artifact, contrasts in result['contrasts_vs_reference'].items():
        lines.append(f'| {artifact} | ' + ' | '.join(_fmt(contrasts.get(t)) for t in tests) + ' |')
    if result['paired_method_contrasts']:
        lines += ['', '## Method vs sequential, same seed (pp)', '', '| Contrast | Estimate [CI] |', '|---|---|']
        lines += [f'| {k} | {_fmt(v)} |' for k, v in result['paired_method_contrasts'].items()]
    if result['pooled_seed_contrasts']:
        lines += ['', '## Pooled over seeds vs M_A (crossed seed × group bootstrap, exploratory)', '',
                  '| Method/skill | Estimate [CI] |', '|---|---|']
        lines += [f'| {k} | {_fmt(v)} |' for k, v in result['pooled_seed_contrasts'].items()]
    lines += ['', result['interpretation'], '']
    return '\n'.join(lines)

