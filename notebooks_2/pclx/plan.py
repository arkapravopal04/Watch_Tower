"""Find every input by content (as the main study's notebooks do) and plan the pending work.

Markers: `spectrum_summary.json` (one per seed, from `pclx spectrum`),
`next_item.json` (one per arm and seed, from `pclx queue`) and
`replication.json` (from the replication check). Main-study outputs are found
by pcl's own discovery.
"""

from pathlib import Path

from pcl import kaggle
from pcl.io import read_json

from . import arms as arm_specs

SPECTRUM_MARKER = 'spectrum_summary.json'
K256_MARKER = 'k256_summary.json'
ITEM_MARKER = 'next_item.json'
REPLICATION_MARKER = 'replication.json'
# SVD bases per training order: G is the gradient of that order's first update ('b': B only;
# 'replay': the replay arms' first update, which mixes in Stage-A memory).
SVD_BASES = {'b': 'svd_bases.pt', 'replay': 'svd_bases_replay.pt'}
# Estimated minutes per NB7 task on a T4 (model load and hashing included), for balancing the two GPUs.
NB7_MINUTES = {'spectrum': 9, 'k256': 15}


def discover_next(roots, *, exclude=None) -> dict:
    """Merge discovery over several roots (e.g. the notebook inputs and the current working folder)."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots) if Path(r).exists()]
    excluded = Path(exclude) if exclude else None

    def markers(name):
        return [p for root in roots for p in sorted(root.rglob(name)) if not (excluded and p.is_relative_to(excluded))]
    main = {}
    for root in roots:
        for key, value in kaggle.discover(root, exclude=exclude).items():
            main.setdefault(key, []).extend(value)
    found = {'main': main, 'spectra': {}, 'k256': {}, 'items': {}, 'replication': []}
    for key, marker in (('spectra', SPECTRUM_MARKER), ('k256', K256_MARKER)):
        for path in markers(marker):
            summary = read_json(path)
            # Spectra from before the first-update rule (one svd_bases file, 8-update G) are not usable.
            stale = key == 'spectra' and set(summary.get('svd_bases') or {}) != set(SVD_BASES)
            if summary.get('status') == 'complete' and not stale:
                found[key].setdefault(summary['seed'], path.parent)
    for path in markers(ITEM_MARKER):
        item = read_json(path)
        found['items'].setdefault((item['arm'], item['seed']), []).append({**item, 'dir': path.parent})
    found['replication'] = [{**read_json(p), 'dir': p.parent} for p in markers(REPLICATION_MARKER)]
    return found


def main_runs(found, seed, method=None) -> list[Path]:
    """Every main-study Stage-B training directory of a seed (all sessions), optionally for one method."""
    runs = []
    for stage in found['main']['stages']:
        if stage.get('stage') != 'b' or stage.get('seed') != seed:
            continue
        for manifest in sorted(Path(stage['dir']).glob('*/train/run_manifest.json')):
            if method is None or read_json(manifest)['provenance']['method'] == method:
                runs.append(manifest.parent)
    return runs


def stage_a_seeds(found) -> list[int]:
    return sorted({s['seed'] for s in found['main']['stages'] if s.get('stage') == 'a' and s['status'] == 'complete'})


def item_complete(found, arm, seed) -> bool:
    return any(entry.get('status') == 'complete' for entry in found['items'].get((arm, seed), []))


def needs(spec) -> str | None:
    """Which NB7 output an arm needs, if any ('spectra' or 'k256')."""
    if spec['init'].startswith('svd'):
        return 'spectra'
    if spec['protection'] == 'k256':
        return 'k256'
    return None


def nb7_tasks(found, *, build_k256=True) -> list[str]:
    seeds = stage_a_seeds(found)
    tasks = [f'spectrum:{s}' for s in seeds if s not in found['spectra']]
    return tasks + ([f'k256:{s}' for s in seeds if s not in found['k256']] if build_k256 else [])


def balance(tasks: list[str], gpus: int = 2, minutes=None) -> list[list[str]]:
    """Longest task first onto the least-loaded GPU, so both GPUs finish at about the same time."""
    minutes = minutes or NB7_MINUTES
    queues, loads = [[] for _ in range(gpus)], [0.0] * gpus
    for task in sorted(tasks, key=lambda t: -minutes[t.split(':')[0]]):
        gpu = loads.index(min(loads))
        queues[gpu].append(task)
        loads[gpu] += minutes[task.split(':')[0]]
    return [q for q in queues if q]


def pending_items(found, *, tiers=(1, 2, 3), run_svd_arms=True, replicate=True) -> list[str]:
    """Work items 'arm:seed' in priority order: tier by tier, seed by seed; the replication check first."""
    items = []
    if replicate and 0 in stage_a_seeds(found) and not any(r.get('status') == 'complete' for r in found['replication']):
        items.append('replicate:0')
    seeds = stage_a_seeds(found)
    for name in arm_specs.by_tier(tiers):
        spec = arm_specs.ARMS[name]
        if spec['init'].startswith('svd') and not run_svd_arms:
            continue
        for seed in seeds:
            requirement = needs(spec)
            if requirement and seed not in found[requirement]:
                continue
            if not item_complete(found, name, seed):
                items.append(f'{name}:{seed}')
    return items


def split(items: list[str], gpus: int = 2) -> list[list[str]]:
    """Round-robin over GPUs, so each GPU gets a mix of priorities."""
    return [items[i::gpus] for i in range(gpus) if items[i::gpus]]
