"""Kaggle glue: session clock, environment record, input discovery, data banks, study spec.

Kaggle mounts each attached Dataset or notebook output somewhere under
/kaggle/input; the layout is not something to hard-code. Every pcl output
carries a marker file, so notebooks discover their inputs by content.
"""

import importlib.metadata
import platform
import shutil
import time
from pathlib import Path

import yaml

from .io import file_hash, read_json, write_json

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
DATA_MARKER = 'pcl_data_root.json'
TEST_MARKER = 'pcl_test_root.json'
STAGE_B_METHODS = ('sequential', 'protected', 'random_q', 'replay')
SEEDS = (0, 1, 2)


class TimeGuard:
    def __init__(self, session_seconds: float, reserve_seconds: float):
        if not 0 < reserve_seconds < session_seconds:
            raise ValueError("reserve must be positive and shorter than the session")
        self.deadline = time.monotonic() + session_seconds - reserve_seconds

    def should_stop(self) -> bool:
        return time.monotonic() >= self.deadline


def environment() -> dict:
    import torch
    packages = {}
    for name in ("torch", "numpy", "transformers", "accelerate", "peft", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "free_disk_bytes": shutil.disk_usage('.').free}


def _dirs(root, marker, exclude):
    return sorted({p.parent for p in Path(root).rglob(marker) if not (exclude and p.is_relative_to(exclude))})


def discover(root='/kaggle/input', *, exclude=None) -> dict:
    """Find every pcl output under root; `exclude` (the code dataset) is never searched."""
    find = lambda marker: _dirs(root, marker, Path(exclude) if exclude else None)  # noqa: E731
    found = {'models': [p for p in find('snapshot_manifest.json') if (p / 'config.json').is_file()],
             'data': find(DATA_MARKER), 'tests': find(TEST_MARKER), 'pilot': find('pilot_report.json'),
             'smoke': [p / 'smoke_report.json' for p in find('smoke_report.json')],
             'registries': [p for p in find('study.json') if p.name == 'registry'], 'stages': []}
    for directory in find('stage_summary.json'):
        found['stages'].append({**read_json(directory / 'stage_summary.json'), 'dir': directory})
    return found


def one(found, key, what):
    items = found[key]
    if len(items) != 1:
        raise FileNotFoundError(f'expected exactly one {what} among the notebook inputs, found {[str(i) for i in items]}')
    return items[0]


def stage_dirs(found, stage, seed, *, complete=None) -> list[Path]:
    return [s['dir'] for s in found['stages'] if s.get('stage') == stage and s.get('seed') == seed
            and (complete is None or (s['status'] == 'complete') == complete)]


def stage_a_for(found, seed) -> Path:
    complete = stage_dirs(found, 'a', seed, complete=True)
    if not complete:
        raise FileNotFoundError(f'attach the completed Stage-A output for seed {seed}')
    return complete[0]


def stage_b_done(found, seed) -> set:
    return {method for s in found['stages'] if s.get('stage') == 'b' and s.get('seed') == seed
            for method, info in s['methods'].items() if info.get('status') == 'complete'}


def plan_stage_a(found) -> list[int]:
    """The (at most two) seeds whose Stage A is not finished yet."""
    return [seed for seed in SEEDS if not stage_dirs(found, 'a', seed, complete=True)][:2]


def plan_stage_b(found):
    """The lowest seed whose Stage A is done but whose Stage-B methods are not."""
    for seed in SEEDS:
        if stage_dirs(found, 'a', seed, complete=True) and not set(STAGE_B_METHODS) <= stage_b_done(found, seed):
            return seed
    return None


# Amendments to a pilot's study.yaml made after the pilot and before any stage ran, each with its reason.
# study_settings applies them deterministically, so every stage and session sees the same file.
AMENDMENTS = [{
    'applies_if_missing': 'max_accuracy_shift_pp',
    'merge': {'max_changed_answers': 20, 'max_accuracy_shift_pp': 1.5},
    'reason': ('2026-10-14, after the pilot and before Stage A: the old rule derived the allowed changed answers '
               'from the pilot probe (7/64 reworded long answers) and allowed 100/400. The gate now counts answers '
               'whose correctness flips (<= 20/400 = 5%; batch composition alone flips about 3%) and requires '
               '|accuracy shift| <= 1.5 pp per skill. Reworded answers with the same correctness are reported only.'),
}]


def study_settings(pilot_dir, out) -> Path:
    """The pilot's study.yaml plus the recorded amendments, written to `out` (identical in every session)."""
    settings = yaml.safe_load((Path(pilot_dir) / 'study.yaml').read_text(encoding='utf-8'))
    applied = []
    for amendment in AMENDMENTS:
        if amendment['applies_if_missing'] not in settings['merge']:
            settings['merge'].update(amendment['merge'])
            applied.append(amendment['reason'])
    if applied:
        settings['amendments'] = applied
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('# The pilot study.yaml with its recorded amendments (pcl.kaggle.AMENDMENTS).\n'
                   + yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    return out


def write_model_config(template, out, **updates) -> Path:
    config = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    config.update({key: str(value) for key, value in updates.items()})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return out


def m0_config(model_dir, out) -> Path:
    """configs/model.yaml (the study protocol: FP16 on CUDA) pointed at the attached model."""
    return write_model_config(CONFIGS / 'model.yaml', out, path=model_dir,
                              snapshot_manifest=Path(model_dir) / 'snapshot_manifest.json')


def _generate(skill, out, tokenizer, settings, sizes, *extra):
    from .cli import invoke
    max_length = yaml.safe_load((CONFIGS / 'model.yaml').read_text())['max_length']
    invoke(['generate', '--skill', skill, '--out', out, '--tokenizer', tokenizer, '--seed', settings['seed'],
            '--max-length', max_length, '--train', sizes['train'], '--dev', sizes['dev'], '--memory', sizes['memory'],
            '--dev-subset', sizes['dev_subset'], *extra])


def make_banks(out, *, tokenizer, data_config=None, audit_sample=100) -> dict:
    """Development banks for every skill, the independent audit and a root marker."""
    from .cli import invoke
    settings = yaml.safe_load(Path(data_config or CONFIGS / 'data.yaml').read_text(encoding='utf-8'))
    out = Path(out)
    for skill, sizes in settings['skills'].items():
        _generate(skill, out / skill, tokenizer, settings, sizes)
    audit = invoke(['audit', '--data', *[out / s for s in settings['skills']], '--out', out / 'audit',
                    '--sample', audit_sample])
    write_json(out / DATA_MARKER, {'data_config': settings, 'tokenizer_files': read_json(Path(tokenizer) / 'snapshot_manifest.json')['files'],
                                   'manifests': {s: file_hash(out / s / 'split_manifest.json') for s in settings['skills']}})
    return {'out': str(out), 'audit_passed': audit['passed'], 'records': audit['records']}


def seal_banks(data_root, out, *, tokenizer) -> dict:
    """Regenerate each bank with its locked tests; train/dev must reproduce byte for byte."""
    from .cli import invoke
    data_root, out = Path(data_root), Path(out)
    settings = read_json(data_root / DATA_MARKER)['data_config']
    for skill, sizes in settings['skills'].items():
        _generate(skill, out / skill, tokenizer, settings, sizes, '--test', sizes['test'], '--test-hard',
                  sizes['test_hard'], '--seal-test', '--expect-manifest', data_root / skill / 'split_manifest.json')
    audit = invoke(['audit', '--data', *[out / s for s in settings['skills']], '--out', out / 'audit', '--sample', 0])
    files = {}
    for skill in settings['skills']:
        for split, suffix in (('test', ''), ('test_hard', '_hard')):
            if (out / skill / f'{split}.jsonl').is_file():
                files[f'{skill}{suffix}'] = str(out / skill / f'{split}.jsonl')
    write_json(out / TEST_MARKER, {'files': {k: file_hash(v) for k, v in files.items()}})
    return {'out': str(out), 'test_files': files, 'audit_passed': audit['passed']}


def study_spec(found, out, *, primary='protected_s0',
               selection_rule='Dev-only (pcl.study.select_checkpoint): Stage A maximizes A dev200 subject to general '
                              '>= -2 pp vs M0; Stage B maximizes B dev200 subject to A >= -2 pp vs M_A and general '
                              '>= -2 pp vs M0; joint maximizes mean(A, B) subject to general >= -2 pp; ties to the '
                              'earliest update. Primary candidate fixed before any test access.') -> dict:
    """Register every completed artifact found in the inputs; nothing is chosen by score."""
    pilot = one(found, 'pilot', 'pilot output')
    tests = Path(one(found, 'tests', 'sealed test bank'))
    artifacts = {'M0': {'hashes': str(pilot / 'm0_hashes.json'), 'role': 'base'}}
    for stage in found['stages']:
        seed = stage['seed']
        if stage['stage'] == 'a' and stage['status'] == 'complete':
            artifacts[f'MA_s{seed}'] = {'patch': str(stage['dir'] / 'merged_a'), 'role': 'stage_a', 'seed': seed,
                                        'eligible': stage['selection']['eligible']}
        elif stage['stage'] == 'joint' and stage['status'] == 'complete':
            artifacts[f'joint_s{seed}'] = {'patch': str(stage['dir'] / 'merged'), 'role': 'joint', 'seed': seed}
        elif stage['stage'] == 'b':
            for method, info in stage['methods'].items():
                # A later session may have adopted the same verified patch; register it once.
                if info.get('status') == 'complete' and f'{method}_s{seed}' not in artifacts:
                    artifacts[f'{method}_s{seed}'] = {'patch': str(stage['dir'] / method / 'merged'), 'role': 'stage_b',
                                                      'method': method, 'seed': seed, 'eligible': info['eligible']}
    test_files = {}
    for name, digest in read_json(tests / TEST_MARKER)['files'].items():
        skill, _, hard = name.partition('_')
        path = tests / skill / ('test_hard.jsonl' if hard else 'test.jsonl')
        if file_hash(path) != digest:
            raise ValueError(f'sealed test file changed: {path}')
        test_files[name] = str(path)
    if primary not in artifacts:
        raise ValueError(f'primary candidate {primary} has no completed artifact among the inputs')
    spec = {'artifacts': artifacts, 'primary_candidate': primary, 'test_files': test_files,
            'gates': str(CONFIGS / 'gates.yaml'), 'selection_rule': selection_rule}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    return spec


def latest_registry(found) -> Path:
    """The registry copy with the most completed locked evaluations (sessions build on each other)."""
    if not found['registries']:
        raise FileNotFoundError('attach the output that contains registry/study.json')

    def completed(path):
        return sum(read_json(p)['status'] == 'completed' for p in (Path(path) / 'locked').glob('*.json'))
    return max(found['registries'], key=completed)
