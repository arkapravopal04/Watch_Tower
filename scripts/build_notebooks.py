"""Generate the seven Kaggle notebooks from one source, so the setup cell never drifts.

    python scripts/build_notebooks.py

Notebook logic stays thin: discovery, bank building and the study spec live in
pcl.kaggle and are exercised by tests/test_end_to_end.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# --- Setup (identical in every notebook; nothing to edit) --------------------
import json, os, shutil, subprocess, sys, time
from pathlib import Path
NOTEBOOK_STARTED = time.time()
KAGGLE = Path(os.environ.get('PCL_KAGGLE_ROOT', '/kaggle'))   # override only for a local rehearsal
INPUT, WORKING = KAGGLE / 'input', KAGGLE / 'working'

code_roots = sorted({p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'pcl' / 'cli.py').is_file()})
assert len(code_roots) == 1, f'attach exactly one copy of the pcl code dataset; found {code_roots}'
WORK = Path(os.environ.get('PCL_WORK', '/tmp/protected-cl'))
if not WORK.exists():
    shutil.copytree(code_roots[0], WORK, ignore=shutil.ignore_patterns('outputs', '__pycache__', '*.egg-info', '.pytest_cache'))
sys.path.insert(0, str(WORK))   # `python -m pcl` runs from WORK, so no package install is needed

import importlib.metadata as metadata
def _version(name):
    try:
        return tuple(int(x) for x in metadata.version(name).split('.')[:2])
    except Exception:
        return (0,)
needed = {'transformers': (4, 51), 'safetensors': (0, 4), 'accelerate': (1, 0), 'PyYAML': (6, 0)}
missing = [f'{n}>={".".join(map(str, v))}' for n, v in needed.items() if _version(n) < v]
if missing:
    print('Installing', missing, '(needs Internet ON for this notebook)')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)

import yaml
from pcl import kaggle
from pcl.io import read_json

def pcl(*args):
    """Run one pcl command in a fresh process (its GPU memory is released when it exits)."""
    print('$ pcl', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, '-m', 'pcl', *map(str, args)], check=True, cwd=WORK)

def on_gpus(commands, log_dir):
    """One pcl command per T4, logs streamed live; fails if either command fails."""
    commands = [json.dumps([sys.executable, '-m', 'pcl', *map(str, c)]) for c in commands]
    subprocess.run([sys.executable, str(WORK / 'scripts' / 'run_pair.py'), '--out', str(log_dir), *commands],
                   check=True, cwd=WORK)

def show(path):
    try:
        from IPython.display import Markdown, display
        display(Markdown(Path(path).read_text()))
    except ImportError:
        print(Path(path).read_text())

def remaining_seconds():
    """Session time left, from SESSION_HOURS in the first cell."""
    return int(SESSION_HOURS * 3600 - (time.time() - NOTEBOOK_STARTED))

found = kaggle.discover(INPUT, exclude=code_roots[0])
print({key: len(value) for key, value in found.items()})'''

MODEL = '''MODEL = kaggle.one(found, 'models', 'pinned model Dataset (made from the NB1 output)')
M0_CONFIG = kaggle.m0_config(MODEL, WORKING / 'configs' / 'm0.yaml')'''

DATA_AND_PILOT = '''DATA = kaggle.one(found, 'data', 'NB2 output (data)')
PILOT = kaggle.one(found, 'pilot', 'NB2 output (pilot)')
SETTINGS = kaggle.study_settings(PILOT, WORKING / 'study.yaml')
settings = yaml.safe_load(SETTINGS.read_text())
assert settings['precision_validated'], 'The pilot did not validate FP16. Read pilot/pilot_report.md first.'
print(settings)'''

GPU_HOURS = 'SESSION_HOURS = 12   # your GPU session limit (see the notebook settings panel)\n'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


def header(title, purpose, accelerator, internet, inputs, after):
    return md(f'''# {title}

{purpose}

**Accelerator:** {accelerator} · **Internet:** {internet} · **Attach:** {inputs}

Run with **Save Version → Save & Run All (Commit)**. Afterwards: {after}
''')


NOTEBOOKS = {
    'NB0_check': [
        header('NB0 — check the Kaggle environment', 'Runs every test (including the whole study in miniature and '
               'the DoRA-vs-PEFT parity tests) and the smoke test on this Kaggle image.',
               'GPU T4 x2', 'ON', 'the code Dataset',
               'the last line must say `NB0 PASSED`. NB6 needs this output (its `smoke/` folder).'),
        code(GPU_HOURS + SETUP),
        code('''(WORKING / 'env').mkdir(exist_ok=True)
pcl('check-env', '--out', WORKING / 'env' / 'environment.json')   # record the image before adding PEFT
lock = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
(WORKING / 'env' / 'requirements-lock.txt').write_text(lock)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'peft==0.18.0'], check=True)   # parity tests only
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider',
                         '--basetemp', '/tmp/pytest'], cwd=WORK, capture_output=True, text=True)
(WORKING / 'env' / 'pytest.log').write_text(result.stdout + result.stderr)
print(result.stdout[-3000:])
assert result.returncode == 0, 'tests failed on this image; see env/pytest.log'
pcl('smoke', '--out', WORKING / 'smoke')
print('NB0 PASSED')'''),
    ],
    'NB1_download': [
        header('NB1 — download the pinned model', 'Downloads `Qwen/Qwen3-4B-Instruct-2507` at the exact commit in '
               '`configs/model.yaml` and writes a SHA-256 manifest of every file (~8 GB).',
               'None', 'ON', 'the code Dataset',
               '**turn this output into a private Dataset** (Output tab → New Dataset). Attach that Dataset, not this '
               'notebook, everywhere the guide says "model".'),
        code(SETUP),
        code('''pcl('download', '--out', WORKING / 'model')
print(sorted(p.name for p in (WORKING / 'model').iterdir()))'''),
    ],
    'NB2_data_and_pilot': [
        header('NB2 — data, audit and GPU pilot', 'Builds the A / B / general development data with the pinned '
               'tokenizer and re-derives every answer independently (audit). Then runs the GPU pilot: FP16 vs FP32 '
               'check, speed and memory, merge tolerances and the untrained-model (M0) baseline. Writes '
               '`pilot/study.yaml`, the settings every later stage uses.',
               'GPU T4 x2', 'OFF', 'code, model',
               'open `pilot/pilot_report.md`. Go on only if it says `precision_validated: True` and lists no '
               'concerns. Optionally skim `data/audit/review_sheet.md`.'),
        code(GPU_HOURS + SETUP),
        code(MODEL + '''
banks = kaggle.make_banks(WORKING / 'data', tokenizer=MODEL)
assert banks['audit_passed'], 'the independent audit found wrong reference answers; see data/audit/audit.json'
print(banks)
pcl('pilot', '--model-config', M0_CONFIG, '--data-root', WORKING / 'data', '--out', WORKING / 'pilot')
show(WORKING / 'pilot' / 'pilot_report.md')'''),
    ],
    'NB3_stage_a': [
        header('NB3 — Stage A: learn skill A', 'Trains ordinary DoRA on skill A for the next unfinished seeds (two at '
               'once, one per GPU), picks a checkpoint on dev, publishes a verified merged patch and records the '
               'protection basis Q. When only one seed is left, the spare GPU trains the joint A+B reference.',
               'GPU T4 x2', 'OFF', 'code, model, NB2 (+ every earlier NB3 output)',
               'run it again (attaching this output too) until it prints "Stage A finished for all seeds".'),
        code(GPU_HOURS + SETUP),
        code(MODEL + '\n' + DATA_AND_PILOT),
        code('''commands = []
for seed in kaggle.plan_stage_a(found):
    previous = kaggle.stage_dirs(found, 'a', seed, complete=False)
    commands.append(['stage-a', '--model-config', M0_CONFIG, '--settings', SETTINGS, '--data-root', DATA,
                     '--seed', seed, '--out', WORKING / 'stage_a' / f'seed{seed}', '--session-seconds', remaining_seconds(),
                     *(['--previous', *previous] if previous else [])])
if len(commands) < 2 and not kaggle.stage_dirs(found, 'joint', 0, complete=True):
    previous = kaggle.stage_dirs(found, 'joint', 0, complete=False)
    commands.append(['stage-joint', '--model-config', M0_CONFIG, '--settings', SETTINGS, '--data-root', DATA,
                     '--seed', 0, '--out', WORKING / 'joint' / 'seed0', '--session-seconds', remaining_seconds(),
                     *(['--previous', *previous] if previous else [])])
if commands:
    on_gpus(commands, WORKING / 'logs')
done = kaggle.discover(WORKING)['stages'] + found['stages']
finished = {s['seed'] for s in done if s['stage'] == 'a' and s['status'] == 'complete'}
print('Stage A finished for seeds', sorted(finished))
if finished >= set(kaggle.SEEDS):
    print('Stage A finished for all seeds. Next: NB4.')'''),
    ],
    'NB4_stage_b': [
        header('NB4 — Stage B: learn skill B four ways', 'For the next unfinished seed, starting from its Stage-A '
               'model: ordinary (sequential), protected, random-Q and replay DoRA on skill B — two methods per '
               'GPU, same seed and token budget — each dev-selected and merged into a verified patch.',
               'GPU T4 x2', 'OFF', 'code, model, NB2, all NB3 outputs (+ every earlier NB4 output)',
               'run it again (attaching this output too) until it prints "Stage B finished for all seeds".'),
        code(GPU_HOURS + SETUP),
        code(MODEL + '\n' + DATA_AND_PILOT),
        code('''# Finish the lowest unfinished seed, then keep going with the next seed while at least
# MIN_HOURS_FOR_NEXT_SEED of the session remain (a seed cut off by the clock resumes next run).
MIN_HOURS_FOR_NEXT_SEED = 2
everything, attempted = found, set()
SEED = kaggle.plan_stage_b(everything)
while SEED is not None and SEED not in attempted and remaining_seconds() > MIN_HOURS_FOR_NEXT_SEED * 3600:
    attempted.add(SEED)
    STAGE_A = kaggle.stage_a_for(found, SEED)
    previous = kaggle.stage_dirs(found, 'b', SEED)
    commands = [['stage-b', '--model-config', M0_CONFIG, '--settings', SETTINGS, '--data-root', DATA, '--seed', SEED,
                 '--stage-a', STAGE_A, '--methods', *methods, '--out', WORKING / 'stage_b' / f'seed{SEED}' / f'gpu{gpu}',
                 '--session-seconds', remaining_seconds(), *(['--previous', *previous] if previous else [])]
                for gpu, methods in enumerate([['sequential', 'replay'], ['protected', 'random_q']])]
    on_gpus(commands, WORKING / 'logs' / f'seed{SEED}')
    everything = {'stages': found['stages'] + kaggle.discover(WORKING)['stages']}
    print(f'seed {SEED}: finished', sorted(kaggle.stage_b_done(everything, SEED)))
    SEED = kaggle.plan_stage_b(everything)
if SEED is None:
    print('Stage B finished for all seeds. Next: NB5.')
else:
    print(f'Next NB4 run continues with seed {SEED} (attach this output as a Dataset).')'''),
    ],
    'NB5_freeze': [
        header('NB5 — freeze and register', 'Creates the locked test sets (the training and dev files must come out '
               'byte-for-byte identical) and registers every finished model, the test-file fingerprints, the gates '
               'and the primary candidate. After this, nothing may be changed.',
               'None', 'OFF', 'code, model, NB2, all NB3 and NB4 outputs',
               'go straight to NB6. Do not open the test files.'),
        code('''PRIMARY = 'protected_s0'   # the one candidate the accept/reject gates apply to; fixed before any testing
''' + SETUP),
        code('''MODEL = kaggle.one(found, 'models', 'pinned model Dataset')
DATA = kaggle.one(found, 'data', 'NB2 output')
missing = [s for s in kaggle.SEEDS if not set(kaggle.STAGE_B_METHODS) <= kaggle.stage_b_done(found, s)]
assert not missing, f'Stage B is unfinished for seeds {missing}; attach every NB4 output'
sealed = kaggle.seal_banks(DATA, WORKING / 'tests', tokenizer=MODEL)
assert sealed['audit_passed']
found['tests'] = kaggle.discover(WORKING)['tests']
spec = kaggle.study_spec(found, WORKING / 'study_spec.yaml', primary=PRIMARY)
print('registering', sorted(spec['artifacts']))
pcl('register', '--spec', WORKING / 'study_spec.yaml', '--registry', WORKING / 'registry')
print('NB5 done. Next: NB6.')'''),
    ],
    'NB6_test_and_decide': [
        header('NB6 — locked test, decision and report', 'Scores every registered model exactly once on every locked '
               'test file (both GPUs). When everything is scored, applies the gates to the primary candidate '
               '(accept, or keep the Stage-A model) and writes the report.',
               'GPU T4 x2', 'OFF', 'code, model, NB0, NB2, NB5, all NB3 and NB4 outputs (+ any earlier NB6 output)',
               'if it says "Not finished", run it again with this output attached too. Otherwise read '
               '`report/report.md`.'),
        code(GPU_HOURS + SETUP),
        code(MODEL + '''
TESTS = kaggle.one(found, 'tests', 'NB5 output (tests)')
REGISTRY = WORKING / 'registry'
if not REGISTRY.exists():
    shutil.copytree(kaggle.latest_registry(found), REGISTRY)
study = read_json(REGISTRY / 'study.json')
batch = yaml.safe_load((kaggle.one(found, 'pilot', 'NB2 output') / 'study.yaml').read_text())['eval_batch_size']
artifacts = sorted(study['artifacts'])
commands = [['evaluate-locked', '--registry', REGISTRY, '--artifact-id', *half, '--model-config', M0_CONFIG,
             '--test-root', TESTS, '--search', INPUT, '--batch-size', batch, '--session-seconds', remaining_seconds()]
            for half in (artifacts[0::2], artifacts[1::2]) if half]
on_gpus(commands, WORKING / 'logs')
scored = sum(read_json(p)['status'] == 'completed' for p in (REGISTRY / 'locked').glob('*.json'))
total = len(artifacts) * len(study['test_files'])
print(scored, 'of', total, 'scored')'''),
        code('''if scored < total:
    print('Not finished: run NB6 again with this output attached as an extra input.')
else:
    if not list(REGISTRY.glob('decision-*.json')):
        pcl('decide', '--registry', REGISTRY, '--smoke-report', kaggle.one(found, 'smoke', 'NB0 output'), '--search', INPUT)
    pcl('report', '--registry', REGISTRY, '--out', WORKING / 'report')
    show(WORKING / 'report' / 'report.md')'''),
    ],
}


def main():
    out = ROOT / 'notebooks'
    out.mkdir(exist_ok=True)
    for stale in out.glob('*.ipynb'):
        if stale.stem not in NOTEBOOKS:
            stale.unlink()
    for name, cells in NOTEBOOKS.items():
        notebook = {'cells': cells, 'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 4}
        (out / f'{name}.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote', ', '.join(NOTEBOOKS))


if __name__ == '__main__':
    main()
