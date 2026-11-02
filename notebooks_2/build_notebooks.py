"""Generate the follow-up notebooks (NB7-NB9) from one source.

    python notebooks_2/build_notebooks.py

The setup cell is the main study's own (imported from scripts/build_notebooks.py,
read-only), plus a few lines that make the `pclx` add-on importable. Notebook
logic stays thin: planning lives in pclx.plan and is exercised by
notebooks_2/tests/test_next_miniature.py. Edit this script, not the .ipynb.
"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('main_builder', HERE.parent / 'scripts' / 'build_notebooks.py')
main_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main_builder)
SETUP, MODEL, DATA_AND_PILOT, GPU_HOURS = (main_builder.SETUP, main_builder.MODEL, main_builder.DATA_AND_PILOT,
                                           main_builder.GPU_HOURS)
md, code, header = main_builder.md, main_builder.code, main_builder.header

NEXT_SETUP = '''# --- Follow-up add-on (pclx lives in notebooks_2/ of the same code dataset) ------------
NEXT = WORK / 'notebooks_2'
assert (NEXT / 'pclx' / 'cli.py').is_file(), 'attach the pcl-next code dataset (it contains notebooks_2/)'
sys.path.insert(0, str(NEXT))
os.environ['PYTHONPATH'] = os.pathsep.join(p for p in (str(WORK), str(NEXT), os.environ.get('PYTHONPATH', '')) if p)
from pclx import plan

def pclx(*args):
    print('$ pclx', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, '-m', 'pclx', *map(str, args)], check=True, cwd=WORK)

def pclx_on_gpus(commands, log_dir):
    """One pclx command per T4 (at most two), logs streamed live."""
    commands = [json.dumps([sys.executable, '-m', 'pclx', *map(str, c)]) for c in commands]
    subprocess.run([sys.executable, str(WORK / 'scripts' / 'run_pair.py'), '--out', str(log_dir), *commands],
                   check=True, cwd=WORK)

found_next = plan.discover_next([INPUT, WORKING], exclude=code_roots[0])
print({'stage_a_seeds': plan.stage_a_seeds(found_next), 'spectra': sorted(found_next['spectra']),
       'finished_items': sorted(k for k in found_next['items'] if plan.item_complete(found_next, *k))})'''

NOTEBOOKS = {
    'NB7_spectrum': [
        header('NB7 — gradient spectrum at M_A (go / no-go for the SVD arms)',
               'Checks the add-on on this image, then for every seed with a finished Stage A: the skill-B gradient '
               'at M_A for each adapted layer, how concentrated it is, how much of it falls inside A\'s protected '
               'subspace, the predicted first-step speed-up of SVD over Kaiming initialization (SGD and AdamW), the '
               'SVD bases the SVD arms start from, and (tier 3) a larger protection basis with up to 256 directions. '
               'The work is balanced over both GPUs while the tests run on the CPU: about 35–40 minutes for three seeds.',
               'GPU T4 x2', 'OFF', 'pcl-next code, model, NB2, all NB3 outputs',
               'read `spectrum/spectrum_report.md`. Its last line says whether NB8 will run the SVD-initialized arms. '
               'NB8 needs this output.'),
        code(GPU_HOURS + SETUP + '\n' + NEXT_SETUP),
        code(MODEL + '\n' + DATA_AND_PILOT),
        code('''BUILD_K256 = True   # the tier-3 arm protected_k256 needs it (~15 min per seed); False skips that arm
# The add-on's tests (unit tests + the follow-up round in miniature, ~3 min of CPU) run while the GPUs work.
tests = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', '/tmp/pytest-next',
                          str(NEXT / 'tests')], cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
tasks = plan.nb7_tasks(found_next, build_k256=BUILD_K256)
queues = plan.balance(tasks)   # longest task first onto the less-loaded GPU
print('NB7 tasks per GPU:', queues)
commands = [['spectrum', '--search', INPUT, WORKING, '--exclude', code_roots[0], '--tasks', *queue,
             '--model-config', M0_CONFIG, '--data-root', DATA, '--out', WORKING / 'spectrum'] for queue in queues]
if commands:
    pclx_on_gpus(commands, WORKING / 'logs')
log = tests.communicate()[0]
print(log[-2000:])
if tests.returncode != 0:
    for marker in [*WORKING.rglob('spectrum_summary.json'), *WORKING.rglob('k256_summary.json')]:
        marker.unlink()   # outputs from an image that fails the tests must not be used by NB8
    raise AssertionError('the follow-up add-on failed its tests on this image; outputs were invalidated')
pclx('spectrum-report', '--search', INPUT, WORKING, '--exclude', code_roots[0], '--out', WORKING / 'spectrum')
show(WORKING / 'spectrum' / 'spectrum_report.md')'''),
    ],
    'NB8_stage_b_next': [
        header('NB8 — the follow-up Stage-B arms',
               'Trains the follow-up arms from each seed\'s verified M_A, paired with the main study\'s runs '
               '(same seed, data order, token budget, evaluation batches): tier 1 `protected_svd`, '
               '`protected_replay`; tier 2 `protected_replay_svd`, `sequential_svd`; tier 3 `protected_k256`. '
               'The first run also checks that this driver reproduces the main study\'s protected run. Dev scoring '
               'only at update 35 and the final update (update 0 is reused), no merges: about 1.9 GPU-hours per arm and '
               'seed, so one 12-hour session finishes tiers 1 and 2 for three seeds.',
               'GPU T4 x2', 'OFF', 'pcl-next code, model, NB2, all NB3, all NB4, NB7 (+ every earlier NB8 output)',
               'if it says items are left, run it again with this output attached too. Then run NB9.'),
        code(GPU_HOURS + SETUP + '\n' + NEXT_SETUP),
        code(MODEL + '\n' + DATA_AND_PILOT),
        code('''TIERS = (1, 2, 3)        # drop 3 (or 2 and 3) to stop earlier; see notebooks_2/README.md for each tier's cost
FORCE_SVD_ARMS = False   # NB7's go/no-go rule decides whether the SVD-initialized arms run
decisions = sorted(INPUT.rglob('go_no_go.json'))
assert decisions, 'attach the NB7 output (spectrum/go_no_go.json)'
run_svd = FORCE_SVD_ARMS or read_json(decisions[-1])['run_svd_arms']
print('SVD-initialized arms:', 'running' if run_svd else 'skipped (NB7 rule)')
items = plan.pending_items(found_next, tiers=TIERS, run_svd_arms=run_svd)
print('pending:', items)
commands = [['queue', '--search', INPUT, '--exclude', code_roots[0], '--items', *queue, '--model-config', M0_CONFIG,
             '--settings', SETTINGS, '--data-root', DATA, '--session-seconds', remaining_seconds(),
             '--out', WORKING / 'next' / f'gpu{gpu}'] for gpu, queue in enumerate(plan.split(items))]
if commands:
    pclx_on_gpus(commands, WORKING / 'logs')
after = plan.discover_next([INPUT, WORKING], exclude=code_roots[0])
left = plan.pending_items(after, tiers=TIERS, run_svd_arms=run_svd)
for rep in after['replication']:
    print('replication check:', 'PASSED' if rep['passed'] else 'FAILED', {m: r['max_relative_difference'] for m, r in rep['modes'].items()})
print('Follow-up arms finished. Next: NB9.' if not left else f'Items left {left}: run NB8 again with this output attached.')'''),
    ],
    'NB9_next_report': [
        header('NB9 — follow-up report',
               'Pairs every follow-up arm with its baseline by seed and dev item: dev accuracy on the common grid, '
               'paired group-bootstrap contrasts at update 35, at the final update and at the dev-selected '
               'checkpoint (the better of the two), early learning speed from the training losses, the gradient '
               'spectrum and run costs. '
               'CPU only, a few minutes.',
               'None', 'OFF', 'pcl-next code, NB2, all NB3, all NB4, NB7, all NB8 outputs',
               'read `report_next/report.md`.'),
        code(SETUP + '\n' + NEXT_SETUP),
        code('''pclx('report', '--search', INPUT, '--exclude', code_roots[0], '--out', WORKING / 'report_next')
show(WORKING / 'report_next' / 'report.md')
try:
    from IPython.display import Image, display
    for figure in sorted((WORKING / 'report_next').glob('*.png')):
        display(Image(filename=str(figure)))
except ImportError:
    pass'''),
    ],
}


def main():
    for stale in HERE.glob('*.ipynb'):
        if stale.stem not in NOTEBOOKS:
            stale.unlink()
    for name, cells in NOTEBOOKS.items():
        notebook = {'cells': cells, 'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 4}
        (HERE / f'{name}.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote', ', '.join(NOTEBOOKS))


if __name__ == '__main__':
    main()
