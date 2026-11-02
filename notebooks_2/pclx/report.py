"""The follow-up report: every arm next to the main study's runs, paired by seed and dev item.

Comparisons use the common dev grid (update 0, the `score_at` updates of
configs/next.yaml, and the final update) that both the main study and the
follow-up scored. Three readings per contrast:
* update 35, where most of B is learned (the early effect SVD init targets);
* the final update (fixed in advance, so free of selection);
* the dev-selected checkpoint on that grid (the main study's rule; optimistic).
Intervals are paired scenario-group bootstraps (pcl.stats), pooled over seeds
with seed resampling. All of this is dev data and exploratory: a promotion
claim needs a locked test on fresh items.
"""

from pathlib import Path

from pcl.io import read_json, read_jsonl, write_json
from pcl.stats import paired_bootstrap
from pcl.study import _find

from . import arms as arm_specs
from .plan import main_runs, stage_a_seeds

SETS = ('B', 'A', 'general')
COLORS = {'reference': '#2a78d6', 'arm': '#eb6834', 'third': '#1baf7a'}
INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781', 'grid': '#e1e0d9', 'surface': '#fcfcfb'}


def _latest(dirs):
    dirs = [Path(d) for d in dirs if (Path(d) / 'run_manifest.json').is_file()]
    if not dirs:
        return None
    best = max(dirs, key=lambda d: read_json(d / 'run_manifest.json')['state']['attempted_updates'])
    return {'manifest': read_json(best / 'run_manifest.json'), 'runs': dirs}


def collect(found) -> dict:
    """{(name, seed): run} for the main-study baselines and every follow-up arm found."""
    table = {}
    for seed in stage_a_seeds(found):
        for method in arm_specs.BASELINES:
            run = _latest(main_runs(found, seed, method))
            if run:
                table[(method, seed)] = {**run, 'source': 'main study'}
    for (name, seed), entries in found['items'].items():
        run = _latest([Path(e['dir']) / 'train' for e in entries])
        if run:
            table[(name, seed)] = {**run, 'source': 'follow-up'}
    return table


def _evaluations(run):
    return {e['update']: e for e in run['manifest'].get('evaluations', [])}


def score_points() -> list[int]:
    from .items import next_settings
    return [0, *next_settings()['score_at']]


def grid(run) -> list[int]:
    """Update 0, the follow-up's scoring points, and the final update (all scored by the main study too)."""
    updates = sorted(_evaluations(run))
    if not updates:
        return []
    points = score_points()
    final = updates[-1] if run['manifest']['complete'] else None
    return [u for u in points if u in updates] + ([final] if final and final not in points else [])


def predictions(run, update, name):
    meta = _evaluations(run)[update]['sets'][name]
    return read_jsonl(_find(run['runs'], meta['predictions'], meta['predictions_hash']))


def accuracy(rows) -> float:
    return sum(r['correct'] for r in rows) / len(rows)


def select_on_grid(run, general_reference, *, retain_pp=-2.0, general_pp=-2.0) -> dict | None:
    """The main study's dev rule restricted to the common grid: max B s.t. A and general constraints."""
    points = [u for u in grid(run) if u > 0]
    if not run['manifest']['complete'] or not points:
        return None
    a0 = accuracy(predictions(run, 0, 'A'))
    g0 = accuracy(general_reference)
    rows = []
    for u in points:
        scores = {name: accuracy(predictions(run, u, name)) for name in SETS}
        eligible = 100 * (scores['A'] - a0) >= retain_pp and 100 * (scores['general'] - g0) >= general_pp
        rows.append({'update': u, 'eligible': eligible, **scores})
    pool = [r for r in rows if r['eligible']] or rows
    best = max(pool, key=lambda r: (r['B'], -r['update']))
    return {**best, 'eligible': best['eligible'] and any(r['eligible'] for r in rows), 'table': rows}


def _point(run, which, selection):
    if which == 'selected':
        return selection['update'] if selection else None
    if which == 'final':
        points = grid(run)
        return points[-1] if run['manifest']['complete'] and points else None
    return which if which in _evaluations(run) else None


def contrast(table, arm, baseline, seeds, which, name, selections) -> dict | None:
    """Pooled (seed-resampled) and per-seed paired bootstrap of arm minus baseline on one dev set."""
    cand, ref, groups, used = [], [], None, []
    for seed in seeds:
        a, b = table.get((arm, seed)), table.get((baseline, seed))
        if not a or not b:
            continue
        ua, ub = _point(a, which, selections.get((arm, seed))), _point(b, which, selections.get((baseline, seed)))
        if ua is None or ub is None:
            continue
        pa = {r['id']: r for r in predictions(a, ua, name)}
        pb = {r['id']: r for r in predictions(b, ub, name)}
        ids = sorted(pa)
        if set(ids) != set(pb):
            raise ValueError(f'{arm} and {baseline} were scored on different {name} items')
        cand.append([pa[i]['correct'] for i in ids])
        ref.append([pb[i]['correct'] for i in ids])
        groups = [pa[i]['group'] for i in ids]
        used.append(seed)
    if not cand:
        return None
    result = paired_bootstrap(cand, ref, groups, repetitions=10000, seed=0, resample_seeds=len(cand) > 1)
    return {**result, 'seeds_used': used}


def _mean_loss(run, start, stop):
    losses = [h['loss'] for h in run['manifest']['history'] if start <= h['update'] <= stop]
    return sum(losses) / len(losses) if losses else None


def _fmt(c):
    if not c:
        return '—'
    return f"{c['estimate_pp']:+.1f} [{c['lower_pp']:+.1f}, {c['upper_pp']:+.1f}]"


def _figures(table, found, out, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'axes.edgecolor': INK['grid'], 'axes.labelcolor': INK['secondary'],
                         'xtick.color': INK['muted'], 'ytick.color': INK['muted'], 'axes.titlecolor': INK['primary'],
                         'figure.facecolor': INK['surface'], 'axes.facecolor': INK['surface'], 'axes.grid': True,
                         'grid.color': INK['grid'], 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'lines.linewidth': 2})
    figures = []
    # 1. Early training loss, SVD init vs its Kaiming pair (same data order), mean over seeds.
    pairs = [(a, s['contrast'][0]) for a, s in arm_specs.ARMS.items() if s['init'].startswith('svd')]
    pairs = [(a, b) for a, b in pairs if any((a, s) in table and (b, s) in table for s in seeds)]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.4), squeeze=False)
        for ax, (a, b) in zip(axes[0], pairs):
            length = 0
            for name, color in ((b, COLORS['reference']), (a, COLORS['arm'])):
                curves = [[h['loss'] for h in table[(name, s)]['manifest']['history'][:60]]
                          for s in seeds if (a, s) in table and (b, s) in table]
                length = min(map(len, curves))
                mean = [sum(c[i] for c in curves) / len(curves) for i in range(length)]
                ax.plot(range(1, length + 1), mean, color=color, label=name)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK['secondary'])
            ax.set_title(f'{a} vs {b}', fontsize=10)
            ax.set_xlabel('optimizer update')
            ax.set_ylabel('training loss (mean over seeds)')
            ax.set_xlim(1, max(length, 2))
        fig.tight_layout()
        path = out / 'early_training_loss.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(('Early training loss: SVD initialization vs its Kaiming pair (same data order)', path.name))
    # 2. Learning vs retention at the final update: one point per arm (mean over seeds), direct-labeled.
    points = []
    for (name, seed), run in table.items():
        g = grid(run)
        if not run['manifest']['complete'] or len(g) < 2:
            continue
        ev = _evaluations(run)
        db = 100 * (ev[g[-1]]['sets']['B']['accuracy'] - ev[0]['sets']['B']['accuracy'])
        da = 100 * (ev[g[-1]]['sets']['A']['accuracy'] - ev[0]['sets']['A']['accuracy'])
        points.append((name, run['source'], db, da))
    if points:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        names = sorted({p[0] for p in points})
        means, placed = [], []
        for name in names:
            mine = [p for p in points if p[0] == name]
            mb, ma = sum(p[2] for p in mine) / len(mine), sum(p[3] for p in mine) / len(mine)
            source = mine[0][1]
            color = COLORS['reference'] if source == 'main study' else COLORS['arm']
            ax.scatter([p[2] for p in mine], [p[3] for p in mine], s=14, color=color, alpha=0.35, linewidths=0)
            ax.scatter([mb], [ma], s=64, color=color, edgecolors=INK['surface'], linewidths=2, zorder=3,
                       label=source if source not in [m[1] for m in means] else None)
            means.append((name, source, mb, ma))
        span_x = (max(m[2] for m in means) - min(m[2] for m in means)) or 1.0
        span_y = (max(m[3] for m in means) - min(m[3] for m in means)) or 1.0
        for name, _, mb, ma in sorted(means, key=lambda m: (m[2], m[3])):
            # Stagger labels whose anchors nearly coincide, so none is drawn over another.
            near = sum(abs(mb - x) < 0.08 * span_x and abs(ma - y) < 0.06 * span_y for x, y in placed)
            ax.annotate(name, (mb, ma), xytext=(6, 4 - 11 * near), textcoords='offset points', fontsize=8,
                        color=INK['primary'])
            placed.append((mb, ma))
        ax.legend(frameon=False, fontsize=8, loc='lower left', labelcolor=INK['secondary'])
        ax.axhline(-2, color=INK['muted'], linewidth=1, linestyle='--')
        ax.annotate('retention margin (−2 pp)', (ax.get_xlim()[0], -2), xytext=(4, 4), textcoords='offset points',
                    fontsize=8, color=INK['muted'])
        ax.set_xlabel('B learned at the final update (pp vs update 0)')
        ax.set_ylabel('A change at the final update (pp vs update 0)')
        ax.set_title('Learning vs retention at the final update (large: mean over seeds; small: seeds)', fontsize=10)
        fig.tight_layout()
        path = out / 'learning_vs_retention.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(('Learning vs retention at the final update', path.name))
    # 3. Gradient concentration: top-r share of the projected / unprojected gradient vs a random subspace.
    spectra = {s: read_json(found['spectra'][s] / 'spectrum.json')['summary'] for s in sorted(found['spectra'])}
    if spectra:
        fig, axes = plt.subplots(1, len(spectra), figsize=(4.2 * len(spectra), 3.3), squeeze=False)
        for ax, (seed, summary) in zip(axes[0], spectra.items()):
            ranks = sorted(int(r) for r in summary['median_top_projected'])
            for key, label, color in (('median_top_projected', 'projected gradient (H = G P)', COLORS['reference']),
                                      ('median_top_unprojected', 'unprojected gradient (G)', COLORS['arm']),
                                      ('median_random_baseline', 'random r-dim subspace', COLORS['third'])):
                values = [100 * summary[key][str(r)] for r in ranks]
                ax.plot(ranks, values, color=color, marker='o', markersize=4, label=label)
            ax.set_xscale('log', base=2)
            ax.set_ylim(0, 105)
            ax.legend(frameon=False, fontsize=8, loc='upper left', labelcolor=INK['secondary'])
            ax.set_title(f'seed {seed} (median layer)', fontsize=10)
            ax.set_xlabel('top r singular directions')
            ax.set_ylabel('share of gradient energy (%)')
        fig.tight_layout()
        path = out / 'gradient_concentration.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(('How concentrated the skill-B gradient is', path.name))
    return figures


def build_report(found, out) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = collect(found)
    seeds = stage_a_seeds(found)
    selections = {}
    for (name, seed), run in table.items():
        stage_a = next(Path(s['dir']) for s in found['main']['stages'] if s.get('stage') == 'a' and s['seed'] == seed
                       and s['status'] == 'complete')
        reference = read_jsonl(stage_a / 'references' / 'm0_general_dev200.jsonl')
        selections[(name, seed)] = select_on_grid(run, reference)
    names = [m for m in arm_specs.BASELINES] + [a for a in arm_specs.ARMS]
    lines = ['# Follow-up report: SVD initialization, protection + replay, larger protection basis', '',
             'Dev data only (dev200 sets), exploratory. Baselines are the main study\'s own runs; every arm is paired '
             'with them by seed (same M_A, data order, token budget, evaluation batches) and by dev item.', '']
    # Status
    lines += ['## What was run', '', '| arm | source | ' + ' | '.join(f'seed {s}' for s in seeds) + ' |',
              '|---|---|' + '---|' * len(seeds)]
    for name in names:
        row = [table.get((name, s)) for s in seeds]
        if not any(row):
            continue
        cells = ['—' if r is None else ('complete' if r['manifest']['complete']
                                        else f"update {r['manifest']['state']['attempted_updates']}") for r in row]
        lines.append(f"| {name} | {next(r['source'] for r in row if r)} | " + ' | '.join(cells) + ' |')
    # Spectrum and replication
    if found['spectra']:
        lines += ['', '## Gradient spectrum at M_A', '']
        for seed in sorted(found['spectra']):
            s = read_json(found['spectra'][seed] / 'spectrum.json')['summary']
            lines.append(f"- seed {seed}: B-gradient energy inside A's protected subspace {100 * s['median_in_q']:.1f}% (median layer); "
                         f"top-8 share of the projected gradient {100 * s['median_top_projected']['8']:.1f}% vs "
                         f"{100 * s['median_random_baseline']['8']:.2f}% for a random 8-dim subspace; predicted first-step "
                         f"speed-up of SVD over Kaiming init: SGD {s['first_order_projected_sgd']['ratio']:.2f}×, "
                         f"AdamW {s['first_order_projected_adam']['ratio']:.2f}×")
    for rep in found['replication']:
        modes = ', '.join(f"{m}: max relative loss difference {r['max_relative_difference']:.2e}"
                          f"{' (checkpointing off)' if not r['checkpointing'] else ''}" for m, r in rep['modes'].items())
        lines += ['', f"**Replication check (seed {rep['seed']}, first {rep['updates']} updates of the main study's "
                      f"protected run): {'PASSED' if rep['passed'] else 'FAILED'}** — {modes}."]
    # Accuracy table
    lines += ['', '## Dev accuracy (%) on the common grid', '',
              f"B / A / general at update {' → '.join(map(str, score_points()))} → final. "
              'Selected = the main study\'s dev rule on this grid '
              '(max B with A ≥ −2 pp vs update 0 and general ≥ −2 pp vs M0); * = no eligible checkpoint.', '',
              '| arm | seed | B | A | general | selected (update: B / A) |', '|---|---|---|---|---|---|']
    for name in names:
        for seed in seeds:
            run = table.get((name, seed))
            if not run:
                continue
            ev, g = _evaluations(run), grid(run)
            cells = [' → '.join(f"{100 * ev[u]['sets'][n]['accuracy']:.1f}" for u in g) for n in SETS]
            sel = selections[(name, seed)]
            chosen = '—' if not sel else f"{sel['update']}: {100 * sel['B']:.1f} / {100 * sel['A']:.1f}{'' if sel['eligible'] else ' *'}"
            lines.append(f'| {name} | {seed} | ' + ' | '.join(cells) + f' | {chosen} |')
    # Paired contrasts
    contrasts = {}
    lines += ['', '## Paired contrasts, arm − its pair (pp, pooled over seeds) [95% CI]', '',
              '| arm vs pair | reading | B | A | general | seeds |', '|---|---|---|---|---|---|']
    for name, spec in arm_specs.ARMS.items():
        pair = spec['contrast'][0]
        for which in (score_points()[1], 'final', 'selected'):
            row = {n: contrast(table, name, pair, seeds, which, n, selections) for n in SETS}
            if not any(row.values()):
                continue
            contrasts[f'{name}-vs-{pair}/{which}'] = row
            used = next(r['seeds_used'] for r in row.values() if r)
            label = f'update {which}' if isinstance(which, int) else which
            lines.append(f"| {name} vs {pair} | {label} | " + ' | '.join(_fmt(row[n]) for n in SETS) + f" | {used} |")
    # Early learning speed
    lines += ['', '## Early learning speed (training loss; same data order only)', '',
              'Mean training loss over updates 1–8 and 1–35, arm minus pair, per seed. Replay arms see A examples, '
              'so their losses are only compared with another replay arm.', '',
              '| arm vs pair | seed | Δ loss, updates 1–8 | Δ loss, updates 1–35 |', '|---|---|---|---|']
    speed = {}
    def method(name):
        return arm_specs.ARMS[name]['method'] if name in arm_specs.ARMS else name
    for name, spec in arm_specs.ARMS.items():
        pair = spec['contrast'][0]
        if ('replay' in method(name)) != ('replay' in method(pair)):
            continue  # different training streams: losses are not comparable
        for seed in seeds:
            a, b = table.get((name, seed)), table.get((pair, seed))
            if not a or not b:
                continue
            d8 = [_mean_loss(a, 1, 8), _mean_loss(b, 1, 8)]
            d35 = [_mean_loss(a, 1, 35), _mean_loss(b, 1, 35)]
            if None in d8 or None in d35:
                continue
            speed[f'{name}-vs-{pair}/seed{seed}'] = {'updates_1_8': d8[0] - d8[1], 'updates_1_35': d35[0] - d35[1]}
            lines.append(f"| {name} vs {pair} | {seed} | {d8[0] - d8[1]:+.4f} | {d35[0] - d35[1]:+.4f} |")
    # Cost
    lines += ['', '## Cost of each follow-up run', '', '| arm | seed | training h | dev scoring h | checkpointing |', '|---|---|---|---|---|']
    for (name, seed), run in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if run['source'] != 'follow-up':
            continue
        m = run['manifest']
        lines.append(f"| {name} | {seed} | {m['metrics']['training_wall_seconds'] / 3600:.2f} | "
                     f"{sum(m['metrics'].get('scoring_seconds', [])) / 3600:.2f} | "
                     f"{'on' if m['provenance'].get('gradient_checkpointing', True) else 'off'} |")
    figures = _figures(table, found, out, seeds)
    if figures:
        lines += ['', '## Figures', '']
        lines += [f'### {title}\n\n![{title}]({name})\n' for title, name in figures]
    lines += ['', '## How to read this', '',
              '- Update 35 is where SVD initialization should show up if it matters (the main study learned most of B by '
              'then). The final update is free of selection; "selected" repeats the main study\'s optimistic dev rule.',
              '- An interval that crosses 0 is inconclusive, not "no effect". Three seeds and 200 items per set detect '
              'only effects of several points.',
              '- Everything here is dev data, which the main study also used for selection. Anything worth adopting '
              'still needs a locked test on fresh items.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out / 'report.json', {'contrasts': contrasts, 'early_learning': speed,
                                     'selections': {f'{n}/seed{s}': v for (n, s), v in selections.items()},
                                     'figures': [f for _, f in figures]})
    return {'report': str(out / 'report.md'), 'runs': len(table), 'contrasts': len(contrasts)}
