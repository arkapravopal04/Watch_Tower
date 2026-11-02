"""Independent re-derivation of every reference answer from the prompt text.

This replaces a human aerospace reviewer with checks that share no
code with `solvers.py`: values are parsed from the text the model sees, and
the physics is solved a different way (shocks from the Rankine-Hugoniot
conservation laws, inverses by bisection, stagnation ratios from the energy
equation in dimensional form). It also writes a short review sheet so a
person can still skim a stratified sample.
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .grader import UNITS, grade_answer
from .splits import stratified_subset

NUM = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
TOLERANCE = 1e-6


def _get(pattern, text):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f'prompt does not match the audited template: {pattern}')
    return float(match.group(1))


def _gamma(text):
    return _get(r'constant gamma=' + NUM + r';', text)


def _bisect(function, low, high, iterations=200):
    f_low = function(low)
    for _ in range(iterations):
        middle = (low + high) / 2
        if (function(middle) > 0) == (f_low > 0):
            low, f_low = middle, function(middle)
        else:
            high = middle
    return (low + high) / 2


def _stagnation(mach, gamma):
    """Energy equation in dimensional form: cp T0 = cp T + V^2 / 2; entropy constant."""
    gas, temperature = 287.0, 300.0
    cp = gamma * gas / (gamma - 1)
    velocity = mach * math.sqrt(gamma * gas * temperature)
    t0 = temperature + velocity ** 2 / (2 * cp)
    ratio_t = t0 / temperature
    ratio_p = math.exp(cp / gas * math.log(ratio_t))  # ds = cp dT/T - R dp/p = 0
    return ratio_t, ratio_p


def _shock(mach, gamma):
    """Rankine-Hugoniot: mass, momentum and total enthalpy across the shock.

    With rho1 = p1 = 1, energy conservation is a quadratic in u2 whose roots are
    u1 (no shock) and the shocked velocity; the product of the roots gives u2.
    """
    rho1, p1 = 1.0, 1.0
    u1 = mach * math.sqrt(gamma * p1 / rho1)
    k = gamma / (gamma - 1)
    h0 = k * p1 / rho1 + u1 ** 2 / 2
    a = 0.5 - k
    c = -h0
    u2 = c / (a * u1)
    rho2 = rho1 * u1 / u2
    p2 = p1 + rho1 * u1 ** 2 - rho1 * u1 * u2
    return u2 / math.sqrt(gamma * p2 / rho2), p2 / p1


def _expected(record) -> dict:
    text, family = record['prompt'], record['family']
    if family == 'mach':
        gamma = _gamma(text)
        v, t, r = _get(r'V=' + NUM + r' m/s', text), _get(r'T=' + NUM + r' K', text), _get(r'R=' + NUM, text)
        return {'M': v / math.sqrt(gamma * r * t)}
    if family == 'isentropic':
        t0, p0 = _stagnation(_get(r'At Mach ' + NUM + r',', text), _gamma(text))
        return {'T0_T': t0, 'p0_p': p0}
    if family == 'pressure_inverse':
        gamma, ratio = _gamma(text), _get(r'p0/p=' + NUM + r'\.', text)
        return {'M': _bisect(lambda m: _stagnation(m, gamma)[1] - ratio, 0.0, 20.0)}
    if family == 'shock':
        m2, p21 = _shock(_get(r'upstream Mach ' + NUM + r'\.', text), _gamma(text))
        return {'M2': m2, 'p2_p1': p21}
    if family == 'applicability':
        return {'status': 'applicable' if _get(r'At upstream Mach ' + NUM + r',', text) > 1 else 'not_applicable'}
    if family == 'dynamic_pressure':
        speed = _get(r'speed ' + NUM, text)
        speed *= 1000 / 3600 if re.search(r'speed ' + NUM + r' km/h', text) else 1
        return {'q': 0.5 * _get(r'density ' + NUM, text) * speed ** 2}
    if family in ('lift', 'drag'):
        coefficient = _get((r'CL=' if family == 'lift' else r'CD=') + NUM + r', find', text)
        force = 0.5 * _get(r'rho=' + NUM, text) * _get(r'V=' + NUM, text) ** 2 * _get(r'S=' + NUM, text) * coefficient
        return {'L' if family == 'lift' else 'D': force}
    if family == 'speed_inverse':
        q, rho = _get(r'Dynamic pressure is ' + NUM + r' Pa', text), _get(r'density ' + NUM, text)
        return {'V': _bisect(lambda v: 0.5 * rho * v * v - q, 0.0, 1e4)}
    if family == 'missing_input':
        if re.search(r'\bC[LD]=', text):
            raise ValueError('a missing-input item states a coefficient')
        return {'status': 'missing_input', 'missing': 'CL'}
    if family == 'arithmetic':
        a, b, c = map(int, re.search(r'result=\((\d+)\+(\d+)\)-(\d+)\.', text).groups())
        return {'result': a + b - c}
    if family == 'instruction':
        return {'code': re.search(r'codes (\d+), (\d+), (\d+),', text).group(2)}
    if family == 'reasoning':
        numbers = re.search(r'starts with (\d+) objects, receives (\d+), then receives (\d+)\.', text).groups()
        return {'total': sum(map(int, numbers))}
    raise ValueError(f'no independent check for family {family}')


def check_record(record) -> list[str]:
    problems = []
    try:
        expected = _expected(record)
    except (ValueError, AttributeError) as error:
        return [f'{record["id"]}: {error}']
    reference = record['reference']
    if expected.keys() != reference.keys():
        return [f'{record["id"]}: fields {sorted(reference)} != independent {sorted(expected)}']
    for key, value in expected.items():
        spec = reference[key]
        if isinstance(spec, str):
            if str(value) != spec:
                problems.append(f'{record["id"]}: {key}={spec!r}, independent {value!r}')
            continue
        target = spec['value'] * UNITS[spec['unit']][1]
        if not math.isclose(target, value, rel_tol=TOLERANCE, abs_tol=1e-9):
            problems.append(f'{record["id"]}: {key}={target!r}, independent {value!r}')
    if not grade_answer(record['response'], reference).correct:
        problems.append(f'{record["id"]}: the training response does not pass its own grader')
    return problems


def audit(records, *, sample=100) -> tuple[dict, str]:
    problems = [p for record in records for p in check_record(record)]
    prompts = defaultdict(set)
    for record in records:
        prompts[record['prompt']].add(record['split'])
    problems += [f'identical prompt in splits {sorted(s)}' for s in prompts.values() if len(s) > 1]
    lengths = [r['token_length'] for r in records if r.get('token_length')]
    summary = {'records': len(records), 'problems': problems, 'passed': not problems,
               'by_skill_split': {f'{k[0]}/{k[1]}': v for k, v in Counter((r['skill'], r['split']) for r in records).items()},
               'by_family': dict(Counter(r['family'] for r in records)),
               'groups': len({r['group'] for r in records}),
               'token_length': {'max': max(lengths), 'mean': sum(lengths) / len(lengths)} if lengths else None,
               'method': 'values parsed from prompt text; physics re-derived without solvers.py'}
    lines = ['# Review sheet', '',
             'Every item below already passed the independent re-derivation. Skim for wording, ',
             'physical plausibility and whether the question is fair. Note anything odd.', '']
    for skill in sorted({r['skill'] for r in records}):
        pool = [r for r in records if r['skill'] == skill and r['split'] in ('train', 'dev')]
        chosen = stratified_subset(pool, min(sample, len(pool)), seed=0)
        lines += [f'## Skill {skill} ({len(chosen)} items)', '']
        for i, record in enumerate(chosen, 1):
            lines += [f"### {i}. {record['family']} — `{record['id']}`", '', record['prompt'], '',
                      '```', record['response'], '```', '', '- [ ] looks right', '']
    return summary, '\n'.join(lines)


def audit_directories(directories, out, *, sample=100) -> dict:
    from ..io import read_jsonl, write_json
    records = [record for directory in directories for path in sorted(Path(directory).glob('*.jsonl'))
               if path.stem not in ('dev200', 'memory')  # subsets of dev/train
               for record in read_jsonl(path)]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate record IDs across the audited banks')
    summary, sheet = audit(records, sample=sample)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'audit.json', summary)
    (out / 'review_sheet.md').write_text(sheet, encoding='utf-8')
    return summary
