"""Paired scenario-cluster bootstrap in percentage points.

Groups are sampled uniformly; scores are item-weighted within each resample.
The same group draw is used across methods and seeds. Optional seed resampling
is crossed with groups; repeated evaluation items are never independent rows.
"""

import numpy as np


def paired_bootstrap(candidate, reference, groups, *, repetitions: int = 10000,
                     seed: int = 0, resample_seeds: bool = False) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.ndim == 1:
        candidate, reference = candidate[None, :], reference[None, :]
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != len(groups):
        raise ValueError("paired scores must match (seeds, items) and group labels")
    if not candidate.size or repetitions < 100:
        raise ValueError("require nonempty scores and at least 100 resamples")
    if not np.isin(candidate, [0, 1]).all() or not np.isin(reference, [0, 1]).all():
        raise ValueError("scores must be binary and finite")
    if candidate.shape[0] > 1 and not resample_seeds:
        raise ValueError("pooled intervals must also resample paired seeds; run per-seed intervals separately")
    unique, membership = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two independent scenario groups are required")
    sizes = np.bincount(membership)
    delta = candidate - reference
    totals = np.stack([np.bincount(membership, weights=row) for row in delta])
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        selected = rng.integers(len(unique), size=len(unique))
        seeds = rng.integers(len(delta), size=len(delta)) if resample_seeds else np.arange(len(delta))
        draws[i] = totals[np.ix_(seeds, selected)].sum() / (sizes[selected].sum() * len(seeds)) * 100
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"estimate_pp": float(delta.mean() * 100), "lower_pp": float(lower), "upper_pp": float(upper),
            "groups": len(unique), "items": len(groups), "seeds": len(delta),
            "correct_to_wrong": ((reference == 1) & (candidate == 0)).sum(axis=1).tolist(),
            "wrong_to_correct": ((reference == 0) & (candidate == 1)).sum(axis=1).tolist(),
            "per_seed_pp": (delta.mean(axis=1) * 100).tolist(), "bootstrap_seed": seed,
            "repetitions": repetitions, "resample_seeds": resample_seeds,
            "weighting": "item-weighted; uniform scenario-group draws; no strata"}


def compare_predictions(candidate: list[dict], reference: list[dict], **kwargs) -> dict:
    c = {row['id']: row for row in candidate}
    r = {row['id']: row for row in reference}
    if len(c) != len(candidate) or len(r) != len(reference) or c.keys() != r.keys():
        raise ValueError("prediction IDs must match one-to-one")
    ids = sorted(c)
    for key in ids:
        if any(c[key][field] != r[key][field] for field in ('group', 'split', 'skill', 'family')):
            raise ValueError("paired prediction metadata mismatch")
    if len({(row['split'], row['skill']) for row in candidate}) != 1:
        raise ValueError("compare one skill/split at a time; do not pool hard cases")
    return paired_bootstrap([c[key]['correct'] for key in ids], [r[key]['correct'] for key in ids],
                            [c[key]['group'] for key in ids], **kwargs)


def acceptance_gates(contrasts: dict, numerical_checks: dict[str, bool], *, learn_pp: float = 5,
                     retain_pp: float = -2, general_pp: float = -2) -> dict:
    required = {'initialization', 'frozen_parameters', 'projection', 'export', 'reload', 'merge', 'resume'}
    checks_pass = required <= numerical_checks.keys() and all(value is True for value in numerical_checks.values())
    for key in ('A', 'B', 'general'):
        result = contrasts[key]
        if result.get('seeds') != 1:
            raise ValueError("an artifact gate requires a single candidate, not a pooled seed interval")
        values = [result[name] for name in ('lower_pp', 'estimate_pp', 'upper_pp')]
        if not np.isfinite(values).all() or values[0] > values[2]:
            raise ValueError("invalid confidence interval")
    gates = {"learned_B": contrasts['B']['estimate_pp'] >= learn_pp and contrasts['B']['lower_pp'] > 0,
             "kept_A": contrasts['A']['lower_pp'] >= retain_pp,
             "general": contrasts['general']['lower_pp'] >= general_pp,
             "numerical": checks_pass}
    return {"accepted": all(gates.values()), "gates": gates}
