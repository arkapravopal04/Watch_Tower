"""Scenario-group allocation and deterministic stratified subsets."""

import hashlib
import json
import random
from collections import defaultdict


def allocate_groups(families: list[str], sizes: dict[str, int], seed: int,
                    variants: int) -> list[tuple[str, str, str, int]]:
    """Assign groups before sampling any physical inputs.

    A group is one physical scenario, with up to `variants` wordings. Equation
    families are intentionally shared; this is NOT a held-out-family benchmark.
    """
    if variants < 1 or not families or len(set(families)) != len(families):
        raise ValueError("require distinct families and a positive variant count")
    rng = random.Random(seed)
    result = []
    for split, size in sizes.items():
        if size < 0:
            raise ValueError("split sizes cannot be negative")
        assignments = [families[i % len(families)] for i in range((size + variants - 1) // variants)]
        rng.shuffle(assignments)
        remaining = size
        for i, family in enumerate(assignments):
            result.append((f"{split}-{family}-{i:06d}", family, split, min(variants, remaining)))
            remaining -= variants
    return result


def stratified_subset(records: list[dict], count: int, seed: int) -> list[dict]:
    if not 0 <= count <= len(records):
        raise ValueError("subset size outside source size")
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[record["family"]].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = []
    while len(result) < count:
        for key in sorted(buckets):
            if buckets[key] and len(result) < count:
                result.append(buckets[key].pop())
    return result


def problem_key(record: dict) -> str:
    # Ignore wording, split, and requested output, keeping inverse variants together.
    canonical = json.dumps([record["skill"], record["family"], record["inputs"]], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

