"""The follow-up arms: Stage-B method, A initialization and protection basis.

Each arm starts from the same seed's verified M_A, shares its seed, data order,
token budget and evaluation batches with the main study's runs, and differs
from its paired baseline in exactly the field named in `contrast`.
"""

ARMS = {
    # Tier 1: the questions this round exists for.
    'protected_svd': {'method': 'protected', 'init': 'svd_projected', 'protection': 'a', 'tier': 1,
                      'contrast': ('protected', 'A initialized from the projected B gradient instead of Kaiming')},
    'protected_replay': {'method': 'protected_replay', 'init': 'kaiming', 'protection': 'a', 'tier': 1,
                         'contrast': ('protected', 'adds 10% replay of Stage-A memory')},
    # Tier 2: does SVD initialization help with replay, and without protection?
    'protected_replay_svd': {'method': 'protected_replay', 'init': 'svd_projected', 'protection': 'a', 'tier': 2,
                             'contrast': ('protected_replay', 'SVD initialization on top of protection + replay')},
    'sequential_svd': {'method': 'sequential', 'init': 'svd_unprojected', 'protection': None, 'tier': 2,
                       'contrast': ('sequential', 'A initialized from the (unprojected) B gradient')},
    # Tier 3: is the residual forgetting under protection a capacity limit of k <= 128?
    'protected_k256': {'method': 'protected', 'init': 'kaiming', 'protection': 'k256', 'tier': 3,
                       'contrast': ('protected', 'protection basis with up to 256 directions (99% energy target)')},
}

# Main-study Stage-B methods, reused unchanged as baselines.
BASELINES = ('sequential', 'protected', 'random_q', 'replay')
INITS = ('kaiming', 'svd_projected', 'svd_unprojected')


def arm(name: str) -> dict:
    if name not in ARMS:
        raise ValueError(f'unknown arm {name!r}; choose from {sorted(ARMS)}')
    return ARMS[name]


def by_tier(tiers) -> list[str]:
    """Arm names of the given tiers, in priority order (lower tier first)."""
    return [name for tier in sorted(tiers) for name, spec in ARMS.items() if spec['tier'] == tier]
