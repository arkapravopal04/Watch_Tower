import pytest

from pcl.stats import acceptance_gates, paired_bootstrap


def test_paired_group_interval_and_transitions():
    result = paired_bootstrap([1, 1, 0, 1], [1, 0, 1, 0], ['a', 'a', 'b', 'b'], repetitions=500)
    assert result['estimate_pp'] == 25
    assert result['correct_to_wrong'] == [1]
    assert result['wrong_to_correct'] == [2]
    assert result['groups'] == 2


def test_pooled_requires_crossed_bootstrap():
    with pytest.raises(ValueError, match='paired seeds'):
        paired_bootstrap([[1, 0], [1, 0]], [[0, 0], [0, 0]], ['a', 'b'])


def test_gates_fail_closed():
    contrasts = {key: {'estimate_pp': 6, 'lower_pp': 1, 'upper_pp': 10, 'seeds': 1}
                 for key in ('A', 'B', 'general')}
    checks = {key: True for key in ('initialization', 'frozen_parameters', 'projection', 'export', 'reload', 'merge', 'resume')}
    assert acceptance_gates(contrasts, checks)['accepted']
    assert not acceptance_gates(contrasts, {})['accepted']
    contrasts['A']['lower_pp'] = -2.01
    assert not acceptance_gates(contrasts, checks)['accepted']
