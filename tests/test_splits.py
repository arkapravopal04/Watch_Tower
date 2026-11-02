import pytest

from pcl.data.generator import generate
from pcl.data.grader import grade_answer
from pcl.data.splits import stratified_subset


@pytest.mark.parametrize('skill', ['A', 'B', 'general'])
def test_reproducible_disjoint_and_self_grade(skill):
    sizes = {'train': 60, 'dev': 20, 'test': 20, 'test_hard': 10}
    records = generate(skill, sizes, 7, allow_test=True)
    assert records == generate(skill, sizes, 7, allow_test=True)
    assert len({r['id'] for r in records}) == len(records)
    for split in sizes:  # scenario groups never cross splits
        assert all(r['group'].split('-')[1] == split for r in records if r['split'] == split)
    assert all(grade_answer(r['response'], r['reference']).correct for r in records)
    for split, count in sizes.items():
        assert sum(r['split'] == split for r in records) == count
    subset = stratified_subset([r for r in records if r['split'] == 'train'], 10, 8)
    assert len({r['id'] for r in subset}) == 10


def test_locked_tests_need_an_explicit_opt_in():
    with pytest.raises(ValueError, match='locked'):
        generate('A', {'test': 5})
