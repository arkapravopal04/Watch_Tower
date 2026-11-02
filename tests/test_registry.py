import pytest

from pcl.io import file_hash, read_json, write_jsonl
from pcl.registry import Registry


def test_locked_artifact_hashes_and_rollback(tmp_path):
    registry = Registry(tmp_path / 'registry')
    data = tmp_path / 'test.jsonl'
    write_jsonl(data, [{'id': 1}])
    hashes = {'weight': 'abc'}
    registry.register({'artifacts': {'candidate': hashes}, 'primary_candidate': 'candidate',
                       'test_files': {'A': file_hash(data)}, 'gates': {'retain_pp': -2},
                       'selection_rule': 'frozen dev choice'})
    with pytest.raises(ValueError, match='hashes'):
        registry.reserve_test('candidate', {'weight': 'wrong'}, data)
    reservation = registry.reserve_test('candidate', hashes, data)
    with pytest.raises(FileExistsError):
        registry.reserve_test('candidate', hashes, data)
    predictions = registry.root / 'predictions' / 'candidate.jsonl'
    write_jsonl(predictions, [{'id': 1, 'correct': False}])
    registry.finish_test(reservation, predictions)
    registry.record_decision('candidate', hashes, {'accepted': False,
                              'gates': {'learned_B': False, 'kept_A': True, 'general': True, 'numerical': True}})
    assert not (registry.root / 'accepted.json').exists()
    assert read_json(reservation)['status'] == 'completed'


def test_no_promotion_without_scoring(tmp_path):
    registry = Registry(tmp_path)
    registry.register({'artifacts': {'x': {'w': 'abc'}}, 'primary_candidate': 'x',
                       'test_files': {'A': 'hash'}, 'gates': {}, 'selection_rule': 'dev'})
    with pytest.raises(ValueError, match='completed'):
        registry.record_decision('x', {'w': 'abc'}, {'accepted': True, 'gates': {'made_up': True}})
