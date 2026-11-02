"""Immutable study registration and artifact-linked accept/rollback records.

This is a single-writer local registry, not an access-control security boundary.
Locked evaluation must reserve its artifact/data pair before generating answers.
"""

import hashlib
import json
import re
from pathlib import Path

from .io import file_hash, read_json, write_json


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _safe(artifact_id: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', artifact_id):
        raise ValueError(f'artifact IDs must be short file-safe names: {artifact_id!r}')
    return artifact_id


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, study: dict) -> str:
        required = {'artifacts', 'test_files', 'gates', 'selection_rule', 'primary_candidate'}
        if not required <= study.keys() or not study['selection_rule']:
            raise ValueError("freeze artifact hashes, data hashes, gates and a dev-only selection rule")
        for artifact_id in study['artifacts']:
            _safe(artifact_id)
        if study['primary_candidate'] not in study['artifacts']:
            raise ValueError("primary candidate is not preregistered")
        path = self.root / 'study.json'
        if path.exists():
            raise FileExistsError("study is already registered; never overwrite locked choices")
        write_json(path, study)
        return identity(study)

    def reserve_test(self, artifact_id: str, tensor_hashes: dict, data_path) -> Path:
        study = read_json(self.root / 'study.json')
        if study['artifacts'].get(artifact_id) != tensor_hashes:
            raise ValueError("artifact hashes do not match the preregistration")
        data_hash = file_hash(data_path)
        if data_hash not in study['test_files'].values():
            raise ValueError("test file is not preregistered")
        # Short readable names (Windows path limits); full identities live inside the record.
        reservation = self.root / 'locked' / f'{_safe(artifact_id)}__{data_hash[:16]}.json'
        reservation.parent.mkdir(exist_ok=True)
        # Exclusive creation prevents concurrent or repeated unregistered scoring.
        with reservation.open('x', encoding='utf-8') as handle:
            json.dump({'artifact_id': artifact_id, 'data_hash': data_hash,
                       'study_hash': identity(study), 'status': 'reserved'}, handle)
        return reservation

    def finish_test(self, reservation, prediction_path) -> None:
        record = read_json(reservation)
        if record['status'] != 'reserved':
            raise ValueError("test reservation already completed")
        # Stored relative to the registry so it can move between Kaggle notebooks.
        resolved, root = Path(prediction_path).resolve(), self.root.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('locked predictions must be written inside the registry')
        stored = resolved.relative_to(root).as_posix()
        record.update(status='completed', predictions=stored, predictions_hash=file_hash(prediction_path))
        write_json(reservation, record)

    def predictions_path(self, stored) -> Path:
        return self.root / stored

    def completed(self) -> list[dict]:
        records = []
        for path in sorted((self.root / 'locked').glob('*.json')):
            record = read_json(path)
            if record['status'] == 'completed':
                if file_hash(self.predictions_path(record['predictions'])) != record['predictions_hash']:
                    raise ValueError(f"scored predictions were modified: {record['predictions']}")
                records.append(record)
        return records

    def record_decision(self, artifact_id: str, tensor_hashes: dict, result: dict) -> None:
        study = read_json(self.root / 'study.json')
        if artifact_id != study['primary_candidate'] or study['artifacts'].get(artifact_id) != tensor_hashes:
            raise ValueError("promotion must target the preselected, exact scored artifact")
        expected_data = set(study['test_files'].values())
        completed = {record['data_hash'] for record in self.completed() if record['artifact_id'] == artifact_id}
        if not expected_data <= completed:
            raise ValueError("candidate has not completed all registered locked evaluations")
        gates = result.get('gates', {})
        if (set(gates) != {'learned_B', 'kept_A', 'general', 'numerical'}
                or any(type(value) is not bool for value in gates.values())
                or type(result.get('accepted')) is not bool or result['accepted'] != all(gates.values())):
            raise ValueError("inconsistent gate result")
        decision_path = self.root / f'decision-{_safe(artifact_id)}.json'
        if decision_path.exists():
            raise FileExistsError("decision already recorded")
        write_json(decision_path, {'artifact_id': artifact_id, 'hashes': tensor_hashes,
                                   'study_hash': identity(study), **result})
        if result['accepted']:
            # A rejected candidate never replaces this pointer.
            write_json(self.root / 'accepted.json', {'artifact_id': artifact_id, 'hashes': tensor_hashes})
