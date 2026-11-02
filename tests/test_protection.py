import pytest
import torch

from pcl.activations import ActivationCollector, sample_positions
from pcl.protection import build_basis


def test_covariance_matches_svd_and_capture_mask():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 8) + 5  # uncentered mean direction must be retained
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, -2:] = False
    layer = torch.nn.Linear(8, 3)
    with ActivationCollector({'layer': layer}) as capture:
        capture.select(mask)
        layer(x)
    h = x[mask]
    assert capture.counts['layer'] == 16
    assert torch.allclose(capture.moments['layer'], h.T @ h)
    q, info = build_basis(capture.moments['layer'], max_k=3)
    _, _, vh = torch.linalg.svd(h.double(), full_matrices=False)
    assert torch.allclose(q @ q.T, (vh[:info['k']].T @ vh[:info['k']]).float(), atol=1e-5)


def test_rank_cap_and_negative_energy():
    q, info = build_basis(torch.eye(5), max_k=2)
    assert q.shape == (5, 2)
    assert info['captured_energy'] == pytest.approx(0.4)
    assert not info['target_met']
    with pytest.raises(ValueError):
        build_basis(torch.zeros(5, 5))
    with pytest.raises(ValueError):
        build_basis(torch.diag(torch.tensor([1., -0.1])))




def test_sampling_uses_entire_memory_and_rejects_test():
    records = [{'id': str(i), 'group': str(i), 'split': 'train'} for i in range(5)]
    selected = sample_positions(records, [10] * 5, 10, 1)
    assert len(selected) == 5
    assert all(len(value) == 2 for value in selected.values())
    assert selected == sample_positions(records, [10] * 5, 10, 1)
    records[0]['split'] = 'test'
    with pytest.raises(ValueError):
        sample_positions(records, [10] * 5, 10, 1)
