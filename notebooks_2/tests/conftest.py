import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE.parent):  # the repository (pcl) and notebooks_2 (pclx)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: the follow-up round in miniature')


def pytest_sessionstart(session):
    torch.set_num_threads(1)
