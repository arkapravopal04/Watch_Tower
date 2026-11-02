import torch


def pytest_sessionstart(session):
    # Tiny numerical fixtures are slower with a full CPU thread pool.
    torch.set_num_threads(1)
