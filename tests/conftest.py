import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import project modules
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random

import pytest


@pytest.fixture
def fixed_seed():
    seed = 42
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
    return seed
