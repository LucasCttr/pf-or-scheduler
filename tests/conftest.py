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
