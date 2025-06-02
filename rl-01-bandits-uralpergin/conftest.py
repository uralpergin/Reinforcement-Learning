from math import log

import numpy as np
import pytest

from exercise01 import MAB


class DummyMab(MAB):
    def __init__(self):
        self.num_actions = 2
        self.num_rounds = 10
        self.step_counter = 5

    @property
    def bandit_est_q_values(self) -> np.ndarray:
        return np.array([log(4), 0])

    @property
    def bandit_counters(self) -> np.ndarray:
        return np.array([1, 4])


@pytest.fixture
def mab():
    return DummyMab()
