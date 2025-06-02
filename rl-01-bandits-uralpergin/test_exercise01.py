"""Tests for exercise01.py"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal


from exercise01 import (
    random,
    epsilon_greedy,
    decaying_epsilon_greedy,
    ucb,
    softmax,
)


# The following code is provided for the student.
#
# class DummyMab(MAB):
#     def __init__(self):
#         self.num_actions = 2
#         self.num_rounds = 10
#         self.step_counter = 5
#
#     @property
#     def bandit_est_q_values(self) -> np.ndarray:
#         return np.array([log(4), 0])
#
#     @property
#     def bandit_counters(self) -> np.ndarray:
#         return np.array([1, 4])


def test_random(mab):
    probs = random(mab)

    assert_array_almost_equal(probs, np.array([0.5, 0.5]))


def test_epsilon_greedy(mab):
    probs = epsilon_greedy(mab, 0.1)

    assert_array_almost_equal(probs, np.array([0.95, 0.05]))


def test_decaying_epsilon_greedy(mab):
    probs = decaying_epsilon_greedy(mab, 0.4)

    assert_array_almost_equal(probs, np.array([0.9, 0.1]))


def test_ucb(mab):
    probs = ucb(mab, 1)

    assert_array_almost_equal(probs, np.array([1.0, 0.0]))


def test_softmax(mab):
    probs = softmax(mab, 1)

    assert_array_almost_equal(probs, np.array([0.8, 0.2]))
