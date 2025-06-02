import pytest 
import numpy as np
import policy_iteration
import value_iteration

def test_policy_eval():
    env = pytest.global_env
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_iteration.policy_eval(random_policy, env)
    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


def test_policy_improvement_p():
    env = pytest.global_env
    policy, v = policy_iteration.policy_improvement(env)
    verify_policy(policy)

def test_policy_improvement_v():
    env = pytest.global_env
    policy, v = policy_iteration.policy_improvement(env)
    verify_value(v)

def test_value_iteration_p():
    env = pytest.global_env
    policy, v = value_iteration.value_iteration(env)
    verify_policy(policy)

def test_value_iteration_v():
    env = pytest.global_env
    policy, v = value_iteration.value_iteration(env)
    verify_value(v)


def verify_policy(policy):
    expected_policy = [[1., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 1., 0., 0.],
                       [1., 0., 0., 0.]]
    np.testing.assert_array_equal(policy, expected_policy)

def verify_value(v):
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)



