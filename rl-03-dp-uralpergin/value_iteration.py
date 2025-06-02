import numpy as np


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
      env: OpenAI environment. env.P represents the transition probabilities of the environment.
      theta: Stopping threshold. If the value of all states changes less than theta
        in one iteration we are done.
      discount_factor: lambda time discount factor.

    Returns:
      A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        delta = 0

        for state in range(env.nS):

            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[state][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])

            max_value = np.max(action_values)

            delta = max(delta, abs(max_value - V[state]))

            V[state] = max_value

        if delta < theta:
            break

    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                action_values[a] += prob * (reward + discount_factor * V[next_state])

        best_action = np.argmax(action_values)

        policy[state, best_action] = 1.0
    # TODO: Implement this!
    return policy, V
