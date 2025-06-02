"""Exercise01: Implement different bandit exploration strategies."""

import sys

import numpy as np
from typing import Tuple, List

from plot import plot_bandit


class Bandit:
    """Bandit class representing a single arm of the multi-armed bandit.

    We assume that the reward is drawn from a shifted uniform distribution.

    Attributes:
        q_value: true mean of the bandit
        est_q_value: estimated mean of the bandits
        counter: number of times the bandit was pulled
    """

    def __init__(self, bias, counter=0) -> None:
        """Initialize the bandit.

        Args:
            bias: bias of the bandit
            counter: number of times the bandit was pulled
        """
        self.q_value = (
            bias + 0.5
        )  # mean of the bandit, 0.5 for the uniform distribution
        self.est_q_value = 0  # estimated mean of the bandit
        self.counter = counter

        self._bias = bias

    def pull(self) -> float:
        """Pulls the bandit.

        Returns:
            The reward of pulling the bandit
        """
        self.counter += 1
        reward = self._bias + np.random.uniform()
        self.est_q_value += (reward - self.est_q_value) / self.counter
        return reward


class MAB:
    """Multi-armed bandit class.

    Hint: The attributes below is what you should use in your implementation.

    Attributes:
        num_actions: number of actions
        step_counter: number of rounds taken
        num_rounds: number of rounds to play
        bandit_counters: number of times each bandit was pulled
        bandit_q_values: q_values of the bandits
    """

    def __init__(self, num_rounds: int, *bandits) -> None:
        """Multi-armed bandit.

        Args:
            num_rounds: number of rounds to play
            bandits: list of bandits to play with
        """
        self.num_actions = len(bandits)
        self.step_counter = 0
        self.num_rounds = num_rounds

        self._bandits = bandits
        self._best_action_value = max(bandit.q_value for bandit in bandits)

    def pull(self, action) -> Tuple[List[float], List[float]]:
        """Pull arm with index action.

        Args:
            action: index of the arm to pull

        Returns:
            A tuple (reward, q_value) containing the reward and the Q-value of the pulled arm.
        """
        self.step_counter += 1
        reward = self._bandits[action].pull()
        return reward, self._bandits[action].q_value

    def run(self, exploration_strategy, **strategy_parameters):
        """Run the multi-armed bandit with the given exploration strategy.

        Args:
            exploration_strategy: callable exploration strategy
            strategy_parameters: parameters for the exploration strategy

        Returns:
            A tuple (regrets, est_q_values) containing the regrets and estimated
            Q-values for each round.
        """
        regrets = []
        est_q_values = []
        for i in range(self.num_rounds):
            if (i + 1) % 100 == 0:
                print("\rRound {}/{}".format(i + 1, self.num_rounds), end="")
                sys.stdout.flush()
            prob_action = exploration_strategy(self, **strategy_parameters)
            action = np.random.choice(self.num_actions, p=prob_action)
            _, q_value = self.pull(action)
            regret = self._best_action_value - q_value
            regrets.append(regret)
            est_q_values.append(self.bandit_est_q_values)

        regrets = np.array(regrets)
        est_q_values = np.array(est_q_values)

        return regrets, est_q_values

    @property
    def bandit_counters(self) -> np.ndarray:
        """Return the number of times each bandit was pulled.

        Returns:
            An array containing the number of times each bandit was pulled.
        """
        return np.array([bandit.counter for bandit in self._bandits])

    @property
    def bandit_est_q_values(self) -> np.ndarray:
        """Return the estimated Q-values of the bandits.

        Returns:
            An array containing the Q-values of the bandits.
        """
        return np.array([bandit.est_q_value for bandit in self._bandits])


def random(mab: MAB) -> np.ndarray:
    """Random strategy.

    Args:
        mab: MAB object

    Returns:
        A numpy array containing the probabilities of selecting each action."""
    # As an example we already implemented the random strategy :)
    return np.ones(mab.num_actions) / mab.num_actions


def epsilon_greedy(mab: MAB, epsilon):
    """Epsilon strategy.

    Hint:
        Helpful comments are np.ones, np.full, np.zeros and np.argmax to
        create the probabilities.

    Args:
        mab: MAB object

    Returns:
        A numpy array containing the probabilities of selecting each action.
    """
    p_act = np.full(mab.num_actions, epsilon / mab.num_actions)
    best_act = np.argmax(mab.bandit_est_q_values)

    p_act[best_act] += 1 - epsilon
    return p_act

def decaying_epsilon_greedy(mab: MAB, epsilon_init):
    """Decaying epsilon strategy.

    Hint:
        You can use the epsilon_greedy function to implement this strategy.

    Args:
        mab: MAB object

    Returns:
        A numpy array containing the probabilities of selecting each action.
    """
    epsilon = (1 - mab.step_counter / mab.num_rounds) * epsilon_init

    return epsilon_greedy(mab, epsilon)


def ucb(mab: MAB, c):
    """UCB strategy.

    Hint:
        There is only one action that maximizes the UCB, thus the
        probabilities are 0 for all other actions.

    Args:
        mab: MAB object

    Returns:
        A numpy array containing the probabilities of selecting each action.
    """
    best_act= np.argmax(mab.bandit_est_q_values + c * np.sqrt(np.log(mab.step_counter) / mab.num_actions))

    prob_action = np.zeros(mab.num_actions)
    prob_action[best_act] = 1.0

    return prob_action


def softmax(mab: MAB, tau):
    """Softmax strategy.

    Hint:
        The softmax can be numerically unstable. To implement a stable version
        take a look at: https://jaykmody.com/blog/stable-softmax/

    Args:
        mab: MAB object

    Returns:
        A numpy array containing the probabilities of selecting each action.
    """
    q_vals = mab.bandit_est_q_values / tau

    q_max = np.max(q_vals)

    exp_val = np.exp(q_vals - q_max)

    return exp_val / np.sum(exp_val)


if __name__ == "__main__":
    num_rounds = 100000  # modify this for debugging
    epsilon = 0.40
    epsilon_init = 0.20
    tau = 0.05
    c = 1.0
    num_actions = 5
    biases = [1.0 / k - 0.3 for k in range(5, 5 + num_actions)]

    # setup the different exploration strategies
    strategies = {}
    strategies[random] = {}
    strategies[epsilon_greedy] = {"epsilon": epsilon}
    strategies[decaying_epsilon_greedy] = {"epsilon_init": epsilon_init}
    strategies[ucb] = {"c": c}
    strategies[softmax] = {"tau": tau}

    est_q_values = {}
    total_regrets = {}

    for strategy, parameters in strategies.items():
        print(strategy.__name__)
        # setup multi armed bandit
        bandits = [Bandit(bias) for bias in biases]
        mab = MAB(num_rounds, *bandits)
        total_regret, est_q_value = mab.run(strategy, **parameters)
        print("\n")
        est_q_values[strategy.__name__] = est_q_value
        total_regrets[strategy.__name__] = total_regret

    q_values = [bandit.q_value for bandit in mab._bandits]  # type: ignore

    plot_bandit(total_regrets, est_q_values, q_values, num_actions)
