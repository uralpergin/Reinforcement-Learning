[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/s1sW9uPe)


# RL Lecture - Exercise 04

This week, we provide code snippets that are to be filled by you. Please follow the coding instructions in each task. You will also find tests you can check against.

---

## Task 01: Monte Carlo Prediction

![Cliff MDP](./images/cliff.svg)

Consider the MDP in Figure 1 ("Cliff MDP"), where all actions (an action moves the agent in a desired direction: up, down, left, or right) succeed with a probability of $0.8$. With a probability of $0.2$, the agent moves randomly in another direction. All transitions result in a reward of $-1$, except when the coffee shop is reached (terminal state $s_{2,5}$: reward of $10$) or if the agent falls off the cliff (terminal states $s_{3,1}$ through $s_{3,5}$: reward of $-100$). The agent always starts in state $s_{2,1}$ as indicated in Figure 1.

Using Monte Carlo policy evaluation, calculate $V(i)$ for all states $i$ based on the illustrated episodes 1 to 3 (right part of Figure 1). Use the first-visit method, i.e., every state is updated only once -- on the first visit -- per episode, even if the state is visited again during the episode. In this task, we estimate the value by a running mean with $\alpha_t = \frac{1}{t}$ for episode $t$ and initialize $V(i) = 0$ for all $i$. We do not discount, i.e., $\gamma = 1$. 

---

## Task 02: Off-Policy MC Control with Importance Sampling

This task is based on the Blackjack example from the lecture ([Sutton & Barto's RL book](http://www.incompleteideas.net/book/RLbook2018.pdf#page=115)) and an implementation can be found in `blackjack.py`. The state is a tuple containing the player's current sum, the dealer's one showing card (1-10 where 1 is ace), and whether or not the player holds a usable ace (0 or 1). The value is a float.

Implement Off-Policy MC Control in `off_policy_mc.py` as introduced in the lecture:

```python
mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0)
```

You find the tests in `test_exercise04.py`. They expect an average return. Run them by:
```
pytest test_exercise04.py
```

Additionally, you will find a visualization script for the predicted value functions. To use it, you need [matplotlib](https://matplotlib.org/users/installing.html). Run it using: 

```python
python visualization.py
```



Notably, the Off-Policy MC Control algorithm uses **Weighted Importance Sampling**. Instead of estimating $v_\pi$ by the empirical mean:
$$
V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)} G_t}{|\mathcal{T}(s)|},
$$
**Weighted Importance Sampling** uses a weighted average:
$$
V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)}},
$$
to reduce variance.

---

## Submission Instructions

- Upload your solution by adding a file `exercise04.pdf`. You can upload handwritten notes or do your submission via latex.
- Implement the tasks in `off_policy_mc.py.py`;
- (Possibly) test them locally with `pytest test_exercise04.py` and `pthon visualization.py`;
- Push your changes to the Github server;
- See the testing results in Actions page;
- Raise questions and feedbacks in Pull Requests page;
- Share your experiences in the forum.

---

**Supervisors:** Prof. Joschka Boedecker, Dr. Gabriel Kalweit, Philipp Bordne, Julien Brosseit, Jasper Hoffmann, Yuan Zhang

