[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jx2ZGnLw)


# RL Lecture - Exercise 03

This week, we provide code snippets that are to be filled by you. Please follow the coding instructions in each task. You will also find tests you can check against.

## Task 1: Dynamic Programming

The tests for the following tasks are based on the Gridworld environment from Sutton's Reinforcement Learning book [Chapter 4](www.incompleteideas.net/book/RLbook2018.pdf). The agent moves on an $m\times n$ grid and the goal is to reach one of the terminal states at the top left or the bottom right corner. An example of a $4\times4$ grid with terminal states $T$ and agent $A$ shows as below, 

$$\begin{bmatrix}
T&\cdot&\cdot&\cdot\\
\cdot&A&\cdot&\cdot\\
\cdot&\cdot&\cdot&\cdot\\
\cdot&\cdot&\cdot&T\\
\end{bmatrix}.$$

The agent can go `up`, `down`, `left` and `right`. Actions leading off the edge do not change the state. The agent receives a reward of $-1$ in each step until it reaches a terminal state. An implementation of this environment is given in `gridworld.py`. 

### Policy Iteration

* Implement the Policy Evaluation function, 

  ```python
  policy_eval(policy, env, discount_factor=1.0, theta=0.00001)
  ```

  in `policy_iteration.py`, where

  * `policy` is a $[S, A]$ ($S$ states and $A$ actions) shaped matrix representing the policy, and 
  * `env` is a discrete MDP environment, and 
  * `env.P[s][a]` is a transition tuple (transition probability, next\_state, reward, done) for state $s$ and action $a$, and
  * `theta` is the stopping threshold. We stop the evaluation once our value-function change (difference between two iterations) is less than `theta` for all states.

  It returns a vector of length $S$ representing the value-function. 

* Implement the Policy Improvement function, 

  ```python
  policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0), 
  ```

  in `policy_iteration.py`. It returns a tuple `(policy, V)` where 

  * `policy` is the optimal policy - a matrix of shape $[S, A]$ where each state $s$ contains a valid probability distribution over actions, and 
  * `V` is the value-function for the optimal policy. 

### Value Iteration

* Implement the Value Iteration function, 

  ```python
  value_iteration(env, theta=0.0001, discount_factor=1.0),
  ```

  in `value_iteration.py`. It again returns a tuple `(policy, V)` of the optimal policy and the optimal value-function.

* What are similarities and differences between Value Iteration and Policy Iteration? Compare the two methods. 



## Submission Instructions

- Implement the tasks in `policy_iteration.py` and `value_iteration.py`;
- (Possibly) test them locally with `pytest test_exercise03.py`;
- Push your changes to the Github server;
- See the testing results in Actions page;
- Raise questions and feedbacks in Pull Requests page;
- Share your experiences in the forum.


**Supervisors** Prof. Joschka Boedecker, Dr. Gabriel Kalweit, Philipp Bordne, Julien Brosseit, Jasper Hoffmann, Yuan Zhang









