[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MvKtpyoh)
# RL Lecture - Exercise 02

![plot](./images/mdp_resized.png)

This is a theory exercise. You can submit your solution by uploading a file `exercise02.pdf`. Before you start with the exercise, please read the submission instructions below carefully.

## Task 1: Markov Decision Processes

Assume there are $5$ parking spaces and you start at parking space $5$. In each step,
you can either try to park or drive on. A parking space is free with probability $p$. If a parking space $i$
was occupied or you drove on, you move to the next parking space $i − 1$. You want to be as close to your home – which is at parking space $1$ – as possible. However, you want to avoid to reach the end of parking spaces without parking successfully.

- a) Formalize the above problem as an Markov Decision Process (MDP).
	_Hint:_ Write down the transition probabilities down in the form $p(s', r | s, a)$.
- b) Draw the transition graph.
- c) Do we have to discount? Explain your answer.

## Task 2: Markov Property

Assume a biased slot machine in a casino. Each round, the player can win 1€. However, whenever the outcome of the last two rounds is larger than 1€, the machine lowers the probability of winning. Is the Markov property fulfilled?

## Task 3: Optimal Value Function

**Prove or disprove by counter example:**

For any MDP with optimal value function $v^\star$ , the optimal deterministic policy $\pi^\star$ is unique.

## Share your experience

Share your experience! Provide a brief summary of your experience with this exercise and the corresponding lecture in `feedback.md`. Optionally, you can also make a post in the discussion thread **Exercise 02: MDP** on the [forum](https://ilias.uni-freiburg.de/goto.php?target=frm_3633835&client_id=unifreiburg).

---
### Submission Instructions

- Upload your solution by adding a file `exercise02.pdf`. You can upload handwritten notes or do your submission via latex.
- Share your experiences in the forum.
- For any questions or problems regarding your solutions please let us know through `feedback.md`.
- Also write us in `feedback.md`, for which tasks you want to get detailed feedback. We will only provide detailed feedback if you ask for it!
---

**Supervisors:**  
Prof. Joschka Boedecker, Dr. Gabriel Kalweit, Philipp Bordne, Julien Brosseit, Jasper Hoffmann, Yuan Zhang 
