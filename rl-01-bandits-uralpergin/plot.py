"""Plotting functions for bandit experiments."""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import lines


def plot_bandit(regrets, est_q_values, q_values, num_actions):
    fig = plt.figure(figsize=(10, 6))
    plt.tight_layout()

    # Upper plot (total regret)
    ax_regret = plt.subplot2grid((2, 5), (0, 0), colspan=5)
    for strategy, regret in regrets.items():
        total_regret = np.cumsum(regret)
        ax_regret.plot(
            np.arange(len(total_regret)), total_regret, label=strategy, alpha=0.9
        )
    ax_regret.set_ylabel("Total Regret")
    ax_regret.set_xlabel("Rounds")
    ax_regret.legend(loc="upper left", fontsize=8)

    # Lower plots (Q-values)
    cmap = matplotlib.colormaps["viridis"]

    ax = None
    for plot_idx, (strategy, est_qs_all) in enumerate(est_q_values.items()):
        ax = plt.subplot2grid((2, 5), (1, plot_idx), sharey=ax)
        ax.set_title(strategy, fontsize=10)
        for qs in q_values:
            ax.axhline(y=qs, color="grey", linestyle="--", alpha=0.5)

        for idx, est_qs in enumerate(est_qs_all.T):
            ax.plot(est_qs, alpha=0.9, color=cmap(idx / num_actions))

        if plot_idx == 0:
            ax.set_ylabel("Estimated Q-Values")
        else:
            ax.tick_params(labelleft=False)

        ax.set_xscale("log")
        ax.set_xlabel("Rounds")

    ax.legend(  # type: ignore
        handles=[
            lines.Line2D([0], [0], color="grey", linestyle="--", label="Q-Values"),
            lines.Line2D([0], [0], color=cmap(0.5), label="Est. Q-Values"),
        ],
        loc="upper right",
        fontsize=6,
    )

    # add padding between plots
    plt.subplots_adjust(hspace=0.5)

    plt.savefig("exercise01.pdf")
    fig.clf()
