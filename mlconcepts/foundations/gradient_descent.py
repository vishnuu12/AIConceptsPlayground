"""
3. Gradient descent: the learning rate
======================================

WHY THIS PAGE EXISTS
--------------------
The learning rate is the one hyperparameter every practitioner has to reason
about, and it is easiest to understand by watching it fail. Three runs, same
data, same code - only the step size changes:

    too small  -> crawls. You will run out of patience before it converges.
    about right-> smooth curve down to the bottom.
    too large  -> overshoots, bounces harder each step, and explodes.

WHY IT EXPLODES (the part interviews ask about)
-----------------------------------------------
Gradient descent assumes the loss is roughly a smooth bowl. Take a step bigger
than the width of that bowl and you land higher up the far side; the next
gradient is bigger, so the next step overshoots further. Formally, for MSE the
update is stable while learning_rate < 2 / (largest curvature), and diverges
above it.

THE OTHER AXIS: BATCH vs STOCHASTIC
-----------------------------------
Full-batch descent uses every example per step - smooth, but expensive and it
can settle into a flat spot. Stochastic (SGD) uses one example or a mini-batch -
the curve is noisy but each step is cheap, and the noise itself helps escape
bad regions. Real training almost always uses mini-batches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def fit(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    steps: int = 40,
) -> list[float]:
    """Full-batch gradient descent; returns the loss after each step."""
    w, b = 0.0, 0.0
    n = len(x)
    history: list[float] = []

    for _ in range(steps):
        error = (w * x + b) - y
        loss = float(np.mean(error**2))
        if not np.isfinite(loss):
            history.append(float("inf"))
            continue

        w -= learning_rate * float(2.0 / n * np.sum(error * x))
        b -= learning_rate * float(2.0 / n * np.sum(error))
        history.append(loss)

    return history


def fit_mini_batch(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    batch_size: int = 8,
    steps: int = 40,
    seed: int = 0,
) -> list[float]:
    """
    Mini-batch SGD: same update, but the gradient comes from a random handful of
    points. Noisier - and often faster in wall-clock time.
    """
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    history: list[float] = []

    for _ in range(steps):
        idx = rng.choice(len(x), size=batch_size, replace=False)
        xb, yb = x[idx], y[idx]

        error = (w * xb + b) - yb
        w -= learning_rate * float(2.0 / len(xb) * np.sum(error * xb))
        b -= learning_rate * float(2.0 / len(xb) * np.sum(error))

        history.append(float(np.mean(((w * x + b) - y) ** 2)))  # full-data loss

    return history


def demo(plots_dir: Path | None = None) -> dict:
    x, y = data.make_linear(n=120, noise=0.6, seed=0)

    runs = {
        "0.005 (too small)": fit(x, y, learning_rate=0.005),
        "0.05  (about right)": fit(x, y, learning_rate=0.05),
        "1.0   (too large)": fit(x, y, learning_rate=1.0),
    }
    sgd = fit_mini_batch(x, y)

    data.heading("3. Gradient descent: choosing the learning rate")
    for label, history in runs.items():
        start, end = history[0], history[-1]
        verdict = "diverged" if not np.isfinite(end) or end > start else f"{end:.4f}"
        print(f"  lr {label:22} start {start:8.3f}  ->  end {verdict}")
    print(f"  lr 0.05 (mini-batch)     start {sgd[0]:8.3f}  ->  end {sgd[-1]:.4f}   (noisy but arrives)")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    for label, history in runs.items():
        ax1.plot(history, lw=2, label=label)
    ax1.set_title("Learning rate decides whether you arrive")
    ax1.set_xlabel("step")
    ax1.set_ylabel("MSE")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(runs["0.05  (about right)"], lw=2, label="full batch")
    ax2.plot(sgd, lw=2, alpha=0.8, label="mini-batch (size 8)")
    ax2.set_title("Full batch is smooth; SGD is noisy")
    ax2.set_xlabel("step")
    ax2.set_ylabel("MSE")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    data.save(fig, "03-learning-rate", plots_dir)

    return {
        "too_small_end": runs["0.005 (too small)"][-1],
        "good_end": runs["0.05  (about right)"][-1],
        "too_large_end": runs["1.0   (too large)"][-1],
        "sgd_end": sgd[-1],
        "histories": runs,
        "sgd_history": sgd,
    }


if __name__ == "__main__":
    demo()
