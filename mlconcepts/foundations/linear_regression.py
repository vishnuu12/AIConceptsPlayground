"""
2. Linear regression, from scratch
==================================

WHY THIS PAGE EXISTS
--------------------
Linear regression is the "hello world" of supervised learning, and it is worth
writing by hand once because every later model is the same code with a fancier
guess function:

    prediction = w * x + b        <- the model (2 parameters)
    loss       = mean((pred - y)^2)
    d loss / d w, d loss / d b    <- two gradients now, one per parameter
    w, b      -= lr * gradient    <- update them together

WHAT BREAKS WITHOUT IT
----------------------
If you skip this and jump straight to a framework, you never see that "training"
is just: compute two derivatives, subtract them, repeat. When a real model fails
to converge, you have no mental model for why.

NOTE ON FEATURES
----------------
Real data has many features, so the model is `X @ w + b` with X of shape
(n_samples, n_features) and w of shape (n_features,). The maths is identical -
the gradient becomes a vector - which is why this single-feature version is the
right place to start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def fit(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    steps: int = 600,
) -> tuple[float, float, list[float]]:
    """
    Gradient descent for y = w*x + b.

    Both parameters are updated using gradients computed from the SAME
    prediction. Updating w first and then recomputing for b would be a subtle
    and very common bug.
    """
    w = 0.0
    b = 0.0
    history: list[float] = []

    n = len(x)
    for _ in range(steps):
        prediction = w * x + b
        error = prediction - y
        loss = float(np.mean(error**2))

        # The two partial derivatives of the MSE loss.
        grad_w = float(2.0 / n * np.sum(error * x))
        grad_b = float(2.0 / n * np.sum(error))

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        history.append(loss)

    return w, b, history


def r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    """How much of the variance in y the model explains. 1.0 is perfect."""
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def demo(plots_dir: Path | None = None) -> dict:
    x, y = data.make_linear(n=120, noise=0.6, seed=0)

    w, b, history = fit(x, y)
    prediction = w * x + b
    score = r_squared(y, prediction)

    data.heading("2. Linear regression from scratch")
    print(f"the data was generated as : y = 2.0*x + 1.0 + noise")
    print(f"learned w                 : {w:.3f}")
    print(f"learned b                 : {b:.3f}")
    print(f"loss (MSE)                : {history[-1]:.4f}")
    print(f"R^2                       : {score:.4f}")
    print()
    print("Nothing was told that the slope was 2 or the intercept 1 - both came")
    print("out of 600 gradient steps on 120 noisy points.")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax1.scatter(x, y, s=18, alpha=0.6, label="data")
    grid = np.linspace(x.min(), x.max(), 50)
    ax1.plot(grid, w * grid + b, color="crimson", lw=2, label=f"fit: {w:.2f}x + {b:.2f}")
    ax1.set_title("The line gradient descent found")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history, lw=2, color="darkgreen")
    ax2.set_title("Loss per step")
    ax2.set_xlabel("step")
    ax2.set_ylabel("MSE")
    ax2.grid(alpha=0.3)

    data.save(fig, "02-linear-regression", plots_dir)

    return {
        "w": w,
        "b": b,
        "loss": history[-1],
        "r2": score,
        "history": history,
    }


if __name__ == "__main__":
    demo()
