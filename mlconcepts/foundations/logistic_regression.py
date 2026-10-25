"""
4. Logistic regression and the sigmoid
======================================

WHY THIS PAGE EXISTS
--------------------
Linear regression predicts a number. Classification needs a probability, and a
probability has to live between 0 and 1 - so we squash the linear output through
the sigmoid:

    z = w . x + b            (any real number)
    p = 1 / (1 + e^-z)       (always between 0 and 1)

Then we train with cross-entropy instead of squared error, and the gradient
comes out beautifully simple:

    d loss / d z = (p - y)

That is the same "prediction minus truth" you saw in linear regression. It is
why logistic regression is the natural next step: the loss changed, the shape of
the update did not.

WHAT BREAKS WITHOUT IT
----------------------
The sigmoid itself has a cost, and it is the reason deep networks use ReLU:
saturate it (z very large or very small) and its derivative goes to zero, so
learning stalls. This is the "vanishing gradient" problem in miniature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    The squashing function.

    Written in a numerically careful way: for large negative z, exp(-z) overflows,
    so we use the equivalent form on the other side.
    """
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    """The classification loss. Clipped so log(0) cannot produce -inf."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.5,
    steps: int = 400,
) -> tuple[np.ndarray, float, list[float]]:
    """Gradient descent on the cross-entropy loss."""
    n, n_features = x.shape
    w = np.zeros(n_features)
    b = 0.0
    history: list[float] = []

    for _ in range(steps):
        p = sigmoid(x @ w + b)
        history.append(binary_cross_entropy(p, y))

        # The gradient. Note how short this is: it is (p - y) times the inputs.
        error = p - y
        w -= learning_rate * (x.T @ error) / n
        b -= learning_rate * float(np.mean(error))

    return w, b, history


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(float) == y))


def demo(plots_dir: Path | None = None) -> dict:
    x, y = data.make_classification(n=200, seed=0)

    # Standardising the features matters: gradient descent on un-scaled inputs
    # is slow and the learning rate becomes hard to pick.
    x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)

    w, b, history = fit(x_scaled, y)
    p = sigmoid(x_scaled @ w + b)
    acc = accuracy(p, y)

    data.heading("4. Logistic regression")
    print(f"training points     : {len(y)}")
    print(f"loss, step 0        : {history[0]:.4f}")
    print(f"loss, final         : {history[-1]:.4f}")
    print(f"training accuracy   : {acc:.3f}")
    print()
    print("A straight line through a squashing function gets a probability, and")
    print("the decision boundary is where that probability crosses 0.5.")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Decision boundary: colour the whole plane by predicted probability.
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sigmoid(grid @ w + b).reshape(xx.shape)

    ax1.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.65)
    ax1.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
    ax1.scatter(x_scaled[y == 0, 0], x_scaled[y == 0, 1], s=16, c="white", edgecolor="k", label="class 0")
    ax1.scatter(x_scaled[y == 1, 0], x_scaled[y == 1, 1], s=16, c="black", label="class 1")
    ax1.set_title(f"Decision boundary (accuracy {acc:.2f})")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    ax2.plot(history, lw=2, color="purple")
    ax2.set_title("Cross-entropy loss per step")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.grid(alpha=0.3)

    data.save(fig, "04-logistic-regression", plots_dir)

    return {"accuracy": acc, "loss": history[-1], "w": w, "b": b, "history": history}


if __name__ == "__main__":
    demo()
