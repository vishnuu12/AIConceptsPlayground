"""
5. Loss functions: why cross-entropy, not squared error
=======================================================

WHY THIS PAGE EXISTS
--------------------
People use cross-entropy for classification because "that is what you do". This
demo shows the actual reason, with numbers: squared error stops caring exactly
when the model is most wrong.

THE MATHS, IN ONE LINE EACH
---------------------------
Both losses are functions of the probability p that the model outputs, with the
true label y being 0 or 1.

    squared error   L = (p - y)^2
    cross-entropy   L = -( y*log(p) + (1-y)*log(1-p) )

Now differentiate each with respect to the pre-sigmoid value z, because that is
what actually drives the weights. Using p = sigmoid(z), dp/dz = p(1-p):

    dL/dz  (squared error) = 2(p - y) * p * (1 - p)
    dL/dz  (cross-entropy) =     (p - y)

See the extra p(1-p) in the first one? That is the sigmoid's derivative, and it
goes to ZERO when the model is confident. So:

    label is 1, model confidently says 0.01:
        cross-entropy gradient = -0.9900   (a big shove in the right direction)
        squared-error gradient = -0.0196   (about 50x smaller - barely moves)

    ...and it gets worse the more confident and more wrong the model is, because
the p(1-p) factor keeps shrinking.

A model that is confidently wrong should be corrected hard. Squared error does
the opposite, which is why training with it on a classification task is slow and
sometimes simply stalls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data
from .logistic_regression import sigmoid


def squared_error_grad_z(p: float, y: float) -> float:
    """d/dz of (p - y)^2, where p = sigmoid(z)."""
    return 2.0 * (p - y) * p * (1.0 - p)


def cross_entropy_grad_z(p: float, y: float) -> float:
    """d/dz of the cross-entropy, where p = sigmoid(z)."""
    return p - y


def squared_error(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def train_with(loss_grad, steps: int = 300, learning_rate: float = 0.5, seed: int = 0):
    """Train logistic regression using whichever gradient we are handed."""
    x, y = data.make_classification(n=200, seed=seed)
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    n, n_features = x.shape
    w = np.zeros(n_features)
    b = 0.0
    history: list[float] = []

    for _ in range(steps):
        p = sigmoid(x @ w + b)
        # Apply the per-example gradient rule, then average.
        error = np.array([loss_grad(pi, yi) for pi, yi in zip(p, y)])
        w -= learning_rate * (x.T @ error) / n
        b -= learning_rate * float(np.mean(error))
        history.append(cross_entropy(p, y))  # always MEASURE with cross-entropy

    return history


def demo(plots_dir: Path | None = None) -> dict:
    data.heading("5. Loss functions: squared error vs cross-entropy")

    print("How hard each loss pushes, when the model is CONFIDENTLY WRONG")
    print("(true label = 1, model predicts p)")
    print()
    print(f"  {'p':>8}  {'CE gradient':>12}  {'MSE gradient':>13}   ratio")
    rows = []
    for p in (0.01, 0.10, 0.30, 0.50, 0.70, 0.99):
        ce = cross_entropy_grad_z(p, 1.0)
        mse = squared_error_grad_z(p, 1.0)
        ratio = ce / mse if mse else float("inf")
        rows.append((p, ce, mse, ratio))
        print(f"  {p:>8.2f}  {ce:>12.4f}  {mse:>13.4f}   {ratio:>6.1f}x")

    print()
    print("At p = 0.01 the cross-entropy gradient is ~50x larger, and it is the")
    print("model that is MOST wrong that gets the biggest correction.")

    ce_history = train_with(cross_entropy_grad_z)
    mse_history = train_with(squared_error_grad_z)

    print()
    print("And in practice, training the SAME model for the SAME 300 steps:")
    print(f"  with cross-entropy gradients : final loss {ce_history[-1]:.4f}")
    print(f"  with squared-error gradients : final loss {mse_history[-1]:.4f}")
    print()
    print("Honest note: on a problem this easy the gap stays small - both losses")
    print("eventually pull the model into place. The squared-error version pays for")
    print("it when the model STARTS confidently wrong (a bad init, an unbalanced")
    print("dataset, a deeper network), which is exactly the gradient table above.")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ps = np.linspace(0.001, 0.999, 300)
    ax1.plot(ps, [cross_entropy_grad_z(p, 1.0) for p in ps], lw=2, label="cross-entropy")
    ax1.plot(ps, [squared_error_grad_z(p, 1.0) for p in ps], lw=2, label="squared error")
    ax1.set_xlabel("predicted probability p  (true label = 1)")
    ax1.set_ylabel("gradient size")
    ax1.set_title("Cross-entropy pulls hardest when most wrong")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(ce_history, lw=2, label="cross-entropy")
    ax2.plot(mse_history, lw=2, label="squared error")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss (measured as cross-entropy)")
    ax2.set_title("Same model, same steps, different gradients")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    data.save(fig, "05-loss-functions", plots_dir)

    return {
        "rows": rows,
        "ce_final": ce_history[-1],
        "mse_final": mse_history[-1],
        "ce_history": ce_history,
        "mse_history": mse_history,
    }


if __name__ == "__main__":
    demo()
