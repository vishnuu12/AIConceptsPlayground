"""
9. A neural network from scratch (forward pass + backpropagation)
=================================================================

WHY THIS PAGE EXISTS
--------------------
This is the centrepiece. Everything before it was a model whose gradient you
could write down by hand. Here the model has 33 parameters and a hidden layer,
and the gradient is computed by the chain rule applied backwards through the
network - which is exactly what PyTorch does for you, and exactly what you should
be able to explain in an interview.

THE NETWORK
-----------
    2 inputs -> 8 hidden units (tanh) -> 1 output (sigmoid)

    forward:
        Z1 = X @ W1 + b1
        A1 = tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)              <- probability of class 1
        loss = binary cross-entropy

    backward (the chain rule, applied from the output back):
        dZ2 = (A2 - y) / n            <- same "prediction minus truth" as before
        dW2 = A1.T @ dZ2
        db2 = sum(dZ2)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (1 - A1**2)       <- the tanh derivative
        dW1 = X.T @ dZ1
        db1 = sum(dZ1)

Read the backward pass and notice it is the forward pass in reverse, with each
step multiplied by the local derivative of the function it passed through. That
is backpropagation. There is nothing else.

WHY A HIDDEN LAYER MATTERS
--------------------------
The data is a ring: a blob surrounded by a circle. No straight line separates
them, so logistic regression is stuck around 50-60% no matter how long you train
it. One hidden layer with a non-linear activation bends the decision boundary
into a shape that can wrap around - and jumps to ~98%.

That gap is the reason deep learning exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data
from . import logistic_regression


def sigmoid(z: np.ndarray) -> np.ndarray:
    return logistic_regression.sigmoid(z)


def init_params(n_in: int, n_hidden: int, seed: int = 0) -> dict:
    """
    Small random weights, scaled by 1/sqrt(fan_in).

    Zero initialisation would be a bug: every hidden unit would compute the same
    thing and receive the same gradient, so they would stay identical forever.
    The scaling keeps the pre-activations near zero, where tanh actually has a
    gradient.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1)),
        "b2": np.zeros(1),
    }


def forward(x: np.ndarray, params: dict) -> dict:
    """Everything the backward pass will need, kept in one place."""
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = sigmoid(z2)
    return {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}


def backward(x: np.ndarray, y: np.ndarray, params: dict, cache: dict) -> dict:
    """The chain rule, one line per arrow, from the loss back to W1."""
    n = x.shape[0]
    y_col = y.reshape(-1, 1)

    dz2 = (cache["A2"] - y_col) / n
    dw2 = cache["A1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1.0 - cache["A1"] ** 2)  # derivative of tanh
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def loss_of(a2: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(a2.reshape(-1), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(a2: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((a2.reshape(-1) >= 0.5).astype(float) == y))


def train(
    x: np.ndarray,
    y: np.ndarray,
    n_hidden: int = 8,
    learning_rate: float = 1.0,
    steps: int = 3000,
    seed: int = 0,
) -> tuple[dict, list[float], list[float]]:
    """Gradient descent on a two-layer network."""
    params = init_params(x.shape[1], n_hidden, seed=seed)
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for _ in range(steps):
        cache = forward(x, params)
        grads = backward(x, y, params, cache)

        for name in params:
            params[name] -= learning_rate * grads[name]

        if _ % 20 == 0 or _ == steps - 1:
            loss_history.append(loss_of(cache["A2"], y))
            accuracy_history.append(accuracy(a2=cache["A2"], y=y))

    return params, loss_history, accuracy_history


def predict_grid(params: dict, resolution: int = 160):
    """Evaluate the network over a grid, so we can draw the decision boundary."""
    span = np.linspace(-3.0, 3.0, resolution)
    xx, yy = np.meshgrid(span, span)
    points = np.c_[xx.ravel(), yy.ravel()]
    probs = forward(points, params)["A2"].reshape(xx.shape)
    return xx, yy, probs


def demo(plots_dir: Path | None = None) -> dict:
    x, y = data.make_ring(n=400, seed=0)
    x = (x - x.mean(axis=0)) / x.std(axis=0)  # standardise: same reason as before

    # --- the baseline: what a model with no hidden layer can do ---------------
    w, b, _ = logistic_regression.fit(x, y, learning_rate=0.5, steps=2000)
    linear_acc = logistic_regression.accuracy(sigmoid(x @ w + b), y)

    # --- the network ----------------------------------------------------------
    params, loss_history, acc_history = train(x, y, n_hidden=8, learning_rate=1.0, steps=3000)
    net_acc = accuracy(forward(x, params)["A2"], y)

    data.heading("9. A neural network from scratch")
    print(f"architecture            : 2 -> 8 (tanh) -> 1 (sigmoid)")
    print(f"parameters              : {sum(p.size for p in params.values())}")
    print()
    print(f"logistic regression     : {linear_acc * 100:5.1f}% accuracy   <- no hidden layer")
    print(f"this network            : {net_acc * 100:5.1f}% accuracy   <- one hidden layer")
    print()
    print(f"loss   : {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    print(f"acc    : {acc_history[0] * 100:.1f}% -> {acc_history[-1] * 100:.1f}%")
    print()
    print("The data is a ring: a blob inside a circle. No straight line separates")
    print("them, which is exactly why the hidden layer is worth 33 parameters.")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.9))

    # Linear boundary
    span = np.linspace(-3, 3, 200)
    xx, yy = np.meshgrid(span, span)
    linear_probs = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
    ax1.contourf(xx, yy, linear_probs, levels=20, cmap="RdBu_r", alpha=0.6)
    ax1.contour(xx, yy, linear_probs, levels=[0.5], colors="k", linewidths=2)
    ax1.scatter(x[y == 0, 0], x[y == 0, 1], s=12, c="white", edgecolor="k")
    ax1.scatter(x[y == 1, 0], x[y == 1, 1], s=12, c="black")
    ax1.set_title(f"No hidden layer: {linear_acc * 100:.0f}%")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    # Network boundary
    xx, yy, probs = predict_grid(params)
    ax2.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
    ax2.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
    ax2.scatter(x[y == 0, 0], x[y == 0, 1], s=12, c="white", edgecolor="k")
    ax2.scatter(x[y == 1, 0], x[y == 1, 1], s=12, c="black")
    ax2.set_title(f"8 hidden units: {net_acc * 100:.0f}%")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    ax3.plot(np.arange(len(loss_history)) * 20, loss_history, lw=2, label="loss")
    ax3.set_xlabel("step")
    ax3.set_ylabel("cross-entropy")
    ax3.set_title("Training")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    data.save(fig, "09-neural-network", plots_dir)

    return {
        "linear_accuracy": linear_acc,
        "net_accuracy": net_acc,
        "n_parameters": int(sum(p.size for p in params.values())),
        "loss_start": loss_history[0],
        "loss_end": loss_history[-1],
        "params": params,
    }


if __name__ == "__main__":
    demo()
