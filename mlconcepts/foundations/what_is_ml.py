"""
1. What machine learning actually is
====================================

WHY THIS PAGE EXISTS
--------------------
"Machine learning" sounds like a mystery and is usually taught as a catalogue of
algorithms. It is neither. It is one loop, repeated:

    1. make a guess          (the model has parameters)
    2. measure how wrong it is   (the loss)
    3. work out which way to nudge the parameters   (the gradient)
    4. nudge them            (the update)
    5. repeat

Everything else - linear regression, a neural network, a large language model -
is a variation on steps 1 and 3. Get this loop in your head and the rest of the
repo is detail.

THE SMALLEST POSSIBLE EXAMPLE
-----------------------------
Rather than start with a model, start with a single number. The model has ONE
parameter, w, and always predicts w. There are no features at all. Gradient
descent then has to discover the one number that minimises squared error - which
turns out to be the mean of the data.

That is a real machine-learning run: a parameter, a loss, a gradient, an update
loop. It is just small enough to check by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


# --------------------------------------------------------------------------
# The model and the loss
# --------------------------------------------------------------------------


def predict(w: float, n: int) -> np.ndarray:
    """The model: 'always guess w'. That is the whole model."""
    return np.full(n, w, dtype=float)


def mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """
    Mean squared error - the loss.

    A loss is a single number that says how wrong we are. Training means making
    it smaller. Squaring does two useful things: it makes all errors positive,
    and it punishes big mistakes far more than small ones.
    """
    return float(np.mean((prediction - truth) ** 2))


def gradient_of_mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """
    The derivative of the loss with respect to w.

    This is the ONLY piece of calculus in the whole demo, and it is worth
    reading slowly:

        loss = mean((w - y)^2)
        d loss / d w = mean( 2 * (w - y) )

    It says: if the prediction is above the truth on average, the gradient is
    positive, so decreasing w lowers the loss. The sign of the gradient IS the
    direction to move. That is all a gradient is - a direction.
    """
    return float(np.mean(2.0 * (prediction - truth)))


# --------------------------------------------------------------------------
# The training loop
# --------------------------------------------------------------------------


def fit(
    y: np.ndarray, learning_rate: float = 0.1, steps: int = 200
) -> tuple[float, list[float]]:
    """Run the loop. Returns the learned w and the loss after every step."""
    w = 0.0  # start from a deliberately wrong guess
    history: list[float] = []

    for _ in range(steps):
        prediction = predict(w, len(y))
        loss = mse(prediction, y)
        grad = gradient_of_mse(prediction, y)

        w -= learning_rate * grad  # the update - the whole trick
        history.append(loss)

    return w, history


def demo(plots_dir: Path | None = None) -> dict:
    y = data.make_linear(n=200)[1]  # just "some numbers" - no features needed

    w, history = fit(y, learning_rate=0.1, steps=200)

    data.heading("1. What machine learning actually is")
    print(f"data points      : {len(y)}")
    print(f"true mean of y   : {y.mean():.4f}   <- the answer, known in advance")
    print(f"learned w        : {w:.4f}   <- discovered by gradient descent")
    print(f"loss, step 0     : {history[0]:.4f}")
    print(f"loss, step 10    : {history[10]:.4f}")
    print(f"loss, step 50    : {history[50]:.4f}")
    print(f"loss, final      : {history[-1]:.6f}")
    print()
    print("The parameter started at 0, knew nothing, and walked to the mean by")
    print("repeatedly asking 'which way is downhill?'. That is the loop.")

    # A plot of the loss falling is the single most useful picture in ML.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(history, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (MSE)")
    ax.set_title("The loss falling as the parameter is nudged downhill")
    ax.grid(alpha=0.3)
    data.save(fig, "01-what-is-ml-loss", plots_dir)

    return {
        "learned": w,
        "true_mean": float(y.mean()),
        "loss_start": history[0],
        "loss_end": history[-1],
        "history": history,
    }


if __name__ == "__main__":
    demo()
