"""
6. Overfitting, the train/test split, and regularisation
========================================================

WHY THIS PAGE EXISTS
--------------------
A model that is perfect on the data it was trained on is worthless if it is
wrong on everything else. This demo makes that concrete: fit polynomials of
increasing degree to noisy data, and watch training error fall while test error
turns around and climbs.

That U-shape is the single most important picture in machine learning.

    degree 1   underfits   - too simple to capture the shape
    degree 3   about right - captures the signal, ignores the noise
    degree 15  overfits    - memorises the noise, useless on new points

THE TWO FIXES
-------------
1. Split the data. Never judge a model on data it has seen. (And keep a third,
   untouched set for the final report - that is what people mean by validation
   vs test.)
2. Regularise. Add a penalty on the size of the weights, so a large weight has
   to earn its place. That is ridge (L2) - the same idea as Lasso (L1), which
   also pushes weights all the way to zero and so doubles as feature selection.

BIAS AND VARIANCE, IN ONE LINE
------------------------------
Underfitting is bias: the model's assumptions are too rigid. Overfitting is
variance: the model's answer moves too much when the data moves. Regularisation
trades a little bias for a lot less variance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


def make_data(n: int = 20, noise: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = true_function(x) + rng.normal(0.0, noise, size=n)

    # A separate, larger test set drawn the same way.
    x_test = rng.uniform(0.0, 1.0, size=400)
    y_test = true_function(x_test) + rng.normal(0.0, noise, size=400)
    return x, y, x_test, y_test


def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """[1, x, x^2, ... x^degree] - one column per power."""
    return np.vstack([x**d for d in range(degree + 1)]).T


def fit_poly(X: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """
    Fit a linear model on polynomial features.

    lam = 0 uses the least-squares solution directly; lam > 0 adds ridge
    regularisation. Note that the bias column (index 0) is NOT penalised -
    shrinking the intercept would just shift the whole curve, which is not what
    regularisation is for.
    """
    if lam <= 0:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def demo(plots_dir: Path | None = None) -> dict:
    x, y, x_test, y_test = make_data()

    data.heading("6. Overfitting, train/test split, regularisation")
    print(f"  {'degree':>7}  {'train MSE':>10}  {'test MSE':>10}   verdict")
    print()

    results = []
    for degree in (1, 3, 9, 15):
        coef = fit_poly(poly_features(x, degree), y)
        train = mse(poly_features(x, degree) @ coef, y)
        test = mse(poly_features(x_test, degree) @ coef, y_test)
        if train > 0.2:
            verdict = "underfit"
        elif test > 4 * train:
            verdict = "OVERFIT"
        else:
            verdict = "good"
        results.append((degree, train, test))
        print(f"  {degree:>7}  {train:>10.4f}  {test:>10.4f}   {verdict}")

    print()
    print("Degree 15 drives TRAINING error to almost nothing - and TEST error up.")
    print("It has stopped learning the curve and started memorising the noise.")

    # Now regularise the worst offender.
    degree = 15
    X, X_test = poly_features(x, degree), poly_features(x_test, degree)
    print()
    print("Same degree-15 model, with a ridge penalty on the weights:")
    print(f"  {'lambda':>10}  {'train MSE':>10}  {'test MSE':>10}  {'max |w|':>9}")
    ridge_rows = []
    for lam in (0.0, 1e-6, 1e-4, 1e-2):
        coef = fit_poly(X, y, lam=lam)
        train = mse(X @ coef, y)
        test = mse(X_test @ coef, y_test)
        ridge_rows.append((lam, train, test, float(np.max(np.abs(coef)))))
        print(f"  {lam:>10.0e}  {train:>10.4f}  {test:>10.4f}  {np.max(np.abs(coef)):>9.2f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    grid = np.linspace(0, 1, 200)

    for ax, degree in zip(axes[:2], (3, 15)):
        coef = fit_poly(poly_features(x, degree), y)
        ax.scatter(x, y, s=30, c="k", zorder=3, label="train data")
        ax.plot(grid, true_function(grid), "g--", lw=2, label="true function")
        ax.plot(grid, poly_features(grid, degree) @ coef, "r-", lw=2, label=f"degree {degree} fit")
        if degree <= 3:
            ax.set_ylim(-2.2, 2.2)
        # For degree 15 the axis is deliberately left to auto-scale: the fit
        # shooting off the top and bottom of the frame IS the overfitting.
        ax.set_title(f"degree {degree}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    degrees = [r[0] for r in results]
    axes[2].plot(degrees, [r[1] for r in results], "o-", label="train MSE")
    axes[2].plot(degrees, [r[2] for r in results], "o-", label="test MSE")
    axes[2].set_xlabel("polynomial degree")
    axes[2].set_ylabel("MSE")
    axes[2].set_yscale("log")
    axes[2].set_title("The U-shape")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    data.save(fig, "06-overfitting", plots_dir)

    return {
        "results": results,
        "ridge": ridge_rows,
        "deg1_test": results[0][2],
        "deg3_test": results[1][2],
        "deg15_test": results[-1][2],
        "deg15_train": results[-1][1],
        "deg15_ridge_test": ridge_rows[-1][2],
    }


if __name__ == "__main__":
    demo()
