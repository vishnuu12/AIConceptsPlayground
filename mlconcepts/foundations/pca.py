"""
8. PCA: finding the directions that matter
==========================================

WHY THIS PAGE EXISTS
--------------------
Five features that are mostly driven by two hidden factors are painful to plot
and wasteful to model. PCA finds the directions along which the data actually
varies, and lets you throw the rest away.

HOW IT WORKS, IN FOUR STEPS
---------------------------
    1. centre the data            (so the covariance is about spread, not offset)
    2. compute the covariance matrix  (which features move together)
    3. take its eigenvectors      (the directions of pure variance)
       and eigenvalues            (how much variance each direction carries)
    4. project onto the top few   (keep the directions that matter)

The eigenvectors of a covariance matrix ARE the principal components. That is
the whole idea - no iteration, no learning rate, one eigen-decomposition.

WHAT YOU GET
------------
- A scree plot: variance explained per component, so you can say "2 of 5
  components carry 96% of the signal".
- Compression: store 2 numbers per row instead of 5.
- A projection you can actually plot and look at.
- The reconstruction error, which is the honest measure of what you lost.

WHEN NOT TO USE IT
------------------
PCA finds LINEAR directions of maximum VARIANCE - and variance is not the same
as usefulness. If the label depends on a small-variance direction, PCA will
happily delete it. It is also sensitive to scale, which is why you standardise
first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def pca_fit(x: np.ndarray, n_components: int) -> dict:
    """
    Standardise, then eigen-decompose the covariance matrix.

    We standardise (subtract the mean, divide by the standard deviation) because
    PCA maximises variance, and a feature measured in thousands would otherwise
    dominate one measured in fractions - purely because of its units.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_standard = (x - mean) / std

    covariance = np.cov(x_standard, rowvar=False)

    # eigh is for symmetric matrices and returns eigenvalues ASCENDING, so we
    # flip to descending order to get "most important first".
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained = eigenvalues / eigenvalues.sum()

    return {
        "mean": mean,
        "std": std,
        "components": eigenvectors[:, :n_components],  # (n_features, n_components)
        "explained": explained,
        "eigenvalues": eigenvalues,
    }


def transform(x: np.ndarray, model: dict) -> np.ndarray:
    """Project the data onto the kept components - the compression step."""
    x_standard = (x - model["mean"]) / model["std"]
    return x_standard @ model["components"]


def inverse_transform(z: np.ndarray, model: dict) -> np.ndarray:
    """Go back to the original feature space - with whatever was dropped, gone."""
    x_standard = z @ model["components"].T
    return x_standard * model["std"] + model["mean"]


def demo(plots_dir: Path | None = None) -> dict:
    x = data.make_correlated(n=300, seed=0)

    data.heading("8. PCA")
    print(f"input shape                : {x.shape}  (300 rows, 5 features)")
    print("the data really has 2 hidden factors - PCA has to find them\n")

    model = pca_fit(x, n_components=2)

    print("variance explained by each component:")
    running = 0.0
    for i, ratio in enumerate(model["explained"]):
        running += ratio
        print(f"    PC{i + 1}: {ratio * 100:5.1f}%   cumulative {running * 100:5.1f}%")

    z = transform(x, model)
    x_back = inverse_transform(z, model)

    total_variance = float(np.var(x, axis=0).sum())
    residual = float(np.var(x - x_back, axis=0).sum())
    kept = float(1.0 - residual / total_variance)

    print()
    print(f"kept 2 of 5 numbers per row : {x.shape[1]} -> {z.shape[1]}")
    print(f"variance retained           : {kept * 100:.1f}%")
    print(f"reconstruction error        : {residual:.4f} (of {total_variance:.4f} total)")
    print()
    print("Two numbers per row instead of five, and almost all of the signal.")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.6))

    ax1.bar(range(1, len(model["explained"]) + 1), model["explained"] * 100, color="steelblue")
    ax1.plot(range(1, len(model["explained"]) + 1), np.cumsum(model["explained"]) * 100,
             "o-", color="crimson")
    ax1.set_xlabel("component")
    ax1.set_ylabel("% variance")
    ax1.set_title("Scree plot (and cumulative)")
    ax1.grid(alpha=0.3)

    ax2.scatter(z[:, 0], z[:, 1], s=14, alpha=0.7, c="teal")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("The data, in 2 dimensions")
    ax2.grid(alpha=0.3)

    ax3.scatter(x[:, 0], x[:, 1], s=14, alpha=0.5, label="original (2 of 5 features)")
    ax3.scatter(x_back[:, 0], x_back[:, 1], s=14, alpha=0.5, label="reconstructed from 2 PCs")
    ax3.set_title("What the compression lost")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    data.save(fig, "08-pca", plots_dir)

    return {
        "explained": model["explained"],
        "top2_variance": float(model["explained"][:2].sum()),
        "variance_retained": kept,
        "residual": residual,
        "projection": z,
    }


if __name__ == "__main__":
    demo()
