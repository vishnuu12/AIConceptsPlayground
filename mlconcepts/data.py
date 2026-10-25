"""
Shared helpers: the toy datasets and the plotting/output conventions.

Every demo in this repo follows the same shape:

    demo(plots_dir=...) -> dict of metrics

The demo prints a readable trace, optionally saves a plot, and RETURNS its
metrics. That last part is what makes the repo testable: tests/test_*.py calls
the same demo function and asserts on the numbers (the loss really fell, the
accuracy really cleared 0.9, the clusters really came out right).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Use a non-interactive backend so plots work on a headless CI machine.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)


# --------------------------------------------------------------------------
# Toy datasets
#
# All of them are tiny, seeded and noise-free-ish on purpose: you want to SEE
# the algorithm work, not fight the data.
# --------------------------------------------------------------------------


def make_linear(n: int = 120, noise: float = 0.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A straight line with noise: y = 2x + 1 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    y = 2.0 * x + 1.0 + rng.normal(0.0, noise, size=n)
    return x, y


def make_classification(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Two overlapping Gaussian blobs - a binary classification problem where a
    straight line can do a decent, but not perfect, job.
    """
    rng = np.random.default_rng(seed)
    n_half = n // 2
    a = rng.normal(loc=(-1.2, -1.0), scale=0.85, size=(n_half, 2))
    b = rng.normal(loc=(1.2, 1.0), scale=0.85, size=(n - n_half, 2))
    x = np.vstack([a, b])
    y = np.concatenate([np.zeros(n_half), np.ones(n - n_half)])
    return x, y


def make_ring(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    An inner blob surrounded by a ring - the classic case a LINEAR model cannot
    solve and a hidden layer can. This is what the neural-network demo uses to
    show why depth matters.
    """
    rng = np.random.default_rng(seed)
    n_inner = n // 2
    inner_r = rng.normal(0.0, 0.35, size=n_inner)
    inner_a = rng.uniform(0, 2 * np.pi, size=n_inner)
    inner = np.stack([inner_r * np.cos(inner_a), inner_r * np.sin(inner_a)], axis=1)

    outer_r = rng.normal(1.6, 0.25, size=n - n_inner)
    outer_a = rng.uniform(0, 2 * np.pi, size=n - n_inner)
    outer = np.stack([outer_r * np.cos(outer_a), outer_r * np.sin(outer_a)], axis=1)

    x = np.vstack([inner, outer])
    y = np.concatenate([np.zeros(n_inner), np.ones(n - n_inner)])
    return x, y


def make_clusters(
    k: int = 3, per_cluster: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    k well-separated Gaussian blobs, UNLABELLED - for clustering.

    Returns (points, true_centres). The true centres are returned only so the
    demo (and its test) can check that k-means recovered them - a clustering
    algorithm never gets to see them.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-6.0, 6.0, size=(k, 2))
    points = [rng.normal(loc=c, scale=0.7, size=(per_cluster, 2)) for c in centres]
    return np.vstack(points), centres


def make_correlated(n: int = 300, seed: int = 0) -> np.ndarray:
    """
    Five features where the real structure lives in two directions - the point
    of PCA is to find them.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    mixing = np.array(
        [
            [1.0, 0.6],
            [0.9, -0.4],
            [0.2, 1.1],
            [-0.7, 0.9],
            [0.5, 0.5],
        ]
    )
    return latent @ mixing.T + rng.normal(scale=0.05, size=(n, 5))


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def plots_dir() -> Path:
    """Where generated PNGs go. Overridable so CI and the site build agree."""
    path = Path(os.environ.get("AICONCEPTS_PLOTS", "docs/plots"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, name: str, out_dir: Path | None = None) -> Path | None:
    """
    Save a figure if we are writing plots at all.

    Returns the path (so the site build can copy it) or None when plotting is
    switched off - tests set AICONCEPTS_NO_PLOTS=1 to stay fast.
    """
    if os.environ.get("AICONCEPTS_NO_PLOTS") == "1":
        plt.close(fig)
        return None

    target_dir = out_dir or plots_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.png"
    fig.savefig(target, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return target


def heading(text: str) -> None:
    """A section banner, so the console trace is easy to follow."""
    print()
    print(text)
    print("-" * len(text))


def rule() -> None:
    print("-" * 68)
