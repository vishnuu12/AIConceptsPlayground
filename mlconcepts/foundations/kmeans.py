"""
7. k-means clustering
=====================

WHY THIS PAGE EXISTS
--------------------
Everything so far was supervised: we had labels. Most data does not have labels,
and k-means is the simplest useful thing you can do without them.

The algorithm is four lines, repeated:

    1. guess k centres
    2. assign every point to its nearest centre
    3. move each centre to the mean of the points assigned to it
    4. repeat until nothing moves

WHY IT WORKS, AND WHERE IT DOESN'T
----------------------------------
Step 2 and step 3 each strictly reduce the total squared distance to the nearest
centre, so it always converges - but only to a LOCAL minimum. That is why the
initialisation matters so much, and why k-means++ exists: instead of picking k
random points, it spreads the starting centres out, which is a few lines of code
and dramatically fewer bad runs.

k-means also assumes clusters are round and roughly the same size. Give it two
crescents and it will cut them down the middle.

HOW YOU PICK k
--------------
You cannot know k in advance. The usual answer is the elbow plot: run it for
k = 1..6, plot the total distance (inertia), and look for the bend. The bend is
a heuristic, not a proof - but it is a good conversation to have in an interview.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import data


def inertia(points: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    """Total squared distance from each point to its assigned centre."""
    return float(np.sum((points - centres[labels]) ** 2))


def kmeans_plus_plus(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ initialisation.

    Pick the first centre at random, then each next centre with probability
    proportional to its squared distance from the centres chosen so far. Points
    that are already well covered are unlikely to be picked again.
    """
    n = len(points)
    centres = [points[rng.integers(n)]]

    for _ in range(k - 1):
        # Squared distance from every point to the nearest centre so far.
        dist_sq = np.min(
            [np.sum((points - c) ** 2, axis=1) for c in centres], axis=0
        )
        total = dist_sq.sum()
        if total == 0:  # fewer distinct points than clusters
            centres.append(points[rng.integers(n)])
            continue
        probs = dist_sq / total
        centres.append(points[rng.choice(n, p=probs)])

    return np.array(centres)


def assign(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Step 2: nearest centre for every point."""
    distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Step 3: each centre becomes the mean of its members."""
    centres = np.empty((k, points.shape[1]))
    for j in range(k):
        members = points[labels == j]
        centres[j] = members.mean(axis=0) if len(members) else points[j]
    return centres


def fit(
    points: np.ndarray, k: int, seed: int = 0, max_steps: int = 100
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run assign/update until the labels stop changing."""
    rng = np.random.default_rng(seed)
    centres = kmeans_plus_plus(points, k, rng)
    labels = assign(points, centres)
    history = [inertia(points, labels, centres)]

    for _ in range(max_steps):
        centres = update(points, labels, k)
        new_labels = assign(points, centres)
        history.append(inertia(points, new_labels, centres))

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    return centres, labels, history


def demo(plots_dir: Path | None = None) -> dict:
    points, true_centres = data.make_clusters(k=3, per_cluster=60, seed=0)

    data.heading("7. k-means clustering")
    print(f"points                : {len(points)}")
    print("true centres          : (k-means never sees these)")
    for c in true_centres:
        print(f"    ({c[0]:6.2f}, {c[1]:6.2f})")

    centres, labels, history = fit(points, k=3, seed=0)

    print()
    print("centres k-means found :")
    for c in centres:
        print(f"    ({c[0]:6.2f}, {c[1]:6.2f})")
    print(f"\ninertia               : {history[-1]:.2f}")
    print(f"steps to converge     : {len(history) - 1}")

    # How far off were we? Compare as a sorted set - cluster numbers are arbitrary.
    order_found = np.lexsort((centres[:, 1], centres[:, 0]))
    order_true = np.lexsort((true_centres[:, 1], true_centres[:, 0]))
    error = float(np.max(np.abs(centres[order_found] - true_centres[order_true])))
    print(f"max centre error      : {error:.3f}   (recovered, without ever seeing them)")

    print()
    print("Choosing k with the elbow plot:")
    sweep = {}
    for k in range(1, 7):
        c, lab, h = fit(points, k=k, seed=0)
        sweep[k] = h[-1]
        marker = "   <- the bend" if k == 3 else ""
        print(f"    k = {k}   inertia = {h[-1]:8.2f}{marker}")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", s=18, alpha=0.8)
    ax1.scatter(centres[:, 0], centres[:, 1], marker="X", s=220, c="red",
                edgecolor="white", linewidth=1.5, label="found centres")
    ax1.scatter(true_centres[:, 0], true_centres[:, 1], marker="+", s=200, c="black",
                label="true centres")
    ax1.set_title("k-means, k = 3")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(list(sweep.keys()), list(sweep.values()), "o-", lw=2)
    ax2.axvline(3, ls="--", c="gray")
    ax2.set_xlabel("k")
    ax2.set_ylabel("inertia (total squared distance)")
    ax2.set_title("The elbow plot")
    ax2.grid(alpha=0.3)

    data.save(fig, "07-kmeans", plots_dir)

    return {
        "centres": centres,
        "true_centres": true_centres,
        "labels": labels,
        "inertia": history[-1],
        "centre_error": error,
        "sweep": sweep,
        "history": history,
    }


if __name__ == "__main__":
    demo()
