"""
10. Embeddings and cosine similarity
====================================

WHY THIS PAGE EXISTS
--------------------
An embedding turns text into a list of numbers - a point in space - such that
texts that mean similar things land close together. That single idea is what
makes semantic search, clustering, recommendations and RAG possible.

Once text is a vector, everything from Part 1 applies: you can measure distance,
find neighbours, cluster, and reduce dimensions.

THE ONLY TWO THINGS YOU NEED
----------------------------
    cosine similarity = (a . b) / (|a| * |b|)

It is the cosine of the angle between the two vectors, so it ignores their
length: "cat" and "the cat" point the same way and score ~1.0 even though one is
longer. That is why cosine is the default for text, and Euclidean distance is
not.

WHAT THIS DEMO ACTUALLY COMPUTES (be honest about this)
-------------------------------------------------------
Training a real embedding model needs a lot of data and compute, so this demo
uses the honest small-scale stand-in: it hashes words into a fixed-size vector
("feature hashing"), normalises it, and compares those.

That means it captures SHARED WORDS, not meaning. "car" and "automobile" will
score near zero here, because they share no characters. A real embedding would
place them next to each other. Everything else you learn from this page - the
geometry, the cosine, the neighbour search, the failure modes - is identical.

To use real embeddings you would swap `embed()` for a call to a model, and the
rest of the file would not change. That is the point of keeping it in one place.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Words this short, and words this common, carry almost no signal and a lot of
# noise: every question contains "what", "the" and "is", and every document does
# too, so matching on them ranks at random. Dropping them is the cheapest
# improvement you can make to a keyword-based retriever - and it is the same
# instinct that leads to IDF weighting in a real one.
MIN_TOKEN_LENGTH = 3
STOPWORDS = frozenset(
    "the and for are was with that this from you not but has can its all does what how why when "
    "which into than they them then there these some such only also more most other over same "
    "very will just about after before being both each few had have here him his may our out she "
    "should their under upon were who would your".split()
)


def tokens(text: str) -> list[str]:
    """Lowercase content words. Real embedding models use subword tokens instead."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]


def _bucket(token: str, dim: int) -> int:
    """
    A STABLE hash from token to bucket index.

    Python's built-in hash() is randomised per process, which would make
    embeddings different on every run. crc32 is stable forever.
    """
    return zlib.crc32(token.encode("utf-8")) % dim


def embed(text: str, dim: int = DIM) -> np.ndarray:
    """
    Feature hashing: words and word pairs, counted into buckets, then L2
    normalised.

    Normalising matters. Cosine similarity divides by the lengths anyway, so
    normalising up front means the dot product alone is the similarity - and it
    keeps long documents from dominating.
    """
    vector = np.zeros(dim, dtype=float)
    words = tokens(text)

    for word in words:
        vector[_bucket(word, dim)] += 1.0

    # Word pairs carry a little more signal than single words.
    for first, second in zip(words, words[1:]):
        vector[_bucket(f"{first}_{second}", dim)] += 0.5

    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two vectors, as a number from -1 to 1."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def nearest(
    query: str, corpus: list[str], k: int = 3, dim: int = DIM
) -> list[tuple[float, str]]:
    """The k corpus entries closest to the query."""
    q = embed(query, dim)
    scored = [(cosine_similarity(q, embed(text, dim)), text) for text in corpus]
    scored.sort(key=lambda pair: -pair[0])
    return scored[:k]


def similarity_matrix(texts: list[str], dim: int = DIM) -> np.ndarray:
    vectors = np.vstack([embed(text, dim) for text in texts])
    return vectors @ vectors.T  # already normalised, so this IS the cosine


CORPUS = [
    "Gradient descent moves the parameters downhill along the loss.",
    "The learning rate decides how big each step of gradient descent is.",
    "Logistic regression predicts a probability using the sigmoid function.",
    "Cross-entropy is the loss function for classification problems.",
    "k-means groups unlabelled points by repeatedly moving cluster centres.",
    "PCA finds the directions in which the data varies the most.",
    "An embedding turns text into a vector so similarity becomes geometry.",
    "Retrieval augmented generation grounds an answer in retrieved documents.",
    "Tokenization splits text into the subword units a model actually reads.",
]


def demo(plots_dir: Path | None = None) -> dict:
    data_heading = "10. Embeddings and cosine similarity"
    print()
    print(data_heading)
    print("-" * len(data_heading))

    print(f"embedding dimension        : {DIM}")
    print(f"corpus size                : {len(CORPUS)} sentences\n")

    queries = [
        "how big should the learning rate be?",
        "what loss do you use for classification?",
        "how do you find similar text?",
    ]

    for query in queries:
        print(f"query: {query!r}")
        for score, text in nearest(query, CORPUS, k=3):
            print(f"   {score:5.3f}  {text}")
        print()

    matrix = similarity_matrix(CORPUS)
    off_diagonal = matrix[~np.eye(len(CORPUS), dtype=bool)]

    print(f"self-similarity (should be 1.0) : {matrix[0, 0]:.4f}")
    print(f"average similarity between DIFFERENT sentences : {off_diagonal.mean():.3f}")
    print()
    print("Notice how low that average is. Shared-word matching is a blunt")
    print("instrument: real embeddings score paraphrases highly, this one cannot.")

    # The honest failure case, stated in numbers.
    print()
    print("The failure mode, demonstrated:")
    for a, b in (("car", "automobile"), ("happy", "glad"), ("learning rate", "learning rate")):
        print(f"   cos({a!r}, {b!r}) = {cosine_similarity(embed(a), embed(b)):.3f}")
    print("   The first two are synonyms and score ~0. A real model would not.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CORPUS)))
    ax.set_yticks(range(len(CORPUS)))
    ax.set_xticklabels([str(i + 1) for i in range(len(CORPUS))])
    ax.set_yticklabels([text[:34] + "..." for text in CORPUS], fontsize=7)
    ax.set_title("Cosine similarity between every pair of sentences")
    fig.colorbar(image, ax=ax, shrink=0.8, label="cosine similarity")

    save_path = _save(fig, "10-embeddings", plots_dir)

    return {
        "dim": DIM,
        "self_similarity": float(matrix[0, 0]),
        "mean_off_diagonal": float(off_diagonal.mean()),
        "matrix": matrix,
        "neighbours": {q: nearest(q, CORPUS, k=1)[0][1] for q in queries},
        "plot": str(save_path) if save_path else None,
    }


def _save(fig, name: str, out_dir: Path | None):
    from .. import data as shared

    return shared.save(fig, name, out_dir)


if __name__ == "__main__":
    demo()
