"""
11. Tokenization and sampling
=============================

WHY THIS PAGE EXISTS
--------------------
Two things sit between your text and a model, and both are usually invisible:

    1. Tokenization - the model does not see characters or words, it sees
       TOKENS from a fixed vocabulary learned from data. Token counts decide
       your cost, your context limit and how much fits in a prompt.
    2. Sampling - the model outputs a score for every token in the vocabulary,
       and one is chosen. Temperature and top-k control how that choice is made.

Both are implemented for real here - no simulator, no API.

BYTE PAIR ENCODING, IN FOUR LINES
---------------------------------
    1. start with every word split into characters
    2. count how often each adjacent pair of symbols appears
    3. merge the most frequent pair into a single symbol
    4. repeat

Do that a few thousand times on a large corpus and you get GPT's vocabulary.
The reason it works: common words end up as one token, rare words stay as
several pieces, and nothing is ever "unknown".

WHY TOKEN COUNT IS NOT WORD COUNT
---------------------------------
"unbelievable" might be one token, or un + bel + iev + able - four. A non-English
word or a code identifier can cost five tokens where an English word costs one.
That is why the same prompt costs more in some languages, and why "just fit it
in the context window" is a real engineering constraint.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from .mock_llm import count_tokens, softmax

END = "</w>"  # marks the end of a word, so "the" and "theory" do not share a tail


# --------------------------------------------------------------------------
# Byte pair encoding, from scratch
# --------------------------------------------------------------------------


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every adjacent occurrence of `pair` with the joined symbol."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, num_merges: int = 60) -> list[tuple[str, str]]:
    """
    Learn the merge rules. Returns them in the order they were learned, which
    matters: encoding applies them in exactly that order.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for word in text.lower().split():
        counts[tuple(word) + (END,)] += 1

    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, frequency in counts.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += frequency

        if not pair_counts:
            break

        # Ties broken deterministically so runs are reproducible.
        best = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best)

        counts = Counter(
            {_merge_symbols(symbols, best): frequency for symbols, frequency in counts.items()}
        )

    return merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize one word by applying the learned merges in order."""
    symbols = tuple(word.lower()) + (END,)
    for pair in merges:
        symbols = _merge_symbols(symbols, pair)
    return [s for s in symbols]


def tokenize(text: str, merges: list[tuple[str, str]]) -> list[str]:
    return [token for word in text.split() for token in encode(word, merges)]


CORPUS = (
    "the model learns from the data. the model predicts the next token from the "
    "previous tokens. training the model changes the parameters of the model. "
    "the loss measures how wrong the model is, and the model is trained to lower "
    "the loss. a large language model is trained on a large amount of text."
)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def next_token_distribution(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits, temperature)


def entropy(probabilities: np.ndarray) -> float:
    """How spread out the distribution is. Low = confident, high = random."""
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= len(logits):
        return logits.astype(float)
    keep = np.argsort(logits)[-k:]
    masked = np.full(len(logits), -np.inf)
    masked[keep] = logits[keep]
    return masked


VOCAB = ["the", "a", "model", "data", "loss", "token", "cat", "bicycle"]
LOGITS = np.array([4.2, 3.1, 2.6, 2.2, 1.4, 0.9, 0.2, -0.6])


def demo(plots_dir: Path | None = None) -> dict:
    title = "11. Tokenization and sampling"
    print()
    print(title)
    print("-" * len(title))

    merges = train_bpe(CORPUS, num_merges=60)

    print("The first 12 merges BPE learned from this small corpus:")
    for index, (a, b) in enumerate(merges[:12], start=1):
        print(f"   {index:>2}. {a!r} + {b!r}  ->  {a + b!r}")

    print()
    print("Now tokenize some words with those rules:")
    for word in ("model", "models", "training", "tokenization", "unbelievable"):
        pieces = encode(word, merges)
        print(f"   {word:<14} -> {len(pieces)} token(s)  {pieces}")

    print()
    print("And the whole point - token count is not word count:")
    sample = "the model learns from the data"
    print(f"   text            : {sample!r}")
    print(f"   words           : {len(sample.split())}")
    print(f"   bpe tokens      : {len(tokenize(sample, merges))}")
    print(f"   approx api count: {count_tokens(sample)}")

    # ---- sampling ---------------------------------------------------------
    print()
    print("Sampling. The model scores every token in the vocabulary:")
    for word, logit in zip(VOCAB, LOGITS):
        print(f"   {word:<9} {logit:+.1f}")

    print()
    print(f"   {'temperature':>11}  {'entropy':>8}  top token   distribution")
    rows = []
    for temperature in (0.2, 1.0, 2.0):
        probabilities = next_token_distribution(LOGITS, temperature)
        rows.append((temperature, probabilities, entropy(probabilities)))
        top = VOCAB[int(np.argmax(probabilities))]
        pretty = " ".join(f"{p:.2f}" for p in probabilities)
        print(f"   {temperature:>11.1f}  {entropy(probabilities):>8.3f}  {top:<10}  {pretty}")

    print()
    print("Temperature is a dial on the same scores: near 0 it always picks the")
    print("argmax, near 2 it flattens towards uniform. That is the whole mechanism")

    print()
    print("top-k filtering, at temperature 1.0:")
    for k in (1, 3, len(VOCAB)):
        masked = top_k_filter(LOGITS, k)
        probabilities = softmax(masked, 1.0)
        allowed = [VOCAB[i] for i in np.argsort(probabilities)[-k:][::-1]]
        print(f"   top_k={k:<2} allowed: {allowed}")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    width = 0.26
    positions = np.arange(len(VOCAB))
    for offset, (temperature, probabilities, _) in enumerate(rows):
        ax1.bar(positions + (offset - 1) * width, probabilities, width,
                label=f"T={temperature}")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(VOCAB)
    ax1.set_ylabel("probability")
    ax1.set_title("Temperature reshapes the distribution")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    lengths = {word: len(encode(word, merges)) for word in ("model", "models", "training", "tokenization", "unbelievable")}
    ax2.bar(list(lengths.keys()), list(lengths.values()), color="steelblue")
    ax2.set_ylabel("tokens")
    ax2.set_title("Common words cost 1 token; rare ones cost several")
    ax2.grid(alpha=0.3, axis="y")

    from .. import data as shared

    shared.save(fig, "11-tokenization", plots_dir)

    return {
        "merges": merges[:12],
        "token_counts": {word: len(encode(word, merges)) for word in ("model", "models", "training", "tokenization", "unbelievable")},
        "temperature_rows": rows,
        "entropy_at_1": rows[1][2],
        "entropy_at_2": rows[2][2],
    }


if __name__ == "__main__":
    demo()
