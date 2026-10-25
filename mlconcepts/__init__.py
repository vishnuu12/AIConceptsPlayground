"""
AIConceptsPlayground
====================

Runnable, commented implementations of the ideas behind machine learning and
LLM engineering - written so that every number you see was produced by code you
can read.

Part 1 (mlconcepts.foundations) implements the algorithms from scratch with
NumPy: gradient descent, logistic regression, k-means, PCA, and a small neural
network with hand-written backpropagation.

Part 2 (mlconcepts.llm) builds the pieces of a modern LLM application - mock LLM,
tokenization, embeddings, retrieval, tool calling, evaluation - against a
deterministic fake model, so everything runs with no API key and no network.

Run everything:

    python run_all.py
"""

__version__ = "1.0.0"
