"""
Run every demo, in order, and print a summary.

    python run_all.py

This is what the deploy workflow calls before building the site: it is the
"prove it actually works" step. If a demo raises, the script exits non-zero and
the build stops, so a broken demo can never be published as a working one.

Plots are written to docs/plots/ (override with AICONCEPTS_PLOTS) because the
site embeds them.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

# Every demo module, in the order the site lists them.
DEMOS: list[tuple[str, str]] = [
    ("1. What machine learning actually is", "mlconcepts.foundations.what_is_ml"),
    ("2. Linear regression from scratch", "mlconcepts.foundations.linear_regression"),
    ("3. Gradient descent: the learning rate", "mlconcepts.foundations.gradient_descent"),
    ("4. Logistic regression and the sigmoid", "mlconcepts.foundations.logistic_regression"),
    ("5. Loss functions", "mlconcepts.foundations.loss_functions"),
    ("6. Overfitting, splits and regularisation", "mlconcepts.foundations.overfitting"),
    ("7. k-means clustering", "mlconcepts.foundations.kmeans"),
    ("8. PCA", "mlconcepts.foundations.pca"),
    ("9. A neural network from scratch", "mlconcepts.foundations.neural_network"),
    ("10. Embeddings and cosine similarity", "mlconcepts.llm.embeddings"),
    ("11. Tokenization and sampling", "mlconcepts.llm.tokenization"),
    ("12. Prompting and structured output", "mlconcepts.llm.prompting"),
    ("13. Retrieval augmented generation", "mlconcepts.llm.rag"),
    ("14. Tool calling and a small agent loop", "mlconcepts.llm.tools_agent"),
    ("15. Evaluation", "mlconcepts.llm.evaluation"),
]


def main() -> int:
    plots = Path(os.environ.get("AICONCEPTS_PLOTS", "docs/plots"))
    plots.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"AIConceptsPlayground - running all {len(DEMOS)} demos")
    print(f"plots -> {plots.resolve()}")
    print("=" * 72)

    results: list[tuple[str, bool, float, str]] = []

    for label, module_name in DEMOS:
        started = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            module.demo()
            ok, detail = True, ""
        except Exception:  # noqa: BLE001 - we want to report, not crash
            ok, detail = False, traceback.format_exc(limit=3)
        elapsed = time.perf_counter() - started
        results.append((label, ok, elapsed, detail))
        print(f"\n[{'ok  ' if ok else 'FAIL'}] {label}  ({elapsed:.1f}s)")
        if not ok:
            print(detail)

    print()
    print("=" * 72)
    passed = sum(1 for _, ok, _, _ in results if ok)
    total_time = sum(elapsed for _, _, elapsed, _ in results)
    for label, ok, elapsed, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<46} {elapsed:>6.1f}s")
    print("=" * 72)
    print(f"  {passed}/{len(results)} demos ran  |  total {total_time:.1f}s")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
