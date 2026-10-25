"""
15. Evaluation
==============

WHY THIS PAGE EXISTS
--------------------
Everything else in Part 2 is a component. This is the thing that tells you
whether a change to any of them made things better or worse - and it is the
piece most people skip.

    collect a golden set  ->  run the system over all of it  ->  score each answer
                          ->  compare variants  ->  look for regressions

Without it, "I improved the prompt" means "I tried three examples and they looked
nicer". With it, it means a number.

WHAT TO SCORE
-------------
Start with the crudest thing that could possibly work:

    - did the answer contain the fact we expected?      (fact presence)
    - did it cite a source?                             (citation rate)
    - was it the exact string?                          (exact match - rare)

Only reach for a model-as-judge once the simple scores stop being useful, and
remember that a judge is itself a model with its own failure modes. The golden
set is the valuable artefact here; the scorer is cheap to swap.

THE RULE
--------
A change is an improvement only if the pass rate goes up AND nothing that used
to pass now fails. That second half is what people forget, and it is exactly
what this demo prints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import knowledge
from .mock_llm import MockLLM
from .rag import (
    Chunk,
    answer_with_retrieval,
    answer_without_retrieval,
    build_index,
    retrieve,
)


@dataclass
class GoldenCase:
    question: str
    expected_fact: str
    doc_id: str


# A golden set. Note that each expected fact is a distinctive phrase that really
# does appear in the knowledge base - if it did not, the eval would be wrong, not
# the system.
GOLDEN_SET: list[GoldenCase] = [
    GoldenCase(
        "What happens if the learning rate is too large?",
        "diverge",
        "gradient-descent",
    ),
    GoldenCase(
        "Why is cross-entropy better than squared error for classification?",
        "confident",
        "loss",
    ),
    GoldenCase(
        "How do you detect overfitting?",
        "test set",
        "overfitting",
    ),
    GoldenCase(
        "What are the steps of retrieval augmented generation?",
        "embed",
        "rag",
    ),
    GoldenCase(
        "Why is there an overlap between chunks?",
        "boundary",
        "chunking",
    ),
    GoldenCase(
        "How do you know a prompt change did not break anything?",
        "golden set",
        "evaluation",
    ),
]


@dataclass
class CaseResult:
    case: GoldenCase
    answer: str
    contains_fact: bool
    cites_source: bool

    @property
    def passed(self) -> bool:
        return self.contains_fact


# A citation has to LOOK like a citation. Checking for the word "source" is not
# enough - a model that says "I was not given a source" would score a hit, which
# is exactly the kind of false pass a sloppy scorer produces.
CITATION_RE = re.compile(r"\(source:\s*[^)]+\)")


def score(answer: str, case: GoldenCase) -> CaseResult:
    """
    Two crude, cheap, deterministic scores.

    Crude is fine. A scorer you can explain in one sentence is a scorer people
    will actually run on every change.
    """
    return CaseResult(
        case=case,
        answer=answer,
        contains_fact=case.expected_fact.lower() in answer.lower(),
        cites_source=bool(CITATION_RE.search(answer)),
    )


def retrieval_hit(case: GoldenCase, index: list[Chunk], k: int = 3) -> bool:
    """
    Did retrieval put the RIGHT document in the top k?

    This is the score people forget to compute, and it is the one that tells you
    whether to fix your chunking or your prompt. If the right chunk was never
    retrieved, no prompt can save the answer.
    """
    return any(chunk.doc_id == case.doc_id for _, chunk in retrieve(case.question, index, k))


def run_variant(name: str, answer_fn) -> list[CaseResult]:
    return [score(answer_fn(case), case) for case in GOLDEN_SET]


def summarise(name: str, results: list[CaseResult]) -> dict:
    passed = sum(1 for r in results if r.passed)
    cited = sum(1 for r in results if r.cites_source)
    return {
        "name": name,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "citation_rate": cited / len(results),
    }


def demo(plots_dir: Path | None = None) -> dict:
    title = "15. Evaluation"
    print()
    print(title)
    print("-" * len(title))

    index = build_index(knowledge.DOCUMENTS)

    grounded_llm = MockLLM(seed=2, allow_hallucination=False)
    ungrounded_llm = MockLLM(seed=2, allow_hallucination=True)

    variants = {
        "A: no retrieval": lambda case: answer_without_retrieval(ungrounded_llm, case.question),
        "B: RAG (retrieval + grounding)": lambda case: answer_with_retrieval(
            grounded_llm, case.question, index, k=3
        )[0],
    }

    results: dict[str, list[CaseResult]] = {}
    summaries: list[dict] = []

    for name, answer_fn in variants.items():
        case_results = run_variant(name, answer_fn)
        results[name] = case_results
        summaries.append(summarise(name, case_results))

    # ---- retrieval quality, on its own ------------------------------------
    hits = [retrieval_hit(case, index) for case in GOLDEN_SET]
    print(f"golden set: {len(GOLDEN_SET)} questions, each with a required fact")
    print(f"retrieval hit rate (right doc in top 3): {sum(hits)}/{len(hits)}\n")

    # ---- per-case table ---------------------------------------------------
    header = f"  {'question':<50} {'retr':>5} {'A':>5} {'B':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    names = list(results)
    for i, case in enumerate(GOLDEN_SET):
        a = results[names[0]][i]
        b = results[names[1]][i]
        question = case.question[:48]
        verdict = lambda ok: "PASS" if ok else "fail"  # noqa: E731
        print(
            f"  {question:<50} {verdict(hits[i]):>5} {verdict(a.passed):>5} {verdict(b.passed):>5}"
        )

    # ---- totals -----------------------------------------------------------
    print()
    print(f"  {'variant':<32} {'pass rate':>10} {'citations':>10}")
    for summary in summaries:
        print(
            f"  {summary['name']:<32} "
            f"{summary['passed']}/{summary['total']:<8} "
            f"{summary['citation_rate'] * 100:>8.0f}%"
        )

    baseline, improved = summaries
    gain = improved["pass_rate"] - baseline["pass_rate"]

    print()
    print(f"Grounding is worth {gain * 100:+.0f} percentage points of factual accuracy on")
    print("this set - and it is the difference between an answer you can check and")
    print("one you cannot.")

    retrieval_rate = sum(hits) / len(hits)
    print()
    print(f"But look at the three columns together. Retrieval found the right")
    print(f"document in {sum(hits)}/{len(hits)} cases, while the generated answer contained the")
    print(f"fact in {improved['passed']}/{improved['total']}. Retrieval is doing its job; the")
    print("GENERATOR is the weak link - it quotes one sentence and misses the rest.")
    print()
    print("That is why you score them separately. 'RAG is bad' is not a diagnosis.")
    print(
        f"'Retrieval {retrieval_rate * 100:.0f}% and generation {improved['pass_rate'] * 100:.0f}%' "
        "tells you exactly what to fix:"
    )
    print("the answer step, not the index.")

    # ---- regression check -------------------------------------------------
    regressions = [
        case.question
        for i, case in enumerate(GOLDEN_SET)
        if results[names[0]][i].passed and not results[names[1]][i].passed
    ]

    print()
    if regressions:
        print("REGRESSIONS (passed in A, failed in B):")
        for question in regressions:
            print(f"   - {question}")
    else:
        print("No regressions: every case that passed in A still passes in B.")

    print()
    print("That last check is the one that makes evals worth the effort. A change")
    print("that fixes three cases and silently breaks two is a net loss, and you")
    print("only see it if you run the whole set every time.")

    return {
        "summaries": summaries,
        "retrieval_hit_rate": retrieval_rate,
        "pass_rate_no_retrieval": baseline["pass_rate"],
        "pass_rate_rag": improved["pass_rate"],
        "citation_rate_no_retrieval": baseline["citation_rate"],
        "citation_rate_rag": improved["citation_rate"],
        "regressions": regressions,
        "per_case": [
            {
                "question": case.question,
                "no_retrieval": results[names[0]][i].passed,
                "rag": results[names[1]][i].passed,
            }
            for i, case in enumerate(GOLDEN_SET)
        ],
    }


if __name__ == "__main__":
    demo()
