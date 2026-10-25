"""
All site content, in one place.

The build script imports CONCEPTS and INTERVIEW_TOPICS from here and renders them.
Nothing about the HTML lives in this package - content and presentation stay apart,
so you can rewrite a page without touching the generator.
"""

from __future__ import annotations

from .interview import EXTRA_TOPICS
from .part1 import PART_1
from .part2 import PART_2

CONCEPTS: list[dict] = PART_1 + PART_2

PARTS: dict[int, dict] = {
    1: {
        "number": 1,
        "title": "Part 1 - Foundations",
        "blurb": "The algorithms, written from scratch with NumPy. No scikit-learn, no PyTorch, "
        "no hidden steps: every number on these pages came out of code you can read.",
        "note": "Run any one of them with `python -m mlconcepts.foundations.NAME` (for example, "
        "`python -m mlconcepts.foundations.pca`), or all of them at once with `python run_all.py`.",
    },
    2: {
        "number": 2,
        "title": "Part 2 - LLM engineering",
        "blurb": "The engineering around a language model: embeddings, tokenization, prompting, "
        "retrieval, tool calling and evaluation.",
        "note": "These run against a deterministic mock model, so they need no API key and no network. "
        "What is real is the plumbing - and the plumbing is the part you write.",
    },
}


def interview_topics() -> list[dict]:
    """
    Build the interview page: every concept's questions, grouped, plus the
    cross-cutting topics.

    Grouping by concept rather than by keyword is deliberate - when a question
    comes up in an interview you want to remember which PAGE it belongs to, so
    you can go and re-read the code.
    """
    topics: list[dict] = []

    for concept in CONCEPTS:
        if not concept.get("interview"):
            continue
        topics.append(
            {
                "topic": f"{concept['number']}. {concept['title']}",
                "part": concept["part"],
                "slug": concept["slug"],
                "questions": concept["interview"],
            }
        )

    for extra in EXTRA_TOPICS:
        topics.append(
            {
                "topic": extra["topic"],
                "part": 3,
                "slug": None,
                "questions": extra["questions"],
            }
        )

    return topics


def total_questions() -> int:
    return sum(len(topic["questions"]) for topic in interview_topics())
