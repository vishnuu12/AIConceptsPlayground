"""
13. Retrieval augmented generation
==================================

WHY THIS PAGE EXISTS
--------------------
A model only knows what is in its parameters, and its parameters are frozen.
Ask it about your documentation and it has two options: admit it does not know,
or invent something that sounds right. Retrieval fixes that by putting the
relevant text into the prompt.

    chunk -> embed -> store -> retrieve by similarity -> ground the prompt

Every step is real code here: the chunking, the embeddings, the cosine ranking,
the prompt assembly, the citation. The only simulated part is the model, and it
is simulated honestly - it answers strictly from the context it is given, and
when it is given none it hallucinates on purpose so you can see the difference.

WHY IT REDUCES HALLUCINATION
----------------------------
Not because the model becomes truthful - because you stopped asking it to
remember. You gave it the source text and told it to answer from that. The
failure mode changes from "confidently invented" to "the retrieved passage did
not contain the answer", which is a much easier failure to detect and fix.

THE TWO THINGS THAT BREAK IN PRACTICE
-------------------------------------
1. CHUNKING. Too big and the embedding is diluted by irrelevant sentences, so
   retrieval matches the wrong chunk. Too small and the answer's context is cut
   in half. That is why there is an overlap.
2. RETRIEVAL QUALITY. If the right chunk is not in the top k, no amount of
   prompt engineering saves you. This is why retrieval is evaluated separately
   from generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import knowledge
from .embeddings import cosine_similarity, embed
from .mock_llm import Message, MockLLM


@dataclass
class Chunk:
    doc_id: str
    title: str
    index: int
    text: str
    vector: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.doc_id}#{self.index}"


# --------------------------------------------------------------------------
# 1. Chunking
# --------------------------------------------------------------------------


def chunk_text(text: str, size: int = 45, overlap: int = 12) -> list[str]:
    """
    Split on words with an overlap between neighbours.

    The overlap is the important detail: without it, a sentence that straddles a
    boundary appears in neither chunk, and that is exactly the sentence someone
    will ask about.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or chunking never advances")

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap  # step back so the next chunk overlaps

    return chunks


def build_index(documents: list[dict], size: int = 45, overlap: int = 12) -> list[Chunk]:
    """Chunk every document and embed every chunk."""
    index: list[Chunk] = []
    for document in documents:
        for i, text in enumerate(chunk_text(document["text"], size, overlap)):
            index.append(
                Chunk(
                    doc_id=document["id"],
                    title=document["title"],
                    index=i,
                    text=text,
                    vector=embed(text),
                )
            )
    return index


# --------------------------------------------------------------------------
# 2. Retrieval
# --------------------------------------------------------------------------


def retrieve(question: str, index: list[Chunk], k: int = 3) -> list[tuple[float, Chunk]]:
    """Rank chunks by cosine similarity to the question, best first."""
    question_vector = embed(question)
    scored = [(cosine_similarity(question_vector, chunk.vector), chunk) for chunk in index]
    scored.sort(key=lambda pair: -pair[0])
    return scored[:k]


# --------------------------------------------------------------------------
# 3. Grounding the prompt
# --------------------------------------------------------------------------


def build_rag_prompt(question: str, passages: list[Chunk]) -> list[Message]:
    """
    The shape that matters. The context is fenced, numbered, and the model is
    told to answer from it and cite the number. Without the fence, models happily
    treat retrieved text as instructions - which is how prompt injection works.
    """
    context = "\n".join(f"[{i + 1}] ({p.label}) {p.text}" for i, p in enumerate(passages))

    system = (
        "Answer the question using ONLY the context below. "
        "Cite the passage number you used. If the context does not contain the "
        "answer, say so - do not use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    return [Message(role="system", content=system), Message(role="user", content=user)]


def answer_with_retrieval(
    llm: MockLLM, question: str, index: list[Chunk], k: int = 3
) -> tuple[str, list[tuple[float, Chunk]]]:
    passages = retrieve(question, index, k)
    messages = build_rag_prompt(question, [chunk for _, chunk in passages])
    response = llm.chat(messages, temperature=0.0)
    return response.text, passages


def answer_without_retrieval(llm: MockLLM, question: str) -> str:
    """The same question, with nothing retrieved - the baseline."""
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=question),
    ]
    return llm.chat(messages, temperature=0.0).text


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------


def demo(plots_dir: Path | None = None) -> dict:
    title = "13. Retrieval augmented generation"
    print()
    print(title)
    print("-" * len(title))

    index = build_index(knowledge.DOCUMENTS)
    sizes = [len(chunk.text.split()) for chunk in index]

    print(f"documents      : {len(knowledge.DOCUMENTS)}")
    print(f"chunks         : {len(index)}  (size 45 words, overlap 12)")
    print(f"chunk sizes    : min {min(sizes)}, max {max(sizes)} words")

    # Show one document splitting, so the overlap is visible.
    example_doc = knowledge.DOCUMENTS[0]
    example_chunks = chunk_text(example_doc["text"], 45, 12)
    print()
    print(f"how '{example_doc['id']}' splits into {len(example_chunks)} chunks:")
    for i, text in enumerate(example_chunks):
        print(f"   [{i}] {text[:64]}...")
    if len(example_chunks) > 1:
        tail = example_chunks[0].split()[-4:]
        head = example_chunks[1].split()[:4]
        print(f"   overlap between [0] and [1]: ...{' '.join(tail)} | {' '.join(head)}...")

    question = "What is the learning rate and what happens if it is too large?"

    print()
    print(f"question: {question!r}")
    passages = retrieve(question, index, k=3)
    print("retrieved (cosine similarity):")
    for score, chunk in passages:
        print(f"   {score:.3f}  {chunk.label:<18} {chunk.title}")

    llm = MockLLM(allow_hallucination=False)
    grounded, _ = answer_with_retrieval(llm, question, index)

    print()
    print("--- WITH retrieval (grounded) ---")
    print(f"   {grounded}")

    hallucinating = MockLLM(allow_hallucination=True)
    ungrounded = answer_without_retrieval(hallucinating, question)

    print()
    print("--- WITHOUT retrieval (same question) ---")
    print(f"   {ungrounded}")

    print()
    print("Same question, same model, same parameters. The only difference is")
    print("that one of them was handed the source text.")
    print()
    print("To make retrieval better you do NOT start by rewriting the prompt. You")
    print("measure retrieval separately: is the correct chunk in the top k? The")
    print("evaluation demo does exactly that, over a golden set.")

    cited = "source:" in grounded.lower() or "[1]" in grounded

    return {
        "n_chunks": len(index),
        "chunk_sizes": sizes,
        "question": question,
        "retrieved": [(score, chunk.label) for score, chunk in passages],
        "top_score": passages[0][0],
        "top_doc": passages[0][1].doc_id,
        "grounded_answer": grounded,
        "ungrounded_answer": ungrounded,
        "cites_source": cited,
    }


if __name__ == "__main__":
    demo()
