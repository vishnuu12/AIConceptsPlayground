"""
Tests for Part 2 - the LLM engineering.

The model is a simulator, so these tests are not about answer quality. They are
about the PLUMBING, which is real:

    cosine similarity behaves like a similarity
    the BPE tokenizer merges what it should
    temperature genuinely changes the distribution and top-k genuinely filters
    the structured-output validator rejects bad payloads and the repair loop
    terminates instead of spinning
    chunking overlaps, retrieval ranks the right document first
    the agent loop is bounded and a tool error does not crash it
    safe_eval refuses anything that is not arithmetic
    the eval harness measures a real difference between two variants
"""

from __future__ import annotations

import json

import numpy as np

from mlconcepts.llm import knowledge
from mlconcepts.llm.embeddings import cosine_similarity, embed, nearest
from mlconcepts.llm.evaluation import GOLDEN_SET, demo as evaluation_demo
from mlconcepts.llm.evaluation import retrieval_hit
from mlconcepts.llm.mock_llm import Message, MockLLM, softmax
from mlconcepts.llm.prompting import extract, validate
from mlconcepts.llm.rag import (
    answer_with_retrieval,
    answer_without_retrieval,
    build_index,
    chunk_text,
    retrieve,
)
from mlconcepts.llm.tokenization import encode, entropy, tokenize, top_k_filter, train_bpe
from mlconcepts.llm.tools_agent import safe_eval, run_agent


# -------------------------------------------------------------- 10. embeddings


def test_cosine_similarity_is_a_similarity():
    a = embed("gradient descent and the learning rate")
    b = embed("the learning rate controls the step size")
    c = embed("k-means groups unlabelled points")

    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    assert -1.0 <= cosine_similarity(a, b) <= 1.0
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


def test_embedding_is_normalised_and_stable_across_calls():
    first = embed("the model learns from the data")
    second = embed("the model learns from the data")

    assert np.allclose(first, second), "embeddings must be deterministic"
    assert abs(np.linalg.norm(first) - 1.0) < 1e-9


def test_nearest_returns_the_matching_sentence():
    corpus = [
        "Gradient descent moves the parameters downhill along the loss.",
        "An embedding turns text into a vector so similarity becomes geometry.",
        "Tokenization splits text into the subword units a model reads.",
    ]
    top_score, top_text = nearest("how does cosine similarity work?", corpus, k=1)[0]
    assert "embedding" in top_text
    assert top_score > 0


# ------------------------------------------------------------ 11. tokenization


def test_bpe_learns_whole_common_words():
    merges = train_bpe("the model learns. the model predicts. the model is trained.", 40)
    assert encode("model", merges) == ["model</w>"]
    assert len(encode("model", merges)) < len(encode("models", merges))


def test_rare_words_cost_more_tokens_than_common_ones():
    merges = train_bpe("the model learns from the data and the model is trained", 40)
    assert len(tokenize("model", merges)) <= len(tokenize("unbelievable", merges))


def test_temperature_changes_the_distribution():
    logits = np.array([3.0, 2.0, 1.0, 0.0])

    cold = softmax(logits, 0.1)
    warm = softmax(logits, 1.0)
    hot = softmax(logits, 5.0)

    assert cold[0] > 0.99, "a near-zero temperature should be almost deterministic"
    assert entropy(cold) < entropy(warm) < entropy(hot)
    assert abs(softmax(logits, 1.0).sum() - 1.0) < 1e-9


def test_top_k_filters_to_exactly_k_tokens():
    logits = np.array([4.0, 3.0, 2.0, 1.0, 0.0])

    top1 = softmax(top_k_filter(logits, 1), 1.0)
    assert int(np.argmax(top1)) == 0
    assert (top1 > 0).sum() == 1

    top3 = softmax(top_k_filter(logits, 3), 1.0)
    assert (top3 > 0).sum() == 3
    assert (top3[-2:] == 0).all()


# --------------------------------------------------------------- 12. prompting


def test_validator_rejects_incomplete_payloads():
    assert validate({"name": "A", "email": "a@b.com", "city": "Pune"}) == []
    assert validate({"name": "A", "email": "a@b.com"})  # missing city
    assert validate({"name": "A", "email": "", "city": "Pune"})  # empty
    assert validate({"name": "A", "email": "not-an-email", "city": "Pune"})
    assert validate("just a string")
    assert validate(None)


def test_extraction_succeeds_and_repair_loop_terminates():
    llm = MockLLM()

    payload, _, log = extract(llm, "Hi, I'm Rohith from Chennai - r@example.com works.")
    assert payload is not None
    assert payload["name"] == "Rohith"
    assert payload["email"] == "r@example.com"
    assert len(log) == 1, "a clean extraction should take one attempt"

    # An impossible request must GIVE UP, not loop forever.
    broken, _, log = extract(llm, "I am Vinith, somewhere in Bangalore.", max_attempts=3)
    assert broken is None
    assert len(log) == 3, "the attempt limit must be respected"


# --------------------------------------------------------------------- 13. RAG


def test_chunking_produces_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, size=30, overlap=10)

    assert len(chunks) == 5
    first_tail = chunks[0].split()[-10:]
    second_head = chunks[1].split()[:10]
    assert first_tail == second_head, "the overlap should be identical text"


def test_chunking_rejects_an_impossible_overlap():
    try:
        chunk_text("a b c d", size=2, overlap=2)
    except ValueError:
        return
    raise AssertionError("overlap >= size should raise, or chunking never advances")


def test_retrieval_ranks_the_right_document_first():
    index = build_index(knowledge.DOCUMENTS)

    score, chunk = retrieve("What does ridge regularisation do?", index, k=1)[0]
    assert chunk.doc_id == "overfitting", f"got {chunk.doc_id}"
    assert score > 0


def test_grounded_answers_cite_and_ungrounded_ones_do_not():
    index = build_index(knowledge.DOCUMENTS)
    question = "Why is cross-entropy used for classification?"

    grounded, _ = answer_with_retrieval(MockLLM(), question, index)
    ungrounded = answer_without_retrieval(MockLLM(allow_hallucination=True), question)

    assert "(source:" in grounded
    assert "(source:" not in ungrounded
    assert grounded != ungrounded


# --------------------------------------------------------------- 14. tool loop


def test_safe_eval_does_arithmetic():
    assert safe_eval("2400 * 0.15") == 360.0
    assert safe_eval("(2 + 3) * 4") == 20.0
    assert safe_eval("-5 + 2") == -3.0


def test_safe_eval_refuses_anything_that_is_not_arithmetic():
    """
    The reason this function exists instead of eval(). Model output is untrusted.

    These inputs cover each category safe_eval has to reject: a function call, an
    attribute lookup, a bare name, a conditional, and a container literal. A real
    attack would try to import a module and run a command - which is precisely
    why the tool parses the expression and walks a whitelist of node types
    instead of handing the string to eval().
    """
    for hostile in (
        "print(1)",            # a call
        "(1).__class__",       # attribute access
        "x + 1",               # a name, not a number
        "1 if True else 2",    # a conditional
        "[1, 2]",              # a container
    ):
        try:
            safe_eval(hostile)
        except (ValueError, SyntaxError):
            continue
        raise AssertionError(f"safe_eval should have refused: {hostile}")


def test_agent_loop_runs_multiple_tools_and_terminates():
    llm = MockLLM(seed=1)
    answer, trace = run_agent(llm, "What is 2400 * 0.15, and what is the weather in Chennai?")

    tools_used = [entry["tool"] for entry in trace if entry["kind"] == "tool_call"]
    assert "calculator" in tools_used
    assert "get_weather" in tools_used
    assert trace[-1]["kind"] == "answer"
    assert "360" in answer
    assert "Chennai" in answer


def test_a_failing_tool_does_not_crash_the_loop():
    llm = MockLLM(seed=1)
    answer, trace = run_agent(llm, "What is 5 apples plus 3 oranges?")

    tool_entries = [entry for entry in trace if entry["kind"] == "tool_call"]
    assert tool_entries, "the calculator should have been called"
    assert "error" in tool_entries[0]["result"]
    assert answer  # we still got an answer back


def test_agent_loop_respects_max_steps():
    class EndlessLLM(MockLLM):
        def chat(self, messages, tools=None, temperature=0.0, max_tokens=256):  # type: ignore[override]
            from mlconcepts.llm.mock_llm import Response, ToolCall

            return Response(
                tool_calls=[ToolCall(id="x", name="get_weather", arguments={"city": "Chennai"})],
                finish_reason="tool_calls",
            )

    _, trace = run_agent(EndlessLLM(), "weather", max_steps=3)
    assert trace[-1]["kind"] == "gave_up", "a runaway loop must be stopped"


# --------------------------------------------------------------- 15. evaluation


def test_retrieval_hit_rate_is_measurable():
    index = build_index(knowledge.DOCUMENTS)
    hits = [retrieval_hit(case, index) for case in GOLDEN_SET]
    assert sum(hits) >= len(hits) // 2, "retrieval should find most of them"


def test_the_eval_harness_finds_a_real_difference():
    report = evaluation_demo()

    assert report["pass_rate_rag"] > report["pass_rate_no_retrieval"], (
        "grounding should beat no-retrieval on factual accuracy"
    )
    assert report["citation_rate_rag"] > report["citation_rate_no_retrieval"]
    assert len(report["per_case"]) == len(GOLDEN_SET)
    assert report["regressions"] == [], "the improved variant should not regress"
