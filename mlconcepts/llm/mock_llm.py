"""
A deterministic stand-in for a chat model.
=========================================

WHY THIS FILE EXISTS
--------------------
Part 2 is about the ENGINEERING around a language model - retrieval, prompting,
tool calling, evaluation. That engineering is the part you actually write, and it
should not require an API key, a network connection, or money to run.

So this file implements a fake model with the same *interface* as a real one:

    llm.chat(messages, tools=..., temperature=...) -> Response

with `messages` in the same role/content shape and a `Response` that carries text,
tool calls, a finish reason and token counts. Swap it for a real client and the
rest of Part 2 is unchanged.

WHAT IT IS, HONESTLY
--------------------
This is a SIMULATOR, not a model. It does not understand language. It pattern
matches: it looks for keywords, pulls sentences out of a supplied context, emits
tool calls for known verbs, and produces JSON when asked for JSON.

That is deliberate. The lessons in Part 2 are about the plumbing - chunking,
retrieval, prompt structure, the tool loop, evals - and those are exactly as real
against a simulator as against GPT. What you should NOT take away from this file
is anything about model quality.

The one genuinely real piece of maths here is `softmax` / `temperature` in
`sample_token`: that is the actual sampling rule, and the tokenization demo uses
it for real.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# The message shapes - deliberately the same as the real APIs
# --------------------------------------------------------------------------


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        out: dict = {"role": self.role, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------
# Token counting and sampling (the honest parts)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """
    A rough token count.

    Real models use a learned subword vocabulary (see the tokenization demo), so
    a real count would come from that. This approximation - words plus separate
    punctuation - is close enough for reasoning about prompt sizes, and it is
    what the simulator reports.
    """
    return len(_TOKEN_RE.findall(text))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Turn raw scores into a probability distribution.

    Temperature is the dial: as it goes to 0 the distribution becomes a spike on
    the largest logit (deterministic); as it grows the distribution flattens
    (random). This is the actual rule every sampling API uses.
    """
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=float)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / temperature
    scaled -= scaled.max()  # numerical stability
    exps = np.exp(scaled)
    return exps / exps.sum()


def sample_token(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    """Sample one token index, with temperature and top-k filtering."""
    masked = logits.astype(float).copy()
    if top_k > 0 and top_k < len(masked):
        keep = np.argsort(masked)[-top_k:]
        mask = np.full_like(masked, -np.inf)
        mask[keep] = masked[keep]
        masked = mask
    probs = softmax(masked, temperature)
    return int(rng.choice(len(probs), p=probs))


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


# Words that make the simulator reach for a tool. A real model decides this from
# the tool descriptions; ours keys off verbs.
_TOOL_TRIGGERS = {
    "get_weather": ("weather", "temperature outside", "raining", "forecast"),
    "calculator": ("calculate", "what is", "how much is", "compute", "+", "*"),
    "search_docs": ("search", "look up", "find the", "documentation"),
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Canned phrasings, used to make temperature visibly change the output.
_ACKNOWLEDGEMENTS = (
    "Got it - here is my take.",
    "Understood. Short answer:",
    "Sure - here is what I can tell you.",
    "Okay, let me summarise that.",
)


class MockLLM:
    """
    A fake chat model with a real interface.

    Parameters
    ----------
    name:
        Reported as the model name, so logs look like a real integration.
    seed:
        Sampling is seeded, so a run is reproducible.
    allow_hallucination:
        When no context is supplied and the question is unknown, whether to
        invent a confident-sounding answer. The RAG demo turns this ON on
        purpose, to show what happens without retrieval.
    """

    def __init__(self, name: str = "mock-1", seed: int = 0, allow_hallucination: bool = False):
        self.name = name
        self.seed = seed
        self.allow_hallucination = allow_hallucination
        self.calls = 0
        self._rng = np.random.default_rng(seed)

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> Response:
        self.calls += 1

        prompt_tokens = sum(count_tokens(m.content) for m in messages)
        system = next((m.content for m in messages if m.role == "system"), "")
        user_messages = [m for m in messages if m.role == "user"]
        tool_messages = [m for m in messages if m.role == "tool"]
        user = user_messages[-1].content if user_messages else ""

        # What the user ORIGINALLY asked decides which tools are needed. After a
        # tool call, the latest message is the tool's own output - using that
        # would lose the original request and the loop would stop after one step.
        original_user = user_messages[0].content if user_messages else ""
        already_called = {m.name for m in tool_messages if m.name}

        # 1. Tools are available and something still needs running.
        if tools:
            call = self._maybe_tool_call(original_user, tools, skip=already_called)
            if call is not None:
                return Response(
                    text="",
                    tool_calls=[call],
                    finish_reason="tool_calls",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                )

        # 2. Tool results came back and nothing else is pending: answer from them.
        if tool_messages:
            text = self._summarise_all(tool_messages, original_user)
            return self._respond(text, prompt_tokens, max_tokens)

        # 3. Structured output was requested.
        if "json" in (system + user).lower():
            text = self._structured_output(user)
            return self._respond(text, prompt_tokens, max_tokens)

        # 4. A context was supplied (RAG): answer FROM the context.
        context = self._extract_context(system, user)
        if context is not None:
            text = self._answer_from_context(context, user)
            return self._respond(text, prompt_tokens, max_tokens)

        # 5. Nothing to ground on.
        text = self._answer_ungrounded(user, temperature)
        return self._respond(text, prompt_tokens, max_tokens)

    # -- internals ----------------------------------------------------------

    def _respond(self, text: str, prompt_tokens: int, max_tokens: int) -> Response:
        completion_tokens = count_tokens(text)
        truncated = completion_tokens > max_tokens
        if truncated:
            # Cut on a word boundary, like a real max_tokens stop would.
            words = text.split()[:max_tokens]
            text = " ".join(words)
            completion_tokens = count_tokens(text)
        return Response(
            text=text,
            finish_reason="length" if truncated else "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _maybe_tool_call(
        self, user: str, tools: list[dict], skip: set[str] | None = None
    ) -> ToolCall | None:
        """The first tool whose trigger word appears and which has not run yet."""
        available = {t["name"] for t in tools}
        already = skip or set()
        lowered = user.lower()

        for tool_name, triggers in _TOOL_TRIGGERS.items():
            if tool_name not in available or tool_name in already:
                continue
            if any(trigger in lowered for trigger in triggers):
                arguments = self._tool_arguments(tool_name, user)
                if arguments is None:
                    continue
                return ToolCall(
                    id=f"call_{self.calls}_{tool_name}",
                    name=tool_name,
                    arguments=arguments,
                )
        return None

    def _tool_arguments(self, tool_name: str, user: str) -> dict | None:
        if tool_name == "get_weather":
            # Prefer the word after "in" ("the weather in Chennai"), because the
            # first capitalised word in a question is usually "What".
            match = re.search(r"\bin\s+([A-Za-z]+)", user) or re.search(
                r"\b([A-Z][a-zA-Z]+)\b", user
            )
            return {"city": match.group(1).capitalize() if match else "Chennai"}

        if tool_name == "calculator":
            expression = re.sub(r"[^0-9+\-*/(). ]", " ", user)
            expression = expression.strip()
            return {"expression": expression or "0"}

        if tool_name == "search_docs":
            return {"query": user.strip()[:80]}

        return None

    def _summarise_all(self, tool_messages: list[Message], user: str) -> str:
        """Combine every tool result into one answer, in the order they ran."""
        parts = [self._summarise_tool_result(message, user) for message in tool_messages]
        return " ".join(part for part in parts if part) or "No tool results."

    def _summarise_tool_result(self, tool_message: Message, user: str) -> str:
        result = tool_message.content
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return f"The tool returned: {result}"

        name = tool_message.name or "the tool"
        if name == "calculator" and "result" in parsed:
            return f"{parsed.get('expression', '')} = {parsed['result']}"
        if name == "get_weather" and "summary" in parsed:
            return parsed["summary"]
        if name == "search_docs" and "results" in parsed:
            hits = parsed["results"]
            return f"Found {len(hits)} matching passage(s); the top one is: {hits[0]}" if hits else "No matches."
        return f"{name} returned {json.dumps(parsed)[:120]}"

    def _structured_output(self, user: str) -> str:
        """
        Pull a name and an email out of the text with regex and return JSON.

        A real model would do this by understanding the request; the SHAPE of the
        answer - a JSON object matching a schema - is what the demo is about.
        """
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", user)
        name = re.search(r"\b(?:my name is|I am|I'm)\s+([A-Z][a-z]+)", user)
        city = re.search(r"\b(?:in|from)\s+([A-Z][a-z]+)\b", user)

        payload = {
            "name": name.group(1) if name else None,
            "email": email.group(0) if email else None,
            "city": city.group(1) if city else None,
        }
        return json.dumps(payload)

    def _extract_context(self, system: str, user: str) -> str | None:
        """
        Find a supplied context block.

        The convention used throughout Part 2 is:

            Context:
            <passages>
            Question: <the question>

        This is the shape a RAG prompt takes, and it is what a real model would be
        told to answer from.
        """
        combined = f"{system}\n{user}"
        if "Context:" not in combined:
            return None
        after = combined.split("Context:", 1)[1]
        context = after.split("Question:", 1)[0]
        return context.strip() or None

    def _answer_from_context(self, context: str, question: str) -> str:
        """
        Pick the sentence in the context that overlaps the question most, and
        cite the passage it came from.

        The passages arrive as `[1] (doc#0) text ... [2] (doc#1) text ...`, which
        is the shape `rag.build_rag_prompt` produces. Reading the label back out
        means the citation points at the real chunk instead of an arbitrary
        sentence number - the whole value of grounding is that the user can go
        and check the source.
        """
        marker = re.compile(r"\[(\d+)\]\s*\(([^)]+)\)")
        pieces = marker.split(context)

        passages: list[tuple[str, str]] = []
        # split() with one group pair yields [prefix, n1, label1, text1, n2, ...]
        for i in range(1, len(pieces) - 2, 3):
            passages.append((pieces[i + 1], pieces[i + 2].strip()))
        if not passages:
            passages = [("1", context.strip())]

        question_words = {w.lower() for w in _TOKEN_RE.findall(question) if len(w) > 2}
        scored: list[tuple[float, str, str]] = []

        for label, text in passages:
            for sentence in _SENTENCE_SPLIT.split(text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                words = {w.lower() for w in _TOKEN_RE.findall(sentence) if len(w) > 2}
                overlap = len(words & question_words) / (len(question_words) or 1)
                scored.append((overlap, label, sentence))

        if not scored:
            return "The provided context was empty."

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_label, best_sentence = scored[0]

        if best_score == 0:
            return "The context does not appear to answer that question."

        return f"{best_sentence} (source: {best_label})"

    def _answer_ungrounded(self, user: str, temperature: float) -> str:
        if self.allow_hallucination:
            # The whole point of the RAG demo: a confident answer with no basis.
            return (
                "That is covered in section 4.2 of the handbook: the value is 42, "
                "and it applies from the next release. (No source - I was not given one.)"
            )

        if temperature <= 0:
            opener = _ACKNOWLEDGEMENTS[0]
        else:
            # Temperature > 0 genuinely changes which phrasing comes out.
            index = int(self._rng.integers(len(_ACKNOWLEDGEMENTS)))
            opener = _ACKNOWLEDGEMENTS[index]

        return f"{opener} You asked about: {user.strip()[:120]}"

    # -- introspection ------------------------------------------------------

    def __repr__(self) -> str:
        return f"MockLLM(name={self.name!r}, calls={self.calls})"
