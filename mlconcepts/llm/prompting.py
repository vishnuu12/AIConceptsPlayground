"""
12. Prompting and structured output
===================================

WHY THIS PAGE EXISTS
--------------------
A prompt is not a sentence you type at a model - it is a program. It has parts
with different jobs, and the order and framing change the output:

    system   who the model is, the rules it must follow, the output contract
    user     the actual request, and any data it needs
    assistant few-shot examples of the answer you want
    tool     results of tools the model asked you to run

Two things this demo shows that people get wrong all the time:

1. FEW-SHOT EXAMPLES COST TOKENS, AND THEY ARE WORTH IT. Two examples can pin
   down a format better than a paragraph of instructions - and you pay for them
   on every single request.

2. NEVER TRUST THE OUTPUT TO PARSE. Ask for JSON, then parse it inside a
   try/except, then VALIDATE it against your schema. When it fails, send the
   validation error back and try again (a repair loop) - with a hard attempt
   limit, because some failures cannot be repaired.
"""

from __future__ import annotations

import json
from pathlib import Path

from .mock_llm import Message, MockLLM, count_tokens


# The contract we want the model to satisfy.
REQUIRED_FIELDS = ("name", "email", "city")


def validate(payload: object) -> list[str]:
    """
    Check the parsed object against the schema. Returns a list of problems.

    This is the step people skip. `json.loads` succeeding does NOT mean the
    object is usable - it can be missing fields, or have nulls where you needed
    strings, and that failure shows up much later as a KeyError in production.
    """
    problems: list[str] = []

    if not isinstance(payload, dict):
        return ["output was not a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in payload:
            problems.append(f"missing field '{field}'")
        elif payload[field] in (None, ""):
            problems.append(f"field '{field}' is empty")

    if isinstance(payload.get("email"), str) and "@" not in payload["email"]:
        problems.append("email does not look like an email")

    return problems


def build_messages(text: str, few_shot: bool = True) -> list[Message]:
    """Assemble the prompt. Note how much of it is not the actual request."""
    system = (
        "You are a data extraction service. "
        "Return ONLY a JSON object with exactly these keys: name, email, city. "
        "Use null for anything you cannot find. No prose, no markdown fences."
    )

    messages = [Message(role="system", content=system)]

    if few_shot:
        # Two examples that pin the format down. Real prompts are mostly this.
        messages.append(
            Message(
                role="user",
                content="My name is Asha and you can reach me at asha@example.com. I am in Pune.",
            )
        )
        messages.append(
            Message(
                role="assistant",
                content=json.dumps({"name": "Asha", "email": "asha@example.com", "city": "Pune"}),
            )
        )

    messages.append(Message(role="user", content=text))
    return messages


def extract(
    llm: MockLLM, text: str, few_shot: bool = True, max_attempts: int = 3
) -> tuple[dict | None, list[Message], list[str]]:
    """
    Ask, parse, validate, and repair.

    Returns (payload, final_messages, attempts_log). `payload` is None when we
    gave up - which is a real outcome, not an error: at that point you show the
    user a validation message rather than looping forever.
    """
    messages = build_messages(text, few_shot)
    log: list[str] = []

    for attempt in range(1, max_attempts + 1):
        response = llm.chat(messages, temperature=0.0)

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as error:
            payload = None
            problems = [f"not valid JSON ({error.msg})"]
        else:
            problems = validate(payload)

        log.append(f"attempt {attempt}: {'ok' if not problems else '; '.join(problems)}")

        if not problems:
            return payload, messages, log

        # The repair turn: hand back the exact problem AND the original data, so
        # the model has everything it needs to fix it.
        messages.append(Message(role="assistant", content=response.text))
        messages.append(
            Message(
                role="user",
                content=(
                    f"That output was rejected: {'; '.join(problems)}. "
                    f"Return corrected JSON for this text: {text}"
                ),
            )
        )

    return None, messages, log


def demo(plots_dir: Path | None = None) -> dict:
    llm = MockLLM(name="mock-1")

    title = "12. Prompting and structured output"
    print()
    print(title)
    print("-" * len(title))

    # ---- the prompt itself ------------------------------------------------
    sample = "Hi, I'm Rohith from Chennai - rohith.k@example.com works best."
    with_examples = build_messages(sample, few_shot=True)
    without_examples = build_messages(sample, few_shot=False)

    print("The prompt, message by message:")
    for message in with_examples:
        preview = message.content.replace("\n", " ")[:76]
        print(f"   {message.role:<9} {preview}")

    cost_with = sum(count_tokens(m.content) for m in with_examples)
    cost_without = sum(count_tokens(m.content) for m in without_examples)
    print()
    print(f"prompt tokens with few-shot examples : {cost_with}")
    print(f"prompt tokens without                : {cost_without}")
    print(f"cost of the two examples             : +{cost_with - cost_without} tokens, on EVERY call")

    # ---- a clean extraction ----------------------------------------------
    print()
    print("Extraction on a well-formed request:")
    payload, _, log = extract(llm, sample)
    for line in log:
        print(f"   {line}")
    print(f"   -> {json.dumps(payload)}")

    # ---- a request that cannot be satisfied ------------------------------
    print()
    print("Now a request that is missing a required field:")
    broken = "I'm Vinith, no contact details, somewhere in Bangalore."
    payload, messages, log = extract(llm, broken)
    for line in log:
        print(f"   {line}")
    print(f"   -> {json.dumps(payload)}")
    print()
    print("The repair loop ran, and still failed - correctly. No amount of")
    print("retrying invents an email address. This is where you return a")
    print("validation error to the user instead of looping.")

    # ---- token cost of a repair ------------------------------------------
    repair_tokens = sum(count_tokens(m.content) for m in messages)
    print()
    print(f"the full repaired conversation cost : {repair_tokens} prompt tokens")
    print(f"the first clean attempt cost        : {cost_with} prompt tokens")
    print("Retries are not free, and they grow the conversation every time.")

    return {
        "prompt_tokens_few_shot": cost_with,
        "prompt_tokens_zero_shot": cost_without,
        "clean_payload": json.loads(json.dumps(extract(MockLLM(), sample)[0])),
        "broken_payload": payload,
        "attempts": log,
        "repair_tokens": repair_tokens,
    }


if __name__ == "__main__":
    demo()
