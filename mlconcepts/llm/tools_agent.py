"""
14. Tool calling and a small agent loop
=======================================

WHY THIS PAGE EXISTS
--------------------
A model cannot check the weather, do arithmetic reliably, or read your database.
Tool calling is how you let it ask you to do those things:

    1. you describe your tools to the model (name, purpose, JSON schema)
    2. the model replies with a TOOL CALL instead of prose
    3. YOU run the function - the model never executes anything
    4. you append the result as a `tool` message and call the model again
    5. repeat until it answers in prose, or you hit the step limit

That loop is an "agent". It is not magic and it is not autonomous: it is a while
loop with a budget, and every real agent framework is a variation on it.

THE THREE THINGS THAT BITE YOU
------------------------------
1. THE MODEL NEVER RUNS CODE. You do. Every argument is untrusted input, which
   is why `calculator` below does not call eval() - it parses the expression
   with the `ast` module and allows only numbers and arithmetic.
2. LOOPS MUST BE BOUNDED. A model can ask for the same tool forever. `max_steps`
   is not optional, and neither is noticing when it is hit.
3. TOOLS FAIL. A bad expression, a timeout, a 404 - that error has to go back to
   the model as a result, not crash your loop. A model that is told the tool
   failed can choose something else; a crashed loop cannot.
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

from . import knowledge
from .embeddings import cosine_similarity, embed
from .mock_llm import Message, MockLLM, ToolCall


# --------------------------------------------------------------------------
# Tool schemas - what the model is shown
# --------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search the documentation for a phrase.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


# --------------------------------------------------------------------------
# Tool implementations - what actually runs
# --------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """
    Evaluate arithmetic WITHOUT eval().

    Model output is untrusted input. `eval("__import__('os').system('rm -rf /')")`
    is a real attack, and a calculator tool is the classic place it is attempted.
    Parsing the expression and walking a whitelist of node types makes it
    impossible: anything that is not a number or an arithmetic operator raises.
    """

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return walk(ast.parse(expression, mode="eval"))


def calculator(expression: str) -> dict:
    try:
        return {"expression": expression, "result": safe_eval(expression)}
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        # A tool failure is a RESULT, not an exception that kills the loop.
        return {"expression": expression, "error": str(error)}


_WEATHER = {
    "chennai": ("34C", "humid, light breeze"),
    "bangalore": ("27C", "cloudy"),
    "mumbai": ("31C", "heavy rain"),
    "delhi": ("38C", "hazy"),
}


def get_weather(city: str) -> dict:
    key = city.strip().lower()
    if key not in _WEATHER:
        return {"city": city, "error": f"no data for '{city}'"}
    temperature, conditions = _WEATHER[key]
    return {"city": city, "summary": f"{city} is {temperature} and {conditions}."}


def search_docs(query: str) -> dict:
    query_vector = embed(query)
    scored = [
        (cosine_similarity(query_vector, embed(document["text"])), document)
        for document in knowledge.DOCUMENTS
    ]
    scored.sort(key=lambda pair: -pair[0])
    return {
        "query": query,
        "results": [
            f"{document['title']} ({score:.2f})" for score, document in scored[:3]
        ],
    }


TOOL_IMPLEMENTATIONS = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_docs": search_docs,
}


def execute_tool(call: ToolCall) -> tuple[str, dict]:
    """
    Run one tool call. Returns (json string for the model, the raw result).

    The result is returned as a STRING because that is what goes back into the
    conversation - the model reads text, not Python objects.
    """
    implementation = TOOL_IMPLEMENTATIONS.get(call.name)
    if implementation is None:
        return json.dumps({"error": f"unknown tool '{call.name}'"}), {"error": "unknown tool"}

    try:
        result = implementation(**call.arguments)
    except TypeError as error:  # wrong arguments - a schema mismatch
        result = {"error": f"bad arguments: {error}"}

    return json.dumps(result), result


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_agent(
    llm: MockLLM, question: str, max_steps: int = 6
) -> tuple[str, list[dict]]:
    """
    The agent loop. Returns (final answer, trace).

    Read the `while` carefully - there is nothing here that a framework adds
    except bookkeeping.
    """
    messages: list[Message] = [
        Message(role="system", content="Use the tools when they help. Then answer plainly."),
        Message(role="user", content=question),
    ]
    trace: list[dict] = []

    for step in range(1, max_steps + 1):
        response = llm.chat(messages, tools=TOOL_SCHEMAS, temperature=0.0)

        if not response.tool_calls:
            trace.append({"step": step, "kind": "answer", "text": response.text})
            return response.text, trace

        # The assistant turn that requested the tool has to be recorded, or the
        # next request has a tool result with no matching tool call.
        messages.append(
            Message(
                role="assistant",
                content=json.dumps(
                    [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls]
                ),
            )
        )

        for call in response.tool_calls:
            payload, raw = execute_tool(call)
            trace.append(
                {
                    "step": step,
                    "kind": "tool_call",
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": raw,
                }
            )
            messages.append(
                Message(role="tool", content=payload, name=call.name, tool_call_id=call.id)
            )

    trace.append({"step": max_steps, "kind": "gave_up", "reason": "max_steps reached"})
    return "I could not finish that within the step limit.", trace


def demo(plots_dir: Path | None = None) -> dict:
    title = "14. Tool calling and a small agent loop"
    print()
    print(title)
    print("-" * len(title))

    print("Tools offered to the model:")
    for schema in TOOL_SCHEMAS:
        print(f"   {schema['name']:<12} {schema['description']}")

    questions = [
        "What is 2400 * 0.15, and what is the weather in Chennai?",
        "Search the docs for chunking.",
        "What is 5 apples plus 3 oranges?",
    ]

    results = {}
    for question in questions:
        print()
        print(f"question: {question!r}")
        llm = MockLLM(seed=1)
        answer, trace = run_agent(llm, question)

        for entry in trace:
            if entry["kind"] == "tool_call":
                print(f"   step {entry['step']}  -> call {entry['tool']}({entry['arguments']})")
                print(f"            result {json.dumps(entry['result'])[:88]}")
            elif entry["kind"] == "answer":
                print(f"   step {entry['step']}  <- answer")
            else:
                print(f"   step {entry['step']}  !! {entry['reason']}")

        print(f"   FINAL: {answer[:110]}")
        results[question] = {"answer": answer, "trace": trace, "steps": len(trace)}

    print()
    print("Three things worth noticing:")
    print("  - the model asked for two tools in one question, one after the other")
    print("  - the third question produced a tool ERROR, which went back to the")
    print("    model as a result instead of crashing the loop")
    print("  - every tool call is logged with its arguments, which is what you")
    print("    need when an agent does something surprising in production")

    return results


if __name__ == "__main__":
    demo()
