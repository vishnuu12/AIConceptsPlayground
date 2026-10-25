"""
Run the JavaScript test suite for the browser demos.

The interactive demos on the site are a SECOND implementation of the algorithms -
the browser cannot run NumPy, so the maths had to be written again in JavaScript.
A second implementation is a liability unless something keeps it honest, so
tests/js/math.test.js asserts the same properties tests/test_foundations.py and
tests/test_llm.py assert, and this wrapper makes pytest run it.

It also smoke-tests all fifteen demo UIs against a fake DOM, which is the only
way to catch "the demo throws when the page loads" without a browser.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_site  # noqa: E402

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_demos_math_and_ui(tmp_path: Path):
    # Generate knowledge.js through the SAME code path the site build uses, so
    # the test exercises the real generated data rather than a copy of it.
    knowledge = tmp_path / "knowledge.js"
    build_site.write_knowledge_data(knowledge)

    result = subprocess.run(
        [NODE, str(ROOT / "tests" / "js" / "math.test.js"), str(knowledge)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        pytest.fail(
            "the JavaScript demo tests failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )

    assert "passed" in result.stdout


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_every_demo_file_is_valid_javascript(tmp_path: Path):
    """A syntax error in a demo file would only show up as a blank page."""
    files = sorted((ROOT / "site" / "static").glob("*.js"))
    assert files, "expected the demo JavaScript to exist"

    for path in files:
        result = subprocess.run(
            [NODE, "--check", str(path)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"{path.name} has a syntax error:\n{result.stderr}"
