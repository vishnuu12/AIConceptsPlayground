"""
Tests for the site generator.

The site is part of the deliverable, so it gets tested like the code does. The
important one is the link check: a static site with a broken link is a page that
looks fine to whoever built it and 404s for everyone else.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_site  # noqa: E402

from content import CONCEPTS, interview_topics, total_questions  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """Build the site once and share it across the tests in this file."""
    out = tmp_path_factory.mktemp("site")
    plots = out / "plots-source"
    plots.mkdir()
    build_site.build(out / "dist", plots)
    return out / "dist"


def test_every_concept_gets_a_page(built: Path):
    for concept in CONCEPTS:
        page = built / "concepts" / f"{concept['slug']}.html"
        assert page.exists(), f"missing page for {concept['slug']}"
        assert concept["title"] in page.read_text(encoding="utf-8")


def test_expected_pages_exist(built: Path):
    assert (built / "index.html").exists()
    assert (built / "interview.html").exists()
    assert (built / "static" / "style.css").exists()
    assert (built / "static" / "app.js").exists()


def test_all_internal_links_resolve(built: Path):
    """
    Every relative href must point at a file that exists.

    This is the check that catches the classic GitHub Pages mistake: an absolute
    path like /static/style.css works locally at the domain root and 404s on a
    project site served from /<repo>/.
    """
    href = re.compile(r'href="([^"#]+)(#[^"]*)?"')
    broken: list[str] = []

    for page in built.rglob("*.html"):
        for target, _fragment in href.findall(page.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{page.relative_to(built)} -> {target}")

    assert not broken, "broken internal links:\n" + "\n".join(broken)


def test_no_absolute_asset_paths(built: Path):
    """A leading slash would break a GitHub Pages project site."""
    for page in built.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert 'href="/' not in text, f"{page.name} has an absolute href"
        assert 'src="/' not in text, f"{page.name} has an absolute src"


def test_interview_page_contains_every_question(built: Path):
    raw = (built / "interview.html").read_text(encoding="utf-8")

    count = raw.count('<details class="qa">')
    assert count == total_questions(), f"expected {total_questions()} questions, found {count}"

    # The page is HTML-escaped (an apostrophe becomes &#x27;), so unescape before
    # comparing against the source strings.
    text = html.unescape(raw)

    for topic in interview_topics():
        for qa in topic["questions"]:
            assert qa["q"] in text, f"question missing from the interview page: {qa['q'][:50]}"
            assert qa["a"][:60] in text, f"answer missing: {qa['q'][:50]}"


def test_content_is_escaped(built: Path):
    """Nothing in the content should be able to inject markup."""
    for page in built.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert "<script>alert" not in text
        # Our own template tags must never leak into the output.
        assert "{{" not in text and "}}" not in text


def test_every_concept_has_an_interview_section():
    for concept in CONCEPTS:
        assert concept["interview"], f"{concept['slug']} has no interview questions"
        assert len(concept["interview"]) >= 3, f"{concept['slug']} has too few questions"
        for qa in concept["interview"]:
            assert qa["q"].strip() and qa["a"].strip()
            assert len(qa["a"]) > 60, f"answer too short in {concept['slug']}: {qa['q']}"


def test_content_metadata_is_complete():
    numbers = [c["number"] for c in CONCEPTS]
    assert numbers == list(range(1, len(CONCEPTS) + 1)), "concepts must be numbered 1..n"

    slugs = [c["slug"] for c in CONCEPTS]
    assert len(set(slugs)) == len(slugs), "duplicate slug"

    for concept in CONCEPTS:
        assert concept["why"], f"{concept['slug']} has no 'why it matters'"
        assert concept["sections"], f"{concept['slug']} has no sections"
        assert concept["results"], f"{concept['slug']} quotes no results"
        assert concept["demo"].startswith("python -m "), f"{concept['slug']} demo command looks wrong"
        assert concept["part"] in (1, 2)
