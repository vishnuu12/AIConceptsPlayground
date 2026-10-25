"""
Build the static site.

    python tools/build_site.py

Reads the content package and writes a self-contained folder of plain HTML to
dist/. No framework, no runtime dependencies, no build step beyond this script -
which means it deploys to GitHub Pages as-is and cannot break because a
JavaScript bundle changed shape.

Two deliberate choices:

1. RELATIVE LINKS EVERYWHERE. GitHub Pages serves a project site from
   /<repo>/, not from the domain root, so an absolute /static/style.css would
   404. Relative paths work locally, on Pages, and from a file:// path.

2. EVERYTHING IS ESCAPED. The content is written by hand today, but a generator
   that trusts its input is a generator that breaks the first time someone puts
   an angle bracket in a sentence.
"""

from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

# Make the repo root importable when this is run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from content import CONCEPTS, PARTS, interview_topics, total_questions  # noqa: E402

REPO_URL = "https://github.com/vishnuu12/AIConceptsPlayground"
SITE_TITLE = "AI Concepts Playground"


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def prefix(depth: int) -> str:
    """Relative path back to the site root: '' at the root, '../' one level down."""
    return "../" * depth


def code_block(code: str, label: str | None = None) -> str:
    label_html = f'<div class="code__label">{esc(label)}</div>' if label else ""
    return f'<div class="code">{label_html}<pre><code>{esc(code)}</code></pre></div>'


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def paragraphs(items: list[str]) -> str:
    return "".join(f"<p>{esc(item)}</p>" for item in items)


def page(title: str, depth: int, body: str, description: str = "") -> str:
    """Wrap body content in the shared page chrome."""
    p = prefix(depth)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} &middot; {SITE_TITLE}</title>
<meta name="description" content="{esc(description)}">
<link rel="stylesheet" href="{p}static/style.css">
</head>
<body>
<header class="topbar">
  <a class="topbar__brand" href="{p}index.html">
    <span class="topbar__mark">AI</span>
    <span>{SITE_TITLE}</span>
  </a>
  <nav class="topbar__nav">
    <a href="{p}index.html#part-1">Part 1 &middot; Foundations</a>
    <a href="{p}index.html#part-2">Part 2 &middot; LLM engineering</a>
    <a href="{p}interview.html">Interview prep</a>
    <a class="topbar__ext" href="{REPO_URL}" rel="noopener">GitHub</a>
  </nav>
</header>
<main class="page">
{body}
</main>
<footer class="footer">
  <p>
    Every number on this site was printed by code in the repository.
    <a href="{REPO_URL}" rel="noopener">Read the source</a>.
  </p>
</footer>
<script src="{p}static/app.js" defer></script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# The pages
# --------------------------------------------------------------------------


def concept_card(concept: dict, depth: int) -> str:
    p = prefix(depth)
    return f"""<a class="card" href="{p}concepts/{esc(concept['slug'])}.html">
  <span class="card__num">{concept['number']}</span>
  <h3>{esc(concept['title'])}</h3>
  <p>{esc(concept['blurb'])}</p>
</a>"""


def render_index(depth: int = 0) -> str:
    parts_html = []

    for part_number in (1, 2):
        part = PARTS[part_number]
        concepts = [c for c in CONCEPTS if c["part"] == part_number]
        parts_html.append(
            f"""
<section id="part-{part_number}" class="part">
  <h2>{esc(part['title'])}</h2>
  <p class="lead">{esc(part['blurb'])}</p>
  <p class="note">{esc(part['note'])}</p>
  <div class="cards">
    {''.join(concept_card(c, depth) for c in concepts)}
  </div>
</section>"""
        )

    questions = total_questions()

    body = f"""
<section class="hero">
  <h1>Machine learning, one runnable demo at a time</h1>
  <p>
    Fifteen concepts, two parts. Part 1 implements the algorithms from scratch with NumPy -
    gradient descent, logistic regression, k-means, PCA and a neural network with hand-written
    backpropagation. Part 2 builds the engineering around a language model: embeddings,
    tokenization, prompting, retrieval, tool calling and evaluation.
  </p>
  <p>
    Nothing here is a slide. Every demo <strong>runs</strong>, every demo is <strong>tested</strong>,
    and every number quoted on these pages was printed by the code in this repository.
  </p>
  <div class="hero__actions">
    <a class="btn btn--primary" href="{prefix(depth)}concepts/what-is-ml.html">Start with concept 1</a>
    <a class="btn" href="{prefix(depth)}interview.html">{questions} interview questions</a>
  </div>
</section>

<section class="panel">
  <h2>How to use this</h2>
  <ul>
    <li>Read a page, then run its demo. The command is on every page.</li>
    <li>Each page ends with <em>In an interview</em> - the questions that concept attracts, and
        answers you can say out loud.</li>
    <li>Every claim is tested. <code>python -m pytest tests -q</code> asserts that gradient descent
        finds the mean, that a hidden layer beats a linear model, and that retrieval finds the right
        document.</li>
    <li>Part 2 needs no API key: the model is a deterministic mock, so the plumbing is what you
        actually study.</li>
  </ul>
</section>

{''.join(parts_html)}
"""

    return page(SITE_TITLE, depth, body, "Fifteen runnable, tested demos for machine learning and LLM engineering.")


def render_concept(concept: dict) -> str:
    depth = 1
    p = prefix(depth)

    # Sections
    sections_html = []
    for section in concept["sections"]:
        inner = ""
        if section.get("paragraphs"):
            inner += paragraphs(section["paragraphs"])
        if section.get("bullets"):
            inner += bullets(section["bullets"])
        if section.get("code"):
            inner += code_block(section["code"])
        sections_html.append(
            f'<section class="panel"><h2>{esc(section["heading"])}</h2>{inner}</section>'
        )

    # What the demo printed
    results_html = (
        f'<section class="panel panel--results">'
        f"<h2>What the demo prints</h2>"
        f"{bullets(concept['results'])}"
        f'<div class="run"><span class="run__prompt">$</span> {esc(concept["demo"])}</div>'
        f"</section>"
    )

    # The plot
    plot_html = ""
    if concept.get("plot"):
        plot_html = (
            f'<section class="panel"><h2>What it looks like</h2>'
            f'<img class="plot" src="{p}plots/{esc(concept["plot"])}" '
            f'alt="Plot generated by the {esc(concept["title"])} demo" loading="lazy">'
            f'<p class="caption">Generated by the demo, not drawn by hand.</p></section>'
        )

    # Why it matters
    why_html = f'<section class="panel panel--why"><h2>Why it matters</h2>{paragraphs(concept["why"])}</section>'

    # Interview
    qa_html = ""
    if concept.get("interview"):
        items = "".join(
            f'<details class="qa"><summary>{esc(qa["q"])}</summary><p>{esc(qa["a"])}</p></details>'
            for qa in concept["interview"]
        )
        qa_html = (
            f'<section class="panel panel--interview"><h2>In an interview</h2>{items}'
            f'<p class="caption">All {total_questions()} questions are collected on the '
            f'<a href="{p}interview.html">interview prep page</a>.</p></section>'
        )

    # Prev / next
    index = CONCEPTS.index(concept)
    nav_parts = []
    if index > 0:
        previous = CONCEPTS[index - 1]
        nav_parts.append(
            f'<a class="pager__item" href="{esc(previous["slug"])}.html">&larr; {esc(previous["title"])}</a>'
        )
    if index < len(CONCEPTS) - 1:
        following = CONCEPTS[index + 1]
        nav_parts.append(
            f'<a class="pager__item pager__item--next" href="{esc(following["slug"])}.html">'
            f'{esc(following["title"])} &rarr;</a>'
        )
    pager = f'<nav class="pager">{"".join(nav_parts)}</nav>'

    body = f"""
<article>
  <header class="page__head">
    <p class="crumb">
      <a href="{p}index.html">Home</a> &rsaquo;
      <a href="{p}index.html#part-{concept['part']}">Part {concept['part']}</a> &rsaquo;
      Concept {concept['number']}
    </p>
    <h1>{esc(concept['title'])}</h1>
    <p class="lead">{esc(concept['blurb'])}</p>
  </header>

  {why_html}
  {''.join(sections_html)}
  {results_html}
  {plot_html}
  {qa_html}
  {pager}
</article>
"""

    return page(concept["title"], depth, body, concept["blurb"])


def render_interview(depth: int = 0) -> str:
    topics = interview_topics()
    questions = total_questions()

    groups = {
        1: "Part 1 &middot; Foundations",
        2: "Part 2 &middot; LLM engineering",
        3: "Working as an engineer",
    }

    blocks = []
    for part_number in (1, 2, 3):
        in_group = [t for t in topics if t["part"] == part_number]
        if not in_group:
            continue
        sections = []
        for topic in in_group:
            items = "".join(
                f'<details class="qa"><summary>{esc(qa["q"])}</summary><p>{esc(qa["a"])}</p></details>'
                for qa in topic["questions"]
            )
            link = (
                f'<a class="topic__link" href="{prefix(depth)}concepts/{esc(topic["slug"])}.html">read the code &rarr;</a>'
                if topic["slug"]
                else ""
            )
            sections.append(
                f'<section class="panel" data-topic>'
                f'<h3>{esc(topic["topic"])} {link}</h3>{items}</section>'
            )
        blocks.append(
            f'<h2 class="group">{groups[part_number]}</h2>' + "".join(sections)
        )

    body = f"""
<header class="page__head">
  <h1>Interview prep</h1>
  <p class="lead">
    {questions} questions across {len(topics)} areas - every concept from this repo, plus the
    cross-cutting questions that decide whether you have shipped something or only read about it.
  </p>
</header>

<section class="panel">
  <label class="field">
    <span>Filter</span>
    <input id="qa-filter" type="search" placeholder="try 'overfitting', 'chunking', 'temperature', 'regression'...">
  </label>
  <p class="caption" id="qa-count">{questions} questions shown</p>
</section>

<div id="qa-body">
{''.join(blocks)}
</div>

<p class="caption" id="qa-empty" hidden>Nothing matches that filter.</p>
"""

    return page("Interview prep", depth, body, f"{questions} machine learning and LLM interview questions with answers.")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def clear_directory(directory: Path) -> None:
    """
    Empty a directory without removing the directory itself.

    Written out explicitly rather than reached for from a library, because this
    generator only ever touches files under the output folder and it is worth
    being able to see that at a glance.
    """
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            clear_directory(child)
            child.rmdir()
        else:
            child.unlink()


def build(out_dir: Path, plots_dir: Path) -> dict:
    # Start from a clean folder so a renamed or deleted page cannot linger.
    clear_directory(out_dir)
    (out_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (out_dir / "static").mkdir(parents=True, exist_ok=True)

    # Static assets
    for name in ("style.css", "app.js"):
        source = ROOT / "site" / "static" / name
        shutil.copy(source, out_dir / "static" / name)

    # Plots, if the demos have been run
    copied_plots = 0
    if plots_dir.exists():
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        for png in sorted(plots_dir.glob("*.png")):
            shutil.copy(png, out_dir / "plots" / png.name)
            copied_plots += 1

    # Pages
    (out_dir / "index.html").write_text(render_index(), encoding="utf-8")
    (out_dir / "interview.html").write_text(render_interview(), encoding="utf-8")
    for concept in CONCEPTS:
        target = out_dir / "concepts" / f"{concept['slug']}.html"
        target.write_text(render_concept(concept), encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "pages": 2 + len(CONCEPTS),
        "concepts": len(CONCEPTS),
        "plots_copied": copied_plots,
        "questions": total_questions(),
    }


def main() -> int:
    out_dir = ROOT / "dist"
    plots_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "plots"

    report = build(out_dir, plots_dir)

    print(f"built {report['pages']} pages into {report['out_dir']}")
    print(f"  concepts           : {report['concepts']}")
    print(f"  interview questions: {report['questions']}")
    print(f"  plots copied       : {report['plots_copied']}")
    if report["plots_copied"] == 0:
        print("  note: no plots found - run `python run_all.py` first for the images")

    return 0


if __name__ == "__main__":
    sys.exit(main())
