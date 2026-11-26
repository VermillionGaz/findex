"""A small ranking evaluation: precision@5 for TF-IDF vs. BM25.

Step 1 -- label (interactive, resumable; progress is saved after every answer):
    python -m findex.evalrank label index.bin eval/queries.txt eval/judgments.json

    For each query, the top 5 of BOTH scorers are pooled into one list, and you
    answer for each document: should it be in the top 5 for this query? (y/n)

Step 2 -- report (prints the Markdown table for the README):
    python -m findex.evalrank report index.bin eval/queries.txt eval/judgments.json

precision@5 = (relevant documents among the top 5) / 5. Judgments are stored
by article title, so they survive rebuilding the index.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .index import Index
from .query import parse
from .rank import BM25, Scorer, TfIdf
from .search import Searcher
from .snippets import fetch_texts, make_snippet
from .store import load
from .timing import setup_logging

K = 5
SCORERS: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}
Judgments = dict[str, dict[str, bool]]  # query -> {title: relevant?}


def read_queries(path: Path) -> list[str]:
    """One query per line; blank lines and lines starting with # are ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in lines if q.strip() and not q.lstrip().startswith("#")]


def read_judgments(path: Path) -> Judgments:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_judgments(path: Path, judgments: Judgments) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # write, then rename: a crash mid-write
    tmp.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # can't leave a half-written file


def top_titles(searcher: Searcher, query: str, scorer: Scorer, k: int = K) -> list[str]:
    return [r.title for r in searcher.search(query, scorer, k)]


def pool(searcher: Searcher, query: str) -> list[int]:
    """Doc ids in either scorer's top k, in order of first appearance."""
    seen: dict[int, None] = {}  # a dict keeps insertion order, unlike a set
    for scorer in SCORERS.values():
        for r in searcher.search(query, scorer, K):
            seen.setdefault(r.doc_id)
    return list(seen)


def precision_at_k(titles: list[str], judged: dict[str, bool], k: int = K) -> tuple[float, int]:
    """(precision@k, how many of the top k are unjudged). Unjudged counts as not relevant.
    Dividing by k even when fewer than k results came back is the standard definition:
    an engine that returns nothing useful shouldn't score well."""
    top = titles[:k]
    relevant = sum(judged.get(t, False) for t in top)
    return relevant / k, sum(t not in judged for t in top)


# --- step 1: labeling ------------------------------------------------------------------------
def label(
    index: Index,
    queries: list[str],
    path: Path,
    ask: Callable[[str], str] = input,
) -> Judgments:
    judgments = read_judgments(path)
    searcher = Searcher(index)
    for qi, query in enumerate(queries, 1):
        judged = judgments.setdefault(query, {})
        todo = [d for d in pool(searcher, query) if index.doc_meta[d].title not in judged]
        if not todo:
            continue
        print(
            f"\n=== query {qi}/{len(queries)}: {query}   ({searcher.last_total} matching documents)"
        )
        texts = fetch_texts(index, todo)
        terms = parse(query).positive_terms()
        for d in todo:
            title = index.doc_meta[d].title
            print(f"\n  {title}")
            if d in texts:
                print(f"    {make_snippet(texts[d], terms, radius=120)}")
            while (
                answer := ask("  should this be in the top 5? [y/n, q = save & quit] ")
                .strip()
                .lower()
            ) not in ("y", "n", "q"):
                pass
            if answer == "q":
                write_judgments(path, judgments)
                return judgments
            judged[title] = answer == "y"
            write_judgments(path, judgments)
    print(f"\nall queries labeled; saved to {path}")
    return judgments


# --- step 2: report --------------------------------------------------------------------------
def report(index: Index, queries: list[str], judgments: Judgments) -> str:
    searcher = Searcher(index)
    names = list(SCORERS)
    lines = [
        f"| # | Query | Matches | {' | '.join(f'{n} P@{K}' for n in names)} |",
        "|---|---|---|" + "---|" * len(names),
    ]
    totals = dict.fromkeys(names, 0.0)
    unjudged_total = 0
    for qi, query in enumerate(queries, 1):
        judged = judgments.get(query, {})
        cells: list[str] = []
        for name, scorer in SCORERS.items():
            p, unjudged = precision_at_k(top_titles(searcher, query, scorer), judged)
            totals[name] += p
            unjudged_total += unjudged
            cells.append(f"{p:.1f}" + (f" ({unjudged} unjudged)" if unjudged else ""))
        lines.append(f"| {qi} | `{query}` | {searcher.last_total:,} | {' | '.join(cells)} |")
    means = " | ".join(f"**{totals[n] / max(len(queries), 1):.2f}**" for n in names)
    lines.append(f"| | **mean** | | {means} |")
    if unjudged_total:
        lines.append(
            f"\n{unjudged_total} top-{K} results are unjudged (counted as not relevant): "
            f"run `label` again to judge them."
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.evalrank")
    ap.add_argument("command", choices=["label", "report"])
    ap.add_argument("index", type=Path)
    ap.add_argument("queries", type=Path, help="text file, one query per line")
    ap.add_argument("judgments", type=Path, help="JSON file to read/write judgments")
    ap.add_argument("--trust-pickle", action="store_true")
    args = ap.parse_args()
    setup_logging()  # once, at the program's entry point

    index = load(args.index, trust_pickle=args.trust_pickle)
    queries = read_queries(args.queries)
    if args.command == "label":
        label(index, queries, args.judgments)
    else:
        print(report(index, queries, read_judgments(args.judgments)))


if __name__ == "__main__":
    main()
