"""The `findex` command-line tool.

    findex index data/simplewiki.jsonl --out index.bin --limit 20000
    findex search index.bin '"solar system" NOT moon' --k 5
    findex search index.bin "volcano" --json | jq .title
    findex stats index.bin
    findex -v search ...        # -v: info logs, -vv: debug (timings, cache hits)

Results go to stdout; progress bars, logs and errors go to stderr, so `--json`
output can be piped into other programs without noise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .corpus import corpus_bytes, plan_chunks
from .crawler import USER_AGENT
from .crawler.crawl import CrawlStats, crawl_to_jsonl
from .crawler.urls import normalize
from .index import Index
from .parallel import CHUNKS_PER_WORKER, build_partials, merge
from .query import ParseError, parse
from .rank import SCORERS
from .search import Searcher
from .snippets import fetch_texts, make_snippet
from .store import load, save

log = logging.getLogger("findex.cli")

app = typer.Typer(
    help="findex: a small full-text search engine.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,  # expected errors get one clean line, not a traceback
)
out = Console()  # results -> stdout
err = Console(stderr=True)  # progress, logs, errors -> stderr

# Snippet highlights travel as two control characters that can't occur in article text;
# _rich_snippet() turns them into styled spans. (Marking with rich markup like
# "[bold]" would break on articles that contain square brackets.)
_HL = ("\x02", "\x03")


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


# --- root command: logging setup ---------------------------------------------------------
@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="More logging on stderr: -v info, -vv debug."
        ),
    ] = 0,
) -> None:
    """Runs before every subcommand; configures logging once for the whole program."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    screen = RichHandler(console=err, show_time=False, show_path=False)
    # The handler's own level decides what reaches the screen. (A logger's level only
    # filters records logged *at* that logger: records from a child logger set to INFO,
    # like the crawler for its log file, would otherwise all reach the screen too.)
    screen.setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[screen],
        force=True,  # replace any handlers left from an earlier run (tests run many)
    )


# --- findex index -------------------------------------------------------------------------
def _count_documents(corpus: Path, limit: int | None) -> int | None:
    """How many documents the progress bar should expect (None = unknown)."""
    if limit is not None:
        return limit
    if corpus.is_file():  # .jsonl: one document per non-empty line; counting is fast
        with corpus.open("rb") as f:
            return sum(1 for line in f if line.strip())
    return sum(1 for _ in corpus.rglob("*.txt"))


class ExecutorChoice(StrEnum):
    auto = "auto"
    serial = "serial"
    threads = "threads"
    processes = "processes"


# Below this, starting worker processes costs more than it saves. A rough break-even from
# the Lab 5 benchmark: serial indexing reads a few MB per second, while spawning a pool of
# fresh interpreters costs about a second.
PARALLEL_MIN_BYTES = 8 * 1024 * 1024


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="A .jsonl file or a folder of .txt files.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Where to save the index.")],
    positions: Annotated[
        bool, typer.Option(help="Store word positions (needed for phrase queries).")
    ] = True,
    limit: Annotated[int | None, typer.Option(help="Index only the first N documents.")] = None,
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            help="Index in one thread, in threads, or in processes. "
            "auto: processes for corpora of 8 MB or more, serial below that."
        ),
    ] = ExecutorChoice.auto,
    workers: Annotated[
        int | None,
        typer.Option(min=1, help="Threads/processes to use (default: one per CPU core)."),
    ] = None,
) -> None:
    """Build an index from a corpus and save it."""
    if not corpus.exists():
        _fail(f"corpus not found: {corpus}")
    total = _count_documents(corpus, limit)
    if executor is ExecutorChoice.auto:
        size = corpus_bytes(corpus, limit)
        executor = ExecutorChoice.processes if size >= PARALLEL_MIN_BYTES else ExecutorChoice.serial
        log.info("auto: %.1f MB of text -> %s executor", size / 1e6, executor.value)
    n_workers = 1 if executor is ExecutorChoice.serial else (workers or os.cpu_count() or 1)
    n_chunks = 16 if executor is ExecutorChoice.serial else CHUNKS_PER_WORKER * n_workers
    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=err, transient=True) as progress:
        indexing = progress.add_task(
            f"indexing ({executor.value}, {n_workers} workers)", total=total
        )
        chunks = plan_chunks(corpus, n_chunks, limit)
        log.info("%d chunks, %s executor, %d workers", len(chunks), executor.value, n_workers)
        partials = build_partials(
            chunks,
            executor.value,
            n_workers,
            positions,
            on_chunk_done=lambda part: progress.advance(indexing, part.num_docs),
        )
        progress.add_task("merging", total=None)  # no bar: merge can't report progress
        index = merge(partials)
    if index.num_docs == 0:
        _fail(f"no documents found in {corpus}")
    save(index, out_path)
    err.print(
        f"indexed [bold]{index.num_docs:,}[/] documents, {len(index):,} terms "
        f"-> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)"
    )


# --- findex crawl -------------------------------------------------------------------------
def _crawl_display(stats: CrawlStats, out_path: Path) -> Table:
    table = Table(title=f"crawling -> {out_path}", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("pages", f"{stats.pages:,}")
    table.add_row("pages/s", f"{stats.pages_per_second:.1f}")
    table.add_row("fetched", f"{stats.fetched:,}")
    table.add_row("errors", f"{stats.errors:,}")
    table.add_row("skipped", f"{stats.skipped:,}")
    table.add_row("in flight", f"{stats.in_flight:,}")
    table.add_row("queued", f"{stats.queued:,}")
    table.add_row("latest", Text(stats.last_url[-70:], style="dim"))
    return table


@app.command("crawl")
def crawl_cmd(
    seeds: Annotated[list[str], typer.Argument(help="Where to start: one or more URLs.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="JSON-lines output.")] = Path(
        "data/crawl.jsonl"
    ),
    max_pages: Annotated[int, typer.Option(min=1, help="Stop after this many pages.")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, help="Requests in flight at most.")] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="Requests in flight per site at most.")] = 2,
    delay: Annotated[
        float,
        typer.Option(min=0, help="Seconds between requests to one site (or its Crawl-delay)."),
    ] = 1.0,
    timeout: Annotated[
        float, typer.Option(min=0.1, help="Seconds before a request is abandoned (then retried).")
    ] = 10.0,
    domain: Annotated[
        list[str] | None,
        typer.Option(
            help="Allowed domain (repeatable; subdomains included) [default: the seeds']."
        ),
    ] = None,
) -> None:
    """Crawl websites politely and save the pages for `findex index`.

    Obeys robots.txt, identifies itself, rate-limits per site, and stops at --max-pages.
    Only crawl sites you are allowed to.
    """
    if not all(normalize(s) for s in seeds):
        _fail("seeds must be http(s) URLs", code=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    stats = CrawlStats()

    # Every URL's outcome goes to the log file; the screen shows only warnings (or -v).
    crawler_log = logging.getLogger("findex.crawler")
    file_log = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    crawler_log.addHandler(file_log)
    old_level = crawler_log.level
    crawler_log.setLevel(logging.INFO)

    err.print(f"identifying as [bold]{USER_AGENT}[/]", highlight=False)
    started = time.monotonic()
    try:
        with Live(
            get_renderable=lambda: _crawl_display(stats, out_path),
            console=err,
            refresh_per_second=4,
            transient=True,
        ):
            pages = asyncio.run(
                crawl_to_jsonl(
                    out_path,
                    seeds,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    min_delay=delay,
                    timeout=timeout,
                    allowed_domains=domain,
                    stats=stats,
                )
            )
    except KeyboardInterrupt:
        err.print(f"[yellow]interrupted[/]: {stats.pages:,} pages saved in {out_path}")
        raise typer.Exit(130) from None
    finally:
        crawler_log.removeHandler(file_log)
        file_log.close()
        crawler_log.setLevel(old_level)

    elapsed = time.monotonic() - started
    err.print(
        f"crawled [bold]{pages:,}[/] pages in {elapsed:.1f}s ({pages / max(elapsed, 1e-9):.1f}/s), "
        f"{stats.errors:,} errors, {stats.skipped:,} skipped -> {out_path}; log: {log_path}"
    )
    if pages == 0:
        _fail(f"no pages crawled: see {log_path} (robots.txt, allowed domains, errors)")


# --- findex search ------------------------------------------------------------------------
def _load(index_path: Path, trust_pickle: bool) -> Index:
    if not index_path.exists():
        _fail(f"index not found: {index_path}")
    try:
        return load(index_path, trust_pickle=trust_pickle)
    except (ValueError, TypeError, PermissionError) as exc:
        log.debug("load failed", exc_info=True)  # -vv shows the full traceback
        _fail(f"cannot load {index_path}: {exc}")


def _rich_snippet(snippet: str) -> Text:
    text = Text()
    highlighted = False
    for part in snippet.replace(_HL[1], _HL[0]).split(_HL[0]):
        text.append(part, style="bold yellow" if highlighted else None)
        highlighted = not highlighted
    return text


@app.command("search")
def search_cmd(
    index_path: Annotated[
        Path, typer.Argument(metavar="INDEX", help="An index file from `findex index`.")
    ],
    query: Annotated[str, typer.Argument(help='Query: AND, OR, NOT, (...), "phrases".')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="How many results.")] = 10,
    scorer: Annotated[ScorerChoice, typer.Option(help="Ranking function.")] = ScorerChoice.bm25,
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON object per line on stdout, for piping.")
    ] = False,
    snippets: Annotated[bool, typer.Option(help="Show highlighted snippets.")] = True,
    trust_pickle: Annotated[
        bool, typer.Option(help="Allow loading a pickle index (only files you made).")
    ] = False,
) -> None:
    """Search an index and print the best-ranked documents."""
    try:
        tree = parse(query)  # check the query before the (slow) index load
    except ParseError as exc:
        _fail(str(exc), code=2)
    index = _load(index_path, trust_pickle)
    searcher = Searcher(index)
    try:
        results = searcher.search(query, SCORERS[scorer.value], k)
    except ValueError as exc:  # e.g. a phrase query on an index built --no-positions
        log.debug("search failed", exc_info=True)
        _fail(str(exc))
    log.info("%d documents match %s", searcher.last_total, tree)

    texts = fetch_texts(index, [r.doc_id for r in results]) if snippets else {}
    terms = tree.positive_terms()

    if as_json:
        for rank, r in enumerate(results, 1):
            record = {
                "rank": rank,
                "score": round(r.score, 4),
                "doc_id": r.doc_id,
                "id": index.doc_meta[r.doc_id].ext_id,
                "title": r.title,
            }
            if r.doc_id in texts:
                record["snippet"] = make_snippet(texts[r.doc_id], terms, mark=("", ""))
            # print(), not the rich console: no styling or wrapping may touch the data
            print(json.dumps(record, ensure_ascii=False))
        return

    table = Table(
        title=f"{searcher.last_total:,} matches for {query!r} ({scorer.value})",
        show_lines=bool(texts),
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("title", style="bold cyan")
    if texts:
        table.add_column("snippet", ratio=1)
    for rank, r in enumerate(results, 1):
        row: list[str | Text] = [str(rank), f"{r.score:.3f}", r.title]
        if texts:
            snippet = make_snippet(texts[r.doc_id], terms, mark=_HL) if r.doc_id in texts else ""
            row.append(_rich_snippet(snippet))
        table.add_row(*row)
    out.print(table)


# --- findex stats -------------------------------------------------------------------------
@app.command("stats")
def stats_cmd(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="An index file.")],
    top: Annotated[int, typer.Option(min=1, help="How many top terms to list.")] = 15,
    trust_pickle: Annotated[bool, typer.Option(help="Allow loading a pickle index.")] = False,
) -> None:
    """Corpus statistics (the Lab 1 numbers), computed from a saved index."""
    index = _load(index_path, trust_pickle)
    tokens = sum(index.doc_lengths.values())
    collection_freq = Counter({term: sum(p.tf for p in plist) for term, plist in index.items()})

    summary = Table(title=f"findex stats: {index_path}", show_header=False)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("documents", f"{index.num_docs:,}")
    summary.add_row("tokens", f"{tokens:,}")
    summary.add_row("vocabulary (distinct terms)", f"{len(index):,}")
    summary.add_row("postings", f"{sum(len(p) for p in index.values()):,}")
    summary.add_row("average document length", f"{index.avg_doc_length:,.1f} tokens")
    summary.add_row("positions stored", "yes" if index.has_positions else "no")
    summary.add_row("file size", f"{index_path.stat().st_size / 1e6:,.1f} MB")
    out.print(summary)

    terms = Table(title=f"top {top} terms")
    terms.add_column("term", style="bold cyan")
    terms.add_column("occurrences", justify="right")
    terms.add_column("documents", justify="right")
    terms.add_column("% of tokens", justify="right")
    for term, count in collection_freq.most_common(top):
        terms.add_row(term, f"{count:,}", f"{index.df(term):,}", f"{100 * count / tokens:.2f}%")
    out.print(terms)


# --- errors -------------------------------------------------------------------------------
def _fail(message: str, code: int = 1) -> NoReturn:
    """Report an expected error as one line on stderr and exit with a nonzero code."""
    err.print(f"[bold red]error:[/] {message}", highlight=False, soft_wrap=True)
    raise typer.Exit(code)
