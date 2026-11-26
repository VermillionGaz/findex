"""Lab 6 benchmark: the same crawl at concurrency 1, 5 and 20.

    uv run --group bench python benchmarks/bench_crawl.py run      # ~30-60 minutes
    uv run --group bench python benchmarks/bench_crawl.py report   # Markdown for the README

Seeds: the Python Software Foundation's documentation sites (override with --seed).
Before crawling, each seed site's robots.txt is fetched and the seed checked against it;
the crawler also obeys robots.txt on every request.

Methodology:
  * Each run is a fresh `python -m findex crawl` process. Wall time, pages and errors
    come from its own summary line; this process samples the memory of the crawler and
    its page-parsing processes every 50 ms with psutil.
  * Configurations are interleaved (round-robin), 3 runs each, so a slow moment of the
    network or a site hits all of them evenly.
  * Same politeness in every run (--per-host, --delay): the benchmark measures what
    concurrency buys *within* those limits.
  * A final run in asyncio debug mode (PYTHONASYNCIODEBUG=1) counts event-loop steps
    over 100 ms: evidence that nothing blocks the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "crawl.json"
SEEDS = [
    "https://docs.python.org/3/",
    "https://peps.python.org/",
    "https://packaging.python.org/en/latest/",
    "https://devguide.python.org/",
]
# crawl.log lines worth showing: retries, timeouts, redirects, robots blocks, 4xx/5xx
HANDLED = re.compile(r"retry|timeout| -> |robots\.txt disallows|^\S+ \S+ \S+\s+[45]\d\d ")
# the local site's deliberate faults, as they appear in crawl.log
FAULTS = re.compile(r"/(flaky|limited|slow|loop-[ab])\b|private")
SUMMARY = re.compile(r"crawled ([\d,]+) pages in ([\d.]+)s .*?([\d,]+) errors, ([\d,]+) skipped")


# --- the robots.txt check --------------------------------------------------------------------
async def robots_check(seeds: list[str]) -> list[dict[str, Any]]:
    from findex.crawler.fetch import fetch, make_client
    from findex.crawler.robots import RobotsCache

    results: list[dict[str, Any]] = []
    async with make_client() as client:
        robots = RobotsCache(client)
        for seed in seeds:
            origin = "/".join(seed.split("/")[:3])
            raw = await fetch(client, f"{origin}/robots.txt", follow_redirects=True)
            results.append(
                {
                    "seed": seed,
                    "robots_status": raw.status,
                    "robots_txt": raw.content.decode("utf-8", errors="replace")[:2000],
                    "seed_allowed": await robots.allowed(seed),
                    "crawl_delay": await robots.crawl_delay(seed),
                }
            )
    return results


# --- one run ---------------------------------------------------------------------------------
def watch(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    import psutil

    proc = psutil.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    peaks: dict[int, int] = {}  # pid -> peak memory seen
    while proc.poll() is None:
        try:
            for p in [proc, *proc.children(recursive=True)]:
                try:
                    mem = p.memory_info()
                    peaks[p.pid] = max(peaks.get(p.pid, 0), mem.rss, getattr(mem, "peak_wset", 0))
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    _, err = proc.communicate()
    match = SUMMARY.search(" ".join(err.split()))
    if proc.returncode != 0 or match is None:
        sys.exit(f"crawl failed (exit {proc.returncode}):\n{err[-3000:]}")
    pages, secs, errors, skipped = (float(g.replace(",", "")) for g in match.groups())
    largest = max(peaks.values(), default=0)
    return {
        "pages": int(pages),
        "wall_s": secs,
        "errors": int(errors),
        "skipped": int(skipped),
        "peak_mb": largest / 1e6,  # the crawler process (the largest one)
        "tree_peak_mb": sum(peaks.values()) / 1e6,  # crawler + page-parsing processes
        "slow_steps": err.count("Executing <"),  # asyncio debug mode's slow-step warnings
        "stderr_tail": err[-1500:],
    }


def crawl_cmd(args: argparse.Namespace, concurrency: int, out: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "findex",
        "crawl",
        *args.seed,
        "--out",
        str(out),
        "--max-pages",
        str(args.max_pages),
        "--concurrency",
        str(concurrency),
        "--per-host",
        str(args.per_host),
        "--delay",
        str(args.delay),
    ]


def run(args: argparse.Namespace) -> None:
    import psutil

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "COLUMNS": "200",
    }
    print("checking robots.txt of every seed site...", file=sys.stderr)
    robots = asyncio.run(robots_check(args.seed))
    for r in robots:
        verdict = "allowed" if r["seed_allowed"] else "NOT ALLOWED"
        print(
            f"  {r['seed']}: robots.txt HTTP {r['robots_status']}, seed {verdict}, "
            f"crawl-delay {r['crawl_delay']}",
            file=sys.stderr,
        )
    if not all(r["seed_allowed"] for r in robots):
        sys.exit("a seed is disallowed by its robots.txt: choose other seeds (--seed)")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(1, args.runs + 1):
            for c in args.concurrency:
                out = Path(tmp) / f"c{c}-r{rep}.jsonl"
                r = watch(crawl_cmd(args, c, out), env)
                r.update(concurrency=c, run=rep)
                results.append(r)
                print(
                    f"  run {rep}/{args.runs}  concurrency {c:>2}: {r['pages']} pages in "
                    f"{r['wall_s']:.1f}s ({r['pages'] / r['wall_s']:.2f}/s), "
                    f"{r['errors']} errors",
                    file=sys.stderr,
                )
        print(
            f"debug-mode run (PYTHONASYNCIODEBUG=1, concurrency {max(args.concurrency)})...",
            file=sys.stderr,
        )
        out = Path(tmp) / "debug.jsonl"
        debug = watch(
            crawl_cmd(args, max(args.concurrency), out), env | {"PYTHONASYNCIODEBUG": "1"}
        )
        print(f"  slow event-loop steps (>100 ms): {debug['slow_steps']}", file=sys.stderr)
        log_lines = (out.with_suffix(".log")).read_text(encoding="utf-8").splitlines()

    data = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "settings": {
            "seeds": args.seed,
            "max_pages": args.max_pages,
            "per_host": args.per_host,
            "delay": args.delay,
            "runs": args.runs,
        },
        "robots": robots,
        "runs": results,
        "debug_run": {k: v for k, v in debug.items() if k != "stderr_tail"},
        # log lines that show failures being handled, for the README
        "handled": [ln for ln in log_lines if HANDLED.search(ln)][:40],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"saved to {args.out}", file=sys.stderr)


# --- the local experiment: concurrency without politeness limits, and deliberate faults -------
LATENCY = 0.1  # seconds added to every response: a typical internet round trip plus server time
LOCAL_PAGES = 300


class _LocalSite(BaseHTTPRequestHandler):
    """A site on this machine: LOCAL_PAGES linked pages, every response delayed LATENCY
    seconds, plus faults that each happen once per run: a 503, a 429 with Retry-After, a
    response slow enough to time out, and a redirect loop."""

    faults_left: ClassVar[set[str]] = set()
    lock: ClassVar[threading.Lock] = threading.Lock()

    def do_GET(self) -> None:
        time.sleep(LATENCY)
        path = self.path
        if path == "/robots.txt":
            return self._send(200, "text/plain", "User-agent: *\nDisallow: /private/\n")
        with self.lock:
            fault = path if path in self.faults_left else None
            self.faults_left.discard(path)
        if fault == "/flaky":
            return self._send(503, "text/plain", "try again")
        if fault == "/limited":
            return self._send(429, "text/plain", "slow down", {"Retry-After": "1"})
        if fault == "/slow":
            time.sleep(3)  # longer than the crawl's --timeout 2
        if path in ("/loop-a", "/loop-b"):
            target = "/loop-b" if path == "/loop-a" else "/loop-a"
            return self._send(302, "text/plain", "", {"Location": target})
        if path == "/" or path.startswith("/p") or path in ("/flaky", "/limited", "/slow"):
            n = int(path[2:]) if path.startswith("/p") and path[2:].isdigit() else 0
            links = [f"/p{(n * 7 + k) % LOCAL_PAGES}" for k in range(1, 6)]
            if path == "/":
                links += ["/flaky", "/limited", "/slow", "/loop-a", "/private/secret"]
            body = "".join(f'<a href="{link}">{link}</a> ' for link in links)
            html = f"<html><head><title>Local {path}</title></head><body><p>Page {path}.</p>{body}</body></html>"
            return self._send(200, "text/html; charset=utf-8", html)
        return self._send(404, "text/plain", "not found")

    def _send(
        self, status: int, ctype: str, body: str, headers: dict[str, str] | None = None
    ) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the crawler gave up waiting (the /slow timeout) and closed the connection

    def log_message(self, format: str, *args: Any) -> None:  # silence per-request logs
        pass


def local(args: argparse.Namespace) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LocalSite)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seed = f"http://127.0.0.1:{server.server_address[1]}/"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "COLUMNS": "200",
    }
    print(
        f"local site at {seed}: {LATENCY * 1000:.0f} ms per response, no politeness limits",
        file=sys.stderr,
    )
    results: list[dict[str, Any]] = []
    handled: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(1, args.runs + 1):
            for c in args.concurrency:
                _LocalSite.faults_left = {"/flaky", "/limited", "/slow"}  # each fault once per run
                out = Path(tmp) / f"local-c{c}-r{rep}.jsonl"
                cmd = [
                    sys.executable,
                    "-m",
                    "findex",
                    "crawl",
                    seed,
                    "--out",
                    str(out),
                    "--max-pages",
                    str(args.max_pages),
                    "--concurrency",
                    str(c),
                    "--per-host",
                    str(c),
                    "--delay",
                    "0",
                    "--timeout",
                    "2",
                ]
                r = watch(cmd, env)
                r.update(concurrency=c, run=rep)
                results.append(r)
                print(
                    f"  run {rep}/{args.runs}  concurrency {c:>2}: {r['pages']} pages in "
                    f"{r['wall_s']:.1f}s ({r['pages'] / r['wall_s']:.1f}/s)",
                    file=sys.stderr,
                )
                if not handled:
                    lines = out.with_suffix(".log").read_text(encoding="utf-8").splitlines()
                    handled = [ln for ln in lines if FAULTS.search(ln)]
    server.shutdown()
    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    data["local"] = {
        "latency_s": LATENCY,
        "max_pages": args.max_pages,
        "runs": results,
        "handled": handled,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"saved to {args.out}", file=sys.stderr)


# --- report ------------------------------------------------------------------------------------
def report(args: argparse.Namespace) -> None:
    data = json.loads(args.results.read_text(encoding="utf-8"))
    m, s = data["machine"], data["settings"]
    print(
        f"Machine: {m['platform']}, Python {m['python']}, {m['logical_cores']} logical cores, "
        f"{m['ram_gb']} GB RAM.  "
    )
    print(
        f"Seeds: {', '.join(f'`{x}`' for x in s['seeds'])}. Max {s['max_pages']} pages per run; "
        f"politeness: {s['per_host']} requests per site at once, {s['delay']} s between request "
        f"starts per site. Median of {s['runs']} runs (min–max in brackets).\n"
    )
    print(
        "| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS (crawler) | Peak RSS (with parse workers) |"
    )
    print("|---|---|---|---|---|---|---|")
    by_c: dict[int, list[dict[str, Any]]] = {}
    for r in data["runs"]:
        by_c.setdefault(r["concurrency"], []).append(r)

    def med(rows: list[dict[str, Any]], key: str) -> float:
        return statistics.median(r[key] for r in rows)

    for c, rows in sorted(by_c.items()):
        rates = [r["pages"] / r["wall_s"] for r in rows]
        label = f"{c} (effectively sequential)" if c == 1 else str(c)
        print(
            f"| {label} | {med(rows, 'pages'):.0f} | {med(rows, 'wall_s'):.1f} "
            f"({min(r['wall_s'] for r in rows):.1f}–{max(r['wall_s'] for r in rows):.1f}) | "
            f"{statistics.median(rates):.2f} | {med(rows, 'errors'):.0f} | {med(rows, 'peak_mb'):.0f} MB | "
            f"{med(rows, 'tree_peak_mb'):.0f} MB |"
        )
    d = data["debug_run"]
    print(
        f"\nDebug-mode run (PYTHONASYNCIODEBUG=1): {d['pages']} pages, "
        f"**{d['slow_steps']}** event-loop steps over 100 ms.\n"
    )
    print("robots.txt check, before crawling:\n")
    for r in data["robots"]:
        print(
            f"- `{r['seed']}`: robots.txt HTTP {r['robots_status']}, seed "
            f"{'allowed' if r['seed_allowed'] else 'NOT allowed'}, Crawl-delay "
            f"{r['crawl_delay'] if r['crawl_delay'] is not None else 'none'}"
        )
    if "local" in data:
        loc = data["local"]
        print(
            f"\nLocal site (this machine, {loc['latency_s'] * 1000:.0f} ms added to every "
            f"response, no politeness limits), {loc['max_pages']} pages:\n"
        )
        print("| Concurrency | Wall (s) | Pages/s | Speedup | Peak RSS (crawler) |")
        print("|---|---|---|---|---|")
        groups: dict[int, list[dict[str, Any]]] = {}
        for r in loc["runs"]:
            groups.setdefault(r["concurrency"], []).append(r)
        base = None
        for c, rows in sorted(groups.items()):
            wall = med(rows, "wall_s")
            base = base or wall
            print(
                f"| {c} | {wall:.1f} | {med(rows, 'pages') / wall:.1f} | {base / wall:.1f}× | "
                f"{med(rows, 'peak_mb'):.0f} MB |"
            )
        print("\nThe deliberate faults, as handled in crawl.log:\n\n```")
        for line in loc["handled"]:
            print(line)
        print("```")
    print("\nHandled failures and redirects from crawl.log (first lines):\n\n```")
    for line in data["handled"][:15]:
        print(line)
    print("```")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", action="append", help="seed URL (repeatable; default: PSF docs sites)")
    r.add_argument("--max-pages", type=int, default=200)
    r.add_argument(
        "--concurrency", type=lambda v: [int(x) for x in v.split(",")], default=[1, 5, 20]
    )
    r.add_argument("--per-host", type=int, default=2)
    r.add_argument("--delay", type=float, default=1.0)
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    loc = sub.add_parser("local", help="concurrency on a local site with latency and faults")
    loc.add_argument("--max-pages", type=int, default=200)
    loc.add_argument(
        "--concurrency", type=lambda v: [int(x) for x in v.split(",")], default=[1, 5, 20]
    )
    loc.add_argument("--runs", type=int, default=3)
    loc.add_argument("--out", type=Path, default=DEFAULT_OUT)
    rep = sub.add_parser("report")
    rep.add_argument("results", type=Path, nargs="?", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.command == "run":
        args.seed = args.seed or SEEDS
        run(args)
    elif args.command == "local":
        local(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
