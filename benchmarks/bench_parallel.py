"""Lab 5 benchmark: serial vs threads vs processes indexing.

Step 1, measure (appends to a JSON file, so runs can be added later, e.g. free-threaded):
    uv run --group bench python benchmarks/bench_parallel.py run data/simplewiki.jsonl
    uv run --group bench python benchmarks/bench_parallel.py run data/simplewiki.jsonl --limit 20000 --runs 1

Step 2, report (Markdown table on stdout, speedup plot as a PNG):
    uv run --group bench python benchmarks/bench_parallel.py report

Methodology:
  * Every run is a fresh Python process (no warm caches or leftover memory from earlier runs).
    It runs only the standard library plus findex (PYTHONPATH=src), so any interpreter
    works -- including a free-threaded one (--python).
  * This process watches each run from outside with psutil, sampling every 20 ms: the CPU
    time of the run AND its worker processes (Windows doesn't report children's CPU time
    to the parent), and the peak memory of the parent and of all workers together.
  * One untimed warm-up run first, so the corpus is in the OS file cache for every
    measured run. Then configurations are interleaved (round-robin), so a slow moment on
    the machine hits all of them evenly rather than all runs of one configuration.
  * Timings: plan (chunking), build (the partial indexes, parallel), merge (serial, in
    the parent). Saving the index is not included: it is the same for every configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "parallel.json"
DEFAULT_PLOT = ROOT / "docs" / "speedup.png"


# =============================================================================================
# One measured run (executed in a fresh child process: stdlib + findex only)
# =============================================================================================
def one_run(corpus: str, executor: str, workers: int, limit: int | None) -> None:
    from findex.corpus import plan_chunks
    from findex.parallel import CHUNKS_PER_WORKER, build_partials, merge

    t0 = time.perf_counter()
    n_chunks = 1 if executor == "serial" else CHUNKS_PER_WORKER * workers
    chunks = plan_chunks(corpus, n_chunks, limit)
    t1 = time.perf_counter()
    if executor not in ("serial", "threads", "processes"):
        sys.exit(f"unknown executor: {executor}")
    partials = build_partials(chunks, executor, workers)
    t2 = time.perf_counter()
    index = merge(partials)
    t3 = time.perf_counter()
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        json.dumps(
            {
                "plan_s": t1 - t0,
                "build_s": t2 - t1,
                "merge_s": t3 - t2,
                "wall_s": t3 - t0,
                "docs": index.num_docs,
                "chunks": len(chunks),
                "python": platform.python_version(),
                "gil": gil,
            }
        )
    )


# =============================================================================================
# The harness (this process): launch runs, watch them, collect results
# =============================================================================================
def watch(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    """Run cmd, sampling its process tree until it exits."""
    import psutil

    proc = psutil.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu: dict[int, float] = {}  # pid -> last seen CPU seconds (user + system)
    parent_peak = 0
    workers_peak = 0
    while proc.poll() is None:
        try:
            parent_peak = max(parent_peak, _peak_rss(proc))
            cpu[proc.pid] = sum(proc.cpu_times()[:2])
            children = proc.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        workers_rss = 0
        for child in children:
            try:
                cpu[child.pid] = sum(child.cpu_times()[:2])
                workers_rss += child.memory_info().rss
            except psutil.NoSuchProcess:  # a worker exited between listing and reading it
                pass
        workers_peak = max(workers_peak, workers_rss)
        time.sleep(0.02)
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.exit(f"run failed: {' '.join(cmd)}\n{err}")
    result: dict[str, Any] = json.loads(out.strip().splitlines()[-1])
    result.update(
        cpu_s=sum(cpu.values()),
        parent_peak_mb=parent_peak / 1e6,
        workers_peak_mb=workers_peak / 1e6,
    )
    return result


def _peak_rss(proc: Any) -> int:
    mem = proc.memory_info()
    return max(mem.rss, getattr(mem, "peak_wset", 0))  # Windows also reports the true peak


def machine() -> dict[str, Any]:
    import psutil

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }


def configs(executors: list[str], worker_counts: list[int]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for ex in executors:
        for w in [1] if ex == "serial" else worker_counts:
            if (ex, w) not in out:
                out.append((ex, w))
    return out


def run(args: argparse.Namespace) -> None:
    cores = os.cpu_count() or 1
    worker_counts = sorted({w for w in args.workers if w <= cores} | {cores})
    todo = configs(args.executors, worker_counts)
    python = args.python or getattr(sys, "_base_executable", sys.executable)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}

    def cmd(ex: str, w: int) -> list[str]:
        c = [python, __file__, "one", str(args.corpus), ex, str(w)]
        return c + (["--limit", str(args.limit)] if args.limit is not None else [])

    print(f"{len(todo)} configurations x {args.runs} runs, with {python}", file=sys.stderr)
    print("warm-up run (not recorded)...", file=sys.stderr)
    watch(cmd("serial", 1), env)

    results: list[dict[str, Any]] = []
    for rep in range(1, args.runs + 1):
        for ex, w in todo:
            r = watch(cmd(ex, w), env)
            r.update(
                executor=ex,
                workers=w,
                run=rep,
                label=args.label,
                corpus=str(args.corpus),
                limit=args.limit,
            )
            results.append(r)
            gil = "" if r["gil"] else ", no GIL"
            print(
                f"  run {rep}/{args.runs}  {ex:9} x{w:<3} wall {r['wall_s']:7.2f}s  "
                f"cpu {r['cpu_s']:7.2f}s  merge {r['merge_s']:6.2f}s{gil}",
                file=sys.stderr,
            )

    data: dict[str, Any] = (
        json.loads(args.out.read_text())
        if args.out.exists()
        else {"machine": machine(), "runs": []}
    )
    data["runs"].extend(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print(f"saved {len(results)} runs to {args.out}", file=sys.stderr)


# =============================================================================================
# Report: Markdown table + speedup plot
# =============================================================================================
def report(args: argparse.Namespace) -> None:
    data = json.loads(args.results.read_text())
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in data["runs"]:
        groups.setdefault((r["label"], r["executor"], r["workers"]), []).append(r)

    def med(rows: list[dict[str, Any]], key: str) -> float:
        return statistics.median(r[key] for r in rows)

    order = {"serial": 0, "threads": 1, "processes": 2}
    keys = sorted(groups, key=lambda k: (k[0] != "default", k[0], order[k[1]], k[2]))
    serial_wall = {
        lab: med(groups[(lab, "serial", 1)], "wall_s") for lab, ex, _ in keys if ex == "serial"
    }

    m = data["machine"]
    first = data["runs"][0]
    print(
        f"Machine: {m['processor']}, {m['physical_cores']} physical / {m['logical_cores']} "
        f"logical cores, {m['ram_gb']} GB RAM, {m['platform']}.  "
    )
    print(
        f"Corpus: `{first['corpus']}`"
        + (f" (limit {first['limit']:,})" if first["limit"] else "")
        + f", {first['docs']:,} documents. Median of {len(groups[keys[0]])} runs per row.\n"
    )
    print(
        "| Python | Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS parent | "
        "Peak RSS workers | Merge (s) | Speedup |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key in keys:
        lab, ex, w = key
        rows = groups[key]
        wall = med(rows, "wall_s")
        spread = f" ({min(r['wall_s'] for r in rows):.1f}–{max(r['wall_s'] for r in rows):.1f})"
        base = serial_wall.get(lab) or serial_wall.get("default")
        speedup = f"{base / wall:.2f}×" if base and wall > 0 else "–"
        gil = " (no GIL)" if not rows[0]["gil"] else ""
        workers_mb = f"{med(rows, 'workers_peak_mb'):,.0f} MB" if ex == "processes" else "–"
        print(
            f"| {rows[0]['python']}{gil} | {ex} | {w} | {wall:.2f}{spread} | "
            f"{med(rows, 'cpu_s'):.1f} | {med(rows, 'parent_peak_mb'):,.0f} MB | {workers_mb} | "
            f"{med(rows, 'merge_s'):.2f} | {speedup} |"
        )

    # Amdahl: the fraction of a serial run that can run in parallel (the build phase).
    if serial_wall.get("default", 0) > 0:
        s = groups[("default", "serial", 1)]
        p = min(med(s, "build_s") / med(s, "wall_s"), 0.999)
        print(
            f"\nIn the serial run, the parallelizable build phase is {p:.0%} of the time; plan and "
            f"merge ({1 - p:.0%}) stay serial. Amdahl's law caps the speedup at "
            f"1 / {1 - p:.2f} = **{1 / (1 - p):.1f}×**, however many workers."
        )
        plot(groups, keys, serial_wall, p, args.plot)


def plot(
    groups: dict[tuple[str, str, int], list[dict[str, Any]]],
    keys: list[tuple[str, str, int]],
    serial_wall: dict[str, float],
    p: float,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no window: write the file only
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    max_w = max(k[2] for k in keys)
    xs = list(range(1, max_w + 1))
    ax.plot(xs, xs, "--", color="grey", label="ideal (linear)")
    ax.plot(
        xs,
        [1 / ((1 - p) + p / n) for n in xs],
        ":",
        color="black",
        label=f"Amdahl limit ({p:.0%} parallel)",
    )
    for lab, ex in dict.fromkeys((k[0], k[1]) for k in keys if k[1] != "serial"):
        points = sorted(
            (w, statistics.median(r["wall_s"] for r in groups[(lab, ex, w)]))
            for (l2, e2, w) in keys
            if (l2, e2) == (lab, ex)
        )
        base = serial_wall.get(lab) or serial_wall["default"]
        name = ex if lab == "default" else f"{ex} ({lab})"
        ax.plot([w for w, _ in points], [base / max(t, 1e-9) for _, t in points], "o-", label=name)
    ax.set_xticks(sorted({k[2] for k in keys}))  # ticks at the measured worker counts
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial")
    ax.set_title("findex: parallel indexing speedup")
    ax.grid(alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    shown = path.resolve()
    link = shown.relative_to(ROOT).as_posix() if shown.is_relative_to(ROOT) else shown.as_posix()
    print(f"\n![speedup vs workers]({link})")


# =============================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="measure, appending to the results file")
    r.add_argument("corpus", type=Path)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--runs", type=int, default=3)
    r.add_argument(
        "--workers",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 8],
        help="comma-separated worker counts; the CPU count is always added",
    )
    r.add_argument(
        "--executors", type=lambda s: s.split(","), default=["serial", "threads", "processes"]
    )
    r.add_argument("--python", help="interpreter for the runs, e.g. a free-threaded one")
    r.add_argument("--label", default="default", help="names this set of runs in the report")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)

    rep = sub.add_parser("report", help="print the table, write the plot")
    rep.add_argument("results", type=Path, nargs="?", default=DEFAULT_OUT)
    rep.add_argument("--plot", type=Path, default=DEFAULT_PLOT)

    one = sub.add_parser("one")  # internal: a single measured run, in a child process
    one.add_argument("corpus")
    one.add_argument("executor")
    one.add_argument("workers", type=int)
    one.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()
    if args.command == "one":
        one_run(args.corpus, args.executor, args.workers, args.limit)
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":  # spawn-safe: worker processes import this file without running main()
    main()
