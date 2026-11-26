
The corpus is streamed one document at a time (`corpus.py`) through a lazy
tokenizer (`tokenize.py`); `stats.py` computes document count, token count,
vocabulary size and the top terms in a single pass.

| Version | Documents | Peak memory | Time elapsed |
|---|---|---|---|
| Eager | 20,000 | 512.1 MB | 20.50 s |
| Lazy | 20,000 | 20.3 MB | 23.51 s |

Lazy consumes about 25× less memory while taking ~15% longer. The extra time
is likely the cost of suspending and resuming generators for every token. Both
timings are inflated by `tracemalloc`, but equally, so the comparison holds.

---

`store.py` saves and loads the index as pickle or as a custom binary format
(an 8-byte magic number, a JSON header with terms and document metadata, then
raw little-endian `array('I')` buffers). `load()` detects the format from the
file's first bytes.

| Format | Size (MB) | Save (s) | Load (s) | Round trip |
|---|---|---|---|---|
| pickle | 58.9 | 9.19 | 9.74 | OK |
| binary | 33.9 | 0.63 | 5.11 | OK |

The binary format is about 40% smaller and saves 15× faster, because it
writes a few large blocks of raw integers instead of one pickled object per
posting. Loading is still slow for both, since both rebuild a `Posting`
object for every posting.

**Why pickle is dangerous:** a pickle file is not data, it's a program for
the unpickler. It can name any importable function (such as `os.system`) with
arguments, and `pickle.load` will call it, before your code ever sees the
result. Loading a pickle someone else made is remote code execution. That's
why `load()` refuses pickle files unless called with `trust_pickle=True`
(`--trust-pickle` on the CLI). The binary format only parses JSON and
integers, so it's safe to load from anywhere.

---
# Sanity check: 

(findex) PS G:\python projects\findex> uv run python -m findex.sanity index.bin
<!-- Index(terms=909,375, docs=241,787, avg_doc_length=175.9) -->

### Check 1 — a rare term outranks a common one

Query `the OR margarines`. `the` is in 218,158 of 241,787 documents; `margarines` is in 1.

| Scorer | Doc A: only the rare term (tf=1) | Doc B: only the common term (tf=1) | Result |
|---|---|---|---|
| BM25 | 15.168 (Margarine, 94 tokens) | 0.130 (Macromolecule, 94 tokens) | V holds |
| TF-IDF | 12.396 (Margarine, 94 tokens) | 0.103 (Macromolecule, 94 tokens) | V holds |

Why: idf. A term in few documents says much more about the ones it's in.

### Check 2 — BM25 saturates, TF-IDF doesn't

Term `98058` (df=1) in document “Distance” (176 tokens; average is 176).

| tf | BM25 score | BM25 gain per extra occurrence | TF-IDF score | TF-IDF gain per extra occurrence |
|---|---|---|---|---|
| 1 | 11.986 | +11.9865 | 12.396 | +12.3958 |
| 2 | 17.125 | +5.1386 | 24.792 | +12.3958 |
| 3 | 19.980 | +2.8552 | 37.187 | +12.3958 |
| 5 | 23.056 | +1.5376 | 61.979 | +12.3958 |
| 10 | 26.064 | +0.6017 | 123.958 | +12.3958 |
| 19 | 27.781 | +0.1908 | 235.520 | +12.3958 |
| 20 | 27.883 | +0.1021 | 247.916 | +12.3958 |

With BM25 the 20th occurrence adds 0.9% of what the 1st added: V holds. With TF-IDF it adds 100% — score grows linearly forever.

### Check 3 — a short document beats a long one with the same single match

Query `the`; it occurs exactly once in both documents.

| Scorer | Short: “Nitta Shrine” (5 tokens) | Long: “List of minor planets: 1001–2000” (18978 tokens) | Result |
|---|---|---|---|
| BM25 | 0.183 | 0.002 | V holds |
| TF-IDF | 0.103 | 0.103 | tie — TF-IDF ignores length |

The long document is 3796× longer: one mention in it is weaker evidence the document is *about* `the`. BM25's `b` parameter encodes that.

<!-- 3/3 checks hold -->


Ten queries (`eval/queries.txt`). For each, the top 5 of both scorers were
pooled and every pooled article was judged by hand (`eval/judgments.json`):
should it be in the top 5? precision@5 is relevant ÷ 5.

| # | Query | Matches | TF-IDF P@5 | BM25 P@5 |
|---|---|---|---|---|
| 1 | `"solar system"` | 207 | 1.0 | 0.2 |
| 2 | `world war` | 2,364 | 0.2 | 0.4 |
| 3 | `photosynthesis` | 67 | 0.8 | 0.8 |
| 4 | `"programming language"` | 90 | 0.6 | 0.6 |
| 5 | `volcano eruption` | 60 | 0.8 | 1.0 |
| 6 | `olympic games` | 242 | 0.6 | 0.6 |
| 7 | `human heart blood` | 65 | 0.8 | 0.8 |
| 8 | `"united states" president` | 1,290 | 0.6 | 0.6 |
| 9 | `french revolution` | 408 | 0.6 | 0.8 |
| 10 | `piano` | 428 | 0.8 | 0.8 |
| | **mean** | | **0.68** | **0.66** |

Caching: 

(findex) PS G:\python projects\findex> uv run python -m findex.search index.bin -i -v

DEBUG findex.timing: load took 4940.5 ms

loaded 40000 documents in 4.95s

query> switch

DEBUG findex.search: query switch: cache miss (0 hits, 1 misses, 1/256 entries)

DEBUG findex.timing: Searcher.search took 0.7 ms

157 matches for 'switch', showing 10 (bm25, merge, 0.78 ms)
   13.018  [7193] Switch  (22812)
           …switches. Some of these are: DIP switch Reed switch Rocker switch Micro switch Relay switch Rotary switch Surface mount switch Toggle switch Tilt switch Time switch…
   11.956  [35169] Context switch  (139161)
           …minimize the time of context switches. A context switch can mean a register context switch, a task context switch, a thread context switch, or a process context switch.…
   11.743  [35165] Thread switching latency  (139155)
           …There is no virtual memory switching required in a thread switch, unlike a process switch. Related pages Process switching latency Context switch Computer science
   10.476  [2319] Nintendo  (7604)
           …Nintendo 2DS (2013) New Nintendo 3DS (2014) New Nintendo 3DS XL (2014) Nintendo Switch (2017) New Nintendo 2DS XL (2017) Nintendo Switch Lite (2019) Nintendo Switch…
    9.878  [23598] User interface  (90897)
           …intuitive, and simple to use. An example of such a user interface is that of the kill switch. A kill switch must shut off the machine at all costs - the idea is to avoid…
    9.235  [35166] Process switching latency  (139156)
           …latency is the time needed by the operating system to perform the process context switch to continue executing another process. Related pages Thread switching latency…
    8.779  [35070] Real-time operating system  (138707)
           …The basic two designs for RTOS are: Event-driven (priority scheduling) designs: switch tasks only when an event of higher priority needs service, called pre-emptive…
    8.604  [11818] Transistor  (43338)
           …useful for electronics that require a switch to be turned on, making it an electronic switch. This rivals the mechanical switch, which requires a constant force pressing…
    8.283  [35550] Multiplexer  (140606)
           …An electronic multiplexer can be considered as a multiple-input, single-output switch, and a demultiplexer as a single-input, multiple-output switch. In telecommunications…
    8.201  [28411] Pikmin (video game)  (114998)
           …developed and published by Nintendo for the Nintendo GameCube, and later the Wii and Switch. The plot involves an alien captain named Captain Olimar who crash-lands on…

query> switch

DEBUG findex.search: query switch: cache HIT (1 hits, 1 misses, 1/256 entries)

DEBUG findex.timing: Searcher.search took 0.4 ms

(same matches as above)

**Coverage: 82%** (`uv run pytest --cov=findex`, branch coverage on). At 100%:
`tokenize`, `corpus`, `query`, `rank`, `snippets`; `cli` 99%. Not covered, and why:

- The `main()` functions of the Lab 1–3 research scripts (`python -m findex.search`'s
  interactive mode and benchmark, `findex.index`, `findex.stats`, `findex.store`,
  `findex.memstudy`, `findex.sanity`, `findex.evalrank`). They are one-off measurement
  tools whose output is recorded in this README; the logic they call is tested, and the
  user-facing tool is the `findex` command, which is.
- The big-endian byte swaps in `store.py`, marked `# pragma: no cover`: they only run on
  big-endian CPUs.
- Abstract `raise NotImplementedError` methods, excluded in the coverage config.
