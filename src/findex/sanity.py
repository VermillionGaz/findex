"""The three ranking sanity checks from Lab 3 M2, demonstrated on your corpus.

    python -m findex.sanity index.bin
    python -m findex.sanity data/simplewiki.jsonl --limit 20000

1. A document containing a RARE query term outranks one containing only a COMMON one.
2. With BM25, the 20th repetition of a term adds almost nothing (TF-IDF: as much as the 1st).
3. A SHORT document with one match outranks a very LONG one with one match.

The terms and documents are picked automatically from the index; the output is
Markdown, ready to paste into the README.
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from .corpus import iter_documents
from .index import Index, Posting, build_index
from .rank import BM25, Scorer, TfIdf, score_docs
from .store import detect_format, load

SCORERS: dict[str, Scorer] = {"BM25": BM25(), "TF-IDF": TfIdf()}


def _mark(ok: bool) -> str:
    return "✅ holds" if ok else "❌ does NOT hold"


# --- check 1: rare beats common -------------------------------------------------------------
def check_rare_beats_common(index: Index) -> bool:
    by_df = sorted(index, key=index.df)

    # Common term: the most widespread one that still has a doc where it occurs once,
    # and that isn't in literally every document (then nothing could lack it).
    common, b_doc = None, None
    for term in reversed(by_df):
        if index.df(term) < index.num_docs:
            once = [p.doc_id for p in index[term] if p.tf == 1]
            if once:
                common, b_candidates = term, once
                break
    if common is None:
        print("(corpus too small for check 1)\n")
        return False
    common_docs = {p.doc_id for p in index[common]}

    # Rare term: the rarest one occurring once in some doc that does NOT contain the
    # common term. Then doc A matches only the rare term, doc B only the common one.
    rare = a_doc = None
    for term in by_df:
        for p in index[term]:
            if p.tf == 1 and p.doc_id not in common_docs:
                rare, a_doc = term, p.doc_id
                break
        if rare:
            break
    if rare is None:
        print("(no suitable rare term for check 1)\n")
        return False
    rare_docs = {p.doc_id for p in index[rare]}
    # Doc B: contains the common term once, not the rare one, length closest to A's,
    # so document length doesn't decide the outcome.
    a_len = index.doc_length(a_doc)
    b_doc = min((d for d in b_candidates if d not in rare_docs),
                key=lambda d: abs(index.doc_length(d) - a_len))

    print("### Check 1 — a rare term outranks a common one\n")
    print(f"Query `{common} OR {rare}`. `{common}` is in {index.df(common):,} of "
          f"{index.num_docs:,} documents; `{rare}` is in {index.df(rare):,}.\n")
    print("| Scorer | Doc A: only the rare term (tf=1) | Doc B: only the common term (tf=1) | Result |")
    print("|---|---|---|---|")
    all_ok = True
    for name, scorer in SCORERS.items():
        scores = score_docs(index, [common, rare], scorer)
        ok = scores[a_doc] > scores[b_doc]
        all_ok &= ok
        print(f"| {name} | {scores[a_doc]:.3f} ({index.doc_meta[a_doc].title}, {a_len} tokens) "
              f"| {scores[b_doc]:.3f} ({index.doc_meta[b_doc].title}, "
              f"{index.doc_length(b_doc)} tokens) | {_mark(ok)} |")
    print("\nWhy: idf. A term in few documents says much more about the ones it's in.\n")
    return all_ok


# --- check 2: saturation --------------------------------------------------------------------
def check_saturation(index: Index) -> bool:
    # A mid-frequency term and the document whose length is closest to average, so
    # the length factor is ~1 and only tf varies. tf values are hypothetical: we
    # feed synthetic postings to the scorers with the real term and real document.
    by_df = sorted(index, key=index.df)
    term = by_df[len(by_df) // 2]
    doc = min(index.doc_lengths, key=lambda d: abs(index.doc_lengths[d] - index.avg_doc_length))

    tfs = [1, 2, 3, 5, 10, 19, 20]
    score = {name: {tf: scorer(term, Posting(doc, tf), index) for tf in [0, *tfs]}
             for name, scorer in SCORERS.items()}

    print("### Check 2 — BM25 saturates, TF-IDF doesn't\n")
    print(f"Term `{term}` (df={index.df(term):,}) in document “{index.doc_meta[doc].title}” "
          f"({index.doc_length(doc)} tokens; average is {index.avg_doc_length:.0f}).\n")
    print("| tf | BM25 score | BM25 gain per extra occurrence | TF-IDF score | TF-IDF gain per extra occurrence |")
    print("|---|---|---|---|---|")
    prev = 0
    for tf in tfs:
        b, t = score["BM25"], score["TF-IDF"]
        # gain per occurrence since the previous row (rows skip some tf values)
        gain_b = (b[tf] - b[prev]) / (tf - prev)
        gain_t = (t[tf] - t[prev]) / (tf - prev)
        print(f"| {tf} | {b[tf]:.3f} | +{gain_b:.4f} | {t[tf]:.3f} | +{gain_t:.4f} |")
        prev = tf
    first_gain = score["BM25"][1] - score["BM25"][0]
    last_gain = score["BM25"][20] - score["BM25"][19]
    ok = last_gain < 0.05 * first_gain
    print(f"\nWith BM25 the 20th occurrence adds {last_gain / first_gain:.1%} of what the 1st "
          f"added: {_mark(ok)}. With TF-IDF it adds 100% — score grows linearly forever.\n")
    return ok


# --- check 3: length normalization ------------------------------------------------------------
def check_short_beats_long(index: Index) -> bool:
    # A term occurring exactly once in both a short and a much longer document.
    best = None
    for term in sorted(index, key=index.df, reverse=True):
        once = [p.doc_id for p in index[term] if p.tf == 1]
        if len(once) < 2:
            continue
        short = min(once, key=index.doc_length)
        long = max(once, key=index.doc_length)
        ratio = index.doc_length(long) / max(index.doc_length(short), 1)
        if best is None or ratio > best[3]:
            best = (term, short, long, ratio)
        if ratio >= 10:
            break
    if best is None:
        print("(corpus too small for check 3)\n")
        return False
    term, short, long, ratio = best

    print("### Check 3 — a short document beats a long one with the same single match\n")
    print(f"Query `{term}`; it occurs exactly once in both documents.\n")
    print("| Scorer | Short: “{}” ({} tokens) | Long: “{}” ({} tokens) | Result |".format(
        index.doc_meta[short].title, index.doc_length(short),
        index.doc_meta[long].title, index.doc_length(long)))
    print("|---|---|---|---|")
    ok_bm25 = False
    for name, scorer in SCORERS.items():
        s = scorer(term, Posting(short, 1), index)
        l = scorer(term, Posting(long, 1), index)
        ok = s > l
        if name == "BM25":
            ok_bm25 = ok
        verdict = _mark(ok) if name == "BM25" else ("tie — TF-IDF ignores length" if s == l else _mark(ok))
        print(f"| {name} | {s:.3f} | {l:.3f} | {verdict} |")
    print(f"\nThe long document is {ratio:.0f}× longer: one mention in it is weaker evidence "
          f"the document is *about* `{term}`. BM25's `b` parameter encodes that.\n")
    return ok_bm25


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.sanity")
    ap.add_argument("source", type=Path, help="a saved index file, or a corpus to index")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trust-pickle", action="store_true")
    args = ap.parse_args()

    if detect_format(args.source):
        index = load(args.source, trust_pickle=args.trust_pickle)
    else:
        index = build_index(islice(iter_documents(args.source), args.limit))
    print(f"<!-- {index!r} -->\n")

    results = [check_rare_beats_common(index), check_saturation(index), check_short_beats_long(index)]
    print(f"<!-- {sum(results)}/3 checks hold -->")


if __name__ == "__main__":
    main()
