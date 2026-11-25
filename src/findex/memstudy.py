"postings": postings,
        "n_postings": n_postings,
        "n_terms": len(postings),
        "peak": peak,
        "retained": retained,
        "size": size,
        "load_s": load_s,
    }


def _real_bytes_each(make: Callable[[int], object], n: int = 100_000) -> float:
    """Actual allocated bytes per element (incl. list slot), via tracemalloc.

    Unlike sys.getsizeof, this sees allocator rounding and anything the object
    points to that's created alongside it (e.g. the doc_id int).
    """
    gc.collect()
    tracemalloc.start()
    xs = [make(10**6 + i) for i in range(n)]
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del xs
    return current / n


def per_object_sizes() -> str:
    """Per-element costs that explain the bytes-per-posting column.

    Note: don't use sys.getsizeof(obj.__dict__) for this. Since CPython 3.11
    a plain instance stores attribute values inline and only builds a real
    dict if something asks for dict -- so asking for it creates the cost
    you were trying to measure.
    """
    plain = _real_bytes_each(lambda i: PlainPosting(i, 3))
    slots = _real_bytes_each(lambda i: SlotPosting(i, 3))
    return (
        f"- plain @dataclass posting: {plain:.0f} B each, measured "
        f"(= 8 B list pointer + {sys.getsizeof(10**6)} B doc_id int + instance with inline values)\n"
        f"- slots=True posting: {slots:.0f} B each, measured "
        f"(same, but the instance is just header + 2 fixed slots)\n"
        f"- array('{TYPECODE}') posting: {2 * array(TYPECODE).itemsize} B "
        f"(two raw {array(TYPECODE).itemsize}-byte ints, no objects, no pointers)\n"
        f"- ints -5..256 (most tf values) are cached singletons: shared, ~0 B each"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    results = []
    for name in VARIANTS:
        r = measure(name, args.corpus, args.limit, args.out_dir)
        results.append(r)
        print(f"  {name}: done", file=sys.stderr)

    # Sanity: all three representations must hold exactly the same postings.
    reference = _as_pairs(results[0]["postings"])
    for r in results[1:]:
        assert _as_pairs(r["postings"]) == reference, f"{r['name']} differs from plain"
    for r in results:
        del r["postings"]

    n = results[0]["n_postings"]
    print(f"\nCorpus: {args.corpus} (limit={args.limit}), "
          f"{results[0]['n_terms']:,} terms, {n:,} postings\n")
    print("| Postings representation | Peak memory (build) | Retained after build "
          "| Bytes / posting | Index file size | Load time |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {VARIANTS[r['name']][0]} | {r['peak'] / 1e6:.1f} MB "
              f"| {r['retained'] / 1e6:.1f} MB | {r['retained'] / n:.1f} "
              f"| {r['size'] / 1e6:.1f} MB | {r['load_s']:.3f} s |")
    print("\nPer-object sizes (CPython " + sys.version.split()[0] + "):")
    print(per_object_sizes())


if name == "__main__":
    main()