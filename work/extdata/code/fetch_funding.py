"""Per-address funding history from Hyperliquid (`userFunding`), resumable.

Output: raw/funding/<address>.json
  {address, start_ms, end_ms, fetched_at_utc, n_requests, page_sizes, complete: true, entries}
Resume: addresses whose file exists with complete == true are skipped.
Failures (after retries) are appended to logs/funding_failures.jsonl and no file is written,
so the next run retries them.

Pagination: the API returns at most PAGE_FULL (=500, measured) entries per call, ascending by
time. A page with >= PAGE_FULL entries is treated as full: the next call starts at the last
entry's time (inclusive) and entries at that boundary timestamp that were already stored are
dropped by comparing whole serialized entries (multiset). Stop on a short page or on a page
that brings nothing new; a full page made of one single timestamp cannot be paged -> error.

Usage (from the repo root):
  .venv/bin/python work/extdata/code/fetch_funding.py                 # all 1,173 addresses
  .venv/bin/python work/extdata/code/fetch_funding.py --limit 5
  .venv/bin/python work/extdata/code/fetch_funding.py --addresses 0xabc...,0xdef...
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # works under `python -I` too
from common import (RAW, LOGS, CUTOFF_MS, HLClient, FetchError, ts_ms, utc_now_iso, log,  # noqa: E402
                    load_package, atomic_write_json, append_jsonl)

START_MS = ts_ms("2025-07-10 00:00:00")
END_MS = CUTOFF_MS - 1
PAGE_FULL = 500
OUT = RAW / "funding"
FAIL_LOG = LOGS / "funding_failures.jsonl"


def _key(e) -> str:
    return json.dumps(e, sort_keys=True)


def fetch_address(cli: HLClient, addr: str):
    entries: list = []
    page_sizes: list = []
    start = START_MS
    n_req = 0
    while True:
        page = cli.post({"type": "userFunding", "user": addr, "startTime": start, "endTime": END_MS})
        n_req += 1
        if not isinstance(page, list):
            raise FetchError(f"unexpected response type {type(page).__name__}")
        page_sizes.append(len(page))
        times = [e["time"] for e in page]
        if times != sorted(times):
            raise FetchError("page not ascending by time")
        if times and (times[0] < start or times[-1] > END_MS):
            raise FetchError(f"page outside requested range: {times[0]}..{times[-1]}")
        # boundary dedupe: entries at time == start that we already hold (multiset by content)
        held = Counter(_key(e) for e in entries if e["time"] == start) if entries else Counter()
        new = []
        for e in page:
            if e["time"] == start and held[_key(e)] > 0:
                held[_key(e)] -= 1
                continue
            new.append(e)
        entries.extend(new)
        if len(page) < PAGE_FULL or not new:
            break
        if times[0] == times[-1]:
            raise FetchError(f"full page with a single timestamp {times[0]}; cannot paginate")
        start = times[-1]
    assert all(e["time"] < CUTOFF_MS for e in entries)
    return entries, n_req, page_sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N addresses (sorted)")
    ap.add_argument("--addresses", default=None, help="comma-separated subset of package addresses")
    ap.add_argument("--min-interval", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--budget", type=float, default=1100.0, help="API weight per minute to stay under")
    args = ap.parse_args()

    pkg = sorted(load_package()["trader"].unique())
    if args.addresses:
        want = [a.strip().lower() for a in args.addresses.split(",") if a.strip()]
        bad = [a for a in want if a not in set(pkg)]
        if bad:
            raise SystemExit(f"not package addresses: {bad}")
        addrs = want
    else:
        addrs = pkg
    if args.limit is not None:
        addrs = addrs[: args.limit]

    cli = HLClient(min_interval=args.min_interval, budget_per_min=args.budget, items_per_weight=20)
    n_done = n_skip = n_fail = 0
    log(f"funding: {len(addrs)} addresses, range {START_MS}..{END_MS}")
    for i, a in enumerate(addrs, 1):
        path = OUT / f"{a}.json"
        if path.exists():
            try:
                if json.loads(path.read_text()).get("complete") is True:
                    n_skip += 1
                    continue
            except ValueError:
                pass
        try:
            entries, n_req, sizes = fetch_address(cli, a)
        except FetchError as e:
            n_fail += 1
            append_jsonl(FAIL_LOG, {"address": a, "error": str(e), "at": utc_now_iso()})
            log(f"{i}/{len(addrs)} {a} FAILED: {e}")
            continue
        atomic_write_json(path, {"address": a, "start_ms": START_MS, "end_ms": END_MS,
                                 "fetched_at_utc": utc_now_iso(), "n_requests": n_req,
                                 "page_sizes": sizes, "complete": True, "entries": entries})
        n_done += 1
        log(f"{i}/{len(addrs)} {a} entries={len(entries)} requests={n_req} pages={sizes}")
    log(f"done: fetched={n_done} skipped={n_skip} failed={n_fail} "
        f"http_requests={cli.n_requests} backoffs={cli.n_backoffs}")


if __name__ == "__main__":
    main()
