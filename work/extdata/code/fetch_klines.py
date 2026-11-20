"""Market candles, resumable.

(a) Hyperliquid candleSnapshot, 1d and 4h, every package perp symbol (HIP-3 names as-is),
    2025-07-01 00:00 UTC .. cutoff - 1 ms.
    raw/hl_candles/<symbol with ':' -> '__'>_<interval>.json
      {symbol, interval, start_ms, end_ms, fetched_at_utc, n_requests, complete: true, candles}
    Candles with open time >= cutoff are dropped (the last kept candle closes at cutoff - 1 ms).
    An unknown coin (HTTP 500 `null`) or another persistent error -> logs/hl_candles_failures.jsonl,
    no file written (retried next run). An empty answer is stored as complete with candles [].
(b) Binance USDT-M futures 5m klines from data.binance.vision, main-dex symbols only:
    X -> XUSDT, kX -> 1000XUSDT. Monthly zips 2025-07..2025-12, daily zips 2026-01-01..04.
    A monthly 404 between two months that do exist falls back to that month's daily zips.
    raw/binance_5m/<SYM>/<file>.zip + <file>.zip.CHECKSUM (sha256 verified), and
    raw/binance_5m/<SYM>/_status.json {symbol, package_symbol, files: {name: ok|404|error..},
      complete, fetched_at_utc}. complete == true -> symbol skipped on resume; otherwise only
    files without a verified zip are fetched again.

Usage (from the repo root):
  .venv/bin/python work/extdata/code/fetch_klines.py --source hl
  .venv/bin/python work/extdata/code/fetch_klines.py --source binance
  .venv/bin/python work/extdata/code/fetch_klines.py --source all --symbols BTC,HYPE,kPEPE
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # works under `python -I` too
import requests  # noqa: E402
from common import (RAW, LOGS, CUTOFF_MS, HLClient, FetchError, ts_ms, utc_now_iso, log,  # noqa: E402
                    load_package, atomic_write_json, atomic_write_bytes, append_jsonl, check_host)

# ------------------------------------------------------------------------- Hyperliquid
HL_START_MS = ts_ms("2025-07-01 00:00:00")
HL_END_MS = CUTOFF_MS - 1
HL_OUT = RAW / "hl_candles"
HL_FAIL_LOG = LOGS / "hl_candles_failures.jsonl"
INTERVAL_MS = {"1d": 86_400_000, "4h": 14_400_000}


def hl_path(sym: str, interval: str) -> Path:
    return HL_OUT / f"{sym.replace(':', '__')}_{interval}.json"


def fetch_hl_candles(cli: HLClient, sym: str, interval: str):
    """All candles in [HL_START_MS, HL_END_MS]. The response cap is not relied upon: while the
    last candle returned is not the final one of the window, ask again from last open + 1 ms;
    stop when a page is empty or adds nothing (coin delisted before the end)."""
    out: dict = {}
    start, n_req = HL_START_MS, 0
    while True:
        page = cli.post({"type": "candleSnapshot",
                         "req": {"coin": sym, "interval": interval, "startTime": start, "endTime": HL_END_MS}})
        n_req += 1
        if not isinstance(page, list):
            raise FetchError(f"unexpected response type {type(page).__name__}")
        added = 0
        for c in page:
            if c["t"] not in out:
                out[c["t"]] = c
                added += 1
        if not page or not added:
            break
        last_t = max(c["t"] for c in page)
        if last_t + INTERVAL_MS[interval] >= CUTOFF_MS:
            break
        start = last_t + 1
    candles = [out[t] for t in sorted(out) if t < CUTOFF_MS]
    return candles, n_req


def run_hl(symbols, args):
    cli = HLClient(min_interval=args.hl_min_interval, budget_per_min=args.budget, items_per_weight=60)
    log(f"hl candles: {len(symbols)} symbols x {list(INTERVAL_MS)}")
    n_ok = n_empty = n_fail = n_skip = 0
    for i, sym in enumerate(symbols, 1):
        for iv in INTERVAL_MS:
            path = hl_path(sym, iv)
            if path.exists():
                try:
                    if json.loads(path.read_text()).get("complete") is True:
                        n_skip += 1
                        continue
                except ValueError:
                    pass
            try:
                candles, n_req = fetch_hl_candles(cli, sym, iv)
            except FetchError as e:
                n_fail += 1
                append_jsonl(HL_FAIL_LOG, {"symbol": sym, "interval": iv, "error": str(e), "at": utc_now_iso()})
                log(f"{i}/{len(symbols)} {sym} {iv} FAILED: {e}")
                continue
            atomic_write_json(path, {"symbol": sym, "interval": iv, "start_ms": HL_START_MS,
                                     "end_ms": HL_END_MS, "fetched_at_utc": utc_now_iso(),
                                     "n_requests": n_req, "complete": True, "candles": candles})
            n_ok += bool(candles)
            n_empty += not candles
            span = (f"{time.strftime('%Y-%m-%d', time.gmtime(candles[0]['t'] / 1000))}.."
                    f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(candles[-1]['t'] / 1000))}") if candles else "-"
            log(f"{i}/{len(symbols)} {sym} {iv} candles={len(candles)} requests={n_req} span={span}")
    log(f"hl done: nonempty={n_ok} empty={n_empty} failed={n_fail} skipped={n_skip} "
        f"http_requests={cli.n_requests} backoffs={cli.n_backoffs}")


# ------------------------------------------------------------------------- Binance archive
BN_BASE = "https://data.binance.vision/data/futures/um"
BN_OUT = RAW / "binance_5m"
MONTHS = [(2025, m) for m in range(7, 13)]
JAN_DAYS = [f"2026-01-0{d}" for d in range(1, 5)]
_tls = threading.local()


def binance_symbol(pkg_sym: str):
    if ":" in pkg_sym:
        return None
    if pkg_sym.startswith("k") and len(pkg_sym) > 1 and pkg_sym[1:].isupper():
        return f"1000{pkg_sym[1:]}USDT"
    return f"{pkg_sym}USDT"


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def _get(url: str, tries: int = 5):
    """bytes, or None on 404. 429/5xx/connection errors: back off and retry `tries` times."""
    check_host(url)
    wait = 5.0
    for k in range(tries):
        try:
            r = _session().get(url, timeout=120)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = f"{type(e).__name__}"
        else:
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            err = f"HTTP {r.status_code}"
        if k < tries - 1:
            time.sleep(wait)
            wait = min(wait * 3, 300)
    raise FetchError(f"{url}: {err}")


def _verified(zpath: Path) -> bool:
    cpath = zpath.with_name(zpath.name + ".CHECKSUM")
    if not (zpath.exists() and cpath.exists()):
        return False
    want = cpath.read_text().split()[0].strip().lower()
    return hashlib.sha256(zpath.read_bytes()).hexdigest() == want


def fetch_file(sym: str, kind: str, stamp: str) -> str:
    """kind 'monthly' (stamp YYYY-MM) or 'daily' (stamp YYYY-MM-DD). Returns 'ok' / '404' / 'error: ...'."""
    name = f"{sym}-5m-{stamp}.zip"
    zpath = BN_OUT / sym / name
    if _verified(zpath):
        return "ok"
    url = f"{BN_BASE}/{kind}/klines/{sym}/5m/{name}"
    try:
        data = _get(url)
        if data is None:
            return "404"
        chk = _get(url + ".CHECKSUM")
        if chk is None:
            return "error: zip present but CHECKSUM 404"
        want = chk.decode().split()[0].strip().lower()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            return f"error: sha256 mismatch {got} != {want}"
        atomic_write_bytes(zpath, data)
        atomic_write_bytes(zpath.with_name(name + ".CHECKSUM"), chk)
        return "ok"
    except FetchError as e:
        return f"error: {e}"


def fetch_binance_symbol(pool: ThreadPoolExecutor, pkg_sym: str, sym: str) -> dict:
    stamps = [("monthly", f"{y}-{m:02d}") for y, m in MONTHS] + [("daily", d) for d in JAN_DAYS]
    res = dict(zip([s for _, s in stamps], pool.map(lambda ks: fetch_file(sym, *ks), stamps)))
    # fallback: a missing month with existing data both before and after it -> daily files
    order = [s for _, s in stamps]
    ok_idx = [i for i, s in enumerate(order) if res[s] == "ok"]
    fallback = []
    for i, (kind, s) in enumerate(stamps):
        if kind == "monthly" and res[s] == "404" and ok_idx and ok_idx[0] < i < ok_idx[-1]:
            y, m = map(int, s.split("-"))
            fallback += [f"{y}-{m:02d}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    if fallback:
        res.update(dict(zip(fallback, pool.map(lambda d: fetch_file(sym, "daily", d), fallback))))
    return {"symbol": sym, "package_symbol": pkg_sym, "files": res, "fallback_daily": fallback,
            "complete": not any(v.startswith("error") for v in res.values()),
            "fetched_at_utc": utc_now_iso()}


def run_binance(symbols, args):
    pairs = [(s, binance_symbol(s)) for s in symbols if binance_symbol(s)]
    log(f"binance 5m: {len(pairs)} mapped symbols, workers={args.workers}")
    n_skip = n_inc = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (pkg_sym, sym) in enumerate(pairs, 1):
            spath = BN_OUT / sym / "_status.json"
            if spath.exists():
                try:
                    if json.loads(spath.read_text()).get("complete") is True:
                        n_skip += 1
                        continue
                except ValueError:
                    pass
            t0 = time.monotonic()
            st = fetch_binance_symbol(pool, pkg_sym, sym)
            atomic_write_json(spath, st)
            v = list(st["files"].values())
            log(f"{i}/{len(pairs)} {pkg_sym}->{sym} ok={v.count('ok')} 404={v.count('404')} "
                f"err={sum(x.startswith('error') for x in v)} fallback_days={len(st['fallback_daily'])} "
                f"{time.monotonic() - t0:.1f}s")
            n_inc += not st["complete"]
            for k, x in st["files"].items():
                if x.startswith("error"):
                    log(f"   {k}: {x}")
    log(f"binance done: skipped={n_skip} incomplete={n_inc} (rerun to retry incomplete symbols)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hl", "binance", "all"], default="all")
    ap.add_argument("--symbols", default=None, help="comma-separated package symbols")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols (sorted)")
    ap.add_argument("--hl-min-interval", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=1100.0, help="HL API weight per minute to stay under")
    ap.add_argument("--workers", type=int, default=4, help="parallel Binance downloads (max 4)")
    args = ap.parse_args()
    args.workers = max(1, min(args.workers, 4))

    pkg = sorted(load_package()["symbol"].unique())
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in syms if s not in set(pkg)]
        if bad:
            raise SystemExit(f"not package symbols: {bad}")
    else:
        syms = pkg
    if args.limit is not None:
        syms = syms[: args.limit]
    if args.source in ("hl", "all"):
        run_hl(syms, args)
    if args.source in ("binance", "all"):
        run_binance(syms, args)


if __name__ == "__main__":
    main()
