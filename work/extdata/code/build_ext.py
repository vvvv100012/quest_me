"""Tidy tables + checks from the raw external data (no network).

Inputs : raw/funding/*.json, raw/hl_candles/*.json, raw/binance_5m/<SYM>/*.zip (+ _status.json),
         the package perp fills (via v2dev lib) and work/v2dev/build/groups.parquet.
Outputs: build/funding.parquet, build/hl_candles_1d.parquet, build/hl_candles_4h.parquet,
         build/klines_5m/<package_symbol>.parquet (accepted mappings only), build/symbol_map.csv,
         build/hidden_positions.parquet (positions visible only through funding),
         build/manifest.json. Checks are printed at the end; they never stop the build.

Works on a partial raw directory (slice runs): addresses / symbols without raw files are
counted as not_fetched.

Funding record forms: Hyperliquid returns two forms, never both for the same
trader x coin x day (counted in the manifest):
  - daily aggregate: stamped exactly 00:00 UTC of the day whose hourly samples it sums,
    nSamples = number of samples (probe: BTC position opened 2025-08-19 14:27 -> entry
    labelled 2025-08-19 00:00 with nSamples = 9, i.e. 15:00..23:00). Known only at the end
    of that day.
  - hourly record: stamped at the funding hour (a few ms after it), nSamples = null.
    n_samples is stored as nullable Int64 (<NA> for hourly records).
Check 2 matches an entry to the replay on the calendar day of its timestamp (valid for both
forms); the shifted convention (previous day) is printed alongside as a sanity check.
"""
from __future__ import annotations
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # works under `python -I` too
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (RAW, BUILD, LOGS, CUTOFF, CUTOFF_MS, GROUPS_PARQUET, load_package,  # noqa: E402
                    utc_now_iso)
from fetch_klines import binance_symbol, INTERVAL_MS  # noqa: E402

PROBE_ADDR = "0x4648baa72d8cfe3f31882fe52bd9fd8e6d539373"
BN_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
BAR_MS = 300_000
LAST_DAY = CUTOFF - pd.Timedelta(days=1)       # 2026-01-04, last full UTC day before the cutoff
WINDOW_START = pd.Timestamp("2025-07-27")      # UTC day of the first package fill
KL_DIR = BUILD / "klines_5m"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ----------------------------------------------------------------------------- funding
def build_funding(addresses):
    fail_log = LOGS / "funding_failures.jsonl"
    failed_ever = set()
    if fail_log.exists():
        failed_ever = {json.loads(line)["address"] for line in fail_log.read_text().splitlines() if line}
    rows, fetched_at, n_req = [], [], 0
    complete, with_entries, empty = set(), 0, 0
    other_types = {}
    for a in addresses:
        path = RAW / "funding" / f"{a}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if d.get("complete") is not True:
            continue
        complete.add(a)
        fetched_at.append(d["fetched_at_utc"])
        n_req += d["n_requests"]
        if d["entries"]:
            with_entries += 1
        else:
            empty += 1
        for e in d["entries"]:
            de = e["delta"]
            if de.get("type") != "funding":
                other_types[de.get("type")] = other_types.get(de.get("type"), 0) + 1
                continue
            rows.append((a, e["time"], de["coin"], float(de["usdc"]), float(de["szi"]),
                         float(de["fundingRate"]),
                         None if de["nSamples"] is None else int(de["nSamples"])))
    f = pd.DataFrame(rows, columns=["trader", "time_ms", "coin", "usdc", "szi", "funding_rate", "n_samples"])
    f["time"] = pd.to_datetime(f["time_ms"], unit="ms")
    f["n_samples"] = f["n_samples"].astype("Int64")
    f = f[["trader", "time", "coin", "usdc", "szi", "funding_rate", "n_samples"]]
    f = f.sort_values(["trader", "time", "coin"], kind="stable").reset_index(drop=True)
    assert (f["time"] < CUTOFF).all(), "funding entry at/after cutoff"
    n_exact = int(f.duplicated().sum())
    assert n_exact == 0, f"{n_exact} exact duplicate funding rows"
    n_keydup = int(f.duplicated(["trader", "time", "coin"], keep=False).sum())
    hourly = f["n_samples"].isna()
    assert (f.loc[~hourly, "time"] == f.loc[~hourly, "time"].dt.floor("D")).all(), "daily entry not at 00:00"
    by_day = hourly.groupby([f["trader"], f["coin"], f["time"].dt.floor("D")]).agg(["any", "all"])
    n_mixed = int((by_day["any"] & ~by_day["all"]).sum())
    write_parquet(f, BUILD / "funding.parquet")
    stats = {"addresses_requested": len(addresses), "addresses_complete": len(complete),
             "addresses_with_entries": with_entries, "addresses_empty": empty,
             "addresses_failed": len(failed_ever - complete),
             "addresses_not_fetched": len(set(addresses) - complete - failed_ever),
             "funding_rows": len(f), "funding_rows_dup_trader_time_coin": n_keydup,
             "funding_rows_hourly": int(hourly.sum()),
             "funding_addresses_with_hourly": int(f.loc[hourly, "trader"].nunique()),
             "funding_trader_coin_days_mixed_daily_hourly": n_mixed,
             "funding_non_funding_delta_types": other_types, "funding_http_requests": n_req,
             "funding_hip3_rows": int(f["coin"].str.contains(":", regex=False).sum()),
             "funding_hip3_coins": sorted(f.loc[f["coin"].str.contains(":", regex=False), "coin"].unique())}
    return f, complete, fetched_at, stats


# ----------------------------------------------------------------------------- HL candles
def build_hl(symbols):
    out, fetched_at, stats = {}, [], {}
    for iv in INTERVAL_MS:
        rows, n_empty, missing = [], [], []
        for s in symbols:
            path = RAW / "hl_candles" / f"{s.replace(':', '__')}_{iv}.json"
            if not path.exists():
                missing.append(s)
                continue
            d = json.loads(path.read_text())
            if d.get("complete") is not True:
                missing.append(s)
                continue
            fetched_at.append(d["fetched_at_utc"])
            if not d["candles"]:
                n_empty.append(s)
            for c in d["candles"]:
                rows.append((s, c["t"], float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"])))
        h = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume"])
        h = h[h["t"] < CUTOFF_MS]
        h.insert(1, "open_time", pd.to_datetime(h.pop("t"), unit="ms"))
        h = h.sort_values(["symbol", "open_time"]).reset_index(drop=True)
        assert not h.duplicated(["symbol", "open_time"]).any()
        write_parquet(h, BUILD / f"hl_candles_{iv}.parquet")
        out[iv] = h
        stats[f"hl_{iv}_rows"] = len(h)
        stats[f"hl_{iv}_symbols_nonempty"] = int(h["symbol"].nunique())
        stats[f"hl_{iv}_symbols_empty"] = n_empty
        stats[f"hl_{iv}_symbols_missing"] = len(missing)
    return out, fetched_at, stats


# ----------------------------------------------------------------------------- Binance 5m
def read_bn_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        assert len(names) == 1, f"{path.name}: members {z.namelist()}"
        raw = z.read(names[0])
    first = raw.split(b"\n", 1)[0].split(b",")[0].strip()
    header = 0 if not first.isdigit() else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=BN_COLS if header is None else None)
    if header is not None:
        assert list(df.columns) == BN_COLS, f"{path.name}: header {list(df.columns)}"
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    return df


def load_bn(sym: str):
    """(DataFrame with open_time ms int, notes list) or (None, notes) if no data."""
    d = RAW / "binance_5m" / sym
    st = json.loads((d / "_status.json").read_text())
    notes = []
    parts = [read_bn_zip(d / f"{sym}-5m-{stamp}.zip") for stamp, v in st["files"].items() if v == "ok"]
    if not st["complete"]:
        notes.append("binance_incomplete")
    if st.get("fallback_daily"):
        notes.append(f"daily_fallback_{len(st['fallback_daily'])}d")
    if not parts:
        return None, st, notes
    k = pd.concat(parts, ignore_index=True)
    k["open_time"] = k["open_time"].astype("int64")
    if k["open_time"].max() > 10**14:                  # microsecond timestamps
        notes.append("open_time_us")
        k.loc[k["open_time"] > 10**14, "open_time"] //= 1000
    dup = k.duplicated("open_time", keep=False)
    if dup.any():
        same = k[dup].drop_duplicates()
        n_conf = int(same.duplicated("open_time").sum())
        notes.append(f"dup_bars_{int(dup.sum())}_conflicting_{n_conf}")
        k = k.drop_duplicates("open_time", keep="first")
    k = k[k["open_time"] < CUTOFF_MS].sort_values("open_time").reset_index(drop=True)
    return k, st, notes


def validate(k: pd.DataFrame, hl1d: pd.DataFrame):
    """Binance daily close (close of the bar opening 23:55 UTC) vs Hyperliquid 1d close."""
    ot = pd.to_datetime(k["open_time"], unit="ms")
    bd = pd.Series(k["close"].values[(ot.dt.hour == 23) & (ot.dt.minute == 55)],
                   index=ot[(ot.dt.hour == 23) & (ot.dt.minute == 55)].dt.normalize().values)
    hd = pd.Series(hl1d["close"].values, index=hl1d["open_time"].values)
    common = bd.index.intersection(hd.index)
    res = {"n_common_days": len(common), "median_ratio": np.nan, "p5_bps": np.nan,
           "p95_bps": np.nan, "ret_corr": np.nan}
    if len(common) == 0:
        return res
    ratio = bd[common] / hd[common]
    res["median_ratio"] = float(ratio.median())
    res["p5_bps"] = float(np.percentile((ratio - 1) * 1e4, 5))
    res["p95_bps"] = float(np.percentile((ratio - 1) * 1e4, 95))
    days = pd.date_range(common.min(), common.max(), freq="D")
    rb = np.log(bd.reindex(days)).diff()
    rh = np.log(hd.reindex(days)).diff()
    m = rb.notna() & rh.notna()
    if m.sum() >= 3:
        res["ret_corr"] = float(np.corrcoef(rb[m], rh[m])[0, 1])
    return res


def build_binance(symbols, hl1d, notional_share):
    KL_DIR.mkdir(parents=True, exist_ok=True)
    for old in KL_DIR.glob("*.parquet"):           # rebuilt from scratch every run
        old.unlink()
    rows, fetched_at, bn_close = [], [], {}
    for s in symbols:
        r = {"package_symbol": s, "binance_symbol": binance_symbol(s) or "", "status": "",
             "first_bar": pd.NaT, "last_bar": pd.NaT, "n_bars": 0, "missing_bars": np.nan,
             "n_common_days": 0, "median_ratio": np.nan, "p5_bps": np.nan, "p95_bps": np.nan,
             "ret_corr": np.nan, "notional_share": float(notional_share.get(s, 0.0)), "note": ""}
        sym = binance_symbol(s)
        if sym is None:
            r["status"] = "hip3"
            rows.append(r)
            continue
        if not (RAW / "binance_5m" / sym / "_status.json").exists():
            r["status"] = "not_fetched"
            rows.append(r)
            continue
        k, st, notes = load_bn(sym)
        fetched_at.append(st["fetched_at_utc"])
        if k is None or k.empty:
            r["status"] = "not_listed"
            r["note"] = ";".join(notes)
            rows.append(r)
            continue
        r["first_bar"] = pd.to_datetime(k["open_time"].iloc[0], unit="ms")
        r["last_bar"] = pd.to_datetime(k["open_time"].iloc[-1], unit="ms")
        r["n_bars"] = len(k)
        r["missing_bars"] = int((k["open_time"].iloc[-1] - k["open_time"].iloc[0]) // BAR_MS + 1 - len(k))
        h = hl1d[hl1d["symbol"] == s]
        if h.empty:
            notes.append("no_hl_1d")
        r.update(validate(k, h))
        ok = (r["n_common_days"] >= 10 and 0.98 <= r["median_ratio"] <= 1.02
              and r["ret_corr"] >= 0.9)
        r["status"] = "accepted" if ok else "rejected"
        r["note"] = ";".join(notes)
        rows.append(r)
        if ok:
            out = k.copy()                              # scale 1: Binance units == package units when accepted
            out["open_time"] = pd.to_datetime(out["open_time"], unit="ms")
            assert (out["open_time"] < CUTOFF).all()
            write_parquet(out[["open_time", "open", "high", "low", "close", "volume"]], KL_DIR / f"{s}.parquet")
            bn_close[s] = out.set_index("open_time")["close"]
    sm = pd.DataFrame(rows)
    sm.to_csv(BUILD / "symbol_map.csv", index=False)
    return sm, bn_close, fetched_at


# ----------------------------------------------------------------------------- checks
def check_probe(f):
    x = f[(f["trader"] == PROBE_ADDR) & (f["time"] >= "2025-08-19") & (f["time"] < "2025-08-23")]
    n, s = len(x), round(float(x["usdc"].sum()), 6)
    ok = n == 11 and abs(s - (-4.333893)) < 1e-9
    print(f"[check 1] probe {PROBE_ADDR} 2025-08-19..22: entries={n} sum_usdc={s:.6f} "
          f"(expected 11, -4.333893) -> {'PASS' if ok else 'FAIL'}")
    if n:
        print("          by coin:", x.groupby("coin")["usdc"].sum().round(6).to_dict())
    return {"entries": n, "sum_usdc": s, "pass": ok}


def replay_days(traders):
    """One row per trader x symbol x UTC day from the first fill day to LAST_DAY, with the
    replayed position state: day_nz (nonzero at some time), day_pos / day_neg (long / short at
    some time), whole_day (nonzero throughout)."""
    g = pd.read_parquet(GROUPS_PARQUET, columns=["trader", "symbol", "ts", "gstart", "gend"])
    g = g[g["trader"].isin(traders)].sort_values(["trader", "symbol", "ts"])
    g["day"] = g["ts"].dt.floor("D")
    g["nz"] = (g["gstart"] != 0) | (g["gend"] != 0)
    g["pos"] = (g["gstart"] > 0) | (g["gend"] > 0)
    g["neg"] = (g["gstart"] < 0) | (g["gend"] < 0)
    g["end_nz"] = g["gend"] != 0
    d = g.groupby(["trader", "symbol", "day"], sort=True).agg(
        first_start=("gstart", "first"), last_end=("gend", "last"), any_nz=("nz", "any"),
        any_pos=("pos", "any"), any_neg=("neg", "any"), all_end_nz=("end_nz", "all")).reset_index()
    first = d.groupby(["trader", "symbol"])["day"].min().rename("first_day").reset_index()
    first_start = d.groupby(["trader", "symbol"])["first_start"].first().rename("first_gstart").reset_index()
    first = first.merge(first_start, on=["trader", "symbol"])
    grid = first.assign(day=[pd.date_range(x, LAST_DAY, freq="D") for x in first["first_day"]]).explode("day")
    grid["day"] = pd.to_datetime(grid["day"])
    grid = grid.merge(d, on=["trader", "symbol", "day"], how="left")
    grid = grid.sort_values(["trader", "symbol", "day"]).reset_index(drop=True)
    keys = [grid["trader"], grid["symbol"]]
    prev_end = grid.groupby(keys)["last_end"].shift().groupby(keys).ffill()
    start = prev_end.fillna(grid["first_start"])        # first fill day: position before its first fill
    grid["start_pos"] = start
    grid["day_nz"] = (start != 0) | grid["any_nz"].eq(True)
    grid["day_pos"] = (start > 0) | grid["any_pos"].eq(True)
    grid["day_neg"] = (start < 0) | grid["any_neg"].eq(True)
    grid["whole_day"] = (start != 0) & ~grid["all_end_nz"].eq(False)   # no fill that day -> unchanged
    return grid[["trader", "symbol", "day", "first_day", "first_gstart", "start_pos",
                 "day_nz", "day_pos", "day_neg", "whole_day"]], first


def check_replay(f, complete):
    traders = sorted(complete)
    grid, first = replay_days(traders)
    e = f.rename(columns={"coin": "symbol"}).copy()
    e["day"] = e["time"].dt.floor("D")
    e["absu"] = e["usdc"].abs()
    e = e.merge(first, on=["trader", "symbol"], how="left")
    tot_n, tot_u = len(e), e["absu"].sum()
    res = {}

    def share(mask):
        return int(mask.sum()), float(mask.mean()) if tot_n else np.nan, \
            float(e.loc[mask, "absu"].sum() / tot_u) if tot_u else np.nan

    nofill = e["first_day"].isna()
    before = ~nofill & (e["day"] < e["first_day"])
    inr = ~nofill & ~before
    print(f"[check 2] funding vs replay ({len(traders)} addresses, {tot_n} entries, |usdc| {tot_u:,.2f})")
    for name, m in [("trader x coin never filled in package", nofill),
                    ("dated before first fill of trader x coin", before),
                    ("on/after first fill day", inr)]:
        c, sn, su = share(m)
        print(f"          {name:42s} n={c:7d} share_n={sn:.4f} share_|usdc|={su:.4f}")
        res[name] = {"n": c, "share_n": sn, "share_abs_usdc": su}
    for shift_days, label in [(0, "same day (used)"), (1, "label = previous day (alt)")]:
        ee = e.assign(day=e["day"] - pd.Timedelta(days=shift_days))
        m = ee.merge(grid[["trader", "symbol", "day", "day_nz", "day_pos", "day_neg"]],
                     on=["trader", "symbol", "day"], how="left")
        nz = m["day_nz"].eq(True).values
        sg = (((m["szi"] > 0) & m["day_pos"].eq(True)) | ((m["szi"] < 0) & m["day_neg"].eq(True))).values
        for nm, mask in [("nonzero replayed position that day", nz), ("szi sign matches replay that day", sg)]:
            c, sn, su = share(pd.Series(mask, index=e.index))
            print(f"          (i) [{label}] {nm:36s} n={c:7d} share_n={sn:.4f} share_|usdc|={su:.4f}")
            res[f"(i) {label}: {nm}"] = {"n": c, "share_n": sn, "share_abs_usdc": su}
    # (ii) whole-day positions without an entry
    have = e[["trader", "symbol", "day"]].drop_duplicates().assign(has_entry=True)
    w = grid[grid["whole_day"]].merge(have, on=["trader", "symbol", "day"], how="left")
    miss = w["has_entry"].isna()
    n_hip3 = int((miss & w["symbol"].str.contains(":", regex=False)).sum())
    print(f"          (ii) trader x coin x days held nonzero all day: {len(w)}; without funding entry: "
          f"{int(miss.sum())} (HIP-3: {n_hip3})")
    res["(ii)"] = {"whole_days": len(w), "without_entry": int(miss.sum()), "without_entry_hip3": n_hip3}
    # (iii) entries before the first fill (positions carried into the window)
    b = e[before]
    carried = ((b["szi"] > 0) & (b["first_gstart"] > 0)) | ((b["szi"] < 0) & (b["first_gstart"] < 0))
    print(f"          (iii) entries before first fill: n={len(b)} |usdc|={b['absu'].sum():,.2f}; "
          f"of which szi sign == sign of position before first fill: n={int(carried.sum())} "
          f"|usdc|={b.loc[carried, 'absu'].sum():,.2f}")
    res["(iii)"] = {"n": len(b), "abs_usdc": float(b["absu"].sum()), "n_carried_sign_match": int(carried.sum())}
    return res


def check_coverage(sm):
    acc = sm["status"] == "accepted"
    share = float(sm.loc[acc, "notional_share"].sum())
    print(f"[check 3] package perp notional on accepted Binance mappings: {share:.4f} "
          f"({int(acc.sum())} symbols)")
    top = sm[~acc].sort_values("notional_share", ascending=False).head(15)
    print("          15 largest without accepted mapping:")
    for _, r in top.iterrows():
        print(f"            {r['package_symbol']:16s} {r['notional_share']:.4f}  {r['status']}")
    return {"accepted_notional_share": share,
            "top_unmapped": top[["package_symbol", "notional_share", "status"]].to_dict("records")}


def check_fill_vs_binance(p, bn_close, n=20_000, seed=0):
    if not bn_close:
        print("[check 4] no accepted symbols")
        return {}
    q = p[p["symbol"].isin(list(bn_close))].sort_values(["symbol", "ts", "tid"])
    q = q.assign(bar=q["ts"].dt.floor("5min"))
    last_px = q.groupby(["symbol", "bar"])["price"].last().rename("last_px")
    smp = q.sample(n=min(n, len(q)), random_state=seed)[["symbol", "bar"]].join(last_px, on=["symbol", "bar"])
    bc = pd.concat({s: c for s, c in bn_close.items()}, names=["symbol", "bar"]).rename("bn_close")
    smp = smp.join(bc, on=["symbol", "bar"])
    miss = int(smp["bn_close"].isna().sum())
    smp = smp.dropna(subset=["bn_close"])
    dev = (smp["bn_close"] / smp["last_px"] - 1).abs() * 1e4
    out = {"n_sampled": int(len(smp) + miss), "n_no_binance_bar": miss, "median_bps": float(dev.median()),
           "p95_bps": float(dev.quantile(0.95)), "max_bps": float(dev.max())}
    print(f"[check 4] |Binance 5m close / last package fill px in bar - 1| on {out['n_sampled']} sampled fills "
          f"(seed {seed}; {miss} without a Binance bar): median={out['median_bps']:.2f} bps "
          f"p95={out['p95_bps']:.2f} bps max={out['max_bps']:.2f} bps")
    per = dev.groupby(smp["symbol"]).agg(["size", "median", lambda s: s.quantile(0.95), "max"])
    per.columns = ["n", "median_bps", "p95_bps", "max_bps"]
    print(per.sort_values("n", ascending=False).head(20).round(2).to_string())
    return out


def build_hidden(f, p):
    """Positions the package fills cannot show: trader x coin pairs with funding entries inside
    the fill window (>= WINDOW_START) but no fill in the package. One row per pair x UTC day with
    a funding entry; szi = position reported by the last entry of that day. Returns the table."""
    pairs = set(zip(p["trader"], p["symbol"]))
    w = f[f["time"] >= WINDOW_START]
    w = w[[k not in pairs for k in zip(w["trader"], w["coin"])]]
    w = w.assign(day=w["time"].dt.floor("D")).sort_values(["trader", "coin", "time"], kind="stable")
    h = w.groupby(["trader", "coin", "day"], sort=True).agg(
        szi=("szi", "last"), usdc=("usdc", "sum"), n_entries=("usdc", "size")).reset_index()
    write_parquet(h, BUILD / "hidden_positions.parquet")
    g = h.groupby(["trader", "coin"]).agg(first=("day", "min"), last=("day", "max"),
                                          n_days=("day", "size"), n_szi=("szi", "nunique"))
    whole = (g["first"] == WINDOW_START) & (g["last"] == LAST_DAY)
    span_days = (g["last"] - g["first"]).dt.days + 1
    out = {"pairs": len(g), "addresses": int(h["trader"].nunique()), "rows": len(h),
           "pairs_whole_window": int(whole.sum()), "pairs_constant_szi": int((g["n_szi"] == 1).sum()),
           "pairs_with_gap_days": int((span_days != g["n_days"]).sum())}
    print(f"[hidden] trader x coin with funding in the window but no package fill: {out}")
    return out


def check_quiet_days(f):
    """Check 5: on days after the pair's first package fill day with no fill of that pair, the
    position reported by funding must equal the replayed position (last gend before that day)."""
    g = pd.read_parquet(GROUPS_PARQUET, columns=["trader", "symbol", "ts", "gend"])
    g = g.sort_values(["trader", "symbol", "ts"], kind="stable")
    g["day"] = g["ts"].dt.floor("D")
    first_day = g.groupby(["trader", "symbol"])["day"].min().rename("first_day").reset_index()
    ev = g[["trader", "symbol", "day"]].drop_duplicates().assign(ev=True)
    e = f.rename(columns={"coin": "symbol"})
    e = e.assign(day=e["time"].dt.floor("D")).sort_values(["trader", "symbol", "time"], kind="stable")
    d = e.groupby(["trader", "symbol", "day"]).agg(szi=("szi", "last"), szi_min=("szi", "min"),
                                                   szi_max=("szi", "max")).reset_index()
    d = d.merge(first_day, on=["trader", "symbol"]).merge(ev, on=["trader", "symbol", "day"], how="left")
    q = d[d["ev"].isna() & (d["day"] > d["first_day"])].sort_values("day")
    q = pd.merge_asof(q.assign(ts=q["day"]), g[["trader", "symbol", "ts", "gend"]].sort_values("ts"),
                      on="ts", by=["trader", "symbol"], direction="backward", allow_exact_matches=False)
    tol = 1e-6 * np.maximum(q["szi"].abs(), q["gend"].abs()) + 1e-9
    bad = ((q["szi"] - q["gend"]).abs() > tol) | ((q["szi_max"] - q["szi_min"]).abs() > tol)
    out = {"quiet_days": len(q), "mismatch": int(bad.sum()),
           "mismatch_pairs": int(q[bad].groupby(["trader", "symbol"]).ngroups)}
    print(f"[check 5] days without a fill after the pair's first fill day, with a funding entry: "
          f"{out['quiet_days']}; funding position != replayed position: {out['mismatch']} "
          f"({out['mismatch_pairs']} pairs)")
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = load_package()
    addresses = sorted(p["trader"].unique())
    symbols = sorted(p["symbol"].unique())
    notional_share = p.groupby("symbol")["notional"].sum() / p["notional"].sum()

    f, complete, fa_f, st_f = build_funding(addresses)
    hl, fa_h, st_h = build_hl(symbols)
    sm, bn_close, fa_b = build_binance(symbols, hl["1d"], notional_share)

    print("[symbol_map] status counts:", sm["status"].value_counts().to_dict())
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sm[sm["status"].isin(["accepted", "rejected"])].to_string(index=False))

    checks = {"1": check_probe(f), "2": check_replay(f, complete), "3": check_coverage(sm),
              "4": check_fill_vs_binance(p, bn_close), "5": check_quiet_days(f)}
    hidden = build_hidden(f, p)

    built = [BUILD / "funding.parquet", BUILD / "hl_candles_1d.parquet", BUILD / "hl_candles_4h.parquet",
             BUILD / "symbol_map.csv", BUILD / "hidden_positions.parquet"] + sorted(KL_DIR.glob("*.parquet"))
    fetched = sorted(fa_f + fa_h + fa_b)
    raw_bytes = sum(x.stat().st_size for x in RAW.rglob("*") if x.is_file())
    vc = sm["status"].value_counts()
    manifest = {
        "built_at_utc": utc_now_iso(), "cutoff_utc": str(CUTOFF),
        **st_f, **st_h,
        "symbols_package": len(symbols),
        **{f"symbols_{k}": int(vc.get(k, 0)) for k in
           ["accepted", "rejected", "not_listed", "hip3", "not_fetched"]},
        "klines_5m_rows": int(sm.loc[sm["status"] == "accepted", "n_bars"].sum()),
        "earliest_fetch_utc": fetched[0] if fetched else None,
        "latest_fetch_utc": fetched[-1] if fetched else None,
        "raw_bytes": raw_bytes,
        "sha256": {str(x.relative_to(BUILD)): sha256_file(x) for x in built},
        "checks": checks,
        "hidden_positions": hidden,
    }
    (BUILD / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"[manifest] funding rows={st_f['funding_rows']} (dup trader,time,coin={st_f['funding_rows_dup_trader_time_coin']}) "
          f"addresses complete={st_f['addresses_complete']} with_entries={st_f['addresses_with_entries']} "
          f"empty={st_f['addresses_empty']} failed={st_f['addresses_failed']} not_fetched={st_f['addresses_not_fetched']}; "
          f"hourly rows={st_f['funding_rows_hourly']} on {st_f['funding_addresses_with_hourly']} addresses, "
          f"trader x coin x days with both forms={st_f['funding_trader_coin_days_mixed_daily_hourly']}; "
          f"HIP-3 funding rows={st_f['funding_hip3_rows']} coins={st_f['funding_hip3_coins']}; "
          f"raw size={raw_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
