"""Rolling per-address profile ledger: every column of work/codex10/REVIEW.md section 2.2 for many windows
S = [a, b), each computed strictly from the information available at its own cutoff C = b (section 2.1).

Windows: for the 11 Mondays M of section 2.1, L60 = [M-60d, M), F28 = [M, M+28d), F1..F4 (the forward weeks),
R28 = [M-28d, M); plus three reference windows used only by the checks: ref_pre [07-27, 11-03),
ref_post [11-03, 01-05), ref_full [07-27, 01-05). profile_windows / profile_window take any list of
midnight-aligned windows inside [W0, W1], so further lookbacks need no code change.

Availability at C (section 2.1): fills, snapshots and hourly funding with event time < C; a daily funding
aggregate for day D when D + 1d <= C; price bars closed by C. Every input is filtered by these rules, either
explicitly here or by construction (shown in comments where it is by construction).

Reused reviewed logic:
  lib.py         loaders, spot exclusion, forced flags, fill groups, POS_EPS
  profile.py     build_episodes, assign_fills (imported)
  exposure.py    pair_pos, carried_start, av_grid (imported); price priority and funding-only rule (copied,
                 because they read files directly and the prefix check needs in-memory truncated inputs)
  style.py       episode labels, opening legs, 60 s clustering, grid_arrays, A baseline (copied: style.py
                 loads its own inputs inside build_clusters and overwrites ARK_* env vars on import)

Inputs (read-only): ArkStream package (fills, traders), work/extdata/build/{funding.parquet, hl_candles_4h.parquet,
klines_5m/, symbol_map.csv, hidden_positions.parquet (assertion only)}.
Outputs (work/roll/build): ledger.parquet, top_symbol.parquet, support.parquet, windows.csv, decisions.parquet,
episodes.parquet, clusters.parquet, address_sums.parquet, gate_counts.csv, manifest.json; log
work/roll/logs/ledger.log. `--checks` runs the acceptance checks on the built outputs (checks.json,
logs/checks.log).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True      # importing the reviewed scripts must not write __pycache__ next to them
ROOT = Path("/Users/xc/Documents/arkstream_quest")
ROLL = ROOT / "work/roll"
OUT = ROLL / "build"
LOGS = ROLL / "logs"
os.environ["ARK_DATA_DIR"] = str(ROOT / "ArkStream笔试题_v2")
# lib mkdirs its build / tables dirs on import; point both at our own build dir
os.environ["ARK_BUILD_DIR"] = str(OUT)
os.environ["ARK_TABLES_DIR"] = str(OUT)
sys.path.insert(0, str(ROOT / "work/v2dev/code"))
import lib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy  # noqa: E402

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


def _load(name, path):
    """Import a reviewed script by path (`profile` is a stdlib name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROF = _load("profile_v1", ROOT / "work/profile/profile.py")
EXPO = _load("exposure_v2", ROOT / "work/profile2/code/exposure.py")

EXT = ROOT / "work/extdata/build"
KLINES = EXT / "klines_5m"
SPEC = ROOT / "work/codex10/REVIEW.md"

# exposure grid (bar open o on the 5-minute lattice from W0, close c = o + 5 min)
W0, W1 = EXPO.W0, EXPO.W1
W0N, W1N = EXPO.W0N, EXPO.W1N
STEP, DAY, HOUR = EXPO.STEP, EXPO.DAY, EXPO.HOUR
N, PPD = EXPO.N, EXPO.PPD
OPEN, CLOSE = EXPO.OPEN, EXPO.CLOSE
AV_MIN = EXPO.AV_MIN
PRE_END = pd.Timestamp("2025-11-03")
assert lib.CUTOFF == W1 and lib.ASOF == W1, "ARK_CUTOFF / ARK_ASOF must not be set for the ledger"

# alignment grid and constants (style.py)
G0 = pd.Timestamp("2025-07-01 00:05:00")           # close time of the first bar of any kline file
G0N = G0.value
G_END = pd.Timestamp("2026-01-05 00:00:00")
NGRID = int((G_END - G0) / pd.Timedelta(minutes=5)) + 1
HORIZONS = {"1h": 1, "4h": 4, "1d": 24, "7d": 168}  # hours
GAP_NS = 60 * 10**9
MAX_DUR_S = 300
VOL_BARS = 288
A_MIN = 5
A_LO, A_HI = 2 / 3, 1.5
H4 = 4 * HOUR
CLASSES = {"taker": "w_taker", "maker": "w_maker"}
VARIANTS = ("raw", "A")
INT_MAX = np.iinfo(np.int64).max
SEED = 20261120                                     # recorded only: the ledger draws no random numbers

MONDAYS = pd.date_range("2025-09-29", "2025-12-08", freq="7D")
REFS = {"ref_pre": (W0, PRE_END), "ref_post": (PRE_END, W1), "ref_full": (W0, W1)}
EXPECTED_L60_BASE = [320, 327, 338, 358, 362, 360, 380, 382, 392, 396, 393]

FILL_COLS = ["active_day_share", "orders_per_active_day_med", "clusters_per_active_day_med", "orders_per_cluster",
             "fills_per_order", "maker_share", "order_ntl_med", "cluster_ntl_med", "turnover_per_active_day_med",
             "n_symbols", "top1_share", "sym_hhi", "btc_eth_share", "long_open_share", "forced_ntl_share"]
EPISODE_COLS = ["hold_med_h", "hold_wavg_h", "adds_per_ep_med", "single_shot_share", "flip_share"]
ADD_COLS = ["add_ntl_share", "add_loser_share"]
GRID_COLS = ["time_in_market", "gross_exp_med", "gross_exp_p95", "net_bias", "long_time_share", "n_pos_med",
             "hold_turnover_h", "lev_med", "lev_p95", "av_med"]
ALIGN_CELLS = [f"{c}_{h}_{v}" for c in CLASSES for h in HORIZONS for v in VARIANTS]
ALIGN_COLS = [f"align_{k}" for k in ALIGN_CELLS]
VALUE_COLS = FILL_COLS + EPISODE_COLS + ADD_COLS + GRID_COLS + ALIGN_COLS
CELL_SUPPORT = ["n", "days", "w", "wmax_share", "kish", "first_n", "first_w", "cov"]
SUPPORT_COLS = (
    ["no_fill", "base_eligible", "hip3_asof", "n_fills", "n_orders", "n_fill_days", "n_clusters", "turnover",
     "maker_ntl", "forced_ntl", "open_long_ntl", "open_ls_ntl", "first_fill_prefix", "active_day_denom",
     # grid
     "first_appearance", "n_span_points", "n_valid_points", "valid_grid_days", "n_missing_px_points",
     "n_pospts_binance", "n_pospts_hl", "n_pospts_fill",
     "n_inm_points", "n_inm_days", "n_lev_points", "n_lev_days", "n_inm_av_missing", "n_inm_av_lt100",
     "held_ntl_h", "n_snap", "n_snap_days",
     # episodes
     "n_complete", "open_ntl_complete", "n_ep_fill", "n_left_cens_fill", "n_right_cens_fill", "n_fo_runs",
     "n_held_through", "n_episodes_x", "n_left_cens_x", "n_right_cens_x",
     # add behaviour
     "n_inc_groups", "n_inc_days", "n_later_add_groups", "n_later_add_days", "inc_ntl_opened", "later_add_ntl",
     "loser_add_ntl",
     # alignment
     "open_ntl_forced", "open_ntl_hip3", "open_ntl_ambiguous", "n_open_clusters", "cand_w_taker", "cand_w_maker"]
    + [f"{k}_{s}" for k in ALIGN_CELLS for s in CELL_SUPPORT])

_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


def window_table():
    """window_id, kind, M, a, b for the 77 rolling windows and the 3 reference windows."""
    rows = []
    d = pd.Timedelta(days=1)
    for M in MONDAYS:
        ms = f"{M:%Y-%m-%d}"
        rows.append((f"L60_{ms}", "L60", M, M - 60 * d, M))
        rows.append((f"F28_{ms}", "F28", M, M, M + 28 * d))
        for k in range(1, 5):
            rows.append((f"F{k}_{ms}", f"F{k}", M, M + 7 * (k - 1) * d, M + 7 * k * d))
        rows.append((f"R28_{ms}", "R28", M, M - 28 * d, M))
    for wid, (a, b) in REFS.items():
        rows.append((wid, "ref", pd.NaT, a, b))
    return pd.DataFrame(rows, columns=["window_id", "kind", "M", "a", "b"])


# ---------------------------------------------------------------------------------------
# Inputs (raw, possibly physically truncated) and window-independent preparation
# ---------------------------------------------------------------------------------------
def load_inputs():
    p = lib.prep_perp_fills()
    tr = lib.load_traders()[["trader", "ts", "acct_val"]]
    fund = pd.read_parquet(EXT / "funding.parquet")
    kl = {f.name[:-8]: pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(KLINES.glob("*.parquet"))}
    hl = pd.read_parquet(EXT / "hl_candles_4h.parquet", columns=["symbol", "open_time", "close"])
    return dict(p=p, tr=tr, fund=fund, kl=kl, hl=hl)


def truncate_inputs(inp, C):
    """Physically remove everything not available at C: fills / snapshots / hourly funding with time >= C,
    daily funding for D with D + 1d > C, kline / HL bars closing after C. The kline dict keeps its keys (the
    fixed symbol map), possibly with empty frames."""
    C = pd.Timestamp(C)
    fund = inp["fund"]
    daily = fund["n_samples"].notna()
    keep = (daily & (fund["time"] + pd.Timedelta(days=1) <= C)) | (~daily & (fund["time"] < C))
    out = dict(p=inp["p"][inp["p"]["ts"] < C].copy(), tr=inp["tr"][inp["tr"]["ts"] < C].copy(),
               fund=fund[keep].copy(),
               kl={s: k[k["open_time"] + pd.Timedelta(minutes=5) <= C].copy() for s, k in inp["kl"].items()},
               hl=inp["hl"][inp["hl"]["open_time"] + pd.Timedelta(hours=4) <= C].copy())
    removed = dict(fills=len(inp["p"]) - len(out["p"]), snapshots=len(inp["tr"]) - len(out["tr"]),
                   funding=len(fund) - len(out["fund"]),
                   kline_bars=sum(len(inp["kl"][s]) - len(out["kl"][s]) for s in inp["kl"]),
                   hl_bars=len(inp["hl"]) - len(out["hl"]))
    return out, removed


def build_prices(p, kl, hl, symbols):
    """exposure.build_prices on in-memory inputs. px[s, i] at close CLOSE[i]: Binance 5m bar closing at c >
    latest HL 4h candle closed by c > last package fill (any address) at or before c."""
    S = len(symbols)
    px = np.full((S, N), np.nan)
    src = np.zeros((S, N), dtype=np.int8)
    tape = p[["symbol", "ts_ns", "tid", "price"]].sort_values(["symbol", "ts_ns", "tid"], kind="stable")
    tape_by = {s: (d["ts_ns"].to_numpy(), d["price"].to_numpy()) for s, d in tape.groupby("symbol")}
    h = hl[hl["close"].notna()].sort_values(["symbol", "open_time"])
    hl_by = {s: ((d["open_time"] + pd.Timedelta(hours=4)).to_numpy().astype("int64"), d["close"].to_numpy())
             for s, d in h.groupby("symbol")}
    for si, sym in enumerate(symbols):
        if sym in tape_by:
            t, v = tape_by[sym]
            j = np.searchsorted(t, CLOSE, side="right") - 1
            ok = j >= 0
            px[si, ok] = v[j[ok]]
            src[si, ok] = 3
        if sym in hl_by:
            t, v = hl_by[sym]
            j = np.searchsorted(t, CLOSE, side="right") - 1
            ok = j >= 0
            px[si, ok] = v[j[ok]]
            src[si, ok] = 2
        if sym in kl:
            k = kl[sym]
            k = k[(k["open_time"] >= W0) & (k["open_time"] < W1) & k["close"].notna()]
            o = k["open_time"].to_numpy().astype("int64") - W0N
            al = o % STEP == 0
            i = o[al] // STEP
            px[si, i] = k["close"].to_numpy()[al]
            src[si, i] = 1
    return px, src, tape_by


def align_arrays(k):
    """style.grid_arrays on an in-memory kline frame: r_h, sigma_h, q_h on the global close grid."""
    lp = np.full(NGRID, np.nan)
    if len(k):
        idx = (k["open_time"].to_numpy().astype("int64") + STEP - G0N) / STEP
        assert np.all(idx == np.round(idx)) and idx.min() >= 0 and idx.max() < NGRID
        lp[idx.astype(np.int64)] = np.log(k["close"].to_numpy())
    ret = np.full(NGRID, np.nan)
    ret[1:] = lp[1:] - lp[:-1]
    V = np.full(NGRID, np.nan)
    V[VOL_BARS:] = (np.lib.stride_tricks.sliding_window_view(ret[1:], VOL_BARS) ** 2).sum(axis=1)
    V[~(V > 0)] = np.nan                       # zero variance gives no usable scale
    out = {}
    for h, hh in HORIZONS.items():
        L = hh * 12
        r = np.full(NGRID, np.nan)
        r[L:] = lp[L:] - lp[:-L]
        sig = np.sqrt(V * hh / 24)
        out[f"sigma_{h}"] = sig
        out[f"q_{h}"] = r / sig
    return out


def build_prep(inp, U):
    """Window-independent state built only from `inp` (so a truncated input set gives a truncated state)."""
    P = SimpleNamespace()
    P.U = np.asarray(U)
    P.nU = len(U)
    tmap = {t: i for i, t in enumerate(P.U)}
    p = inp["p"].copy()
    assert p["trader"].isin(tmap).all()
    p["ti"] = p["trader"].map(tmap).astype(np.int64)
    p["ts_ns"] = p["ts"].to_numpy().astype("int64")
    p["day_ns"] = p["ts_ns"] - p["ts_ns"] % DAY

    # fill groups (lib) and pairs
    g = lib.fill_groups(p).sort_values(["trader", "symbol", "ts"], kind="stable").reset_index(drop=True)
    pairs = g[["trader", "symbol"]].drop_duplicates().reset_index(drop=True)
    pairs["pid"] = np.arange(len(pairs))
    pairs["ti"] = pairs["trader"].map(tmap).astype(np.int64)
    g = g.merge(pairs, on=["trader", "symbol"], how="left")
    g["ts_ns"] = g["ts"].to_numpy().astype("int64")
    # style.py episode labels on the prefix: new pair, flat start or sign flip starts a label
    new_pair = g["pid"].ne(g["pid"].shift()).to_numpy()
    flip = ((g["gstart"] * g["gend"]) < 0).to_numpy()
    sff = (g["gstart"].eq(0).to_numpy() | flip)
    lab = np.cumsum(new_pair | sff)
    g["episode"] = lab
    # observed opening time of a label = ts of its first group if that group starts flat / flips, else none
    ep_open = np.full(lab.max() + 1 if len(lab) else 1, -1, dtype=np.int64)
    first = new_pair | sff
    ep_open[lab[first & sff]] = g["ts_ns"].to_numpy()[first & sff]
    P.ep_open = ep_open
    p = p.merge(g[["trader", "symbol", "ts", "episode", "mixed"]], on=["trader", "symbol", "ts"], how="left",
                validate="many_to_one")
    assert p["episode"].notna().all()
    # opening quantity per fill (style.py); a flip contributes only the residual new position
    q = p["start_position"].to_numpy().copy()
    q[np.abs(q) < lib.POS_EPS] = 0.0
    d = p["signed"].to_numpy()
    open_q = np.where(q * d >= 0, np.abs(d), np.maximum(np.abs(d) - np.abs(q), 0.0))
    open_q[open_q < lib.POS_EPS] = 0.0
    p["open_ntl"] = open_q * p["price"].to_numpy()
    P.p = p[["ti", "symbol", "ts", "ts_ns", "day_ns", "oid", "tid", "sgn", "dir", "price", "notional", "is_maker",
             "maker_notional", "forced_notional", "is_forced", "is_hip3", "episode", "mixed", "open_ntl"]]
    P.p_ts = P.p["ts_ns"].to_numpy()
    P.g = g[["pid", "ti", "ts", "ts_ns", "gstart", "gend", "signed", "notional", "vwap"]]
    P.g_ts = g["ts_ns"].to_numpy()
    P.g_gend = g["gend"].to_numpy()
    ff = p.groupby("ti")["ts_ns"].min()
    P.first_fill = np.full(P.nU, INT_MAX, dtype=np.int64)
    P.first_fill[ff.index.to_numpy()] = ff.to_numpy()

    # funding: universe only; carried-in start (exposure.carried_start) on all funding days of the pair
    fund = inp["fund"][inp["fund"]["trader"].isin(tmap)].copy()
    fund["day_ns"] = fund["time"].dt.floor("D").to_numpy().astype("int64")
    rowno = pd.Series(np.arange(len(g)), index=g.index)
    pairs = pairs.join(pd.DataFrame({"r0": rowno.groupby(g["pid"]).min(), "r1": rowno.groupby(g["pid"]).max() + 1}),
                       on="pid")
    pairs["first_ts"] = g["ts_ns"].to_numpy()[pairs["r0"].to_numpy()]
    pairs["gstart0"] = g["gstart"].to_numpy()[pairs["r0"].to_numpy()]
    car = pairs[pairs["gstart0"] != 0]
    fd = fund[["trader", "coin", "day_ns"]].drop_duplicates().merge(
        car[["trader", "symbol"]].rename(columns={"symbol": "coin"}), on=["trader", "coin"])
    fdays = {k: set(v.to_numpy()) for k, v in fd.groupby(["trader", "coin"])["day_ns"]}
    carry = {r.pid: EXPO.carried_start(fdays.get((r.trader, r.symbol), set()), r.first_ts - r.first_ts % DAY)
             for r in car.itertuples(index=False)}
    pairs["carry_ns"] = [carry.get(k) for k in pairs["pid"]]
    # funding-only representation (exposure.funding_only): day's last szi by funding time, days >= W0,
    # |szi| >= POS_EPS. With midnight cutoffs every record of day D is usable at C iff D < C.
    f = fund[fund["time"] >= W0].sort_values("time", kind="stable")
    fday = f.groupby(["trader", "coin", "day_ns"])["szi"].last().reset_index()
    fday = fday[fday["szi"].abs() >= lib.POS_EPS].sort_values(["trader", "coin", "day_ns"]).reset_index(drop=True)
    fday["ti"] = fday["trader"].map(tmap).astype(np.int64)
    pf = dict(zip(zip(pairs["trader"], pairs["symbol"]), pairs["first_ts"].tolist()))    # exact int ns
    fday["pkg_first_ns"] = np.array([pf.get(k, INT_MAX) for k in zip(fday["trader"], fday["coin"])], dtype=np.int64)
    P.symbols = sorted(set(g["symbol"]) | set(fday["coin"]))
    sym_i = {s: i for i, s in enumerate(P.symbols)}
    pairs["sym_i"] = pairs["symbol"].map(sym_i)
    fday["sym_i"] = fday["coin"].map(sym_i)
    P.pairs = pairs
    P.pair_ti = pairs["ti"].to_numpy()
    P.pair_sym = pairs["symbol"].to_numpy()
    P.pairs_by_ti = {t: list(d.itertuples(index=False)) for t, d in pairs.groupby("ti")}
    P.fday = fday
    P.fund_by_ti = {}
    for (t, c), d in fday.groupby(["ti", "coin"], sort=True):
        P.fund_by_ti.setdefault(t, []).append((c, d["day_ns"].to_numpy(), d["szi"].to_numpy(), sym_i[c]))

    # HIP-3 evidence: fills / hourly funding usable iff time < C; daily funding iff D + 1d <= C
    h3 = p.loc[p["is_hip3"]].groupby("ti")["ts_ns"].min()
    fh = fund[fund["coin"].str.contains(":", regex=False)]
    hourly = fh["n_samples"].isna()
    fh_ti = fh["trader"].map(tmap)
    hh = fh.loc[hourly, "time"].astype("int64").groupby(fh_ti[hourly]).min()
    hd = (fh.loc[~hourly, "time"] + pd.Timedelta(days=1)).astype("int64").groupby(fh_ti[~hourly]).min()
    P.h3_lt = np.full(P.nU, INT_MAX, dtype=np.int64)
    P.h3_le = np.full(P.nU, INT_MAX, dtype=np.int64)
    for s in (h3, hh):
        P.h3_lt[s.index.to_numpy()] = np.minimum(P.h3_lt[s.index.to_numpy()], s.to_numpy())
    P.h3_le[hd.index.to_numpy()] = hd.to_numpy()

    # snapshots
    tr = inp["tr"][inp["tr"]["trader"].isin(tmap)]
    P.snaps = {tmap[t]: (d["ts"].to_numpy().astype("int64"), d["acct_val"].to_numpy()) for t, d in tr.groupby("trader")}

    # prices
    P.px, P.src, P.tape_by = build_prices(p, inp["kl"], inp["hl"], P.symbols)
    P.kl = inp["kl"]
    P.align = {}
    return P


def price_at_cutoff(P, C):
    """Price of every symbol at the last grid close c = C, with the fallback fill restricted to t < C."""
    i = (C - W0N) // STEP - 1
    out = P.px[:, i].copy()
    for si in np.flatnonzero(P.src[:, i] == 3):
        t, v = P.tape_by[P.symbols[si]]
        j = np.searchsorted(t, C, side="left") - 1
        out[si] = v[j] if j >= 0 else np.nan
    return out


def get_align(P, sym):
    if sym not in P.align:
        P.align[sym] = align_arrays(P.kl[sym]) if sym in P.kl else None
    return P.align[sym]


# ---------------------------------------------------------------------------------------
# Position grid (all windows at once, trader-outer)
# ---------------------------------------------------------------------------------------
def q_(x, pr):
    return float(np.quantile(x, pr)) if len(x) else np.nan


def med_(x):
    return float(np.median(x)) if len(x) else np.nan


def window_grid_stats(w, i0, G, Nn, Pn, Iv, Sx, fa, sn_ts, sn_av):
    a, b = w
    ia, ib = (a - W0N) // STEP - i0, (b - W0N) // STEP - i0
    g, n, pc, iv = G[ia:ib], Nn[ia:ib], Pn[ia:ib], Iv[ia:ib] > 0
    op = OPEN[ia + i0:ib + i0]
    span = (op >= fa) if fa is not None else np.zeros(len(op), dtype=bool)
    I = span & ~iv
    J = I & (g > 0)
    nI, nJ = int(I.sum()), int(J.sum())
    r = {"first_appearance": int(fa) if (fa is not None and fa < b) else -1,     # int ns, -1 = none
         "n_span_points": int(span.sum()), "n_valid_points": nI, "valid_grid_days": nI * STEP / DAY,
         "n_missing_px_points": int((span & iv).sum()), "n_inm_points": nJ,
         "n_inm_days": int(J.reshape(-1, PPD).any(1).sum()),
         "time_in_market": nJ / nI if nI else np.nan}
    for k, name in ((1, "binance"), (2, "hl"), (3, "fill")):  # nonzero position x valid point, by price source
        r[f"n_pospts_{name}"] = int(Sx[k - 1][ia:ib][I].sum())
    gj = g[J]
    r["gross_exp_med"] = med_(gj)
    r["gross_exp_p95"] = q_(gj, 0.95)
    sg = g[I].sum()
    r["net_bias"] = n[I].sum() / sg if sg > 0 else np.nan
    r["long_time_share"] = float((n[J] > 0).mean()) if nJ else np.nan
    r["n_pos_med"] = med_(pc[J])
    r["held_ntl_h"] = sg * STEP / HOUR
    av = EXPO.av_grid(sn_ts, sn_av, b, CLOSE[ia + i0:ib + i0])
    fin = np.isfinite(av)
    lv = J & fin & (av >= AV_MIN)
    lev = g[lv] / av[lv]
    r["lev_med"] = med_(lev)
    r["lev_p95"] = q_(lev, 0.95)
    r["n_lev_points"] = int(lv.sum())
    r["n_lev_days"] = int(lv.reshape(-1, PPD).any(1).sum())
    r["n_inm_av_missing"] = int((J & ~fin).sum())
    r["n_inm_av_lt100"] = int((J & fin & (av < AV_MIN)).sum())
    k0, k1 = np.searchsorted(sn_ts, [a, b], side="left")
    v, t = sn_av[k0:k1], sn_ts[k0:k1]
    ok = np.isfinite(v)
    r["n_snap"] = int(ok.sum())
    r["n_snap_days"] = int(len(np.unique(t[ok] // DAY)))
    r["av_med"] = med_(v[ok])
    return r


def grid_windows(P, wins):
    """Grid columns for every window in wins (list of (a_ns, b_ns)); returns {w: DataFrame indexed by ti}.

    For a cutoff C the address's grid is the sum, in ascending symbol order, of one contribution per symbol:
    the package replay (exposure.pair_pos incl. its carried-in start) if the pair has a fill before C, else the
    funding-only daily szi if the pair has funding days before C. On bars closing before C both are
    prefix-identical to a rebuild from inputs truncated at C (they use only events / fills / candles at
    t <= c < C, and every funding record of a day D < C); the bar closing exactly at C is recomputed with
    events and fallback fills restricted to t < C."""
    cutoffs = sorted({b for _, b in wins})
    by_c = {C: sorted(w for w in wins if w[1] == C) for C in cutoffs}
    lo_c = {C: min(a for a, _ in by_c[C]) for C in cutoffs}
    px_c = {C: price_at_cutoff(P, C) for C in cutoffs}
    res = {w: [] for w in wins}
    empty_sn = (np.array([], dtype=np.int64), np.array([]))
    for ti in range(P.nU):
        contrib = {}
        for r in P.pairs_by_ti.get(ti, []):
            ev_ts, ev_ge = P.g_ts[r.r0:r.r1], P.g_gend[r.r0:r.r1]
            cns = None if r.carry_ns is None or pd.isna(r.carry_ns) else int(r.carry_ns)
            pos = EXPO.pair_pos(ev_ts, ev_ge, r.gstart0, cns)
            nzi = np.flatnonzero(pos)
            arrs = None
            if len(nzi):
                lo, hi = nzi[0], nzi[-1] + 1
                ps = pos[lo:hi]
                val = ps * P.px[r.sym_i, lo:hi]
                inv = np.isnan(val) & (ps != 0)
                val = np.where(np.isnan(val), 0.0, val)
                sr = np.where(ps != 0, P.src[r.sym_i, lo:hi], 0)
                arrs = (lo, hi, np.abs(val), val, (ps != 0).astype(np.int32), inv.astype(np.int32),
                        [(sr == k).astype(np.int32) for k in (1, 2, 3)])
            contrib.setdefault(r.symbol, {})["pkg"] = (r.first_ts, cns if cns is not None else r.first_ts,
                                                      ev_ts, ev_ge, r.sym_i, arrs)
        for coin, days, szi, si in P.fund_by_ti.get(ti, []):
            i_d = (days - W0N) // STEP
            lo, hi = i_d[0], i_d[-1] + PPD
            ps = np.zeros(hi - lo)
            ps[((i_d - lo)[:, None] + np.arange(PPD)[None, :]).ravel()] = np.repeat(szi, PPD)
            val = ps * P.px[si, lo:hi]
            inv = np.isnan(val) & (ps != 0)
            val = np.where(np.isnan(val), 0.0, val)
            sr = np.where(ps != 0, P.src[si, lo:hi], 0)
            contrib.setdefault(coin, {})["fund"] = (days, szi, si, (lo, hi, np.abs(val), val, (ps != 0).astype(np.int32),
                                                                   inv.astype(np.int32),
                                                                   [(sr == k).astype(np.int32) for k in (1, 2, 3)]))
        sn_ts, sn_av = P.snaps.get(ti, empty_sn)
        syms = sorted(contrib)
        for C in cutoffs:
            iC, i0 = (C - W0N) // STEP, (lo_c[C] - W0N) // STEP
            L = iC - i0
            G, Nn = np.zeros(L), np.zeros(L)
            Pn, Iv = np.zeros(L, dtype=np.int32), np.zeros(L, dtype=np.int32)
            Sx = [np.zeros(L, dtype=np.int32) for _ in range(3)]
            gl, nl, pl, il = 0.0, 0.0, 0, 0
            sl = [0, 0, 0]
            fa = None
            last_day = (C - STEP) - (C - STEP) % DAY
            for s in syms:
                d = contrib[s]
                pk, fu = d.get("pkg"), d.get("fund")
                if pk is not None and pk[0] < C:
                    start, si, arrs = pk[1], pk[4], pk[5]
                    pos_last = pk[3][np.searchsorted(pk[2], C, side="left") - 1]
                elif fu is not None and fu[0][0] < C:
                    days, szi, si, arrs = fu
                    start = int(days[0])
                    k = np.searchsorted(days, last_day)
                    pos_last = szi[k] if k < len(days) and days[k] == last_day else 0.0
                else:
                    continue
                fa = start if fa is None else min(fa, start)
                if arrs is not None:
                    lo, hi, ga, na, pa, ia, sa = arrs
                    x0, x1 = max(lo, i0), min(hi, iC)
                    if x1 > x0:
                        G[x0 - i0:x1 - i0] += ga[x0 - lo:x1 - lo]
                        Nn[x0 - i0:x1 - i0] += na[x0 - lo:x1 - lo]
                        Pn[x0 - i0:x1 - i0] += pa[x0 - lo:x1 - lo]
                        Iv[x0 - i0:x1 - i0] += ia[x0 - lo:x1 - lo]
                        for k in range(3):
                            Sx[k][x0 - i0:x1 - i0] += sa[k][x0 - lo:x1 - lo]
                if pos_last != 0:
                    v = pos_last * px_c[C][si]
                    pl += 1
                    if np.isnan(v):
                        il += 1
                    else:
                        sl[int(P.src[si, iC - 1]) - 1] += 1       # 1..3; a strict fill price keeps source 3
                        gl += abs(v)
                        nl += v
            G[-1], Nn[-1], Pn[-1], Iv[-1] = gl, nl, pl, il
            for k in range(3):
                Sx[k][-1] = sl[k]
            for w in by_c[C]:
                r = window_grid_stats(w, i0, G, Nn, Pn, Iv, Sx, fa, sn_ts, sn_av)
                r["ti"] = ti
                res[w].append(r)
    return {w: pd.DataFrame(v).set_index("ti") for w, v in res.items()}


# ---------------------------------------------------------------------------------------
# Fill-based columns, episodes, clusters, alignment (one window, all traders)
# ---------------------------------------------------------------------------------------
def _sdiv(a, b):
    b = b.astype(float)
    return a.astype(float) / b.where(b != 0)


def fill_window(P, a, b):
    """Columns built from in-S fills / groups of S = [a, b) at cutoff b. Returns (frame indexed by ti, records)."""
    nU = P.nU
    fr = pd.DataFrame(index=pd.RangeIndex(nU, name="ti"))
    f = P.p[(P.p_ts >= a) & (P.p_ts < b)]
    gt = f.groupby("ti")
    fr["n_fills"] = gt.size()
    fr["turnover"] = gt["notional"].sum()
    fr["maker_ntl"] = gt["maker_notional"].sum()
    fr["forced_ntl"] = gt["forced_notional"].sum()
    fr["n_fill_days"] = gt["day_ns"].nunique()
    fr["open_long_ntl"] = f["notional"].where(f["dir"] == "Open Long", 0.0).groupby(f["ti"]).sum()
    fr["open_ls_ntl"] = f["notional"].where(f["dir"].isin(["Open Long", "Open Short"]), 0.0).groupby(f["ti"]).sum()
    for c in ("n_fills", "n_fill_days"):
        fr[c] = fr[c].fillna(0).astype(np.int64)
    for c in ("turnover", "maker_ntl", "forced_ntl", "open_long_ntl", "open_ls_ntl"):
        fr[c] = fr[c].fillna(0.0)
    has = fr["n_fills"] > 0
    tov = fr["turnover"].where(has)

    # active-day share over [max(a, first prefix fill day), b)
    ff = P.first_fill
    known = ff < b
    start = np.maximum(a, ff - ff % DAY)
    fr["first_fill_prefix"] = np.where(known, ff, -1)                 # int ns, -1 = none
    fr["active_day_denom"] = np.where(known, (b - start) / DAY, np.nan)
    fr["active_day_share"] = fr["n_fill_days"] / fr["active_day_denom"]

    # orders: (address, symbol, oid) among in-S fills only
    o = f.groupby(["ti", "symbol", "oid"], sort=True).agg(
        first_ts=("ts_ns", "min"), last_ts=("ts_ns", "max"), ntl=("notional", "sum"), sgn=("sgn", "first"),
        nside=("sgn", "nunique"), nep=("episode", "nunique"), episode=("episode", "first")).reset_index()
    o = o.sort_values(["ti", "symbol", "first_ts", "last_ts", "oid"], kind="stable").reset_index(drop=True)
    prev = o.shift()
    single = o["nside"].eq(1) & o["nep"].eq(1)
    join = (o["ti"].eq(prev["ti"]) & o["symbol"].eq(prev["symbol"]) & single & single.shift(fill_value=False)
            & o["sgn"].eq(prev["sgn"]) & o["episode"].eq(prev["episode"])
            & ((o["first_ts"] - prev["last_ts"]) <= GAP_NS))
    o["cid"] = (~join).cumsum()
    o["day_ns"] = o["first_ts"] - o["first_ts"] % DAY
    cl = o.groupby("cid").agg(ti=("ti", "first"), symbol=("symbol", "first"), first_ts=("first_ts", "min"),
                              ntl=("ntl", "sum"), k=("oid", "size"))
    cl["day_ns"] = cl["first_ts"] - cl["first_ts"] % DAY
    fr["n_orders"] = o.groupby("ti").size()
    fr["n_clusters"] = cl.groupby("ti").size()
    for c in ("n_orders", "n_clusters"):
        fr[c] = fr[c].fillna(0).astype(np.int64)
    act = f[["ti", "day_ns"]].drop_duplicates()
    for name, src in (("orders_per_active_day_med", o), ("clusters_per_active_day_med", cl)):
        k = src.groupby(["ti", "day_ns"]).size().rename("k").reset_index()
        od = act.merge(k, on=["ti", "day_ns"], how="left").fillna({"k": 0})
        fr[name] = od.groupby("ti")["k"].median()
    fr["orders_per_cluster"] = _sdiv(fr["n_orders"], fr["n_clusters"])
    fr["fills_per_order"] = _sdiv(fr["n_fills"], fr["n_orders"])
    fr["maker_share"] = fr["maker_ntl"] / tov.where(tov != 0)
    fr["forced_ntl_share"] = fr["forced_ntl"] / tov.where(tov != 0)
    fr["order_ntl_med"] = o.groupby("ti")["ntl"].median()
    fr["cluster_ntl_med"] = cl.groupby("ti")["ntl"].median()
    fr["turnover_per_active_day_med"] = f.groupby(["ti", "day_ns"])["notional"].sum().groupby("ti").median()

    # symbols
    st = f.groupby(["ti", "symbol"], sort=True)["notional"].sum()
    tot = st.groupby("ti").transform("sum")
    sh = st / tot.where(tot > 0)
    fr["n_symbols"] = st.groupby("ti").size()
    top = st.groupby("ti").idxmax()                    # first maximum in ascending symbol order
    top_sym = pd.Series(top.map(lambda x: x[1]), index=top.index).reindex(fr.index)
    fr["top1_share"] = sh.groupby("ti").max()
    fr["sym_hhi"] = (sh ** 2).groupby("ti").sum(min_count=1)
    fr["btc_eth_share"] = f["notional"].where(f["symbol"].isin(["BTC", "ETH"]), 0.0).groupby(f["ti"]).sum() \
        / tov.where(tov != 0)
    fr["long_open_share"] = _sdiv(fr["open_long_ntl"], fr["open_ls_ntl"])
    valid_top = tov.reindex(fr.index) > 0
    for c in ("top1_share", "sym_hhi"):
        fr[c] = fr[c].where(valid_top)
    top_sym = top_sym.where(valid_top)

    # episodes on S's groups (position state carried in through gstart)
    gm = (P.g_ts >= a) & (P.g_ts < b)
    sub = P.g.loc[gm, ["pid", "ts", "ts_ns", "gstart", "gend", "signed", "notional", "vwap"]].reset_index(drop=True)
    ep, _, _ = PROF.build_episodes(sub)
    ep["ti"] = P.pair_ti[ep["pid"].to_numpy()]
    ep["symbol"] = P.pair_sym[ep["pid"].to_numpy()]
    af = PROF.assign_fills(f[["ti", "symbol", "ts", "oid", "dir", "is_forced"]].rename(columns={"ti": "trader"}),
                           ep.assign(trader=ep["ti"]))
    ce = ep[ep["complete"]].copy()
    ce["n_add_orders"] = ce["eid"].map(af["add"]).fillna(0).astype(int)
    ce["n_red_orders"] = ce["eid"].map(af["red"]).fillna(0).astype(int)
    gc = ce.groupby("ti")
    fr["n_complete"] = gc.size()
    fr["hold_med_h"] = gc["dur_h"].median()
    wv = ce[ce["open_ntl"].notna()]
    den = wv["open_ntl"].groupby(wv["ti"]).sum()
    fr["open_ntl_complete"] = den
    fr["hold_wavg_h"] = (wv["open_ntl"] * wv["dur_h"]).groupby(wv["ti"]).sum() / den.where(den != 0)
    fr["adds_per_ep_med"] = gc["n_add_orders"].median()
    fr["single_shot_share"] = ((ce["n_add_orders"] == 1) & (ce["n_red_orders"] == 1)).groupby(ce["ti"]).mean()
    fr["flip_share"] = gc["end_by_flip"].mean()
    nl = ep[~ep["left_cens"]].groupby("ti")           # episodes opened inside S (incl. right-censored)
    fr["inc_ntl_opened"] = nl["open_ntl"].sum()
    fr["later_add_ntl"] = nl["later_add_ntl"].sum()
    fr["loser_add_ntl"] = nl["later_loser_ntl"].sum()
    fr["add_ntl_share"] = _sdiv(fr["later_add_ntl"], fr["inc_ntl_opened"])
    fr["add_loser_share"] = _sdiv(fr["loser_add_ntl"], fr["later_add_ntl"])
    fr["n_ep_fill"] = ep.groupby("ti").size()
    fr["n_left_cens_fill"] = ep[ep["left_cens"]].groupby("ti").size()
    fr["n_right_cens_fill"] = ep[ep["right_cens"]].groupby("ti").size()

    # add-behaviour support: the builder's group roles, on episodes opened inside S
    pid = sub["pid"].to_numpy()
    gs, ge = sub["gstart"].to_numpy(), sub["gend"].to_numpy()
    openflat = (gs == 0) & (ge != 0)
    flipg = (gs != 0) & (ge != 0) & (np.sign(gs) != np.sign(ge))
    addg = (gs != 0) & (ge != 0) & ~flipg & (np.abs(ge) > np.abs(gs))
    opener = openflat | flipg
    seen = pd.Series(opener).groupby(pid).cummax().to_numpy().astype(bool)
    inc_el = opener | (addg & seen)
    later = addg & seen
    sub_ti = P.pair_ti[pid]
    sub_day = sub["ts_ns"].to_numpy() - sub["ts_ns"].to_numpy() % DAY
    for name, m in (("inc", inc_el), ("later_add", later)):
        x = pd.DataFrame({"ti": sub_ti[m], "day": sub_day[m]})
        fr[f"n_{name}_groups"] = x.groupby("ti").size()
        fr[f"n_{name}_days"] = x.groupby("ti")["day"].nunique()
    chk = pd.Series(sub["notional"].to_numpy()[later]).groupby(sub_ti[later]).sum().reindex(fr.index).fillna(0.0)
    assert np.allclose(chk, fr["later_add_ntl"].fillna(0.0), rtol=1e-9, atol=1e-6), "later-add group roles"

    # censoring as in exposure.sample_episodes: funding-only runs and package positions held through S
    before = P.g.loc[P.g_ts < a, ["pid", "ti", "gend"]].groupby("pid").tail(1)
    held = before[(before["gend"] != 0) & ~before["pid"].isin(set(pid))]
    fr["n_held_through"] = held.groupby("ti").size()
    fd = P.fday[(P.fday["pkg_first_ns"] >= b) & (P.fday["day_ns"] < b)]
    if len(fd):
        dn, sz = fd["day_ns"].to_numpy(), fd["szi"].to_numpy()
        pr_ = (fd["ti"].astype(str) + "|" + fd["coin"]).to_numpy()
        new = np.r_[True, (pr_[1:] != pr_[:-1]) | (np.diff(dn) != DAY) | (np.sign(sz[1:]) != np.sign(sz[:-1]))]
        run = np.cumsum(new)
        rs = pd.DataFrame({"ti": fd["ti"].to_numpy(), "run": run, "d": dn}).groupby("run").agg(
            ti=("ti", "first"), s=("d", "min"), e=("d", "max"))
        rs = rs[(rs["s"] < b) & (rs["e"] + DAY > a)]
        fr["n_fo_runs"] = rs.groupby("ti").size()
    else:
        fr["n_fo_runs"] = 0
    for c in ("n_complete", "n_ep_fill", "n_left_cens_fill", "n_right_cens_fill", "n_inc_groups", "n_inc_days",
              "n_later_add_groups", "n_later_add_days", "n_held_through", "n_fo_runs"):
        fr[c] = fr[c].fillna(0).astype(np.int64)
    extra = fr["n_fo_runs"] + fr["n_held_through"]
    fr["n_episodes_x"] = fr["n_ep_fill"] + extra
    fr["n_left_cens_x"] = fr["n_left_cens_fill"] + extra
    fr["n_right_cens_x"] = fr["n_right_cens_fill"] + extra

    # alignment: positive opening legs, eligible opening clusters (style.py) on window-fragment orders
    fo = f.merge(o[["ti", "symbol", "oid", "cid", "nep"]], on=["ti", "symbol", "oid"], how="left",
                 validate="many_to_one")
    opening = fo["open_ntl"] > 0
    amb = fo["mixed"] | fo["nep"].ne(1)
    elig = opening & ~fo["is_forced"] & ~fo["is_hip3"] & ~amb
    for name, m in (("forced", opening & fo["is_forced"]), ("hip3", opening & ~fo["is_forced"] & fo["is_hip3"]),
                    ("ambiguous", opening & ~fo["is_forced"] & ~fo["is_hip3"] & amb)):
        fr[f"open_ntl_{name}"] = fo["open_ntl"].where(m, 0.0).groupby(fo["ti"]).sum()
        fr[f"open_ntl_{name}"] = fr[f"open_ntl_{name}"].fillna(0.0)
    op = fo[elig].copy()
    op["maker_ntl"] = np.where(op["is_maker"], op["open_ntl"], 0.0)
    op["taker_ntl"] = np.where(op["is_maker"], 0.0, op["open_ntl"])
    c = op.groupby("cid").agg(ti=("ti", "first"), symbol=("symbol", "first"), episode=("episode", "first"),
                              t=("ts_ns", "min"), last=("ts_ns", "max"), s=("sgn", "first"), nsgn=("sgn", "nunique"),
                              w=("open_ntl", "sum"), w_maker=("maker_ntl", "sum"),
                              w_taker=("taker_ntl", "sum")).reset_index()
    assert c["nsgn"].eq(1).all(), "cluster with both buy and sell opening legs"
    c = c[c["w"] > 0].sort_values(["ti", "symbol", "t", "cid"], kind="stable").reset_index(drop=True)
    c["is_first"] = c["t"].to_numpy() == P.ep_open[c["episode"].to_numpy()]
    c["duration_s"] = (c["last"] - c["t"]) / 1e9
    c["day_ns"] = c["t"] - c["t"] % DAY
    fr["n_open_clusters"] = c.groupby("ti").size()
    fr["n_open_clusters"] = fr["n_open_clusters"].fillna(0).astype(np.int64)
    has_file = c["symbol"].isin(P.kl.keys()).to_numpy()
    ok_cl = (c["duration_s"].to_numpy() <= MAX_DUR_S) & has_file
    tt = c["t"].to_numpy()
    ci = (tt - tt % STEP - G0N) // STEP                  # c(t): last close <= t
    bi = (tt - tt % H4 - G0N) // STEP                    # first grid point of the 4h block
    s = c["s"].to_numpy()
    for h in HORIZONS:
        c[f"z_{h}"] = np.nan
        c[f"xA_{h}"] = np.nan
    if len(c):
        assert ci.min() >= 0 and (bi + 47).max() < NGRID
    for sym, ix in c.groupby("symbol", sort=True).indices.items():
        ga = get_align(P, sym)
        if ga is None:
            continue
        cand = bi[ix][:, None] + np.arange(48)[None, :]
        for h in HORIZONS:
            sig = ga[f"sigma_{h}"][ci[ix]]
            qv = ga[f"q_{h}"][ci[ix]]
            cq = ga[f"q_{h}"][cand]
            ratio = ga[f"sigma_{h}"][cand] / sig[:, None]
            okc = np.isfinite(cq) & (ratio >= A_LO) & (ratio <= A_HI)
            nA = okc.sum(axis=1)
            meanA = np.where(okc, cq, 0.0).sum(axis=1) / np.maximum(nA, 1)
            z = s[ix] * qv
            bA = np.where((nA >= A_MIN) & np.isfinite(qv), s[ix] * meanA, np.nan)
            c.loc[c.index[ix], f"z_{h}"] = z
            c.loc[c.index[ix], f"xA_{h}"] = z - bA
    for h in HORIZONS:
        c.loc[~ok_cl, [f"z_{h}", f"xA_{h}"]] = np.nan
    ac = {}
    for cls, wc in CLASSES.items():
        cand_w = c.loc[c[wc] > 0].groupby("ti")[wc].sum()
        ac[f"cand_w_{cls}"] = cand_w.reindex(fr.index).fillna(0.0)
        for h in HORIZONS:
            for var in VARIANTS:
                col = f"z_{h}" if var == "raw" else f"xA_{h}"      # A: its own finite decision set
                e = c[c[col].notna() & (c[wc] > 0)]
                w = e[wc]
                k = f"{cls}_{h}_{var}"
                ge_ = e.groupby("ti")
                sw = w.groupby(e["ti"]).sum()
                ac[f"align_{k}"] = (e[col] * w).groupby(e["ti"]).sum() / sw
                ac[f"{k}_n"] = ge_.size().reindex(fr.index).fillna(0).astype(np.int64)
                ac[f"{k}_days"] = ge_["day_ns"].nunique().reindex(fr.index).fillna(0).astype(np.int64)
                ac[f"{k}_w"] = sw.reindex(fr.index).fillna(0.0)
                ac[f"{k}_wmax_share"] = w.groupby(e["ti"]).max() / sw
                ac[f"{k}_kish"] = sw ** 2 / (w * w).groupby(e["ti"]).sum()
                ac[f"{k}_first_n"] = ge_["is_first"].mean()
                ac[f"{k}_first_w"] = w.where(e["is_first"], 0.0).groupby(e["ti"]).sum() / sw
                ac[f"{k}_cov"] = sw / cand_w.reindex(sw.index)
    fr = pd.concat([fr, pd.DataFrame({k: v.reindex(fr.index) for k, v in ac.items()})], axis=1)

    # record tables for the tier-pooled measures (section 2.8)
    zc = [f"z_{h}" for h in HORIZONS] + [f"xA_{h}" for h in HORIZONS]
    dec = c.loc[c[[f"z_{h}" for h in HORIZONS]].notna().any(axis=1),
                ["ti", "symbol", "t", "s", "w_taker", "w_maker", "is_first"] + zc].reset_index(drop=True)
    eps_rec = ce[["ti", "symbol", "open_ts", "close_ts", "dur_h", "open_ntl", "end_by_flip", "n_add_orders",
                  "n_red_orders"]].reset_index(drop=True)
    cl_rec = cl[["ti", "symbol", "first_ts", "k", "ntl"]].rename(columns={"k": "n_orders"}).reset_index(drop=True)
    return fr, top_sym, dict(decisions=dec, episodes=eps_rec, clusters=cl_rec)


# ---------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------
def profile_windows(P, wins, verbose=True):
    """Profile of every address for each window (a, b) in wins (Timestamps, midnight UTC, W0 <= a < b <= W1).
    Returns {(a_ns, b_ns): dict(frame=DataFrame indexed by ti, top=Series, records=dict)}."""
    wn = sorted({(int(pd.Timestamp(a).value), int(pd.Timestamp(b).value)) for a, b in wins})
    for a, b in wn:
        assert a % DAY == 0 and b % DAY == 0 and W0N <= a < b <= W1N, (a, b)
    t0 = time.time()
    grid = grid_windows(P, wn)
    if verbose:
        log(f"  grid pass: {len(wn)} windows, {len({b for _, b in wn})} cutoffs, {time.time() - t0:.1f}s")
    out = {}
    for a, b in wn:
        fr, top, rec = fill_window(P, a, b)
        x = fr.join(grid[(a, b)])
        hip3 = (P.h3_lt < b) | (P.h3_le <= b)
        x["hip3_asof"] = hip3
        x["no_fill"] = x["n_fills"] == 0
        x["base_eligible"] = (x["n_orders"] >= 30) & (x["n_fill_days"] >= 10)
        x["hold_turnover_h"] = x["held_ntl_h"] / (x["turnover"] / 2).where(x["turnover"] > 0)
        for col in ("av_med", "lev_med", "lev_p95"):     # AV / leverage out of scope for HIP-3-as-of-C
            x[col] = x[col].where(~hip3)
        out[(a, b)] = dict(frame=x, top=top, records=rec)
    if verbose:
        log(f"  fill pass done at {time.time() - t0:.1f}s")
    return out


_P_FULL = None


def profile_window(a, b, P=None):
    """One window [a, b) on the full package inputs (or on P)."""
    global _P_FULL
    if P is None:
        if _P_FULL is None:
            inp = load_inputs()
            _P_FULL = build_prep(inp, sorted(inp["p"]["trader"].unique()))
        P = _P_FULL
    return profile_windows(P, [(a, b)], verbose=False)[(pd.Timestamp(a).value, pd.Timestamp(b).value)]


def to_tables(P, res, wt):
    """Long ledger, top-symbol table, support table and record tables for the windows of wt."""
    led, tops, sup, recs = [], [], [], {"decisions": [], "episodes": [], "clusters": []}
    sums = []
    trader = pd.Series(P.U)
    for r in wt.itertuples(index=False):
        x = res[(r.a.value, r.b.value)]
        fr = x["frame"]
        v = fr[VALUE_COLS].copy()
        v.index = trader.to_numpy()
        lv = v.rename_axis("trader").reset_index().melt(id_vars="trader", var_name="column", value_name="value")
        lv.insert(1, "window_id", r.window_id)
        led.append(lv)
        tops.append(pd.DataFrame({"trader": P.U, "window_id": r.window_id, "top_symbol": x["top"].to_numpy()}))
        s = fr[SUPPORT_COLS].copy()
        s.insert(0, "window_id", r.window_id)
        s.insert(0, "trader", P.U)
        sup.append(s.reset_index(drop=True))
        sm = fr[["turnover", "maker_ntl", "inc_ntl_opened", "later_add_ntl", "loser_add_ntl"]].fillna(0.0)
        sm.insert(0, "window_id", r.window_id)
        sm.insert(0, "trader", P.U)
        sums.append(sm.reset_index(drop=True))
        for k, d in x["records"].items():
            d = d.copy()
            d.insert(0, "window_id", r.window_id)
            d.insert(0, "trader", P.U[d["ti"].to_numpy()])
            recs[k].append(d.drop(columns="ti"))
    cat = lambda d, cols: d.astype({c: "category" for c in cols})  # noqa: E731
    out = dict(ledger=cat(pd.concat(led, ignore_index=True), ["trader", "window_id", "column"]),
               top_symbol=cat(pd.concat(tops, ignore_index=True), ["trader", "window_id"]),
               support=pd.concat(sup, ignore_index=True),
               address_sums=pd.concat(sums, ignore_index=True))
    for k, v in recs.items():
        d = pd.concat(v, ignore_index=True)
        if k == "decisions":
            d["t"] = d["t"].to_numpy().astype("datetime64[ns]")
        if k == "clusters":
            d["first_ts"] = d["first_ts"].to_numpy().astype("datetime64[ns]")
        out[k] = cat(d, ["trader", "window_id", "symbol"])
    for c in ("first_fill_prefix", "first_appearance"):           # int ns (-1 = none) -> exact datetime64
        v = out["support"][c].to_numpy(dtype=np.int64)
        out["support"][c] = np.where(v == -1, np.iinfo(np.int64).min, v).view("datetime64[ns]")
    return out


# ---------------------------------------------------------------------------------------
# Gates (section 2.3), for the report: per column, B_M -> A_jM (L60) -> C_jM (F28)
# ---------------------------------------------------------------------------------------
FAMILY = {**{c: "fill" for c in FILL_COLS}, "time_in_market": "tim",
          **{c: "grid_q" for c in ("gross_exp_med", "gross_exp_p95", "net_bias", "long_time_share", "n_pos_med")},
          "lev_med": "lev", "lev_p95": "lev", "av_med": "av", "hold_turnover_h": "turnover",
          **{c: "complete" for c in EPISODE_COLS}, "add_ntl_share": "add", "add_loser_share": "loss_add",
          **{c: "align" for c in ALIGN_COLS}}


def gate(sup, col, kind):
    """Support gate of the column's family on a support frame, kind 'L' (L60, without B_M) or 'F' (F28)."""
    fam = FAMILY[col]
    L = kind == "L"
    grid = sup["valid_grid_days"] >= 14
    ords = (sup["n_orders"] >= 15) & (sup["n_fill_days"] >= 5)
    if fam == "fill":
        return pd.Series(True, index=sup.index) if L else ords
    if fam == "tim":
        return grid
    if fam == "grid_q":
        return grid & (sup["n_inm_days"] >= (10 if L else 5))
    if fam == "lev":
        return grid & (sup["n_lev_days"] >= (10 if L else 5)) & ~sup["hip3_asof"]
    if fam == "av":
        return (sup["n_snap_days"] >= (10 if L else 5)) & ~sup["hip3_asof"]
    if fam == "turnover":
        return grid & ((sup["turnover"] > 0) if L else ords)
    if fam == "complete":
        return sup["n_complete"] >= (10 if L else 5)
    n, d = (30, 10) if L else (15, 5)
    if fam == "add":
        return (sup["n_inc_groups"] >= n) & (sup["n_inc_days"] >= d)
    if fam == "loss_add":
        return (sup["n_later_add_groups"] >= n) & (sup["n_later_add_days"] >= d)
    k = col[len("align_"):]
    return (sup[f"{k}_n"] >= n) & (sup[f"{k}_days"] >= d)


def gate_counts(tabs, wt):
    sup = tabs["support"].set_index(["window_id", "trader"])
    led = tabs["ledger"]
    val = led.pivot_table(index=["window_id", "trader"], columns="column", values="value", aggfunc="first",
                          dropna=False, observed=True)
    rows = []
    for M in MONDAYS:
        ms = f"{M:%Y-%m-%d}"
        sl, sf = sup.loc[f"L60_{ms}"], sup.loc[f"F28_{ms}"]
        vl, vf = val.loc[f"L60_{ms}"].reindex(sl.index), val.loc[f"F28_{ms}"].reindex(sl.index)
        B = sl["base_eligible"]
        for col in VALUE_COLS:
            gl = B & gate(sl, col, "L")
            A = gl & vl[col].notna()
            gf = gate(sf, col, "F")
            Cc = A & gf & vf[col].notna()
            rows.append(dict(M=ms, column=col, family=FAMILY[col], B=int(B.sum()), L_gate=int(gl.sum()),
                             A=int(A.sum()), F_gate_among_A=int((A & gf).sum()), C=int(Cc.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(tabs, wt, runtime):
    data = ROOT / "ArkStream笔试题_v2"
    files = [data / "笔试题数据包-fills", data / "笔试题数据包-traders.csv", EXT / "funding.parquet",
             EXT / "hl_candles_4h.parquet", EXT / "symbol_map.csv", EXT / "hidden_positions.parquet",
             EXT / "manifest.json"]
    src = [ROOT / "work/v2dev/code/lib.py", ROOT / "work/profile/profile.py", ROOT / "work/profile2/code/exposure.py",
           ROOT / "work/style/code/style.py", Path(__file__).resolve(), SPEC]
    schemas = {}
    for f in files:
        if f.suffix == ".parquet":
            schemas[f.name] = {n: str(t) for n, t in zip(pq.read_schema(f).names, pq.read_schema(f).types)}
        elif f.suffix != ".json":
            with open(f, encoding="utf-8") as fh:
                schemas[f.name] = fh.readline().strip().split(",")
    schemas["klines_5m/*.parquet"] = {n: str(t) for n, t in zip(pq.read_schema(next(KLINES.glob("*.parquet"))).names,
                                                                pq.read_schema(next(KLINES.glob("*.parquet"))).types)}
    sup = tabs["support"]
    counts = {}
    for r in wt.itertuples(index=False):
        s = sup[sup["window_id"] == r.window_id]
        counts[r.window_id] = dict(base_eligible=int(s["base_eligible"].sum()), with_fills=int((~s["no_fill"]).sum()),
                                   observed_grid=int((s["n_valid_points"] > 0).sum()),
                                   hip3_asof=int(s["hip3_asof"].sum()),
                                   missing_px_points=int(s["n_missing_px_points"].sum()))
    return dict(
        spec=str(SPEC), spec_sections="2.1, 2.2, 2.3 (ledger only)",
        inputs={str(f.relative_to(ROOT)): sha256(f) for f in files},
        klines_5m={f.name: sha256(f) for f in sorted(KLINES.glob("*.parquet"))},
        sources={str(f.relative_to(ROOT)): sha256(f) for f in src},
        schemas=schemas,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      pyarrow=pyarrow.__version__, scipy=scipy.__version__),
        seed=SEED, seed_note="recorded per section 2.1; the ledger itself draws no random numbers",
        run_utc=pd.Timestamp.now(tz="UTC").isoformat(), runtime_s=round(runtime, 1),
        rows={k: int(len(v)) for k, v in tabs.items()}, counts_per_window=counts)


def build():
    global _logf
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "ledger.log", "w")
    t0 = time.time()
    inp = load_inputs()
    U = sorted(inp["p"]["trader"].unique())
    log(f"loaded in {time.time() - t0:.1f}s: fills {len(inp['p']):,}, snapshots {len(inp['tr']):,}, "
        f"funding {len(inp['fund']):,}, kline files {len(inp['kl'])}, HL bars {len(inp['hl']):,}; U {len(U)}")
    P = build_prep(inp, U)
    log(f"prepared in {time.time() - t0:.1f}s: groups {len(P.g):,}, pairs {len(P.pairs):,}, funding pair-days "
        f"{len(P.fday):,}, symbols {len(P.symbols)}; carried-in pairs with a run "
        f"{int(P.pairs['carry_ns'].notna().sum())}")
    # the funding-only reconstruction at C = W1 must equal hidden_positions.parquet (as asserted in exposure.py)
    hid = pd.read_parquet(EXT / "hidden_positions.parquet")
    fo = P.fday[P.fday["pkg_first_ns"] >= W1N].assign(day=lambda d: pd.to_datetime(d["day_ns"], unit="ns"))
    a_ = fo[["trader", "coin", "day", "szi"]].sort_values(["trader", "coin", "day"]).reset_index(drop=True)
    b_ = hid[["trader", "coin", "day", "szi"]].sort_values(["trader", "coin", "day"]).reset_index(drop=True)
    assert a_.equals(b_), "funding-only as of W1 does not reproduce hidden_positions.parquet"
    log(f"funding-only as of W1 == hidden_positions.parquet ({len(a_):,} pair-days, asserted)")

    wt = window_table()
    wt.assign(M=wt["M"].dt.strftime("%Y-%m-%d"), a=wt["a"].dt.strftime("%Y-%m-%d"),
              b=wt["b"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "windows.csv", index=False)
    res = profile_windows(P, list(zip(wt["a"], wt["b"])))
    log(f"profiles done at {time.time() - t0:.1f}s ({len(res)} distinct windows for {len(wt)} window ids)")
    tabs = to_tables(P, res, wt)
    for k, d in tabs.items():
        d.to_parquet(OUT / f"{k}.parquet", index=False)
        log(f"wrote {k}.parquet: {len(d):,} rows x {d.shape[1]} cols")
    gc = gate_counts(tabs, wt)
    gc.to_csv(OUT / "gate_counts.csv", index=False)
    sup = tabs["support"]
    log("\n=== base-eligible per window ===")
    for r in wt.itertuples(index=False):
        s = sup[sup["window_id"] == r.window_id]
        log(f"{r.window_id:16s} base {int(s['base_eligible'].sum()):4d}  with fills {int((~s['no_fill']).sum()):4d}  "
            f"hip3 {int(s['hip3_asof'].sum()):4d}  missing-price points {int(s['n_missing_px_points'].sum())}")
    l60 = [int(sup.loc[sup["window_id"] == f"L60_{M:%Y-%m-%d}", "base_eligible"].sum()) for M in MONDAYS]
    log(f"L60 base-eligible {l60}; expected {EXPECTED_L60_BASE}; match {l60 == EXPECTED_L60_BASE}")
    runtime = time.time() - t0
    man = manifest(tabs, wt, runtime)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"\nruntime {runtime:.1f}s")


# ---------------------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------------------
def _eq_frames(a, b, cols):
    """Per column: number of cells, number not identical (NaN == NaN)."""
    out = {}
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            x, y = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            same = (x == y) | (np.isnan(x) & np.isnan(y))
        elif pd.api.types.is_datetime64_any_dtype(x):
            same = (x.to_numpy() == y.to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
        else:
            same = (x.astype(object).to_numpy() == y.astype(object).to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
        out[c] = int((~same).sum())
    return out


def _cmp(ours, theirs, rtol=1e-9):
    """Numeric comparison summary of two aligned Series."""
    x, y = ours.astype(float), theirs.astype(float)
    both = x.notna() & y.notna()
    d = (x - y).abs()
    rel = d / np.maximum(y.abs(), 1e-300)
    close = both & ((d <= rtol * y.abs()) | (d == 0))
    return dict(n_both=int(both.sum()), n_ours_only=int((x.notna() & y.isna()).sum()),
                n_theirs_only=int((x.isna() & y.notna()).sum()), n_exact=int((both & (d == 0)).sum()),
                n_close=int(close.sum()), n_differ=int((both & ~close).sum()),
                max_rel=float(rel[both].max()) if both.any() else None,
                max_abs_of_differing=float(d[both & ~close].max()) if (both & ~close).any() else None,
                median_rel_of_differing=float(rel[both & ~close].median()) if (both & ~close).any() else None)


def checks():
    global _logf
    _logf = open(LOGS / "checks.log", "w")
    t0 = time.time()
    res = {}
    wt = pd.read_csv(OUT / "windows.csv", parse_dates=["M", "a", "b"])
    led = pd.read_parquet(OUT / "ledger.parquet")
    top = pd.read_parquet(OUT / "top_symbol.parquet")
    sup = pd.read_parquet(OUT / "support.parquet")
    rec = {k: pd.read_parquet(OUT / f"{k}.parquet") for k in ("decisions", "episodes", "clusters", "address_sums")}
    inp = load_inputs()
    U = sorted(inp["p"]["trader"].unique())
    wide = led.pivot_table(index=["window_id", "trader"], columns="column", values="value", aggfunc="first",
                           dropna=False, observed=True)

    # ---------------- check 1: prefix equality ----------------
    log("=== check 1: prefix equality (L60 and R28 at C = M, inputs physically truncated at C) ===")
    c1 = {}
    for M in (pd.Timestamp("2025-10-06"), pd.Timestamp("2025-11-24")):
        ms = f"{M:%Y-%m-%d}"
        tin, removed = truncate_inputs(inp, M)
        assert tin["p"]["ts"].max() < M and tin["tr"]["ts"].max() < M
        Pt = build_prep(tin, U)
        wl = [(M - pd.Timedelta(days=60), M), (M - pd.Timedelta(days=28), M)]
        rt = profile_windows(Pt, wl, verbose=False)
        sub_wt = pd.DataFrame({"window_id": [f"L60_{ms}", f"R28_{ms}"], "kind": ["L60", "R28"], "M": [M, M],
                               "a": [w[0] for w in wl], "b": [w[1] for w in wl]})
        tt = to_tables(Pt, rt, sub_wt)
        cm = {"removed_from_inputs": removed}
        for wid in sub_wt["window_id"]:
            mine = led[led["window_id"] == wid].reset_index(drop=True)
            new = tt["ledger"][tt["ledger"]["window_id"] == wid].reset_index(drop=True)
            k = ["trader", "column"]
            mine = mine.sort_values(k).reset_index(drop=True)
            new = new.assign(trader=new["trader"].astype(str), column=new["column"].astype(str)).sort_values(k) \
                .reset_index(drop=True)
            assert (mine["trader"].astype(str).to_numpy() == new["trader"].to_numpy()).all()
            assert (mine["column"].astype(str).to_numpy() == new["column"].to_numpy()).all()
            d_led = _eq_frames(mine, new, ["value"])["value"]
            ms_ = sup[sup["window_id"] == wid].sort_values("trader").reset_index(drop=True)
            ns_ = tt["support"].sort_values("trader").reset_index(drop=True)
            ns_ = ns_[ns_["window_id"] == wid].reset_index(drop=True)
            d_sup = _eq_frames(ms_, ns_, SUPPORT_COLS)
            mt = top[top["window_id"] == wid].sort_values("trader").reset_index(drop=True)
            nt = tt["top_symbol"][tt["top_symbol"]["window_id"] == wid].sort_values("trader").reset_index(drop=True)
            d_top = _eq_frames(mt, nt, ["top_symbol"])["top_symbol"]
            d_rec = {}
            for kname, keys in (("decisions", ["trader", "symbol", "t"]), ("episodes", ["trader", "symbol", "open_ts"]),
                                ("clusters", ["trader", "symbol", "first_ts", "ntl"]),
                                ("address_sums", ["trader"])):
                a_ = rec[kname][rec[kname]["window_id"] == wid]
                b_ = tt[kname][tt[kname]["window_id"] == wid]
                a_ = a_.astype({c: str for c in a_.columns if isinstance(a_[c].dtype, pd.CategoricalDtype)})
                b_ = b_.astype({c: str for c in b_.columns if isinstance(b_[c].dtype, pd.CategoricalDtype)})
                a_ = a_.sort_values(keys, kind="stable").reset_index(drop=True)
                b_ = b_.sort_values(keys, kind="stable").reset_index(drop=True)
                if len(a_) != len(b_):
                    d_rec[kname] = f"row count {len(a_)} vs {len(b_)}"
                else:
                    dd = _eq_frames(a_, b_, [c for c in a_.columns if c != "window_id"])
                    d_rec[kname] = dict(rows=len(a_), cells_differing=int(sum(dd.values())))
            cm[wid] = dict(ledger_cells=len(mine), ledger_differ=d_led, support_cells=len(ms_) * len(SUPPORT_COLS),
                           support_differ=int(sum(d_sup.values())),
                           support_differ_cols={k: v for k, v in d_sup.items() if v},
                           top_symbol_differ=d_top, records=d_rec)
            log(f"{wid}: {cm[wid]}")
        log(f"  removed from inputs at {ms}: {removed}")
        c1[ms] = cm
    res["check1_prefix_equality"] = c1

    # ---------------- check 2: a later close does not complete an earlier episode ----------------
    log("\n=== check 2: episodes open at C and closed later are right-censored in the window ending at C ===")
    epf = rec["episodes"][rec["episodes"]["window_id"] == "ref_full"]
    c2 = {}
    tot_cross, tot_leak = 0, 0
    for M in MONDAYS:
        ms = f"{M:%Y-%m-%d}"
        a = M - pd.Timedelta(days=60)
        cross = epf[(epf["open_ts"] >= a) & (epf["open_ts"] < M) & (epf["close_ts"] >= M)]
        el = rec["episodes"][rec["episodes"]["window_id"] == f"L60_{ms}"]
        k = ["trader", "symbol", "open_ts"]
        leak = cross[k].astype(str).merge(el[k].astype(str), on=k)
        tot_cross += len(cross)
        tot_leak += len(leak)
        c2[ms] = dict(full_complete_open_in_L60_closed_after_M=len(cross), found_complete_in_L60=len(leak))
    # example: recompute the L60 window's episodes for one such episode and show its censoring flags
    M = pd.Timestamp("2025-10-06")
    a = M - pd.Timedelta(days=60)
    cross = epf[(epf["open_ts"] >= a) & (epf["open_ts"] < M) & (epf["close_ts"] >= M)] \
        .sort_values(["close_ts", "trader", "symbol"]).iloc[0]
    Pf = build_prep(inp, U)
    ti = int(np.searchsorted(Pf.U, cross["trader"]))
    gsub = Pf.g[(Pf.g["ti"] == ti) & (Pf.g_ts >= a.value) & (Pf.g_ts < M.value)]
    gsub = gsub[Pf.pair_sym[gsub["pid"].to_numpy()] == cross["symbol"]].reset_index(drop=True)
    ep_ex, _, _ = PROF.build_episodes(gsub)
    row = ep_ex[ep_ex["open_ts"] == cross["open_ts"]].iloc[0]
    supx = sup[(sup["window_id"] == f"L60_{M:%Y-%m-%d}") & (sup["trader"] == cross["trader"])].iloc[0]
    ex = dict(trader=cross["trader"][:10] + "...", symbol=cross["symbol"], open_ts=str(cross["open_ts"]),
              close_ts_full=str(cross["close_ts"]), dur_h_full=float(cross["dur_h"]),
              in_L60_right_cens=bool(row["right_cens"]), in_L60_complete=bool(row["complete"]),
              in_L60_close_ts=str(row["close_ts"]), L60_support_n_right_cens_fill=int(supx["n_right_cens_fill"]))
    groups_at_cutoffs = int(np.isin(Pf.g_ts, [m.value for m in MONDAYS] + [W1N, PRE_END.value]).sum())
    c2.update(total_crossing=tot_cross, total_leaked=tot_leak, example=ex,
              groups_exactly_at_a_monday_cutoff=groups_at_cutoffs)
    log(json.dumps(c2, indent=1, default=str))
    res["check2_censoring"] = c2

    # ---------------- check 3: carried position with no new fills stays measurable ----------------
    log("\n=== check 3: positions carried through a whole forward week with no fill in it ===")
    wk = wt[wt["kind"].isin(["F1", "F2", "F3", "F4"])]["window_id"]
    s3 = sup[sup["window_id"].isin(wk)]
    carried = s3[s3["no_fill"] & (s3["n_valid_points"] == 7 * PPD) & (s3["n_inm_points"] == 7 * PPD)]
    w3 = wide.loc[list(zip(carried["window_id"], carried["trader"]))]
    gcols = ["time_in_market", "gross_exp_med", "gross_exp_p95", "net_bias", "long_time_share", "n_pos_med"]
    pkg3 = carried[(carried["n_held_through"] > 0) & (carried["n_fo_runs"] == 0)]
    ex3 = pkg3.sort_values(["window_id", "trader"]).iloc[0]      # example: package replay, no funding-only part
    v3 = wide.loc[(ex3["window_id"], ex3["trader"])]
    # base-eligible in the L60 of the same Monday
    mon = carried["window_id"].str[3:]
    bl = sup[sup["window_id"].str.startswith("L60_")].set_index(["window_id", "trader"])["base_eligible"]
    in_b = bl.reindex(list(zip("L60_" + mon, carried["trader"]))).fillna(False).to_numpy()
    c3 = dict(address_weeks=len(carried), distinct_addresses=int(carried["trader"].nunique()),
              of_which_base_eligible_in_L60=int(in_b.sum()),
              package_held_through_only=len(pkg3),
              funding_only_only=int(((carried["n_held_through"] == 0) & (carried["n_fo_runs"] > 0)).sum()),
              both_kinds=int(((carried["n_held_through"] > 0) & (carried["n_fo_runs"] > 0)).sum()),
              grid_cols_all_finite=int(w3[gcols].notna().all(axis=1).sum()),
              fill_cols_all_nan=int(w3[["maker_share", "fills_per_order", "n_symbols"]].isna().all(axis=1).sum()),
              example=dict(window_id=ex3["window_id"], trader=ex3["trader"][:10] + "...",
                           n_fills=int(ex3["n_fills"]), n_valid_points=int(ex3["n_valid_points"]),
                           n_inm_points=int(ex3["n_inm_points"]), n_held_through=int(ex3["n_held_through"]),
                           n_fo_runs=int(ex3["n_fo_runs"]),
                           **{c: float(v3[c]) for c in gcols + ["hold_turnover_h", "maker_share"]}))
    log(json.dumps(c3, indent=1, default=str))
    res["check3_carried"] = c3

    # ---------------- check 4: reviewed outputs on the reference windows ----------------
    log("\n=== check 4a: work/profile2/build/address_exposure.csv (pre / post / full) ===")
    ex_ = pd.read_csv(ROOT / "work/profile2/build/address_exposure.csv", parse_dates=["first_appearance"],
                      float_precision="round_trip")
    cmap = {c: c for c in ["time_in_market", "gross_exp_med", "gross_exp_p95", "lev_med", "lev_p95", "av_med",
                           "net_bias", "long_time_share", "n_pos_med", "hold_turnover_h", "held_ntl_h",
                           "hold_med_h", "hold_wavg_h", "n_span_points", "n_lev_days", "n_snap", "n_complete",
                           "n_orders", "turnover"]}
    cmap.update(n_active_days="n_fill_days", n_episodes="n_episodes_x", n_left_cens="n_left_cens_x",
                n_right_cens="n_right_cens_x")
    c4a = {}
    for smp, wid in (("pre", "ref_pre"), ("post", "ref_post"), ("full", "ref_full")):
        th = ex_[ex_["sample"] == smp].set_index("trader")
        ou = wide.loc[wid].join(sup[sup["window_id"] == wid].set_index("trader"))
        ou = ou.reindex(th.index)
        cc = {}
        for tc, oc in cmap.items():
            y = th[tc]
            if tc == "n_lev_days":                       # exposure.py nulls it for HIP-3; ours keeps the count
                y = y.where(~th["hip3_asof"])
                cc[tc] = _cmp(ou[oc].where(~ou["hip3_asof"].astype(bool)), y)
            else:
                cc[tc] = _cmp(ou[oc], y)
        cc["enough_vs_base_eligible_disagree"] = int((th["enough"].astype(bool) != ou["base_eligible"].astype(bool)).sum())
        cc["hip3_asof_disagree"] = int((th["hip3_asof"].astype(bool) != ou["hip3_asof"].astype(bool)).sum())
        fa_o = ou["first_appearance"]
        cc["first_appearance_disagree"] = int(((fa_o != th["first_appearance"]) & ~(fa_o.isna() & th["first_appearance"].isna())).sum())
        c4a[smp] = cc
        log(f"--- {smp} ---")
        for k_, v_ in cc.items():
            log(f"  {k_}: {v_}")
    res["check4a_exposure"] = c4a

    log("\n=== check 4b: work/profile/address_profile.csv (full window) ===")
    pr = pd.read_csv(ROOT / "work/profile/address_profile.csv", float_precision="round_trip").set_index("trader")
    ou = wide.loc["ref_full"].join(sup[sup["window_id"] == "ref_full"].set_index("trader")).reindex(pr.index)
    tf = top[top["window_id"] == "ref_full"].set_index("trader")["top_symbol"].reindex(pr.index)
    c4b = {}
    for col in ["maker_share", "fills_per_order", "orders_per_active_day_med", "n_symbols", "top1_share", "sym_hhi",
                "btc_eth_share", "long_open_share", "adds_per_ep_med", "single_shot_share", "add_ntl_share",
                "add_loser_share", "flip_share", "forced_ntl_share", "order_ntl_med", "active_day_share",
                "turnover_per_active_day_med", "hold_med_h", "hold_wavg_h"]:
        c4b[col] = _cmp(ou[col], pr[col])
    c4b["top_symbol_disagree"] = int((tf.astype(str) != pr["top1_symbol"].astype(str)).sum())
    c4b["n_complete_vs_n_episodes"] = _cmp(ou["n_complete"], pr["n_episodes"])
    for col in ["clusters_per_active_day_med", "orders_per_cluster", "cluster_ntl_med"]:
        c = _cmp(ou[col], pr[col])
        r_ = (ou[col] / pr[col]).replace([np.inf, -np.inf], np.nan).dropna()
        c.update(ratio_new_over_old=dict(p05=float(r_.quantile(.05)), p25=float(r_.quantile(.25)),
                                         median=float(r_.median()), p75=float(r_.quantile(.75)),
                                         p95=float(r_.quantile(.95)), share_ne_1=float((r_ != 1).mean())))
        c4b[col] = c
    nc_new = sup.loc[sup["window_id"] == "ref_full", "n_clusters"].sum()
    c4b["clusters_total_new_vs_old_profile"] = dict(new=int(nc_new),
                                                    old=float((pr["n_orders"] / pr["orders_per_cluster"]).sum()))
    for k_, v_ in c4b.items():
        log(f"  {k_}: {v_}")
    res["check4b_profile"] = c4b

    log("\n=== check 4c: work/style/build/address_style.csv (pre / post / full, taker / maker) ===")
    sy = pd.read_csv(ROOT / "work/style/build/address_style.csv", float_precision="round_trip")
    c4c = {}
    for smp, wid in (("pre", "ref_pre"), ("post", "ref_post"), ("full", "ref_full")):
        ou = wide.loc[wid].join(sup[sup["window_id"] == wid].set_index("trader"))
        for cls in CLASSES:
            for h in HORIZONS:
                th = sy[(sy["sample"] == smp) & (sy["cls"] == cls) & (sy["h"] == h)].set_index("trader")
                idx = th.index.union(ou.index[ou[f"{cls}_{h}_raw_n"] > 0])
                o_ = ou.reindex(idx)
                t_ = th.reindex(idx)
                k = f"{smp}|{cls}|{h}"
                nd = (o_[f"{cls}_{h}_raw_n"].fillna(0) != t_["n_dec"].fillna(0))
                na = (o_[f"{cls}_{h}_A_n"].fillna(0) != t_["n_A"].fillna(0))
                gate_a_own = (o_[f"{cls}_{h}_A_n"] >= 30) & (o_[f"{cls}_{h}_A_days"] >= 10)
                gate_style = t_["reportable"].fillna(False).astype(bool) & t_["score_A"].notna()
                c4c[k] = dict(addresses=len(idx), n_dec_differ=int(nd.sum()), n_A_differ=int(na.sum()),
                              score_raw=_cmp(o_[f"align_{cls}_{h}_raw"], t_["score_raw"], rtol=1e-12),
                              score_A=_cmp(o_[f"align_{cls}_{h}_A"], t_["score_A"], rtol=1e-12),
                              score_raw_differ_where_n_dec_equal=int((~nd & ((o_[f"align_{cls}_{h}_raw"] - t_["score_raw"]).abs() > 1e-12 * t_["score_raw"].abs().clip(lower=1e-12))).sum()),
                              A_gate_own_vs_style_reportable_disagree=int((gate_a_own != gate_style).sum()),
                              A_own_pass_style_fail=int((gate_a_own & ~gate_style).sum()),
                              A_own_fail_style_pass=int((~gate_a_own & gate_style).sum()))
                log(f"  {k}: {c4c[k]}")
    res["check4c_style"] = c4c

    # ---------------- check 5: reconciliation ----------------
    log("\n=== check 5: base-eligible addresses per window ===")
    c5 = {r.window_id: int(sup.loc[sup["window_id"] == r.window_id, "base_eligible"].sum())
          for r in wt.itertuples(index=False)}
    l60 = [c5[f"L60_{M:%Y-%m-%d}"] for M in MONDAYS]
    res["check5_base_eligible"] = dict(per_window=c5, L60=l60, expected=EXPECTED_L60_BASE,
                                       match=l60 == EXPECTED_L60_BASE)
    log(json.dumps(res["check5_base_eligible"], indent=1))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"\nchecks done in {res['runtime_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="run the acceptance checks on the built outputs")
    args = ap.parse_args()
    checks() if args.checks else build()
