"""Position size / leverage / holding columns of the address profile on a 5-minute grid, per
address x sample, and how stable each column is between an earlier and a later period.

Grid: bar close times c = W0 + (i+1) * 5 min, i = 0 .. N-1, c in (W0, W1]. A grid point belongs to a
sample by its bar open c - 5 min, so sample [a, b) holds the bars opening in [a, b).

Position of trader x symbol at c: gend of the last package event (fill group) with ts <= c. Before
the first event, a carried-in gstart != 0 is held from 00:00 of the first day of the funding run
that ends the day before the first fill day (none -> from the first event only). Positions known
only from funding as of the sample end E (pair with no package fill before E, funding entries in
[W0, E)) hold each funding day's last szi on that day's 288 bars; for E = W1 this is exactly
hidden_positions.parquet (asserted), for pre (E = PRE_END) it also covers pairs whose first fill
comes later, and nothing from that later fill is used in pre. Synthetic starts (carried-in,
funding-only) take effect on the first bar opening at the start time.

Episodes (pre / post / full only; even / odd leave the episode columns null): build_episodes on the
groups inside the sample, plus funding-only runs and package positions held through the whole
sample with no event inside it, each censored where it crosses the sample boundary.

Price at c: Binance 5m bar closing at c > latest Hyperliquid 4h candle closed by c > last package
fill price of the symbol at or before c.

Inputs (read-only): lib.prep_perp_fills(), lib.load_traders(), v2dev/build/{groups, snap_value_pair,
tape}.parquet, extdata/build/{klines_5m, hl_candles_4h, funding, hidden_positions}.parquet,
profile/profile.py (build_episodes), profile/address_profile.csv (check 3).
Outputs (work/profile2/build): address_exposure.csv, daily.parquet, stability.csv, summary.json;
log in work/profile2/logs/exposure.log. The per-grid-point table is never written.
"""
import importlib.util
import json
import os
import sys
import time
import warnings

os.environ.setdefault("ARK_DATA_DIR", "/Users/xc/Documents/arkstream_quest/ArkStream笔试题_v2")
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/v2dev/code")
import lib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.simplefilter("ignore", FutureWarning)

ROOT = "/Users/xc/Documents/arkstream_quest"
OUT = f"{ROOT}/work/profile2/build"
LOG_PATH = f"{ROOT}/work/profile2/logs/exposure.log"
V2BUILD = f"{ROOT}/work/v2dev/build"
EXT = f"{ROOT}/work/extdata/build"
OLD_PROFILE = f"{ROOT}/work/profile/address_profile.csv"

W0 = pd.Timestamp("2025-07-27 00:00:00")
W1 = pd.Timestamp("2026-01-05 00:00:00")
PRE_END = pd.Timestamp("2025-11-03 00:00:00")
STEP = pd.Timedelta(minutes=5).value
DAY = pd.Timedelta(days=1).value
WEEK = 7 * DAY
HOUR = pd.Timedelta(hours=1).value
W0N, W1N, PREN = W0.value, W1.value, PRE_END.value
N = (W1N - W0N) // STEP                  # 46,656 grid points
NPRE = (PREN - W0N) // STEP              # bars opening before PRE_END
PPD = DAY // STEP                        # grid points per day
NDAYS = N // PPD
OPEN = W0N + np.arange(N, dtype=np.int64) * STEP     # bar open
CLOSE = OPEN + STEP                                   # grid time c
WEEK_IDX = (OPEN - W0N) // WEEK
DAY_IDX = np.arange(N) // PPD
SAMPLES = ["pre", "post", "full", "even", "odd"]
SAMPLE_END = {"pre": PREN, "post": W1N, "full": W1N, "even": W1N, "odd": W1N}
EXP_EPS = 1e-6                           # as in profile.py, for the old snapshot exposure
AV_MIN = 100.0
N_BOOT = 2000
SEED = 20261119
SRC_NAMES = {0: "none", 1: "binance_5m", 2: "hl_4h", 3: "last_fill"}
CHECK4_ADDR = "0x4648baa72d8cfe3f31882fe52bd9fd8e6d539373"

_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


def load_build_episodes():
    """build_episodes from the current profile script (imported by path: `profile` is a stdlib name)."""
    spec = importlib.util.spec_from_file_location("profile_v1", f"{ROOT}/work/profile/profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_episodes


def ts_in_sample(ts_ns, sample):
    """Event times (fills, snapshots) in a sample; even / odd by the week index of the time."""
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    inwin = (ts_ns >= W0N) & (ts_ns < W1N)
    if sample == "pre":
        return inwin & (ts_ns < PREN)
    if sample == "post":
        return inwin & (ts_ns >= PREN)
    if sample == "full":
        return inwin
    wk = (ts_ns - W0N) // WEEK
    return inwin & (wk % 2 == (0 if sample == "even" else 1))


GRID_MASK = {s: ts_in_sample(OPEN, s) for s in SAMPLES}


def sample_blocks(sample):
    """Contiguous time blocks of a sample, as (block id, a_ns, b_ns). even / odd are cut at every week."""
    if sample == "pre":
        return [(0, W0N, PREN)]
    if sample == "post":
        return [(0, PREN, W1N)]
    if sample == "full":
        return [(0, W0N, W1N)]
    par = 0 if sample == "even" else 1
    nwk = -(-(W1N - W0N) // WEEK)
    return [(k, W0N + k * WEEK, min(W0N + (k + 1) * WEEK, W1N)) for k in range(nwk) if k % 2 == par]


# ---------------------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------------------
def build_prices(p, symbols):
    """px[s, i], src[s, i]: price at grid point i and its source (1 Binance, 2 HL 4h, 3 last fill, 0 none)."""
    S = len(symbols)
    px = np.full((S, N), np.nan)
    src = np.zeros((S, N), dtype=np.int8)
    tape = p.sort_values(["symbol", "ts", "tid"], kind="stable")
    tape_by = {s: (d["ts"].to_numpy().astype("int64"), d["price"].to_numpy()) for s, d in tape.groupby("symbol")}
    hl = pd.read_parquet(f"{EXT}/hl_candles_4h.parquet")
    hl = hl[hl["close"].notna()].sort_values(["symbol", "open_time"])
    hl_by = {s: ((d["open_time"] + pd.Timedelta(hours=4)).to_numpy().astype("int64"), d["close"].to_numpy())
             for s, d in hl.groupby("symbol")}
    kdir = f"{EXT}/klines_5m"
    info = {"n_kline_files": 0, "kline_misaligned_rows": 0}
    for si, sym in enumerate(symbols):
        if sym in tape_by:                                       # (iii) last package fill, any address
            t, v = tape_by[sym]
            j = np.searchsorted(t, CLOSE, side="right") - 1
            ok = j >= 0
            px[si, ok] = v[j[ok]]
            src[si, ok] = 3
        if sym in hl_by:                                         # (ii) latest HL 4h candle closed by c
            t, v = hl_by[sym]
            j = np.searchsorted(t, CLOSE, side="right") - 1
            ok = j >= 0
            px[si, ok] = v[j[ok]]
            src[si, ok] = 2
        f = f"{kdir}/{sym}.parquet"
        if os.path.exists(f):                                    # (i) Binance bar closing at c
            info["n_kline_files"] += 1
            k = pd.read_parquet(f, columns=["open_time", "close"])
            k = k[(k["open_time"] >= W0) & (k["open_time"] < W1) & k["close"].notna()]
            o = k["open_time"].to_numpy().astype("int64") - W0N
            al = o % STEP == 0
            info["kline_misaligned_rows"] += int((~al).sum())
            i = o[al] // STEP
            px[si, i] = k["close"].to_numpy()[al]
            src[si, i] = 1
    return px, src, info


# ---------------------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------------------
def event_grid_index(ts_ns):
    """First grid point c_i >= ts: i = ceil((ts - W0) / 5 min) - 1."""
    return np.maximum(-((W0N - ts_ns) // STEP) - 1, 0)


def pair_pos(ev_ts, ev_gend, gstart0, carry_ns):
    """Position on the full grid for one package pair. ev_ts / ev_gend in time order."""
    gi = event_grid_index(ev_ts)
    arr = np.full(N, -1, dtype=np.int64)
    last = np.r_[gi[1:] != gi[:-1], True]               # last event mapped to each grid point wins
    arr[gi[last]] = np.flatnonzero(last)
    arr = np.maximum.accumulate(arr)
    pos = np.where(arr >= 0, ev_gend[np.maximum(arr, 0)], 0.0)
    if carry_ns is not None:
        i0 = (carry_ns - W0N) // STEP                    # first bar opening at the carry start
        pos[i0:gi[0]] = gstart0
    pos[np.abs(pos) < lib.POS_EPS] = 0.0
    return pos


def carried_start(days, first_day_ns):
    """00:00 of the earliest day D >= W0 with a funding entry on every day D .. first fill day - 1."""
    d = first_day_ns - DAY
    if d < W0N or d not in days:
        return None
    while d - DAY >= W0N and (d - DAY) in days:
        d -= DAY
    return d


def funding_only(fund, first_ts, fday_all, sym_i, e_ns):
    """Positions known only from funding as of sample end e_ns: trader x coin pairs with no package fill
    before e_ns but funding entries in [W0, e_ns). Each UTC day with an entry holds that day's last szi.
    Returns (daily table trader, coin, day, szi; runs of consecutive days with one sign, one row each)."""
    f = fund[(fund["time"] >= W0) & (fund["time"] < pd.Timestamp(e_ns))]
    f = f.merge(first_ts, on=["trader", "coin"], how="left")          # first_ts: trader, coin, first_ts
    f = f[f["first_ts"].isna() | (f["first_ts"] >= pd.Timestamp(e_ns))]
    d = (f.sort_values("time", kind="stable").groupby(["trader", "coin", "day"])["szi"].last().reset_index())
    d = d[d["szi"].abs() >= lib.POS_EPS].sort_values(["trader", "coin", "day"]).reset_index(drop=True)
    rows = []
    for (t, c), x in d.groupby(["trader", "coin"], sort=True):
        dn = x["day"].to_numpy().astype("int64")
        sz = x["szi"].to_numpy()
        cut = np.flatnonzero((np.diff(dn) != DAY) | (np.sign(sz[1:]) != np.sign(sz[:-1]))) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(dn)]):
            s_ns, e_run = int(dn[a]), int(dn[b - 1]) + DAY
            rows.append({"trader": t, "coin": c, "sym_i": sym_i[c], "start_ns": s_ns, "end_ns": e_run,
                         "szi": sz[a], "szi_days": sz[a:b],
                         "unk_left": s_ns == W0N and (t, c, W0N - DAY) in fday_all,
                         "unk_right": e_run >= e_ns})
    return d, pd.DataFrame(rows)


def av_grid(snap_ns, snap_av, cutoff_ns, q_ns):
    """av(c) = max(av(s_prev), av(s_next)) if s_next - s_prev <= 48 h; at the edges the one snapshot
    within 24 h of c. Only snapshots before cutoff_ns."""
    keep = snap_ns < cutoff_ns
    S, V = snap_ns[keep], snap_av[keep]
    out = np.full(len(q_ns), np.nan)
    if len(S) == 0:
        return out
    j = np.searchsorted(S, q_ns, side="right") - 1
    hp, hn = j >= 0, j + 1 < len(S)
    jp, jn = np.clip(j, 0, len(S) - 1), np.clip(j + 1, 0, len(S) - 1)
    both = hp & hn & (S[jn] - S[jp] <= 48 * HOUR)
    out[both] = np.maximum(V[jp], V[jn])[both]
    op = hp & ~hn & (q_ns - S[jp] <= 24 * HOUR)
    out[op] = V[jp][op]
    on = ~hp & hn & (S[jn] - q_ns <= 24 * HOUR)
    out[on] = V[jn][on]
    return out


def q_(x, p):
    return float(np.quantile(x, p)) if len(x) else np.nan


def grid_stats(gross, net, npos, av, span):
    """Exposure / leverage columns over the observation span (boolean mask on the given arrays)."""
    r = {}
    n_span = int(span.sum())
    inm = span & (gross > 0)
    n_inm = int(inm.sum())
    r["n_span_points"] = n_span
    r["time_in_market"] = n_inm / n_span if n_span else np.nan
    gi = gross[inm]
    r["gross_exp_med"] = q_(gi, 0.5)
    r["gross_exp_p95"] = q_(gi, 0.95)
    valid = inm & np.isfinite(av) & (av >= AV_MIN)
    lev = gross[valid] / av[valid]
    r["lev_med"] = q_(lev, 0.5)
    r["lev_p95"] = q_(lev, 0.95)
    r["n_lev_days"] = int(len(np.unique(DAY_IDX[:len(gross)][valid])))
    r["av_invalid_share"] = 1.0 - valid.sum() / n_inm if n_inm else np.nan
    sg = gross[span].sum()
    r["net_bias"] = net[span].sum() / sg if sg > 0 else np.nan
    r["long_time_share"] = float((net[inm] > 0).mean()) if n_inm else np.nan
    r["n_pos_med"] = q_(npos[inm], 0.5)
    r["n_pos_p95"] = q_(npos[inm], 0.95)
    r["held_ntl_h"] = sg * STEP / HOUR
    return r


# ---------------------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------------------
def sample_episodes(build_episodes, g, fo_runs, held_through, px, sample):
    """Episodes of one sample (pre / post / full: a single block [a, b)). One row per episode, `kind`:
      fill          build_episodes on the groups inside the block
      funding_only  runs of funding-only positions overlapping the block: always censored on both sides
                    (their opening and closing are not observed as fills)
      held_through  package positions held through the whole block with no event in it: censored both sides
    """
    (_, a_ns, b_ns), = sample_blocks(sample)
    ts = g["ts"].to_numpy().astype("int64")
    sub = g.loc[(ts >= a_ns) & (ts < b_ns), ["pair", "trader", "ts", "gstart", "gend", "signed", "notional", "vwap"]].copy()
    sub["pid"] = sub["pair"]
    ep, _, _ = build_episodes(sub.reset_index(drop=True))
    ep["trader"] = ep["pid"].map(sub.drop_duplicates("pid").set_index("pid")["trader"])
    ep = ep[["trader", "left_cens", "right_cens", "complete", "dur_h", "open_ntl"]].assign(kind="fill")
    rows = []
    for h in fo_runs.itertuples(index=False):
        if h.start_ns < b_ns and h.end_ns > a_ns:
            # the opening and closing of a funding-only run are never observed as fills, so the run
            # is censored on both sides and never enters the complete-episode statistics
            rows.append((h.trader, True, True, False, np.nan, 0.0, "funding_only"))
    rows += [(t, True, True, False, np.nan, 0.0, "held_through") for t in held_through]
    he = pd.DataFrame(rows, columns=["trader", "left_cens", "right_cens", "complete", "dur_h", "open_ntl", "kind"])
    return pd.concat([ep, he], ignore_index=True)


def episode_columns(ep, traders):
    out = pd.DataFrame(index=traders)
    gt = ep.groupby("trader")
    out["n_episodes"] = gt.size()
    out["n_left_cens"] = ep[ep["left_cens"]].groupby("trader").size()
    out["n_right_cens"] = ep[ep["right_cens"]].groupby("trader").size()
    ce = ep[ep["complete"]].copy()
    out["n_complete"] = ce.groupby("trader").size()
    for c in ("n_episodes", "n_left_cens", "n_right_cens", "n_complete"):
        out[c] = out[c].fillna(0).astype(int)
    hq = ce.groupby("trader")["dur_h"].quantile([0.25, 0.5, 0.75]).unstack()
    out["hold_med_h"] = hq[0.5] if len(hq) else np.nan
    out["hold_p25_h"] = hq[0.25] if len(hq) else np.nan
    out["hold_p75_h"] = hq[0.75] if len(hq) else np.nan
    w = ce[ce["open_ntl"].notna()]
    num = (w["open_ntl"] * w["dur_h"]).groupby(w["trader"]).sum()
    den = w["open_ntl"].groupby(w["trader"]).sum()
    out["hold_wavg_h"] = num / den.where(den != 0)
    out["hold_ok"] = out["n_complete"] >= 10
    return out


# ---------------------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------------------
def spearman_boot(x, y):
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry).sum(1) / den
    r = r[np.isfinite(r)]
    lo, hi = (np.percentile(r, [2.5, 97.5]) if len(r) else (np.nan, np.nan))
    return float(rho), float(lo), float(hi)


STAB_COLS = ["time_in_market", "gross_exp_med", "gross_exp_p95", "lev_med", "lev_p95", "av_med", "net_bias",
             "long_time_share", "n_pos_med", "hold_turnover_h", "hold_med_h", "hold_wavg_h"]
HOLD_COLS = ("hold_med_h", "hold_wavg_h")       # episode columns: not computed for even / odd
EPISODE_COLS = ["n_episodes", "n_left_cens", "n_right_cens", "n_complete", "hold_med_h", "hold_p25_h", "hold_p75_h",
                "hold_wavg_h", "hold_ok"]


def eligible(w, col, s):
    """Address eligibility for a column in sample s (w: wide frame indexed by trader, columns (col, sample))."""
    e = w[("enough", s)].astype(bool)
    if col.startswith("lev_") or col == "av_med":
        e &= w[("lev_ok", s)].fillna(False).astype(bool) & ~w[("hip3_asof", s)].astype(bool)
    if col in HOLD_COLS:
        e &= w[("hold_ok", s)].fillna(False).astype(bool)
    return e & w[(col, s)].notna()


def stability(ax):
    w = ax.set_index(["trader", "sample"]).unstack("sample")      # columns (col, sample)
    rows = []
    for col in STAB_COLS:
        if col in HOLD_COLS:
            # no even / odd cells; the restricted variant can only mean "eligible in pre and post", which is
            # the pairwise set again -- labelled so it is not read as the four-sample set
            e = {s: eligible(w, col, s) for s in ("pre", "post")}
            cells = [("pre_post", ("pre", "post"), [("pairwise", e["pre"] & e["post"]),
                                                     ("all4_na_eq_pre_post_only", e["pre"] & e["post"])])]
        else:
            e = {s: eligible(w, col, s) for s in ("pre", "post", "even", "odd")}
            all4 = e["pre"] & e["post"] & e["even"] & e["odd"]
            cells = [(comp, ss, [("pairwise", e[ss[0]] & e[ss[1]]), ("all4", all4)])
                     for comp, ss in (("pre_post", ("pre", "post")), ("even_odd", ("even", "odd")))]
        for comp, (s1, s2), subsets in cells:
            for subset, m in subsets:
                x = w.loc[m, (col, s1)].to_numpy(dtype=float)
                y = w.loc[m, (col, s2)].to_numpy(dtype=float)
                rho, lo, hi = spearman_boot(x, y)
                rows.append({"column": col, "comparison": comp, "subset": subset, "n": int(m.sum()),
                             "spearman": rho, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
def main():
    global _logf
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    _logf = open(LOG_PATH, "w")
    summary = {}
    build_episodes = load_build_episodes()

    p = lib.prep_perp_fills()
    g = pd.read_parquet(f"{V2BUILD}/groups.parquet").sort_values(["trader", "symbol", "ts"]).reset_index(drop=True)
    tr = lib.load_traders()
    fund = pd.read_parquet(f"{EXT}/funding.parquet")
    hidden = pd.read_parquet(f"{EXT}/hidden_positions.parquet")
    log(f"loaded in {time.time() - t0:.1f}s: fills {len(p):,}, groups {len(g):,}, snapshots {len(tr):,}, "
        f"funding {len(fund):,}, hidden rows {len(hidden):,}; grid N {N:,} (pre {NPRE:,})")
    assert p["ts"].min() > W0 and p["ts"].max() < W1
    assert (fund["time"] < W1).all()

    traders = pd.Index(sorted(p["trader"].unique()), name="trader")
    symbols = sorted(set(g["symbol"]) | set(hidden["coin"]))
    sym_i = {s: i for i, s in enumerate(symbols)}

    # ---------------- prices ----------------
    px, src, pinfo = build_prices(p, symbols)
    log(f"prices: {len(symbols)} symbols, {pinfo['n_kline_files']} with Binance files, misaligned kline rows "
        f"{pinfo['kline_misaligned_rows']}; grid points by source (all symbols): "
        + ", ".join(f"{SRC_NAMES[k]} {int((src == k).sum()):,}" for k in range(4)))

    # ---------------- pairs, carried-in, hidden ----------------
    fund["day"] = fund["time"].dt.floor("D")
    pairs = g[["trader", "symbol"]].drop_duplicates().reset_index(drop=True)
    pairs["pair"] = np.arange(len(pairs))
    g = g.merge(pairs, on=["trader", "symbol"], how="left")
    rowno = pd.Series(np.arange(len(g)), index=g.index)
    bounds = pd.DataFrame({"r0": rowno.groupby(g["pair"]).min(), "r1": rowno.groupby(g["pair"]).max() + 1,
                           "first_ts": g.groupby("pair")["ts"].min()})
    pairs = pairs.join(bounds, on="pair")
    pairs["gstart0"] = g["gstart"].to_numpy()[pairs["r0"].to_numpy()]
    pairs["sym_i"] = pairs["symbol"].map(sym_i)
    car = pairs[pairs["gstart0"] != 0]
    fd = fund[["trader", "coin", "day"]].drop_duplicates().merge(
        car[["trader", "symbol"]].rename(columns={"symbol": "coin"}), on=["trader", "coin"])
    fdays = {k: set(v.to_numpy().astype("int64")) for k, v in fd.groupby(["trader", "coin"])["day"]}
    carry = {}
    for r in car.itertuples(index=False):
        carry[r.pair] = carried_start(fdays.get((r.trader, r.symbol), set()), r.first_ts.floor("D").value)
    pairs["carry_ns"] = pd.array([carry.get(k) for k in pairs["pair"]], dtype="Int64")
    pairs["late"] = pairs["first_ts"] >= PRE_END          # no package fill before PRE_END: absent from pre
    n_car = len(car)
    n_car_run = int(pairs["carry_ns"].notna().sum())
    n_car_w0day = int((car["first_ts"].dt.floor("D") == W0).sum())
    log(f"carried-in pairs (first gstart != 0): {n_car:,}; with a funding run {n_car_run:,}; without {n_car - n_car_run:,} "
        f"(of which first fill on the W0 day, so no run possible by definition: {n_car_w0day}); "
        f"carry starts at W0: {int((pairs['carry_ns'] == W0N).sum())}; "
        f"carried-in pairs with a run whose first fill is >= PRE_END (not used in pre): "
        f"{int((pairs['late'] & pairs['carry_ns'].notna()).sum())}")

    # positions known only from funding, as of each sample end (W1 for post / full / even / odd, PRE_END for pre)
    fday_all = set(map(tuple, fund[["trader", "coin"]].assign(d=fund["day"].to_numpy().astype("int64"))
                       .drop_duplicates().to_numpy().tolist()))
    ftab = pairs[["trader", "symbol", "first_ts"]].rename(columns={"symbol": "coin"})
    fo_day_w1, hid = funding_only(fund, ftab, fday_all, sym_i, W1N)
    a_ = fo_day_w1[["trader", "coin", "day", "szi"]].sort_values(["trader", "coin", "day"]).reset_index(drop=True)
    b_ = hidden[["trader", "coin", "day", "szi"]].sort_values(["trader", "coin", "day"]).reset_index(drop=True)
    assert a_.equals(b_), "funding-only positions as of W1 do not reproduce hidden_positions.parquet"
    fo_day_pre, fo_pre = funding_only(fund, ftab, fday_all, sym_i, PREN)
    hid_keys = set(zip(hid["trader"], hid["coin"]))
    fo_pre["added"] = [(t, c) not in hid_keys for t, c in zip(fo_pre["trader"], fo_pre["coin"])]
    log(f"funding-only as of W1: {hid[['trader', 'coin']].drop_duplicates().shape[0]} pairs, {len(hid)} runs, "
        f"{len(fo_day_w1):,} pair-days on {hid['trader'].nunique()} addresses == hidden_positions.parquet (asserted); "
        f"left-censored at W0 {int(hid['unk_left'].sum())}; still open at W1 {int(hid['unk_right'].sum())}")
    # informational: funding szi vs the gstart of the later first fill, on the pre-only funding pair-days
    fdp = fo_day_pre.merge(pairs[["trader", "symbol", "gstart0"]].rename(columns={"symbol": "coin"}),
                           on=["trader", "coin"], how="inner")
    szi_ne = (fdp["szi"] - fdp["gstart0"]).abs() > 1e-9 * np.maximum(fdp["gstart0"].abs(), 1.0)
    fo_pre_info = {"pairs": int(fo_pre[["trader", "coin"]].drop_duplicates().shape[0]), "runs": len(fo_pre),
                   "pair_days": len(fo_day_pre),
                   "added_vs_hidden_pairs": int(fo_pre.loc[fo_pre["added"], ["trader", "coin"]].drop_duplicates().shape[0]),
                   "added_pairs_with_later_fill": int(fdp[["trader", "coin"]].drop_duplicates().shape[0]),
                   "added_pairs_with_later_fill_gstart_zero": int(fdp.loc[fdp["gstart0"] == 0, ["trader", "coin"]]
                                                                  .drop_duplicates().shape[0]),
                   "added_pair_days_with_later_fill": len(fdp), "pair_days_szi_ne_later_gstart": int(szi_ne.sum()),
                   "pairs_szi_ne_later_gstart": int(fdp.loc[szi_ne, ["trader", "coin"]].drop_duplicates().shape[0]),
                   "left_censored_at_W0": int(fo_pre["unk_left"].sum()), "open_at_pre_end": int(fo_pre["unk_right"].sum())}
    log(f"funding-only as of PRE_END: {fo_pre_info}")

    # ---------------- HIP-3 flag times ----------------
    h3 = pd.concat([p.loc[p["is_hip3"], ["trader", "ts"]].rename(columns={"ts": "t"}),
                    fund.loc[fund["coin"].str.contains(":", regex=False), ["trader", "time"]].rename(columns={"time": "t"})])
    hip3_first = h3.groupby("trader")["t"].min().reindex(traders)

    # ---------------- snapshots ----------------
    tr = tr[tr["trader"].isin(traders)]
    snaps = {t: (d["ts"].to_numpy().astype("int64"), d["acct_val"].to_numpy()) for t, d in tr.groupby("trader")}

    # check 2 query points: snapshots the old pipeline used (ts >= trader's first fill), grid point <= snapshot
    sv = pd.read_parquet(f"{V2BUILD}/snap_value_pair.parquet")
    first_fill = p.groupby("trader")["ts"].min()
    svq = sv[["trader", "ts"]].drop_duplicates()
    svq = svq[svq["ts"] >= svq["trader"].map(first_fill)]
    svq["gi"] = (svq["ts"].to_numpy().astype("int64") - W0N) // STEP - 1
    svq = svq[(svq["gi"] >= 0) & (svq["gi"] < N)]
    svq_by = {t: (d["ts"].to_numpy(), d["gi"].to_numpy()) for t, d in svq.groupby("trader")}

    # ---------------- per-trader grid pass ----------------
    ev_ts_all = g["ts"].to_numpy().astype("int64")
    ev_ge_all = g["gend"].to_numpy()
    pairs_by_t = {t: d for t, d in pairs.groupby("trader")}
    hid_by_t = {t: d for t, d in hid.groupby("trader")}
    fo_pre_by_t = {t: d for t, d in fo_pre.groupby("trader")}
    blocks_ep = [(s,) + sample_blocks(s)[0][1:] for s in ("pre", "post", "full")]
    src_exp = np.zeros(4)
    tally = dict(carry_ntl_h=0.0, hidden_ntl_h=0.0, total_ntl_h=0.0, late_pre_ntl_h=0.0, missing_px_points=0,
                 pos_points=0, fo_pre_ntl_h=0.0, fo_pre_added_ntl_h=0.0, fo_pre_missing_px_points=0)
    held_through = {s: [] for s in ("pre", "post", "full")}
    held_ntl = {s: 0.0 for s in ("pre", "post", "full")}
    nz_count = np.zeros(len(pairs), dtype=np.int64)
    grid_rows, daily_rows, chk2_rows, chk2_hid = [], [], [], []
    chk4 = None

    def add_fo(acc, h, n_max):
        """Add one funding-only run (szi per day, 288 bars a day) to acc, clipped to n_max grid points."""
        i0, i1 = (h.start_ns - W0N) // STEP, min((h.end_ns - W0N) // STEP, n_max)
        val = np.repeat(h.szi_days, PPD)[:i1 - i0] * px[h.sym_i, i0:i1]
        miss = np.isnan(val)
        val = np.where(miss, 0.0, val)
        a = np.abs(val)
        acc[0][i0:i1] += a
        acc[1][i0:i1] += val
        acc[2][i0:i1] += 1
        return i0, i1, a, int(miss.sum())

    for ti, t in enumerate(traders):
        # A: package pairs with a fill before PRE_END (every sample); B: the other package pairs (samples ending
        # at W1); H: funding-only as of W1 (samples ending at W1); P: funding-only as of PRE_END (pre, NPRE long)
        A = [np.zeros(N), np.zeros(N), np.zeros(N, dtype=np.int32)]
        B = [np.zeros(N), np.zeros(N), np.zeros(N, dtype=np.int32)]
        H = [np.zeros(N), np.zeros(N), np.zeros(N, dtype=np.int32)]
        P = [np.zeros(NPRE), np.zeros(NPRE), np.zeros(NPRE, dtype=np.int32)]
        starts_full, starts_pre = [], []
        snap_ts, snap_gi = svq_by.get(t, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.int64)))
        pt = pairs_by_t.get(t)
        for r in pt.itertuples(index=False):
            ev_ts = ev_ts_all[r.r0:r.r1]
            cns = None if pd.isna(r.carry_ns) else int(r.carry_ns)
            pos = pair_pos(ev_ts, ev_ge_all[r.r0:r.r1], r.gstart0, cns)
            st = cns if cns is not None else int(ev_ts[0])
            starts_full.append(st)
            if not r.late:
                starts_pre.append(st)
            nzi = np.flatnonzero(pos)
            nz_count[r.pair] = len(nzi)
            if len(snap_gi):
                chk2_rows.append(pd.DataFrame({"trader": t, "symbol": r.symbol, "ts": snap_ts, "pos_new": pos[snap_gi],
                                               "px_new": px[r.sym_i, snap_gi], "src_new": src[r.sym_i, snap_gi]}))
            for name, a_ns, b_ns in blocks_ep:          # held through the whole sample, no event inside
                if name == "pre" and r.late:
                    continue
                ia, ib = (a_ns - W0N) // STEP, (b_ns - W0N) // STEP
                k0, k1 = np.searchsorted(ev_ts, [a_ns, b_ns])
                if k0 == k1 and np.any(pos[ia:ib] != 0):
                    held_through[name].append(t)
                    held_ntl[name] += abs(pos[ia] * px[r.sym_i, ia]) if np.isfinite(px[r.sym_i, ia]) else 0.0
            if not len(nzi):
                continue
            lo, hi = nzi[0], nzi[-1] + 1
            ps = pos[lo:hi]
            val = ps * px[r.sym_i, lo:hi]
            miss = np.isnan(val) & (ps != 0)
            val = np.where(np.isnan(val), 0.0, val)
            a = np.abs(val)
            acc = B if r.late else A
            acc[0][lo:hi] += a
            acc[1][lo:hi] += val
            acc[2][lo:hi] += ps != 0
            sr = src[r.sym_i, lo:hi]
            for k in range(4):
                src_exp[k] += a[sr == k].sum()
            tally["missing_px_points"] += int(miss.sum())
            tally["pos_points"] += int((ps != 0).sum())
            tally["total_ntl_h"] += a.sum() * STEP / HOUR
            if cns is not None:
                i0, i1 = (cns - W0N) // STEP, int(event_grid_index(ev_ts[:1])[0])
                tally["carry_ntl_h"] += a[max(i0 - lo, 0):max(i1 - lo, 0)].sum() * STEP / HOUR
            if r.late:
                tally["late_pre_ntl_h"] += a[:max(NPRE - lo, 0)].sum() * STEP / HOUR
        hidden_here = []
        for h in (hid_by_t[t].itertuples(index=False) if t in hid_by_t else []):
            starts_full.append(h.start_ns)
            i0, i1, a, nmiss = add_fo(H, h, N)
            sr = src[h.sym_i, i0:i1]
            for k in range(4):
                src_exp[k] += a[sr == k].sum()
            tally["missing_px_points"] += nmiss
            tally["pos_points"] += i1 - i0
            tally["total_ntl_h"] += a.sum() * STEP / HOUR
            tally["hidden_ntl_h"] += a.sum() * STEP / HOUR
            hidden_here.append((i0, i1, h.szi_days, h.sym_i))
        for h in (fo_pre_by_t[t].itertuples(index=False) if t in fo_pre_by_t else []):
            starts_pre.append(h.start_ns)
            i0, i1, a, nmiss = add_fo(P, h, NPRE)
            tally["fo_pre_ntl_h"] += a.sum() * STEP / HOUR
            tally["fo_pre_missing_px_points"] += nmiss
            if h.added:
                tally["fo_pre_added_ntl_h"] += a.sum() * STEP / HOUR
        if len(snap_gi) and hidden_here:
            hg = np.zeros(len(snap_gi))
            for i0, i1, szd, si in hidden_here:
                inside = (snap_gi >= i0) & (snap_gi < i1)
                szi = szd[np.clip((snap_gi - i0) // PPD, 0, len(szd) - 1)]
                hg += np.where(inside, np.abs(szi * np.nan_to_num(px[si, snap_gi])), 0.0)
            chk2_hid.append(pd.DataFrame({"trader": t, "ts": snap_ts, "hidden_gross": hg}))

        gross, net, npos = A[0] + B[0] + H[0], A[1] + B[1] + H[1], A[2] + B[2] + H[2]
        pre_arrs = (A[0][:NPRE] + P[0], A[1][:NPRE] + P[1], A[2][:NPRE] + P[2])
        sn_ts, sn_av = snaps.get(t, (np.array([], dtype=np.int64), np.array([])))
        av_full = av_grid(sn_ts, sn_av, W1N, CLOSE)
        av_pre = av_grid(sn_ts, sn_av, PREN, CLOSE[:NPRE])
        fa_full, fa_pre = min(starts_full), (min(starts_pre) if starts_pre else W1N)
        for s in SAMPLES:
            if s == "pre":
                arrs = pre_arrs + (av_pre,)
                span = GRID_MASK[s][:NPRE] & (OPEN[:NPRE] >= fa_pre)
            else:
                arrs = (gross, net, npos, av_full)
                span = GRID_MASK[s] & (OPEN >= fa_full)
            r = grid_stats(*arrs, span)
            fa = fa_pre if s == "pre" else fa_full
            r.update(trader=t, sample=s, first_appearance=pd.Timestamp(fa) if fa < SAMPLE_END[s] else pd.NaT)
            grid_rows.append(r)

        # daily (full view)
        G = gross.reshape(NDAYS, PPD)
        A = av_full.reshape(NDAYS, PPD)
        S_ = (OPEN >= fa_full).reshape(NDAYS, PPD)
        inm = S_ & (G > 0)
        dd = np.flatnonzero(S_.any(1))
        valid = inm & np.isfinite(A) & (A >= AV_MIN)
        L = np.where(valid, G / np.where(valid, A, 1.0), -np.inf)
        first_inm = inm.argmax(1)
        any_inm = inm.any(1)
        daily_rows.append(pd.DataFrame({
            "trader": t, "day": W0 + pd.to_timedelta(dd, unit="D"),
            "gross_max": np.where(S_, G, 0.0).max(1)[dd],
            "gross_mean": (np.where(S_, G, 0.0).sum(1) / np.maximum(S_.sum(1), 1))[dd],
            "n_inm_points": inm.sum(1)[dd],
            "av_first_inm": np.where(any_inm, A[np.arange(NDAYS), first_inm], np.nan)[dd],
            "lev_max": np.where(valid.any(1), L.max(1), np.nan)[dd]}))
        if t == CHECK4_ADDR:
            chk4 = (gross.copy(), av_full.copy(), {r.symbol: pair_pos(ev_ts_all[r.r0:r.r1], ev_ge_all[r.r0:r.r1], r.gstart0,
                                                                      None if pd.isna(r.carry_ns) else int(r.carry_ns))
                                                   for r in pt.itertuples(index=False)})
        if ti % 200 == 0:
            log(f"  trader {ti}/{len(traders)} at {time.time() - t0:.0f}s")

    tot = src_exp.sum()
    shares = {SRC_NAMES[k]: float(src_exp[k] / tot) for k in range(4)}
    log(f"\nexposure valued by source (position-notional x grid points): "
        + ", ".join(f"{k} {v:.4%}" for k, v in shares.items()))
    log(f"position grid points {tally['pos_points']:,}; without any price {tally['missing_px_points']:,}")
    log(f"total notional-hours {tally['total_ntl_h']:.4e}; carried-in segments (before first fill) "
        f"{tally['carry_ntl_h']:.4e} ({tally['carry_ntl_h'] / tally['total_ntl_h']:.3%}); hidden "
        f"{tally['hidden_ntl_h']:.4e} ({tally['hidden_ntl_h'] / tally['total_ntl_h']:.3%})")
    log(f"pre view: package pairs with first fill >= PRE_END held {tally['late_pre_ntl_h']:.4e} notional-hours inside "
        f"the pre range in the W1 view (not used in pre); funding-only as of PRE_END {tally['fo_pre_ntl_h']:.4e}, of "
        f"which pairs not in the W1 hidden set {tally['fo_pre_added_ntl_h']:.4e}; pre funding-only points without "
        f"price {tally['fo_pre_missing_px_points']}")
    log("package positions held through a whole sample with no event in it (one both-censored episode each): "
        + "; ".join(f"{k}: {len(v)} pairs on {len(set(v))} addresses, start notional {held_ntl[k]:,.0f}"
                    for k, v in held_through.items()))
    fo_pre_info.update(ntl_h_in_pre=tally["fo_pre_ntl_h"], added_ntl_h_in_pre=tally["fo_pre_added_ntl_h"],
                       late_package_pairs_ntl_h_in_pre_range_w1_view=tally["late_pre_ntl_h"])
    summary["funding_only_pre"] = fo_pre_info
    summary["held_through_episodes"] = {k: {"pairs": len(v), "addresses": len(set(v)), "start_notional": held_ntl[k]}
                                        for k, v in held_through.items()}
    summary["price_source_shares"] = shares
    summary["position_grid_points"] = tally["pos_points"]
    summary["position_grid_points_without_price"] = tally["missing_px_points"]
    summary["carried_in"] = {"pairs": n_car, "with_funding_run": n_car_run, "without_run": n_car - n_car_run,
                             "without_run_first_fill_on_W0_day": n_car_w0day,
                             "carry_start_at_W0": int((pairs["carry_ns"] == W0N).sum()),
                             "with_run_first_fill_ge_pre_end": int((pairs["late"] & pairs["carry_ns"].notna()).sum()),
                             "ntl_h": tally["carry_ntl_h"], "share_of_total_ntl_h": tally["carry_ntl_h"] / tally["total_ntl_h"]}
    summary["hidden"] = {"pairs": int(hid[["trader", "coin"]].drop_duplicates().shape[0]), "runs": len(hid),
                         "addresses": int(hid["trader"].nunique()), "ntl_h": tally["hidden_ntl_h"],
                         "share_of_total_ntl_h": tally["hidden_ntl_h"] / tally["total_ntl_h"],
                         "left_censored_at_W0": int(hid["unk_left"].sum()), "open_at_W1": int(hid["unk_right"].sum())}

    # ---------------- per address x sample table ----------------
    gr = pd.DataFrame(grid_rows)
    p["ts_ns"] = p["ts"].to_numpy().astype("int64")
    p["day"] = p["ts"].dt.floor("D")
    tr_ns = tr["ts"].to_numpy().astype("int64")
    frames = []
    for s in SAMPLES:
        x = pd.DataFrame(index=traders)
        ps_ = p[ts_in_sample(p["ts_ns"], s)]
        x["n_active_days"] = ps_.groupby("trader")["day"].nunique()
        x["n_orders"] = ps_.groupby(["trader", "symbol", "oid"]).size().groupby("trader").size()
        x["turnover"] = ps_.groupby("trader")["notional"].sum()
        x[["n_active_days", "n_orders"]] = x[["n_active_days", "n_orders"]].fillna(0).astype(int)
        x["turnover"] = x["turnover"].fillna(0.0)
        x["enough"] = (x["n_active_days"] >= 10) & (x["n_orders"] >= 30)
        x["hip3_asof"] = (hip3_first < pd.Timestamp(SAMPLE_END[s])).fillna(False).astype(bool)
        ts_ = tr[ts_in_sample(tr_ns, s)]
        gs_ = ts_.groupby("trader")
        x["n_snap"] = gs_.size()
        x["n_snap"] = x["n_snap"].fillna(0).astype(int)
        x["av_med"] = gs_["acct_val"].median()
        x["av_last"] = gs_["acct_val"].last()
        if s in ("pre", "post", "full"):
            ep = sample_episodes(build_episodes, g, fo_pre if s == "pre" else hid, held_through[s], px, s)
            x = x.join(episode_columns(ep, traders))
            x["hold_ok"] = x["hold_ok"].astype("boolean")
            log(f"sample {s}: episodes {len(ep):,} ("
                + ", ".join(f"{k} {n}" for k, n in ep["kind"].value_counts().sort_index().items())
                + f"), complete {int(ep['complete'].sum()):,}")
        else:                                          # no episode columns for even / odd
            for c in EPISODE_COLS:
                x[c] = pd.array([pd.NA] * len(x), dtype="boolean") if c == "hold_ok" else np.nan
        x = x.join(gr[gr["sample"] == s].set_index("trader").drop(columns="sample"))
        x["hold_turnover_h"] = x["held_ntl_h"] / (x["turnover"] / 2).where(x["turnover"] > 0)
        x["lev_ok"] = x["n_lev_days"] >= 10
        flag = x["hip3_asof"]
        for c in ("av_med", "av_last", "lev_med", "lev_p95", "n_lev_days", "av_invalid_share"):
            x[c] = x[c].where(~flag)
        x["lev_ok"] = x["lev_ok"] & ~flag
        x["sample"] = s
        frames.append(x.reset_index())
    cols = ["trader", "sample", "first_appearance", "n_active_days", "n_orders", "turnover", "enough", "hip3_asof",
            "n_snap", "av_med", "av_last", "n_span_points", "time_in_market", "gross_exp_med", "gross_exp_p95",
            "lev_med", "lev_p95", "n_lev_days", "lev_ok", "av_invalid_share", "net_bias", "long_time_share",
            "n_pos_med", "n_pos_p95", "held_ntl_h", "hold_turnover_h", "n_episodes", "n_left_cens", "n_right_cens",
            "n_complete", "hold_med_h", "hold_p25_h", "hold_p75_h", "hold_wavg_h", "hold_ok"]
    ax = pd.concat(frames, ignore_index=True)[cols]
    ax.to_csv(f"{OUT}/address_exposure.csv", index=False)

    counts = {}
    log("\n=== counts per sample ===")
    for s in SAMPLES:
        x = ax[ax["sample"] == s]
        obs = x["first_appearance"] < pd.Timestamp(SAMPLE_END[s])
        c = {"observed": int(obs.sum()), "with_fills": int((x["n_orders"] > 0).sum()), "enough": int(x["enough"].sum()),
             "hip3_flagged": int(x["hip3_asof"].sum()), "lev_ok": int(x["lev_ok"].sum()),
             "enough_and_lev_ok_not_hip3": int((x["enough"] & x["lev_ok"] & ~x["hip3_asof"]).sum())}
        if s in ("pre", "post", "full"):
            c["hold_ok"] = int(x["hold_ok"].sum())
            c["enough_and_hold_ok"] = int((x["enough"] & x["hold_ok"]).sum())
        else:
            c["hold_ok"] = c["enough_and_hold_ok"] = "n/a"
        counts[s] = c
        log(f"{s:5s} " + ", ".join(f"{k} {v}" for k, v in c.items()))
    summary["counts"] = counts

    # daily
    hip3_full = ax.loc[ax["sample"] == "full"].set_index("trader")["hip3_asof"]
    daily = pd.concat(daily_rows, ignore_index=True)
    daily["hip3_full"] = daily["trader"].map(hip3_full).astype(bool)
    for c in ("av_first_inm", "lev_max"):
        daily[c] = daily[c].where(~daily["hip3_full"])
    daily.to_parquet(f"{OUT}/daily.parquet", index=False)

    # ---------------- stability ----------------
    st = stability(ax)
    st.to_csv(f"{OUT}/stability.csv", index=False)
    log("\n=== stability (Spearman, 95% bootstrap CI) ===")
    log(st.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    summary["stability"] = st.to_dict(orient="records")

    # ---------------- check 1: replay ----------------
    log("\n=== check 1: replay at random nonzero (trader, symbol, grid point) ===")
    rng = np.random.default_rng(SEED)
    cum = np.cumsum(nz_count)
    draws = np.sort(rng.integers(0, cum[-1], size=2000))
    pk = np.searchsorted(cum, draws, side="right")
    kk = draws - np.r_[0, cum][pk]
    pf = p.sort_values(["trader", "symbol", "ts", "tid"], kind="stable")
    pf_by = {k: d for k, d in pf.groupby(["trader", "symbol"])}
    diffs = []
    n_heads = {True: 0, False: 0}                  # pairs whose chain start was unique / ambiguous
    for pid_ in np.unique(pk):
        r = pairs.iloc[pid_]
        pos = pair_pos(ev_ts_all[r.r0:r.r1], ev_ge_all[r.r0:r.r1], r.gstart0,
                       None if pd.isna(r.carry_ns) else int(r.carry_ns))
        gi = np.flatnonzero(pos)[kk[pk == pid_]]
        f = pf_by[(r.trader, r.symbol)]
        fts = f["ts"].to_numpy().astype("int64")
        # row order within a timestamp is not execution order: the chain starts at the row of the
        # first timestamp whose start_position is no other same-timestamp row's end_pos
        f0 = f[fts == fts[0]]
        e0 = f0["end_pos"].to_numpy()
        heads = [s for s in f0["start_position"].to_numpy() if not np.any(np.abs(e0 - s) < 1e-9 * max(1.0, abs(s)))]
        n_heads[len(heads) == 1] += 1
        init = heads[0] if heads else f0["start_position"].iloc[0]
        cpos = init + np.cumsum(f["signed"].to_numpy())
        j = np.searchsorted(fts, CLOSE[gi], side="right") - 1
        rep = np.where(j >= 0, cpos[np.maximum(j, 0)], init)
        rep = np.where(np.abs(rep) < lib.POS_EPS, 0.0, rep)
        diffs.append(np.abs(pos[gi] - rep) / np.maximum(np.abs(rep), 1e-12))
    diffs = np.concatenate(diffs)
    c1 = {"n": int(len(diffs)), "pairs": int(len(np.unique(pk))), "max_rel_diff": float(diffs.max()),
          "n_rel_diff_gt_1e-9": int((diffs > 1e-9).sum()), "chain_start_unique": n_heads[True],
          "chain_start_ambiguous": n_heads[False]}
    log(c1)
    summary["check1_replay"] = c1

    # ---------------- check 2: snapshot gross vs old pipeline ----------------
    log("\n=== check 2: grid gross at snapshot times vs old snap_value_pair exposure ===")
    nw = pd.concat(chk2_rows, ignore_index=True)
    old = sv.merge(svq[["trader", "ts"]], on=["trader", "ts"])
    old["exposure"] = old["exposure"].where(old["exposure"].abs() >= EXP_EPS, 0.0)
    m = old.merge(nw, on=["trader", "symbol", "ts"], how="left")
    tape = pd.read_parquet(f"{V2BUILD}/tape.parquet").rename(columns={"ts": "tape_ts", "price": "px_old"})
    tape["ts"] = tape["tape_ts"]
    m = m.sort_values("ts", kind="stable").reset_index(drop=True)
    tb = tape.sort_values("ts", kind="stable")
    m = pd.merge_asof(m, tb[["symbol", "ts", "px_old"]], on="ts", by="symbol", direction="backward")
    mf = pd.merge_asof(m[["symbol", "ts"]].reset_index(), tb[["symbol", "ts", "px_old"]], on="ts", by="symbol",
                       direction="forward").set_index("index")["px_old"]
    m["px_old"] = m["px_old"].fillna(mf)
    m["g_old"] = m["exposure"].abs()
    m["g_new"] = (m["pos_new"] * m["px_new"]).abs().fillna(0.0)
    m["g_newpos_oldpx"] = (m["pos_new"] * m["px_old"]).abs()
    agg = m.groupby(["trader", "ts"]).agg(g_old=("g_old", "sum"), g_new_pkg=("g_new", "sum"),
                                          g_newpos_oldpx=("g_newpos_oldpx", "sum")).reset_index()
    if chk2_hid:
        agg = agg.merge(pd.concat(chk2_hid, ignore_index=True), on=["trader", "ts"], how="left")
    agg["hidden_gross"] = agg.get("hidden_gross", 0.0)
    agg["hidden_gross"] = agg["hidden_gross"].fillna(0.0)
    agg["g_new"] = agg["g_new_pkg"] + agg["hidden_gross"]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    def qd(s):
        return {f"p{int(q * 100):02d}": float(s.quantile(q)) for q in qs} | {"n": int(s.notna().sum())}

    both = agg[(agg["g_old"] > 0) & (agg["g_new"] > 0)]
    c2 = {"snapshots": len(agg), "both_positive": len(both),
          "old_pos_new_zero": int(((agg["g_old"] > 0) & (agg["g_new"] == 0)).sum()),
          "old_zero_new_pos": int(((agg["g_old"] == 0) & (agg["g_new"] > 0)).sum()),
          "old_zero_new_pos_hidden_only": int(((agg["g_old"] == 0) & (agg["g_new_pkg"] == 0) & (agg["hidden_gross"] > 0)).sum()),
          "ratio_new_over_old": qd(both["g_new"] / both["g_old"]),
          "ratio_position_effect (new pos, old px / old)": qd((both["g_newpos_oldpx"] / both["g_old"])),
          "ratio_price_effect (new pkg / new pos at old px)": qd(both["g_new_pkg"] / both["g_newpos_oldpx"].where(both["g_newpos_oldpx"] > 0)),
          "ratio_hidden_effect (new / new pkg)": qd(both["g_new"] / both["g_new_pkg"].where(both["g_new_pkg"] > 0)),
          "aggregate_sum_new_over_old": float(agg["g_new"].sum() / agg["g_old"].sum()),
          "aggregate_sum_pkg_new_over_old": float(agg["g_new_pkg"].sum() / agg["g_old"].sum()),
          "aggregate_hidden_share_of_new": float(agg["hidden_gross"].sum() / agg["g_new"].sum())}
    # pair-level reasons
    m["pos_old"] = m["pos"].where(m["pos"].abs() >= lib.POS_EPS, 0.0)
    m["pos_new"] = m["pos_new"].fillna(0.0)
    dpos = (m["pos_new"] - m["pos_old"]).abs() > 1e-9 * np.maximum(m["pos_old"].abs(), 1.0)
    pre_first = m["ts"] < m["pair_first_ts"]
    c2["pair_rows"] = len(m)
    c2["pair_rows_pos_differ"] = int(dpos.sum())
    c2["pair_rows_pos_differ_before_pair_first_fill"] = int((dpos & pre_first).sum())
    c2["pair_rows_pos_differ_before_pair_first_fill_old_ntl"] = float(m.loc[dpos & pre_first, "g_old"].sum())
    c2["pair_rows_pos_differ_after_first_fill"] = int((dpos & ~pre_first).sum())
    c2["pair_rows_pos_differ_after_first_fill_old_ntl"] = float(m.loc[dpos & ~pre_first, "g_old"].sum())
    c2["total_old_ntl"] = float(m["g_old"].sum())
    pr = m[(m["pos_new"] != 0) & (m["px_new"] > 0) & (m["px_old"] > 0)].copy()
    pr["pratio"] = pr["px_new"] / pr["px_old"]
    c2["px_ratio_new_over_old_by_new_source"] = {SRC_NAMES[k]: qd(d["pratio"]) for k, d in pr.groupby("src_new")}
    pr["abs_log"] = np.abs(np.log(pr["pratio"]))
    pr["age_bin"] = pd.cut(pr["mark_age_h"], [-np.inf, 1, 24, 24 * 7, np.inf], labels=["<1h", "1-24h", "1-7d", ">7d"])
    c2["px_abs_log_ratio_median_by_old_mark_age"] = {str(k): {"median": float(d["abs_log"].median()), "n": len(d)}
                                                     for k, d in pr.groupby("age_bin", observed=True)}
    for k_, v_ in c2.items():
        log(f"{k_}: {v_}")
    summary["check2_snapshot_gross"] = c2

    # ---------------- check 3: old vs new on full ----------------
    log("\n=== check 3: old profile vs new (full sample, enough addresses) ===")
    oldp = pd.read_csv(OLD_PROFILE).set_index("trader")
    nf = ax[ax["sample"] == "full"].set_index("trader")
    log(f"enough: new {int(nf['enough'].sum())}, old {int(oldp['enough'].sum())}, "
        f"both {int((nf['enough'] & oldp['enough'].reindex(nf.index).fillna(False)).sum())}")
    en = nf.index[nf["enough"]]
    c3 = {}
    for grp_name, idx in (("all", en),
                          ("old_hold_med_lt24h", en[(oldp.loc[en, "hold_med_h"] < 24).to_numpy()]),
                          ("old_hold_med_ge24h", en[(oldp.loc[en, "hold_med_h"] >= 24).to_numpy()])):
        for col in ("lev_med", "time_in_market", "gross_exp_med"):
            a_, b_ = nf.loc[idx, col], oldp.loc[idx, col]
            ok = a_.notna() & b_.notna()
            ratio = (a_[ok] / b_[ok].where(b_[ok] != 0))
            rho = stats.spearmanr(a_[ok], b_[ok]).statistic if ok.sum() > 2 else np.nan
            c3[f"{grp_name}/{col}"] = {"n": int(ok.sum()), "spearman": float(rho),
                                       "median_ratio_new_old": float(ratio.median()),
                                       "p25_ratio": float(ratio.quantile(0.25)), "p75_ratio": float(ratio.quantile(0.75))}
            log(f"{grp_name:20s} {col:15s} n {ok.sum():4d}  spearman {rho:.3f}  median new/old {ratio.median():.3f}  "
                f"(p25 {ratio.quantile(0.25):.3f}, p75 {ratio.quantile(0.75):.3f})")
    summary["check3_old_vs_new"] = c3

    # ---------------- check 4: one address ----------------
    log(f"\n=== check 4: {CHECK4_ADDR} 2025-08-19 .. 08-22 ===")
    c4 = []
    if chk4 is not None:
        gross4, av4, pos4 = chk4
        sn = tr[(tr["trader"] == CHECK4_ADDR) & (tr["ts"] >= "2025-08-17") & (tr["ts"] < "2025-08-24")]
        log(sn[["ts", "pnl", "net_deposit", "acct_val"]].to_string(index=False))
        for d in pd.date_range("2025-08-19", "2025-08-22"):
            i0 = (d.value - W0N) // STEP
            sl = slice(i0, i0 + PPD)
            G, A = gross4[sl], av4[sl]
            k = int(np.argmax(G))
            valid = (G > 0) & np.isfinite(A) & (A >= AV_MIN)
            lev = np.where(valid, G / np.where(valid, A, 1.0), np.nan)
            held = {s: float(v[sl][k] * px[sym_i[s], i0 + k]) for s, v in pos4.items() if v[sl][k] != 0}
            row = {"day": str(d.date()), "gross_max": float(G.max()), "at": str(pd.Timestamp(int(CLOSE[i0 + k]))),
                   "av_at_max": float(A[k]), "lev_at_max": float(G[k] / A[k]) if valid[k] else None,
                   "lev_max_day": float(np.nanmax(lev)) if valid.any() else None,
                   "av_used_values": sorted(set(np.round(A[G > 0], 2).tolist())), "positions_at_max": held}
            c4.append(row)
            log(row)
        dd = daily[(daily["trader"] == CHECK4_ADDR) & (daily["day"] >= "2025-08-19") & (daily["day"] <= "2025-08-22")]
        log(dd.to_string(index=False))
        f4 = p[(p["trader"] == CHECK4_ADDR) & (p["ts"] >= "2025-08-19") & (p["ts"] < "2025-08-23")]
        log(f4.groupby([f4["ts"].dt.date, "symbol", "dir"])["notional"].sum().round(2).to_string())
    summary["check4"] = c4

    summary["runtime_s"] = time.time() - t0
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log(f"\nwrote {OUT}/address_exposure.csv ({len(ax)} rows), daily.parquet ({len(daily):,} rows), stability.csv, "
        f"summary.json; runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
