"""Excursion, realised-result and open-position dimensions of the rolling address profile, for the same 80
windows as ledger.py, each S = [a, b) computed at its own cutoff C = b.

  0. One causal walk over all package fill groups (per pair, by time) reproducing profile.build_episodes state
     exactly (episode id, sign, running average entry before / after, position before / after, role per group).
     The state at a cutoff C is the state after the last group with ts < C.
  1. Excursions of every episode opened inside S (observed opening ts in [a, b)), observed on [open_ts, e],
     e = close_ts if close_ts < b else b: group events at their vwap against the state before the event, plus
     every Binance 5-minute bar fully inside one segment between events (high / low and close-only variants).
     MAE / MFE in price terms, in units of the 288-bar daily volatility at the opening, and MAE in dollars
     over the account-value proxy at the opening.
  2. Realised result of complete episodes: closed_pnl minus fees (flip group pnl to the closed episode, flip
     fees pro rata by size); losing-vs-winning median holding-time ratio.
  3. Package positions still open at C, marked at ledger.price_at_cutoff, and funding-only positions at C.

Reuses ledger.py (imported, not modified): load_inputs, truncate_inputs, build_prep, window_table,
price_at_cutoff, get_align, EXPO.av_grid, constants. Binance high / low come from the kline files, loaded here.

Inputs (read-only): as ledger.py; work/roll/build/{episodes, support}.parquet (checks only).
Outputs (work/roll/build): ledger_x.parquet, support_x.parquet, positions.parquet, open_positions.parquet,
manifest_x.json, checks_x.json; logs work/roll/logs/{excursion, checks_x}.log.
Usage: python excursion.py (build), python excursion.py --checks.
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ next to ledger.py or the reviewed scripts
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/roll/code")
import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402

import ledger as LG  # noqa: E402

lib, PROF, EXPO = LG.lib, LG.PROF, LG.EXPO
ROOT, OUT, LOGS, KLINES = LG.ROOT, LG.OUT, LG.LOGS, LG.KLINES
W0N, W1N, STEP, DAY, HOUR = LG.W0N, LG.W1N, LG.STEP, LG.DAY, LG.HOUR
G0N, AV_MIN, INT_MAX, MONDAYS = LG.G0N, LG.AV_MIN, LG.INT_MAX, LG.MONDAYS
KL0N = G0N - STEP                       # open of the first bar of any kline file (2025-07-01 00:00)
NB = (W1N - KL0N) // STEP               # bar-open lattice KL0 .. W1 - 5 min
NAT = np.iinfo(np.int64).min
SEED = 20261120
OPEN, ADD, REDUCE, CLOSE, FLIP, NONE = 1, 2, 3, 4, 5, 0
ROLE_NAMES = {NONE: "none", OPEN: "open", ADD: "add", REDUCE: "reduce", CLOSE: "close", FLIP: "flip"}
# ledger units on the path class -> (MAE field, MFE field) of positions.parquet
UNITS = {"px": ("mae_px_c", "mfe_px_c"), "vol": ("mae_vol_c", "mfe_vol_c"), "rt": ("mae_rt_c", "mfe_rt_c"),
         "av": ("loss_cum_av", None)}
XCOLS = ["mae_px_med", "mae_px_p90", "mae_vol_med", "mae_vol_p90", "mae_rt_med", "mae_rt_p90", "mae_av_med",
         "mae_av_p90", "mfe_px_med", "mfe_px_p90", "mfe_vol_med", "mfe_vol_p90", "mfe_rt_med", "mfe_rt_p90",
         "hold_loss_win_ratio", "hold_med_loss_h", "hold_med_win_h", "open_loss_share_n", "open_loss_share_ntl",
         "exit_rate_loss", "exit_rate_gain", "exit_rate_ratio"]
SUPPORT_X = ["hip3_asof", "n_ep_opened", "n_scored", "n_scored_days", "n_nobar", "n_scored_complete",
             "n_path", "n_event_only", "path_share", "n_path_lowcov", "n_path_censored",
             "n_px", "d_px", "n_vol", "d_vol", "n_rt", "d_rt", "n_av", "d_av",
             "n_unscored_no_file", "n_unscored_missing_bars", "open_ntl_scored", "open_ntl_total", "open_ntl_scored_share",
             "n_complete", "n_win", "n_loss", "n_zero", "n_open", "n_open_known", "n_open_unknown_entry",
             "n_open_funding_only", "open_known_ntl", "open_known_ntl_maxshare", "open_known_kish",
             "t_loss_h", "t_gain_h", "t_flat_h", "n_exit_loss", "n_exit_gain", "n_exit_flat"]
EXPAND_CHUNK = 4_000_000                # bars expanded at once in the time-in-state pass

_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


# ---------------------------------------------------------------------------------------
# Inputs: high / low / close bars (own loader; ledger.load_inputs keeps closes only)
# ---------------------------------------------------------------------------------------
def load_hilo():
    return {f.name[:-8]: pd.read_parquet(f, columns=["open_time", "high", "low", "close"])
            for f in sorted(KLINES.glob("*.parquet"))}


def truncate_hilo(hl, C):
    """Same rule as ledger.truncate_inputs: keep bars closing by C; keys (the fixed symbol map) kept."""
    C = pd.Timestamp(C)
    return {s: k[k["open_time"] + pd.Timedelta(minutes=5) <= C].copy() for s, k in hl.items()}


def bar_arrays(k):
    """high, low, close on the bar-open lattice KL0 .. W1 (NaN where missing) and the cumulative count of
    present bars (all three finite); arrays padded by one element for reduceat."""
    hi, lo, cl = (np.full(NB + 1, np.nan) for _ in range(3))
    if len(k):
        o = k["open_time"].to_numpy().astype("int64") - KL0N
        assert np.all(o % STEP == 0) and o.min() >= 0
        i = o // STEP
        ok = i < NB
        hi[i[ok]], lo[i[ok]], cl[i[ok]] = (k[c].to_numpy()[ok] for c in ("high", "low", "close"))
    pres = np.isfinite(hi[:NB]) & np.isfinite(lo[:NB]) & np.isfinite(cl[:NB])
    return dict(hi=hi, lo=lo, cl=cl, cum=np.r_[0, np.cumsum(pres)])


# ---------------------------------------------------------------------------------------
# 0. Episode walk
# ---------------------------------------------------------------------------------------
def walk(pid, ts, gs, ge, sg, nt, vw, cash, fee):
    """Causal replay of profile.build_episodes. Per group: ep_before / ep_after (-1 = flat), avg_before /
    avg_after (NaN = flat or entry unknown), role, ontl = ep_after's increasing notional after the group,
    ccum / fcum = ep_after's cumulative cash flow / fees through the group (sequential sums; an opening flip
    contributes its |s1| / |signed| share, as in the ep_cash / ep_fee split).
    Episodes: pid, open_ns (NAT if left-censored in the package), close_ns (NAT if still open at the end of
    the inputs), sign, open_ntl, left, end_by_flip, later_add_ntl, later_loser_ntl, r_first (first row),
    r_close (-1 if not closed)."""
    n = len(pid)
    epb = np.full(n, -1, dtype=np.int64)
    epa = np.full(n, -1, dtype=np.int64)
    avb = np.full(n, np.nan)
    ava = np.full(n, np.nan)
    role = np.zeros(n, dtype=np.int8)
    ontl = np.full(n, np.nan)
    ccum = np.full(n, np.nan)
    fcum = np.full(n, np.nan)
    E = {k: [] for k in ("pid", "open_ns", "close_ns", "sign", "open_ntl", "left", "flip_end", "later", "loser",
                         "avg", "r_first", "r_close")}

    def new(p, t, s, ntl, left, avg, r):
        for k, v in (("pid", p), ("open_ns", t), ("close_ns", NAT), ("sign", s), ("open_ntl", ntl), ("left", left),
                     ("flip_end", False), ("later", 0.0), ("loser", 0.0), ("avg", avg), ("r_first", r),
                     ("r_close", -1)):
            E[k].append(v)
        return len(E["pid"]) - 1

    def close(e, t, flip, r):
        E["close_ns"][e], E["flip_end"][e], E["r_close"][e] = t, flip, r

    cur = -1
    cc = fc = np.nan
    for i in range(n):
        if i == 0 or pid[i] != pid[i - 1]:
            cur = -1                                  # previous pair's open episode stays right-censored
            cc = fc = np.nan
            if gs[i] != 0:                            # position pre-dates the pair's first group
                cur = new(pid[i], NAT, np.sign(gs[i]), 0.0, True, np.nan, i)
        s0, s1 = gs[i], ge[i]
        epb[i] = cur
        avb[i] = E["avg"][cur] if cur >= 0 else np.nan
        if s0 == 0:
            if s1 != 0:
                cur = new(pid[i], ts[i], np.sign(s1), nt[i], False, vw[i], i)
                role[i] = OPEN
                cc, fc = cash[i], fee[i]
        elif s1 == 0:
            close(cur, ts[i], False, i)
            cur = -1
            role[i] = CLOSE
        elif np.sign(s0) != np.sign(s1):
            close(cur, ts[i], True, i)
            cur = new(pid[i], ts[i], np.sign(s1), nt[i] * abs(s1) / abs(sg[i]), False, vw[i], i)
            role[i] = FLIP
            cc, fc = cash[i] * (abs(s1) / abs(sg[i])), fee[i] * (abs(s1) / abs(sg[i]))
        elif abs(s1) > abs(s0):
            E["open_ntl"][cur] += nt[i]
            if not E["left"][cur]:
                E["later"][cur] += nt[i]
                avg = E["avg"][cur]
                if (E["sign"][cur] > 0 and vw[i] < avg) or (E["sign"][cur] < 0 and vw[i] > avg):
                    E["loser"][cur] += nt[i]
                E["avg"][cur] = (avg * abs(s0) + vw[i] * (abs(s1) - abs(s0))) / abs(s1)
            role[i] = ADD
            cc, fc = cc + cash[i], fc + fee[i]
        else:
            role[i] = REDUCE
            cc, fc = cc + cash[i], fc + fee[i]
        epa[i] = cur
        if cur >= 0:
            ava[i] = E["avg"][cur]
            ontl[i] = E["open_ntl"][cur]
            ccum[i], fcum[i] = cc, fc
    E = {k: np.asarray(v) for k, v in E.items()}
    for k in ("pid", "open_ns", "close_ns", "r_first", "r_close"):
        E[k] = E[k].astype(np.int64)
    return dict(epb=epb, epa=epa, avb=avb, ava=ava, role=role, ontl=ontl, ccum=ccum, fcum=fcum), E


def build_x(inp, hilo, U):
    """Ledger prep plus the walk, group pnl / fee / cash contributions and per-episode realised sums."""
    X = SimpleNamespace()
    X.P = P = LG.build_prep(inp, U)
    g = lib.fill_groups(inp["p"]).sort_values(["trader", "symbol", "ts"], kind="stable").reset_index(drop=True)
    assert len(g) == len(P.g)
    for c in ("gstart", "gend", "signed", "notional", "vwap"):
        assert np.array_equal(g[c].to_numpy(), P.g[c].to_numpy(), equal_nan=True), c
    assert np.array_equal(g["ts"].to_numpy().astype("int64"), P.g_ts)
    X.pid = P.g["pid"].to_numpy()
    X.ts = P.g_ts
    X.gs, X.ge = P.g["gstart"].to_numpy(), P.g["gend"].to_numpy()
    X.sg, X.nt, X.vw = P.g["signed"].to_numpy(), P.g["notional"].to_numpy(), P.g["vwap"].to_numpy()
    X.cash_g, X.fee_g = g["cash"].to_numpy(), g["fee"].to_numpy()
    R, E = walk(X.pid, X.ts, X.gs, X.ge, X.sg, X.nt, X.vw, X.cash_g, X.fee_g)
    X.R, X.E = R, E
    X.nE = len(E["pid"])
    X.r0 = P.pairs["r0"].to_numpy()                     # pairs indexed by pid, rows contiguous
    assert np.array_equal(P.pairs["pid"].to_numpy(), np.arange(len(P.pairs))) and np.all(np.diff(X.r0) > 0)

    # realised pnl / fee / cash per episode: a flip's closed_pnl to the episode it closes, its fee and cash
    # split by size (|s0| / |signed| old, |s1| / |signed| new); other groups to their single episode
    flip = R["role"] == FLIP
    one = np.where(R["epb"] >= 0, R["epb"], R["epa"])
    w_old = np.where(flip, np.abs(X.gs) / np.abs(np.where(flip, X.sg, 1.0)), 1.0)
    w_new = np.where(flip, np.abs(X.ge) / np.abs(np.where(flip, X.sg, 1.0)), 0.0)
    X.dropped_none = dict(groups=int((one < 0).sum()), closed_pnl=float(g.loc[one < 0, "closed_pnl"].sum()),
                          fee=float(g.loc[one < 0, "fee"].sum()))
    k = one >= 0
    X.ep_closed_pnl = np.bincount(one[k], weights=g["closed_pnl"].to_numpy()[k], minlength=X.nE)

    def split(v):
        out = np.bincount(one[k], weights=(v * w_old)[k], minlength=X.nE)
        kf = flip & (R["epa"] >= 0)
        return out + np.bincount(R["epa"][kf], weights=(v * w_new)[kf], minlength=X.nE)

    X.ep_fee = split(g["fee"].to_numpy())
    X.fee_close_g = g["fee"].to_numpy() * w_old       # fee share a group gives the episode it closes
    X.ep_cash = split(g["cash"].to_numpy())
    X.ep_sym = P.pair_sym[E["pid"]]
    X.ep_ti = P.pair_ti[E["pid"]]
    X.hilo = hilo
    X.bars = {}
    X.has_file = set(hilo.keys())
    return X


def get_bars(X, sym):
    if sym not in X.bars:
        X.bars[sym] = bar_arrays(X.hilo[sym])
    return X.bars[sym]


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def per_ti(nU, ti, vals, funcs):
    """{name: array over ti} applying numpy funcs to each address's finite values (NaN where none)."""
    out = {name: np.full(nU, np.nan) for name in funcs}
    ok = np.isfinite(vals)
    t, v = ti[ok], vals[ok]
    o = np.lexsort((v, t))
    t, v = t[o], v[o]
    if len(t):
        st = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
        en = np.r_[st[1:], len(t)]
        for s, e in zip(st, en):
            for name, f in funcs.items():
                out[name][t[s]] = f(v[s:e])
    return out


MED = {"med": np.median}
MED90 = {"med": np.median, "p90": lambda x: float(np.quantile(x, 0.9))}


def dur_hours(end_ns, open_ns):
    """Hours exactly as profile.build_episodes computes dur_h."""
    e = pd.Series(np.asarray(end_ns, dtype=np.int64).view("datetime64[ns]"))
    o = pd.Series(np.asarray(open_ns, dtype=np.int64).view("datetime64[ns]"))
    return ((e - o).dt.total_seconds() / 3600.0).to_numpy()


def rows_of(ro, re):
    """Concatenated row ranges ro[i]..re[i] (inclusive) and the local episode index of each row."""
    nr = re - ro + 1
    loc = np.repeat(np.arange(len(ro)), nr)
    rows = ro[loc] + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
    return rows, loc


# ---------------------------------------------------------------------------------------
# 3. Positions open at a cutoff
# ---------------------------------------------------------------------------------------
def open_at(X, C):
    P, R, E = X.P, X.R, X.E
    cnt = np.add.reduceat((X.ts < C).astype(np.int64), X.r0)
    r = X.r0 + cnt - 1
    op = (cnt > 0) & (X.ge[np.maximum(r, 0)] != 0)
    pid = np.flatnonzero(op)
    rr = r[op]
    ep = R["epa"][rr]
    assert np.all(ep >= 0)
    pos = X.ge[rr]
    px = LG.price_at_cutoff(P, C)
    pxs = px[P.pairs["sym_i"].to_numpy()[pid]]
    known = ~E["left"][ep]
    avg = np.where(known, R["ava"][rr], np.nan)
    s = np.sign(pos)
    op_ns = E["open_ns"][ep]
    pk = pd.DataFrame({"ti": P.pair_ti[pid], "symbol": P.pair_sym[pid], "sign": s, "pos": pos, "avg": avg, "px": pxs,
                       "upnl": s * (pxs - avg) * np.abs(pos), "ntl": np.abs(pos) * pxs,
                       "open_ns": np.where(known, op_ns, NAT), "known": known, "kind": "package"})
    # funding-only at C, as on the ledger grid's last bar (day C - 1d, pair without a fill before C)
    fd = P.fday[(P.fday["pkg_first_ns"] >= C) & (P.fday["day_ns"] == C - DAY)]
    fpx = px[fd["sym_i"].to_numpy()]
    fo = pd.DataFrame({"ti": fd["ti"].to_numpy(), "symbol": fd["coin"].to_numpy(), "sign": np.sign(fd["szi"].to_numpy()),
                       "pos": fd["szi"].to_numpy(), "avg": np.nan, "px": fpx, "upnl": np.nan,
                       "ntl": np.abs(fd["szi"].to_numpy()) * fpx, "open_ns": NAT, "known": False,
                       "kind": "funding_only"})
    d = pd.concat([pk, fo], ignore_index=True)
    on = d["open_ns"].to_numpy()
    d["age_h"] = np.where(on != NAT, (C - np.where(on != NAT, on, C)) / HOUR, np.nan)
    return d


def open_stats(X, d):
    nU = X.P.nU
    kn = d[d["known"] & np.isfinite(d["px"])]
    loss = kn["upnl"] < 0
    n_kn = np.bincount(kn["ti"], minlength=nU)
    n_ls = np.bincount(kn["ti"], weights=loss.to_numpy(dtype=float), minlength=nU)
    nt_kn = np.bincount(kn["ti"], weights=kn["ntl"].to_numpy(), minlength=nU)
    nt_ls = np.bincount(kn["ti"], weights=kn["ntl"].where(loss, 0.0).to_numpy(), minlength=nU)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"open_loss_share_n": np.where(n_kn > 0, n_ls / np.maximum(n_kn, 1), np.nan),
               "open_loss_share_ntl": np.where(nt_kn > 0, nt_ls / np.where(nt_kn > 0, nt_kn, 1.0), np.nan)}
    nt2 = np.bincount(kn["ti"], weights=kn["ntl"].to_numpy() ** 2, minlength=nU)
    mx = np.zeros(nU)
    np.maximum.at(mx, kn["ti"].to_numpy(), kn["ntl"].to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_ = nt_kn > 0
        out["open_known_ntl"] = nt_kn
        out["open_known_ntl_maxshare"] = np.where(pos_, mx / np.where(pos_, nt_kn, 1.0), np.nan)
        out["open_known_kish"] = np.where(pos_, nt_kn ** 2 / np.where(pos_, nt2, 1.0), np.nan)
    pk = d["kind"] == "package"
    out["n_open"] = np.bincount(d["ti"], minlength=nU)
    out["n_open_known"] = np.bincount(d.loc[d["known"], "ti"], minlength=nU)
    out["n_open_unknown_entry"] = np.bincount(d.loc[pk & ~d["known"], "ti"], minlength=nU)
    out["n_open_funding_only"] = np.bincount(d.loc[~pk, "ti"], minlength=nU)
    return out


# ---------------------------------------------------------------------------------------
# 1 + 2. One window
# ---------------------------------------------------------------------------------------
def window_x(X, a, b, ostats):
    P, R, E = X.P, X.R, X.E
    nU = P.nU
    cnt = np.add.reduceat((X.ts < b).astype(np.int64), X.r0)
    last_row = X.r0 + cnt - 1
    eids = np.flatnonzero(~E["left"] & (E["open_ns"] >= a) & (E["open_ns"] < b))
    ro = E["r_first"][eids]
    comp = (E["r_close"][eids] >= 0) & (E["close_ns"][eids] < b)
    re = np.where(comp, E["r_close"][eids], last_row[E["pid"][eids]])
    assert np.all(re >= ro)
    sgn = E["sign"][eids].astype(float)
    sym = X.ep_sym[eids]
    end_ns = np.where(comp, E["close_ns"][eids], b)
    open_ns = E["open_ns"][eids]
    ne = len(eids)

    rows, loc = rows_of(ro, re)
    is_first = rows == ro[loc]
    is_last = rows == re[loc]
    n_events = np.bincount(loc, minlength=ne)
    n_adds = np.bincount(loc, weights=(R["role"][rows] == ADD).astype(float), minlength=ne).astype(np.int64)
    # increasing notional up to the end of observation (exact sequential sums of the walk)
    ontl_end = np.where(R["epa"][re] == eids, R["ontl"][re], R["ontl"][np.maximum(re - 1, 0)])

    # event observations: every group after the opening, against the state before it
    ev = rows[~is_first]
    ev_loc = loc[~is_first]
    assert np.array_equal(R["avb"][ev], R["ava"][ev - 1], equal_nan=True)
    ev_avg, ev_q, ev_x, ev_s = R["ava"][ev - 1], np.abs(X.ge[ev - 1]), X.vw[ev], sgn[ev_loc]

    # segments: after every event except a completing close
    sm = ~(is_last & comp[loc])
    sr, sl = rows[sm], loc[sm]
    t_s = X.ts[sr]
    t_e = np.where(is_last[sm], b, X.ts[np.minimum(sr + 1, len(X.ts) - 1)])
    s_avg, s_q, s_s = R["ava"][sr], np.abs(X.ge[sr]), sgn[sl]
    k_lo = -((KL0N - t_s) // STEP)                    # first bar opening at or after t_s
    k_hi = (t_e - KL0N) // STEP                       # bars k < k_hi close by t_e
    nexp = np.maximum(k_hi - k_lo, 0)
    npres = np.zeros(len(sr), dtype=np.int64)
    mx_hi, mn_lo, mx_cl, mn_cl = (np.full(len(sr), np.nan) for _ in range(4))
    seg_sym = sym[sl]
    for s_ in np.unique(seg_sym):
        if s_ not in X.has_file:
            continue
        ix = np.flatnonzero((seg_sym == s_) & (nexp > 0))
        if not len(ix):
            continue
        ba = get_bars(X, s_)
        npres[ix] = ba["cum"][k_hi[ix]] - ba["cum"][k_lo[ix]]
        idx = np.ravel(np.column_stack([k_lo[ix], k_hi[ix]]))
        mx_hi[ix] = np.maximum.reduceat(ba["hi"], idx)[0::2]
        mn_lo[ix] = np.minimum.reduceat(ba["lo"], idx)[0::2]
        mx_cl[ix] = np.maximum.reduceat(ba["cl"], idx)[0::2]
        mn_cl[ix] = np.minimum.reduceat(ba["cl"], idx)[0::2]
    n_bars = np.bincount(sl, weights=nexp, minlength=ne).astype(np.int64)
    n_miss = np.bincount(sl, weights=nexp - npres, minlength=ne).astype(np.int64)
    has_file = np.array([s_ in X.has_file for s_ in sym], dtype=bool)
    scored = has_file & (n_miss == 0)

    # time in state: every bar fully inside a segment of a scored episode, d = sign * (close - segment's running
    # average entry); loss bar d < 0, gain bar d > 0, flat bar d == 0. Expanded per window (no cache across windows)
    seg_cnt = np.zeros((3, len(sr)), dtype=np.int64)          # loss, gain, flat
    want = scored[sl] & (nexp > 0)
    for s_ in np.unique(seg_sym[want]):
        ba = get_bars(X, s_)
        ix_all = np.flatnonzero(want & (seg_sym == s_))
        cs_ = np.cumsum(nexp[ix_all])
        for ix in np.split(ix_all, np.searchsorted(cs_, np.arange(EXPAND_CHUNK, cs_[-1], EXPAND_CHUNK), side="right")):
            if not len(ix):
                continue
            n_ = nexp[ix]
            owner = np.repeat(np.arange(len(ix)), n_)
            kb = np.repeat(k_lo[ix], n_) + (np.arange(n_.sum()) - np.repeat(np.cumsum(n_) - n_, n_))
            d = np.repeat(s_s[ix], n_) * (ba["cl"][kb] - np.repeat(s_avg[ix], n_))
            assert np.isfinite(d).all()                         # scored: every required bar present
            for j, m in enumerate((d < 0, d > 0, d == 0)):
                seg_cnt[j, ix] = np.bincount(owner[m], minlength=len(ix))
    bars_lgf = np.stack([np.bincount(sl, weights=seg_cnt[j], minlength=ne) for j in range(3)]).astype(np.int64)

    def adv(x, avg, s):
        return np.maximum(0.0, -s * (x / avg - 1.0))

    def fav(x, avg, s):
        return np.maximum(0.0, s * (x / avg - 1.0))

    def adv_usd(x, avg, s, q):
        return np.maximum(0.0, -s * (x - avg)) * q

    mae, mfe, mae_c, mfe_c, mae_u = (np.zeros(ne) for _ in range(5))
    np.maximum.at(mae, ev_loc, adv(ev_x, ev_avg, ev_s))
    np.maximum.at(mfe, ev_loc, fav(ev_x, ev_avg, ev_s))
    np.maximum.at(mae_u, ev_loc, adv_usd(ev_x, ev_avg, ev_s, ev_q))
    mae_c[:], mfe_c[:] = mae, mfe
    bm = nexp > 0
    lg = s_s > 0
    x_adv, x_fav = np.where(lg, mn_lo, mx_hi), np.where(lg, mx_hi, mn_lo)
    xc_adv, xc_fav = np.where(lg, mn_cl, mx_cl), np.where(lg, mx_cl, mn_cl)
    with np.errstate(invalid="ignore"):
        for arr, vals in ((mae, adv(x_adv, s_avg, s_s)), (mfe, fav(x_fav, s_avg, s_s)),
                          (mae_c, adv(xc_adv, s_avg, s_s)), (mfe_c, fav(xc_fav, s_avg, s_s)),
                          (mae_u, adv_usd(x_adv, s_avg, s_s, s_q))):
            ok = bm & np.isfinite(vals)
            np.maximum.at(arr, sl[ok], vals[ok])
    # close-sampled against the FIXED opening entry, and peak cumulative episode loss (realised cash, fees and
    # the remaining position marked at x): value = cash_to_j - fee_to_j + pos_j * x for the state j of the
    # segment (bar closes) or of the row before an event (the event at its vwap, before it is applied)
    a_open = R["ava"][ro]
    mo_c, fo_c, lcum = (np.zeros(ne) for _ in range(3))
    np.maximum.at(mo_c, ev_loc, adv(ev_x, a_open[ev_loc], ev_s))
    np.maximum.at(fo_c, ev_loc, fav(ev_x, a_open[ev_loc], ev_s))
    cf_row = R["ccum"][rows] - R["fcum"][rows]
    ev_i = np.flatnonzero(~is_first)
    assert np.array_equal(R["epa"][rows[ev_i - 1]], eids[ev_loc]) and np.array_equal(R["epa"][sr], eids[sl])
    np.maximum.at(lcum, ev_loc, np.maximum(0.0, -(cf_row[ev_i - 1] + X.ge[ev - 1] * ev_x)))
    seg_cf, seg_pos = cf_row[sm], X.ge[sr]
    with np.errstate(invalid="ignore"):
        for arr, vals in ((mo_c, adv(xc_adv, a_open[sl], s_s)), (fo_c, fav(xc_fav, a_open[sl], s_s)),
                          (lcum, np.maximum(0.0, -(seg_cf + seg_pos * xc_adv)))):   # xc_adv: min close long, max short
            ok = bm & np.isfinite(vals)
            np.maximum.at(arr, sl[ok], vals[ok])
    for arr in (mae, mfe, mae_c, mfe_c, mae_u, mo_c, fo_c, lcum):
        arr[~scored] = np.nan

    # units
    sig = np.full(ne, np.nan)
    ci = (open_ns - open_ns % STEP - G0N) // STEP
    for s_ in np.unique(sym):
        ga = LG.get_align(P, s_)
        if ga is not None:
            m = sym == s_
            sig[m] = ga["sigma_1d"][ci[m]]
    ti = X.ep_ti[eids]
    hip3 = (P.h3_lt < b) | (P.h3_le <= b)
    c_open = W0N + (-((W0N - open_ns) // STEP)) * STEP     # first grid close >= open_ts
    av_open = np.full(ne, np.nan)
    empty = (np.array([], dtype=np.int64), np.array([]))
    for t_ in np.unique(ti):
        m = ti == t_
        sn_ts, sn_av = P.snaps.get(t_, empty)
        av_open[m] = EXPO.av_grid(sn_ts, sn_av, b, c_open[m])
    dur = dur_hours(end_ns, open_ns)
    span = end_ns - open_ns
    bar_cov = np.where(span > 0, n_bars * STEP / np.where(span > 0, span, 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        good_sig = np.isfinite(sig) & (sig > 0)
        sig1 = np.where(good_sig, sig, 1.0)
        mae_vol = np.where(good_sig, mae / sig1, np.nan)
        mfe_vol = np.where(good_sig, mfe / sig1, np.nan)
        mae_vol_c = np.where(good_sig, mae_c / sig1, np.nan)
        mfe_vol_c = np.where(good_sig, mfe_c / sig1, np.nan)
        rt_ok = good_sig & (dur >= 1) & (n_bars >= 1)
        den_rt = np.where(rt_ok, sig1 * np.sqrt(dur / 24.0), 1.0)
        mae_rt_c = np.where(rt_ok, mae_c / den_rt, np.nan)
        mfe_rt_c = np.where(rt_ok, mfe_c / den_rt, np.nan)
        good_av = np.isfinite(av_open) & (av_open >= AV_MIN) & ~hip3[ti]
        av1 = np.where(good_av, av_open, 1.0)
        mae_av = np.where(good_av, mae_u / av1, np.nan)
        loss_cum_av = np.where(good_av, lcum / av1, np.nan)

    # 2. realised result of complete episodes
    cp = np.where(comp, X.ep_closed_pnl[eids], np.nan)
    fe = np.where(comp, X.ep_fee[eids], np.nan)
    pnl = cp - fe
    path = scored & (n_bars >= 1)
    # exit state of complete path episodes: closing group's vwap against the running average before it
    rc = np.maximum(E["r_close"][eids], 0)
    cpath = path & comp
    d_exit = np.where(cpath, sgn * (X.vw[rc] - R["avb"][rc]), np.nan)
    exit_state = np.where(cpath, np.sign(d_exit), np.nan)

    pos = pd.DataFrame({
        "ti": ti, "symbol": sym, "open_ts": open_ns.view("datetime64[ns]"), "end_ts": end_ns.view("datetime64[ns]"),
        "complete": comp, "sign": sgn, "dur_h": dur, "open_ntl": ontl_end, "avg_open": a_open,
        "q_open": np.abs(X.ge[ro]), "open_by_flip": R["role"][ro] == FLIP,
        "end_by_flip": comp & E["flip_end"][eids], "n_events": n_events, "n_adds": n_adds,
        "n_bars": n_bars, "n_bars_missing": n_miss, "bar_cov": bar_cov, "scored": scored, "path": path,
        "unscored_reason": np.where(scored, "", np.where(has_file, "missing_bars", "no_file")),
        "mae_px": mae, "mfe_px": mfe, "mae_px_c": mae_c, "mfe_px_c": mfe_c, "mae_open_c": mo_c, "mfe_open_c": fo_c,
        "mae_usd": mae_u, "loss_cum": lcum, "sigma_1d": sig,
        "mae_vol": np.where(scored, mae_vol, np.nan), "mfe_vol": np.where(scored, mfe_vol, np.nan),
        "mae_vol_c": np.where(scored, mae_vol_c, np.nan), "mfe_vol_c": np.where(scored, mfe_vol_c, np.nan),
        "mae_rt_c": np.where(scored, mae_rt_c, np.nan), "mfe_rt_c": np.where(scored, mfe_rt_c, np.nan),
        "av_open": av_open, "mae_av": np.where(scored, mae_av, np.nan),
        "loss_cum_av": np.where(scored, loss_cum_av, np.nan),
        "closed_pnl_e": cp, "fee_e": fe, "pnl_net": pnl, "cash_e": np.where(comp, X.ep_cash[eids], np.nan),
        "fee_close": np.where(comp, X.fee_close_g[np.maximum(E["r_close"][eids], 0)], np.nan),
        "bars_loss": np.where(path, bars_lgf[0], np.nan), "bars_gain": np.where(path, bars_lgf[1], np.nan),
        "bars_flat": np.where(path, bars_lgf[2], np.nan), "d_exit": d_exit, "exit_state": exit_state})

    # per-address ledger columns (path class: scored, n_bars >= 1; close-sampled, running average) and support
    out = {}
    pc = pos[pos["path"]]
    pt = pc["ti"].to_numpy()
    for unit, (fm, ff) in UNITS.items():
        for kind, field in (("mae", fm), ("mfe", ff)):
            if field is None:
                continue
            r_ = per_ti(nU, pt, pc[field].to_numpy(), MED90)
            out[f"{kind}_{unit}_med"], out[f"{kind}_{unit}_p90"] = r_["med"], r_["p90"]
    cpl = pos[pos["complete"]]
    lo_ = cpl[cpl["pnl_net"] < 0]
    wi_ = cpl[cpl["pnl_net"] > 0]
    out["hold_med_loss_h"] = per_ti(nU, lo_["ti"].to_numpy(), lo_["dur_h"].to_numpy(), MED)["med"]
    out["hold_med_win_h"] = per_ti(nU, wi_["ti"].to_numpy(), wi_["dur_h"].to_numpy(), MED)["med"]
    with np.errstate(invalid="ignore", divide="ignore"):
        den = out["hold_med_win_h"]
        out["hold_loss_win_ratio"] = np.where(np.isfinite(den) & (den != 0), out["hold_med_loss_h"] / den, np.nan)
    out.update({k: ostats[k] for k in ("open_loss_share_n", "open_loss_share_ntl")})

    def cnt_(d, w=None):
        return np.bincount(d["ti"].to_numpy(), weights=w, minlength=nU)

    def days_(d):
        return np.bincount(d.assign(d_=d["open_ts"].dt.floor("D")).drop_duplicates(["ti", "d_"])["ti"], minlength=nU)

    sc = pos[pos["scored"]]
    sup = {"hip3_asof": hip3, "n_ep_opened": cnt_(pos), "n_scored": cnt_(sc), "n_scored_days": days_(sc),
           "n_nobar": cnt_(sc[sc["n_bars"] == 0]), "n_scored_complete": cnt_(sc[sc["complete"]]),
           "n_path": cnt_(pc), "n_event_only": cnt_(sc[sc["n_bars"] == 0]),
           "n_path_lowcov": cnt_(pc[pc["bar_cov"] < 0.5]), "n_path_censored": cnt_(pc[~pc["complete"]]),
           "n_unscored_no_file": cnt_(pos[pos["unscored_reason"] == "no_file"]),
           "n_unscored_missing_bars": cnt_(pos[pos["unscored_reason"] == "missing_bars"]),
           "open_ntl_scored": cnt_(sc, sc["open_ntl"].to_numpy()), "open_ntl_total": cnt_(pos, pos["open_ntl"].to_numpy()),
           "n_complete": cnt_(cpl), "n_win": cnt_(wi_), "n_loss": cnt_(lo_),
           "n_zero": cnt_(cpl[cpl["pnl_net"] == 0])}
    for unit, (fm, _) in UNITS.items():
        fin = pc[pc[fm].notna()]
        sup[f"n_{unit}"], sup[f"d_{unit}"] = cnt_(fin), days_(fin)
    with np.errstate(invalid="ignore", divide="ignore"):
        sup["open_ntl_scored_share"] = np.where(sup["open_ntl_total"] > 0,
                                                sup["open_ntl_scored"] / np.where(sup["open_ntl_total"] > 0,
                                                                                  sup["open_ntl_total"], 1.0), np.nan)
        sup["path_share"] = np.where(sup["n_scored"] > 0, sup["n_path"] / np.maximum(sup["n_scored"], 1), np.nan)
    sup.update({k: ostats[k] for k in ("n_open", "n_open_known", "n_open_unknown_entry", "n_open_funding_only",
                                       "open_known_ntl", "open_known_ntl_maxshare", "open_known_kish")})
    # exit rate by state (path class): exits per hour spent in that state
    for j, st_ in enumerate(("loss", "gain", "flat")):
        sup[f"t_{st_}_h"] = np.bincount(pt, weights=pc[f"bars_{st_}"].to_numpy(), minlength=nU) / 12.0
        sup[f"n_exit_{st_}"] = cnt_(pc[pc["exit_state"] == (-1, 1, 0)[j]])
    with np.errstate(invalid="ignore", divide="ignore"):
        for st_ in ("loss", "gain"):
            t_ = sup[f"t_{st_}_h"]
            out[f"exit_rate_{st_}"] = np.where(t_ > 0, sup[f"n_exit_{st_}"] / np.where(t_ > 0, t_, 1.0), np.nan)
        rl, rg = out["exit_rate_loss"], out["exit_rate_gain"]
        ok_ = np.isfinite(rl) & np.isfinite(rg) & (rg > 0)
        out["exit_rate_ratio"] = np.where(ok_, rl / np.where(ok_, rg, 1.0), np.nan)
    frame = pd.DataFrame({**out, **sup})
    for c in SUPPORT_X:
        if c.startswith("n_") or c.startswith("d_"):
            frame[c] = frame[c].astype(np.int64)
    frame["hip3_asof"] = frame["hip3_asof"].astype(bool)
    return frame[XCOLS + SUPPORT_X], pos


def run_windows(X, wt, verbose=True):
    """Ledger, support, positions for the windows of wt (one row per window id) and open positions per cutoff."""
    t0 = time.time()
    U = X.P.U
    cut = sorted({int(b.value) for b in wt["b"]})
    opens, ost = {}, {}
    for C in cut:
        d = open_at(X, C)
        opens[C] = d
        ost[C] = open_stats(X, d)
    res = {}
    for a, b in sorted({(int(a.value), int(b.value)) for a, b in zip(wt["a"], wt["b"])}):
        res[(a, b)] = window_x(X, a, b, ost[b])
    if verbose:
        log(f"  {len(res)} distinct windows, {len(cut)} cutoffs in {time.time() - t0:.1f}s")
    led, sup, pos = [], [], []
    for r in wt.itertuples(index=False):
        fr, ps = res[(r.a.value, r.b.value)]
        v = fr[XCOLS].copy()
        v.insert(0, "trader", U)
        lv = v.melt(id_vars="trader", var_name="column", value_name="value")
        lv.insert(1, "window_id", r.window_id)
        led.append(lv)
        s = fr[SUPPORT_X].copy()
        s.insert(0, "window_id", r.window_id)
        s.insert(0, "trader", U)
        sup.append(s)
        ps = ps.copy()
        ps.insert(0, "window_id", r.window_id)
        ps.insert(0, "trader", U[ps["ti"].to_numpy()])
        pos.append(ps.drop(columns="ti"))
    op = []
    for C, d in opens.items():
        d = d.copy()
        d.insert(0, "cutoff", pd.Timestamp(C))
        d.insert(0, "trader", U[d["ti"].to_numpy()])
        d["open_ts"] = d["open_ns"].to_numpy(dtype=np.int64).view("datetime64[ns]")      # NAT -> NaT
        op.append(d.drop(columns=["ti", "open_ns"]))
    cat = lambda d, cols: d.astype({c: "category" for c in cols})  # noqa: E731
    return dict(ledger_x=cat(pd.concat(led, ignore_index=True), ["trader", "window_id", "column"]),
                support_x=pd.concat(sup, ignore_index=True),
                positions=cat(pd.concat(pos, ignore_index=True), ["trader", "window_id", "symbol"]),
                open_positions=cat(pd.concat(op, ignore_index=True), ["trader", "symbol"]))


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def fee_convention(p):
    """Sign convention of the raw fills (perp rows)."""
    return dict(n_fills=len(p), fee_negative=int((p["fee"] < 0).sum()), fee_zero=int((p["fee"] == 0).sum()),
                fee_positive=int((p["fee"] > 0).sum()),
                fee_over_notional_median_taker=float((p["fee"] / p["notional"])[~p["is_maker"]].median()),
                fee_over_notional_median_maker=float((p["fee"] / p["notional"])[p["is_maker"]].median()),
                fee_tokens={k: int(v) for k, v in p["fee_token"].value_counts().items()},
                fee_tokens_non_usdc_symbols_all_hip3=bool(p.loc[p["fee_token"] != "USDC", "is_hip3"].all()))


def build():
    global _logf
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "excursion.log", "w")
    t0 = time.time()
    inp = LG.load_inputs()
    hilo = load_hilo()
    U = sorted(inp["p"]["trader"].unique())
    X = build_x(inp, hilo, U)
    log(f"prepared in {time.time() - t0:.1f}s: groups {len(X.ts):,}, episodes {X.nE:,} (left-censored "
        f"{int(X.E['left'].sum()):,}, still open at end {int((X.E['close_ns'] == NAT).sum()):,}), kline files {len(hilo)}")
    fc = fee_convention(inp["p"])
    log(f"fee convention: {fc}")
    log(f"flat-to-flat groups (no episode): {X.dropped_none}")
    wt = LG.window_table()
    tabs = run_windows(X, wt)
    for k, d in tabs.items():
        d.to_parquet(OUT / f"{k}.parquet", index=False)
        log(f"wrote {k}.parquet: {len(d):,} rows x {d.shape[1]} cols")
    sx = tabs["support_x"]
    ps = tabs["positions"]
    log("\n=== per L60 window ===")
    for M in MONDAYS:
        wid = f"L60_{M:%Y-%m-%d}"
        s = sx[sx["window_id"] == wid]
        log(f"{wid}: opened {int(s['n_ep_opened'].sum()):,}, scored {int(s['n_scored'].sum()):,}, no-bar "
            f"{int(s['n_nobar'].sum()):,}, unscored no file {int(s['n_unscored_no_file'].sum()):,} / missing bars "
            f"{int(s['n_unscored_missing_bars'].sum()):,}, complete {int(s['n_complete'].sum()):,} (win "
            f"{int(s['n_win'].sum()):,}, loss {int(s['n_loss'].sum()):,}, zero {int(s['n_zero'].sum()):,}), open at M "
            f"{int(s['n_open'].sum()):,} (known {int(s['n_open_known'].sum()):,})")
    runtime = time.time() - t0
    srcs = [LG.Path(__file__).resolve(), ROOT / "work/roll/code/ledger.py", ROOT / "work/v2dev/code/lib.py",
            ROOT / "work/profile/profile.py", ROOT / "work/profile2/code/exposure.py"]
    data = ROOT / "ArkStream笔试题_v2"
    files = [data / "笔试题数据包-fills", data / "笔试题数据包-traders.csv", LG.EXT / "funding.parquet",
             LG.EXT / "hl_candles_4h.parquet", LG.EXT / "symbol_map.csv"]
    man = dict(inputs={str(f.relative_to(ROOT)): LG.sha256(f) for f in files},
               klines_5m={f.name: LG.sha256(f) for f in sorted(KLINES.glob("*.parquet"))},
               sources={str(f.relative_to(ROOT)): LG.sha256(f) for f in srcs},
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             pyarrow=pyarrow.__version__, scipy=scipy.__version__),
               seed=SEED, run_utc=pd.Timestamp.now(tz="UTC").isoformat(), runtime_s=round(runtime, 1),
               rows={k: int(len(v)) for k, v in tabs.items()}, fee_convention=fc, flat_to_flat_groups=X.dropped_none,
               positions_per_window={w: int(n) for w, n in ps.groupby("window_id", observed=True).size().items()})
    (OUT / "manifest_x.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"\nruntime {runtime:.1f}s")


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def _diff(a, b, cols):
    out = {}
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x):
            x, y = x.to_numpy(float), y.to_numpy(float)
            same = (x == y) | (np.isnan(x) & np.isnan(y))
        else:
            same = (x.astype(object).to_numpy() == y.astype(object).to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
        out[c] = int((~same).sum())
    return out


def _norm(d, keys):
    d = d.astype({c: str for c in d.columns if isinstance(d[c].dtype, pd.CategoricalDtype)})
    return d.sort_values(keys, kind="stable").reset_index(drop=True)


def brute(gp, bars, s_hint, b):
    """Independent slow recomputation of one episode. gp: the pair's fill groups from the opening group on
    (ts < b), with gstart, gend, signed, vwap, cash, fee; bars: the symbol's kline frame (open_time, high, low,
    close). Running-average and fixed-entry excursions, dollar MAE and peak cumulative episode loss."""
    tsg = gp["ts"].to_numpy().astype("int64")
    gsl, gel, vwl = gp["gstart"].to_numpy(), gp["gend"].to_numpy(), gp["vwap"].to_numpy()
    sgl, csl, fel = gp["signed"].to_numpy(), gp["cash"].to_numpy(), gp["fee"].to_numpy()
    ot = bars["open_time"].to_numpy().astype("int64")
    hi_, lo_, cl_ = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    s = float(np.sign(gel[0]))
    assert s == s_hint
    avg, q, a0, pos = vwl[0], abs(gel[0]), vwl[0], gel[0]
    if gsl[0] != 0:                                   # opening flip: its |s1| / |signed| share
        w = abs(gel[0]) / abs(sgl[0])
        cash, fee = csl[0] * w, fel[0] * w
    else:
        cash, fee = csl[0], fel[0]
    m = dict(mae=0.0, mfe=0.0, mae_c=0.0, mfe_c=0.0, mae_usd=0.0, mae_open_c=0.0, mfe_open_c=0.0, loss_cum=0.0,
             bars_loss=0, bars_gain=0, bars_flat=0, d_exit=np.nan, exit_state=np.nan)
    n_bars = 0

    def obs(key_adv, key_fav, x_adv, x_fav, ref):
        m[key_adv] = max(m[key_adv], max(0.0, -s * (x_adv / ref - 1.0)))
        m[key_fav] = max(m[key_fav], max(0.0, s * (x_fav / ref - 1.0)))

    def close_obs(x):                                 # close-sampled observation (bar close or event vwap)
        obs("mae_c", "mfe_c", x, x, avg)
        obs("mae_open_c", "mfe_open_c", x, x, a0)
        m["loss_cum"] = max(m["loss_cum"], max(0.0, -((cash - fee) + pos * x)))

    def bars_in(t0, t1):
        nonlocal n_bars
        for j in np.flatnonzero((ot >= t0) & (ot + STEP <= t1)):     # selection only; per-bar loop
            n_bars += 1
            x_adv, x_fav = (lo_[j], hi_[j]) if s > 0 else (hi_[j], lo_[j])
            obs("mae", "mfe", x_adv, x_fav, avg)
            m["mae_usd"] = max(m["mae_usd"], max(0.0, -s * (x_adv - avg)) * q)
            close_obs(cl_[j])
            d = s * (cl_[j] - avg)                    # state of the bar against the segment's running average
            if d < 0:
                m["bars_loss"] += 1
            elif d > 0:
                m["bars_gain"] += 1
            else:
                m["bars_flat"] += 1

    cur_t, closed = tsg[0], False
    for k in range(1, len(tsg)):
        bars_in(cur_t, tsg[k])
        obs("mae", "mfe", vwl[k], vwl[k], avg)
        m["mae_usd"] = max(m["mae_usd"], max(0.0, -s * (vwl[k] - avg)) * q)
        close_obs(vwl[k])
        if gel[k] == 0 or np.sign(gel[k]) != s:
            closed = True
            m["d_exit"] = s * (vwl[k] - avg)
            m["exit_state"] = -1.0 if m["d_exit"] < 0 else (1.0 if m["d_exit"] > 0 else 0.0)
            break
        if abs(gel[k]) > abs(gsl[k]):
            avg = (avg * abs(gsl[k]) + vwl[k] * (abs(gel[k]) - abs(gsl[k]))) / abs(gel[k])
        q, pos = abs(gel[k]), gel[k]
        cash, fee = cash + csl[k], fee + fel[k]
        cur_t = tsg[k]
    if not closed:
        bars_in(cur_t, b)
    m["n_bars"] = n_bars
    return m


def checks():
    global _logf
    _logf = open(LOGS / "checks_x.log", "w")
    t0 = time.time()
    res = {}
    wt = LG.window_table()
    inp = LG.load_inputs()
    hilo = load_hilo()
    U = sorted(inp["p"]["trader"].unique())
    X = build_x(inp, hilo, U)
    led = pd.read_parquet(OUT / "ledger_x.parquet")
    sup = pd.read_parquet(OUT / "support_x.parquet")
    pos = pd.read_parquet(OUT / "positions.parquet")
    opp = pd.read_parquet(OUT / "open_positions.parquet")

    # ---------------- check 1: walk == build_episodes; complete sets == ledger episodes.parquet ----------------
    log("=== check 1: walk vs profile.build_episodes (ref_full) and complete episodes vs episodes.parquet ===")
    ep, _, _ = PROF.build_episodes(X.P.g[["pid", "ts", "gstart", "gend", "signed", "notional", "vwap"]])
    E = X.E
    mine = pd.DataFrame({"pid": E["pid"], "open_ns": E["open_ns"], "close_ns": E["close_ns"], "sign": E["sign"],
                         "open_ntl": E["open_ntl"], "later_add_ntl": np.where(E["left"], 0.0, E["later"]),
                         "later_loser_ntl": np.where(E["left"], 0.0, E["loser"]), "left": E["left"],
                         "end_by_flip": E["flip_end"]})
    theirs = pd.DataFrame({"pid": ep["pid"], "open_ns": ep["open_ts"].to_numpy().astype("int64"),
                           "close_ns": ep["close_ts"].to_numpy().astype("int64"), "sign": ep["sign"],
                           "open_ntl": ep["open_ntl"], "later_add_ntl": ep["later_add_ntl"],
                           "later_loser_ntl": ep["later_loser_ntl"], "left": ep["left_cens"],
                           "end_by_flip": ep["end_by_flip"]})
    k = ["pid", "open_ns"]
    mine, theirs = mine.sort_values(k).reset_index(drop=True), theirs.sort_values(k).reset_index(drop=True)
    c1 = {"episodes_walk": len(mine), "episodes_builder": len(theirs)}
    if len(mine) == len(theirs):
        c1["cells_differing"] = _diff(mine, theirs, list(mine.columns))
    epq = pd.read_parquet(OUT / "episodes.parquet")
    per_w = {}
    for r in wt.itertuples(index=False):
        a_ = pos[(pos["window_id"] == r.window_id) & pos["complete"]][["trader", "symbol", "open_ts", "end_ts", "dur_h"]] \
            .rename(columns={"end_ts": "close_ts"})
        b_ = epq[epq["window_id"] == r.window_id][["trader", "symbol", "open_ts", "close_ts", "dur_h"]]
        kk = ["trader", "symbol", "open_ts"]
        a_, b_ = _norm(a_, kk), _norm(b_, kk)
        per_w[r.window_id] = (len(a_), len(b_), sum(_diff(a_, b_, list(a_.columns)).values()) if len(a_) == len(b_) else None)
    bad = {w: v for w, v in per_w.items() if v[0] != v[1] or v[2] != 0}
    c1["windows_compared"] = len(per_w)
    c1["windows_with_differences"] = bad
    c1["complete_episodes_total"] = int(sum(v[0] for v in per_w.values()))
    log(json.dumps(c1, indent=1, default=str))
    res["check1_walk"] = c1

    # ---------------- check 2: prefix equality ----------------
    log("\n=== check 2: prefix equality (L60, R28 at C = M; inputs and high/low bars truncated at C) ===")
    c2 = {}
    for M in (pd.Timestamp("2025-10-06"), pd.Timestamp("2025-11-24")):
        ms = f"{M:%Y-%m-%d}"
        tin, removed = LG.truncate_inputs(inp, M)
        th = truncate_hilo(hilo, M)
        Xt = build_x(tin, th, U)
        sw = wt[wt["window_id"].isin([f"L60_{ms}", f"R28_{ms}"])].reset_index(drop=True)
        tt = run_windows(Xt, sw, verbose=False)
        cm = {"removed": dict(removed, hilo_bars=sum(len(hilo[s]) - len(th[s]) for s in hilo))}
        for wid in sw["window_id"]:
            a_ = _norm(led[led["window_id"] == wid], ["trader", "column"])
            b_ = _norm(tt["ledger_x"][tt["ledger_x"]["window_id"] == wid], ["trader", "column"])
            d_led = _diff(a_, b_, ["trader", "column", "value"])
            a_ = _norm(sup[sup["window_id"] == wid], ["trader"])
            b_ = _norm(tt["support_x"][tt["support_x"]["window_id"] == wid], ["trader"])
            d_sup = _diff(a_, b_, SUPPORT_X)
            kk = ["trader", "symbol", "open_ts"]
            a_ = _norm(pos[pos["window_id"] == wid], kk)
            b_ = _norm(tt["positions"][tt["positions"]["window_id"] == wid], kk)
            d_pos = (sum(_diff(a_, b_, [c for c in a_.columns if c != "window_id"]).values())
                     if len(a_) == len(b_) else f"rows {len(a_)} vs {len(b_)}")
            cm[wid] = dict(ledger_cells=int((led["window_id"] == wid).sum()), ledger_differ=sum(d_led.values()),
                           support_cells=int((sup["window_id"] == wid).sum()) * len(SUPPORT_X),
                           support_differ=sum(d_sup.values()), positions_rows=len(a_), positions_cells_differ=d_pos)
        a_ = _norm(opp[opp["cutoff"] == M], ["trader", "symbol", "kind"])
        b_ = _norm(tt["open_positions"], ["trader", "symbol", "kind"])
        cm["open_positions"] = dict(rows=len(a_), cells_differ=(sum(_diff(a_, b_, list(a_.columns)).values())
                                                               if len(a_) == len(b_) else f"rows {len(a_)} vs {len(b_)}"))
        log(f"{ms}: {cm}")
        c2[ms] = cm
    res["check2_prefix"] = c2

    # ---------------- check 3: brute force ----------------
    log("\n=== check 3: brute-force recomputation of 300 path-class episodes ===")
    sc = pos[pos["path"]].copy()
    sc["wb"] = sc["window_id"].map(dict(zip(wt["window_id"], wt["b"])))
    sc = sc.sort_values(["window_id", "trader", "symbol", "open_ts"], kind="stable").reset_index(drop=True)
    strata = {"adds": sc["n_adds"] > 0, "reductions_before_close": (sc["n_events"] > 2) & (sc["n_adds"] == 0),
              "other": (sc["n_events"] <= 2) & (sc["n_adds"] == 0)}
    rng = np.random.default_rng(SEED)
    g = lib.fill_groups(inp["p"])
    g_by = {kk: d.sort_values("ts") for kk, d in g.groupby(["trader", "symbol"])}
    fields = (("mae", "mae_px"), ("mfe", "mfe_px"), ("mae_c", "mae_px_c"), ("mfe_c", "mfe_px_c"), ("mae_usd", "mae_usd"),
              ("mae_open_c", "mae_open_c"), ("mfe_open_c", "mfe_open_c"), ("loss_cum", "loss_cum"),
              ("mae_rt_c", "mae_rt_c"), ("bars_loss", "bars_loss"), ("bars_gain", "bars_gain"),
              ("bars_flat", "bars_flat"), ("d_exit", "d_exit"), ("exit_state", "exit_state"))
    c3 = {}
    worst = {f: 0.0 for _, f in fields}
    for name, m in strata.items():
        pick = rng.choice(np.flatnonzero(m.to_numpy()), size=100, replace=False)
        n_bad, n_exact = 0, 0
        for i in pick:
            r = sc.iloc[i]
            b = int(r["wb"].value)
            gp = g_by[(r["trader"], r["symbol"])]
            gp = gp[(gp["ts"] >= r["open_ts"]) & (gp["ts"] < pd.Timestamp(b))]
            bf = brute(gp, hilo[r["symbol"]], r["sign"], b)
            sg_, du_ = r["sigma_1d"], r["dur_h"]
            ok_rt = np.isfinite(sg_) and sg_ > 0 and du_ >= 1 and bf["n_bars"] >= 1
            bf["mae_rt_c"] = bf["mae_c"] / (sg_ * np.sqrt(du_ / 24.0)) if ok_rt else np.nan
            ok_all, ex_all = bf["n_bars"] == r["n_bars"], True
            for kk, col in fields:
                x, y = r[col], bf[kk]
                if np.isnan(x) and np.isnan(y):
                    continue
                d = abs(x - y)
                rel = d / max(abs(y), 1e-300) if d else 0.0
                worst[col] = max(worst[col], rel if np.isfinite(rel) else np.inf)
                ex_all &= d == 0
                ok_all &= d <= 1e-12 * max(abs(y), 1e-300)
            n_bad += not ok_all
            n_exact += ex_all
        c3[name] = dict(sampled=100, all_fields_exact=int(n_exact), within_1e12=int(100 - n_bad), failures=int(n_bad))
    c3["max_rel_diff_by_field"] = worst
    log(json.dumps(c3, indent=1))
    res["check3_brute"] = c3

    # ---------------- check 4: sanity ----------------
    log("\n=== check 4: sanity ===")
    s = pos[pos["scored"]]
    c4 = dict(mae_px_ge_close_variant_violations=int((s["mae_px"] < s["mae_px_c"]).sum()),
              mfe_px_ge_close_variant_violations=int((s["mfe_px"] < s["mfe_px_c"]).sum()), scored_rows=len(s))
    # complete, no adds, realised loss L > 0: every exit leg loses at most mae_px * avg_open per coin and the exits
    # of a no-add episode sum to q_open coins, so  L <= mae_px * avg_open * q_open  (tested with 1e-9 rel + 1e-9 abs).
    # Tested with L = -cash_e (our own exact cash flow, episodes touching no flip) and L = -closed_pnl_e (exchange)
    sd = s[s["complete"]].drop_duplicates(["trader", "symbol", "open_ts"])     # complete rows first, then dedupe
    noflip = ~sd["open_by_flip"] & ~sd["end_by_flip"]
    for name, col, extra in (("loss_bound_cash_noflip", "cash_e", noflip), ("loss_bound_closed_pnl_all", "closed_pnl_e", True),
                             ("loss_bound_closed_pnl_noflip", "closed_pnl_e", noflip)):
        t = sd[sd["complete"] & (sd["n_adds"] == 0) & (sd[col] < 0) & extra]
        bound = t["mae_px"] * t["avg_open"] * t["q_open"]
        exc = -t[col] - bound
        viol = exc > bound * 1e-9 + 1e-9
        c4[name] = dict(tested=len(t), violations=int(viol.sum()),
                        max_excess_usd=float(exc[viol].max()) if viol.any() else 0.0,
                        max_excess_over_open_ntl=float((exc / t["open_ntl"])[viol].max()) if viol.any() else 0.0)
    # loss_cum: the last observation of a complete episode is its closing event at its vwap before it is applied,
    # value = cash_to_(k-1) - fee_to_(k-1) + pos_(k-1) * vwap_close; since the closing leg's cash share equals
    # pos_(k-1) * vwap_close up to rounding, value = cash_e - (fee_e - fee_close). Tested:
    #   loss_cum >= max(0, -(cash_e - (fee_e - fee_close))) - (1e-9 * open_ntl + 1e-9)
    # and, for information, how often loss_cum < max(0, -(cash_e - fee_e)) (the closing fee is never inside loss_cum).
    cs = sd
    tgt = np.maximum(0.0, -(cs["cash_e"] - (cs["fee_e"] - cs["fee_close"])))
    short = tgt - cs["loss_cum"]
    lb = short > 1e-9 * cs["open_ntl"] + 1e-9
    tgt2 = np.maximum(0.0, -(cs["cash_e"] - cs["fee_e"]))
    c4["loss_cum_vs_final_before_close"] = dict(
        tested=len(cs), violations=int(lb.sum()), max_shortfall_usd=float(short.max()),
        max_shortfall_over_open_ntl=float((short / cs["open_ntl"]).max()),
        n_target_positive=int((tgt > 0).sum()),
        n_loss_cum_below_net_result_incl_close_fee=int((cs["loss_cum"] < tgt2 - 1e-9 * cs["open_ntl"] - 1e-9).sum()))
    lx = led.pivot_table(index=["window_id", "trader"], columns="column", values="value", aggfunc="first", observed=True)
    for c in ("open_loss_share_n", "open_loss_share_ntl"):
        v = lx[c].dropna()
        c4[f"{c}_range"] = (float(v.min()), float(v.max()), int(len(v)))
    w1 = opp[(opp["cutoff"] == pd.Timestamp(W1N)) & (opp["kind"] == "package")]
    ep_full = ep
    c4["open_at_W1_vs_right_censored_ref_full"] = dict(
        package_open=len(w1), known=int(w1["known"].sum()),
        builder_right_censored=int(ep_full["right_cens"].sum()),
        builder_right_censored_not_left=int((ep_full["right_cens"] & ~ep_full["left_cens"]).sum()),
        ledger_support_n_right_cens_fill=int(pd.read_parquet(OUT / "support.parquet", columns=["window_id", "n_right_cens_fill"])
                                             .query("window_id == 'ref_full'")["n_right_cens_fill"].sum()),
        funding_only_open=int((opp["cutoff"] == pd.Timestamp(W1N)).sum() - len(w1)))
    # realised: closed_pnl_e vs episode cash flow (sum of -signed*price, flip groups split by size)
    # (all complete episodes of all windows, deduplicated, rather than a sample)
    cf = pos[pos["complete"]].drop_duplicates(["trader", "symbol", "open_ts"])
    c4["closed_pnl_vs_cash"] = {"total_closed_pnl": float(cf["closed_pnl_e"].sum()),
                                "total_cash": float(cf["cash_e"].sum()), "total_fee": float(cf["fee_e"].sum())}
    for name, m in (("no_flip", ~cf["open_by_flip"] & ~cf["end_by_flip"]), ("touches_flip", cf["open_by_flip"] | cf["end_by_flip"])):
        x = cf[m]
        d = (x["closed_pnl_e"] - x["cash_e"]).abs()
        r = d / x["open_ntl"]
        c4["closed_pnl_vs_cash"][name] = dict(
            episodes=len(x), n_abs_le_1em6=int((d <= 1e-6).sum()), n_abs_gt_1em6=int((d > 1e-6).sum()),
            max_abs=float(d.max()), sum_signed=float((x["closed_pnl_e"] - x["cash_e"]).sum()),
            over_open_ntl_p50=float(r.median()), over_open_ntl_p99=float(r.quantile(.99)), over_open_ntl_max=float(r.max()))
    log(json.dumps(c4, indent=1, default=str))
    res["check4_sanity"] = c4

    # ---------------- check 6: exit-rate identities ----------------
    log("\n=== check 6: time-in-state / exit identities ===")
    pp = pos[pos["path"]]
    c6 = {"path_rows": len(pp),
          "bars_sum_ne_n_bars": int((pp["bars_loss"] + pp["bars_gain"] + pp["bars_flat"] != pp["n_bars"]).sum()),
          "nonpath_rows_with_bars_fields": int(pos.loc[~pos["path"], ["bars_loss", "bars_gain", "bars_flat"]].notna().any(axis=1).sum()),
          "exit_state_outside_complete_path": int(pos.loc[~(pos["path"] & pos["complete"]), "exit_state"].notna().sum()),
          "complete_path_without_exit_state": int(pp.loc[pp["complete"], "exit_state"].isna().sum())}
    agg = pp.groupby(["window_id", "trader"], observed=True).agg(
        bl=("bars_loss", "sum"), bg=("bars_gain", "sum"), bf=("bars_flat", "sum"),
        el=("exit_state", lambda x: int((x == -1).sum())), eg=("exit_state", lambda x: int((x == 1).sum())),
        ef=("exit_state", lambda x: int((x == 0).sum())), nc=("complete", "sum"))
    sx = sup.set_index(["window_id", "trader"]).reindex(agg.index)
    c6["address_windows_with_path"] = len(agg)
    c6["support_hours_ne_position_sums"] = int(((sx["t_loss_h"] != agg["bl"] / 12.0) | (sx["t_gain_h"] != agg["bg"] / 12.0)
                                                | (sx["t_flat_h"] != agg["bf"] / 12.0)).sum())
    c6["support_exits_ne_position_counts"] = int(((sx["n_exit_loss"] != agg["el"]) | (sx["n_exit_gain"] != agg["eg"])
                                                  | (sx["n_exit_flat"] != agg["ef"])).sum())
    c6["exits_total_ne_complete_path"] = int((sx["n_exit_loss"] + sx["n_exit_gain"] + sx["n_exit_flat"] != agg["nc"]).sum())
    rest = sup.set_index(["window_id", "trader"]).drop(index=agg.index)
    c6["no_path_address_windows_with_nonzero_support"] = int((rest[["t_loss_h", "t_gain_h", "t_flat_h", "n_exit_loss",
                                                                    "n_exit_gain", "n_exit_flat"]] != 0).any(axis=1).sum())
    log(json.dumps(c6, indent=1))
    res["check6_exit_identities"] = c6

    # ---------------- check 5: coverage ----------------
    log("\n=== check 5: coverage per L60 window (base-eligible addresses from ledger support.parquet) ===")
    lsup = pd.read_parquet(OUT / "support.parquet", columns=["window_id", "trader", "base_eligible"])
    rows = []
    for M in MONDAYS:
        wid = f"L60_{M:%Y-%m-%d}"
        be = lsup[(lsup["window_id"] == wid) & lsup["base_eligible"]]["trader"]
        x = sup[(sup["window_id"] == wid)].set_index("trader").reindex(be)
        rows.append(dict(M=f"{M:%Y-%m-%d}", base=len(be),
                         px_n10_d3=int(((x["n_px"] >= 10) & (x["d_px"] >= 3)).sum()), px_n30=int((x["n_px"] >= 30).sum()),
                         rt_n10_d3=int(((x["n_rt"] >= 10) & (x["d_rt"] >= 3)).sum()),
                         av_n10_d3=int(((x["n_av"] >= 10) & (x["d_av"] >= 3)).sum()),
                         loss5=int((x["n_loss"] >= 5).sum()), win5=int((x["n_win"] >= 5).sum()),
                         loss5_and_win5=int(((x["n_loss"] >= 5) & (x["n_win"] >= 5)).sum()),
                         open_known3=int((x["n_open_known"] >= 3).sum()),
                         median_path_share=float(x["path_share"].median())))
    c5 = pd.DataFrame(rows)
    log(c5.to_string(index=False))
    res["check5_coverage"] = c5.to_dict(orient="records")
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "checks_x.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"\nchecks done in {res['runtime_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    checks() if ap.parse_args().checks else build()
