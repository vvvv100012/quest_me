"""Return-performance dimensions of the rolling address profile (ret_sum, ret_mean_sd, max_dd, gain_share), for the
same 80 windows as ledger.py, each S = [a, b) computed at its own cutoff C = b with only the information available
at C (the availability rules of ledger.truncate_inputs). All four come from one daily series per address and cutoff.

  1. Daily net PnL (USD) of an address for the UTC day D = [d, d + 1d), d + 1d <= C:
        pnl_d = sum over symbols of mtm_s(D) - fees(D) + funding(D)
        mtm_s(D) = q_end * P(d + 1d) - q_start * P(d) + cash_s(D) - entries_s(D)
     cash_s(D) = sum of -signed * price of the symbol's perp fills in D; q_start / q_end = position at the
     boundaries in the representation of ledger.grid_windows at C: the package replay (gend of the last fill group
     with ts < boundary, incl. the carried-in start gstart0 held from carry_ns to the first fill) if the pair has a
     fill before C, else the funding-only daily szi (the day's last szi, constant over that day) if the pair has
     funding days before C. Positions entering the books without a fill carry no PnL at entry (entries_s):
       (i)   a carried-in start enters at the valuation price P(carry_ns);
       (ii)  gstart0 != 0 at the first fill without a carry enters at the first group's vwap;
       (iii) a funding-only position is valued within each day only: szi_D * (P(d + 1d) - P(d)), no mtm term;
       (iv)  a group whose gstart differs from the previous group's gend by more than lib.POS_EPS: the jump enters
             at that group's vwap.
     P(t) at a midnight boundary t = the exposure-grid price at CLOSE == t (Binance 5m bar closing at t > latest HL
     4h candle closed by t > last package fill at or before t). No package fill is stamped exactly 00:00
     (asserted), so "at or before t" equals "before t" and P(C) is causal. The boundary W0 has no grid point:
     its price is missing. fees(D) = sum of `fee` of the address's fills in D; every fee token is taken as USD
     (1,705 of the 864,623 perp fills pay in USDH / USDE, all on HIP-3 symbols). funding(D) = sum of `usdc` (signed,
     negative = paid) of the address's funding records of D over all coins: hourly records by their time, daily
     aggregates by their stamped day (both available at C for every D with d + 1d <= C). HIP-3 symbols are
     included with whatever prices exist (addresses with HIP-3 evidence are gated out in measures.py).
  2. Daily return r_d = pnl_d / av_d, av_d = account-value proxy at the day's start d: max of the two bracketing
     snapshots' acct_val (s_prev <= d < s_next, snapshots < C only) if they are at most 168 h apart (exposure.av_grid
     uses 48 h; the package-wide snapshot gap 2025-11-23 .. 11-27 needs the wider bracket), at an edge the single
     snapshot within 24 h. wide_av = both brackets used and more than 48 h apart.
     Day classes (per address, per cutoff): span = days d >= the day containing the address's first appearance as
     of C (as in grid_windows: min over symbols of carry start / first fill / first funding-only day); active = a
     span day with a nonzero position at either boundary, a fill or a funding record. An inactive span day has
     pnl = r = 0 and is valid. An active day is valid iff every price it needs is finite and av_d is finite and
     >= ledger.AV_MIN; otherwise it is dropped with ONE reason, in this priority: px_bad (a needed price missing),
     noav (av missing), lt100 (av < 100).
  3. Per address-window, over the valid span days of [a, b) in day order:
       ret_sum      sum of r_d (no compounding)
       ret_mean_sd  mean(r_d) / std(r_d, ddof=1); NaN if fewer than 2 valid days or all valid r_d equal (std == 0)
       max_dd       max of (running max of the cumulative sum, the curve starting at 0) - cumulative sum; >= 0
       gain_share   G / (G + L), G = sum of positive pnl_d, L = sum of |negative pnl_d| (USD); NaN if G + L == 0
     All four are NaN without a valid day.

Reconciliation diagnostic (recon.parquet; full information as of W1, not a gate): for consecutive snapshots of an
address <= 48 h apart with identical net_deposit, t1 >= W0 and no HIP-3 evidence as of the later snapshot,
resid = d_av - replay PnL over the interval. Replay PnL = position * price at the grid closes c1, c2 at or before
each snapshot time (positions and prices as above, entries (i), (ii), (iv) at their entry price, funding-only
positions valued within each day), plus the cash of the fills in (c1, c2], minus their fees, plus funding
(hourly records with time in (t1, t2]; daily aggregates prorated by the overlap of (t1, t2] with their day -- an
approximation).

Reuses ledger.py (imported, not modified): load_inputs, truncate_inputs, build_prep, window_table, constants;
lib.fill_groups (cash, fee, closed_pnl of the groups).

Inputs (read-only): as ledger.py; lib.load_traders() (net_deposit, reconciliation only); work/roll/build/
support.parquet (base-eligible counts in the log; checks). Outputs (work/roll/build): ledger_r.parquet,
support_r.parquet, daily_pnl.parquet (the daily table as of the final cutoff W1, all span days -- reference and
inspection only; every window uses its own cutoff), recon.parquet, manifest_r.json, checks_r.json; logs
work/roll/logs/{returns, checks_r}.log.
Usage: python returns.py (build), python returns.py --checks [--before DIR] (DIR: pre-change overview.csv,
len_overview.csv and an optional `shasum -a 256` listing hashes.txt of work/roll/build, for the regression check).
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ next to ledger.py or the reviewed scripts
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/roll/code")
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402

import ledger as LG  # noqa: E402

lib, EXPO = LG.lib, LG.EXPO
ROOT, OUT, LOGS = LG.ROOT, LG.OUT, LG.LOGS
W0N, W1N, STEP, DAY, HOUR, PPD, N = LG.W0N, LG.W1N, LG.STEP, LG.DAY, LG.HOUR, LG.PPD, LG.N
AV_MIN, INT_MAX, MONDAYS = LG.AV_MIN, LG.INT_MAX, LG.MONDAYS
ND = (W1N - W0N) // DAY                 # days W0 .. W1 - 1d; boundaries k = 0 .. ND at W0 + k days
SEED = 20261120
AV_BRACKET, AV_NARROW, AV_EDGE = 168 * HOUR, 48 * HOUR, 24 * HOUR
RECON_GAP = 48 * HOUR
RCOLS = ["ret_sum", "ret_mean_sd", "max_dd", "gain_share"]
SUPPORT_R = ["hip3_asof", "n_span_days", "n_ret_days", "n_act_days", "n_act_noav", "n_act_lt100", "n_px_bad_days",
             "n_wide_av_days", "n_gain_days", "n_loss_days", "gain_usd", "loss_usd", "pnl_usd", "mtm_usd", "fee_usd",
             "funding_usd", "ret_sd"]
REASON = {-1: "outside_span", 0: "", 1: "px_bad", 2: "noav", 3: "lt100"}    # 0 = valid (span days)
GAP_DAYS = pd.date_range("2025-11-23", "2025-11-27", freq="D")
EXISTING = ["ledger", "support", "top_symbol", "decisions", "episodes", "clusters", "address_sums", "ledger_x",
            "support_x", "positions", "open_positions"]

_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


# ---------------------------------------------------------------------------------------
# Account value at day starts
# ---------------------------------------------------------------------------------------
def av_days(sn_ts, sn_av, cutoff_ns, q_ns):
    """exposure.av_grid with a 168 h bracket: av(q) = max(av(s_prev), av(s_next)), s_prev <= q < s_next, if
    s_next - s_prev <= 168 h; at an edge the one snapshot within 24 h of q. Only snapshots before cutoff_ns.
    Returns (av, wide) with wide = both brackets used and more than 48 h apart."""
    keep = sn_ts < cutoff_ns
    S, V = sn_ts[keep], sn_av[keep]
    out = np.full(len(q_ns), np.nan)
    wide = np.zeros(len(q_ns), dtype=bool)
    if len(S) == 0:
        return out, wide
    j = np.searchsorted(S, q_ns, side="right") - 1
    hp, hn = j >= 0, j + 1 < len(S)
    jp, jn = np.clip(j, 0, len(S) - 1), np.clip(j + 1, 0, len(S) - 1)
    gap = S[jn] - S[jp]
    both = hp & hn & (gap <= AV_BRACKET)
    out[both] = np.maximum(V[jp], V[jn])[both]
    wide = both & (gap > AV_NARROW)
    op = hp & ~hn & (q_ns - S[jp] <= AV_EDGE)
    out[op] = V[jp][op]
    on = ~hp & hn & (S[jn] - q_ns <= AV_EDGE)
    out[on] = V[jn][on]
    return out, wide


# ---------------------------------------------------------------------------------------
# Window-independent preparation
# ---------------------------------------------------------------------------------------
def _qv(q, p):
    """q * p where q != 0 (0 where q == 0, also for a missing p); bad = q != 0 with a missing price."""
    bad = (q != 0) & ~np.isfinite(p)
    with np.errstate(invalid="ignore"):
        v = np.where((q != 0) & ~bad, q * p, 0.0)
    return v, bad


def build_r(inp, U, P=None):
    """Ledger prep (or the given one, built by ledger.build_prep from the same inp) plus the per-pair daily
    mark-to-market on all days W0 .. W1 - 1d and the per-address daily fees, funding and counts. Everything here is
    cutoff-independent for the days a cutoff uses (d + 1d <= C); the cutoff enters only in daily_at."""
    R = SimpleNamespace()
    R.P = P = LG.build_prep(inp, U) if P is None else P
    nU = P.nU
    if len(P.p_ts):
        assert P.p_ts.min() >= W0N and np.all(P.p_ts % DAY != 0), "fill before W0 or stamped exactly 00:00"
    g = lib.fill_groups(inp["p"]).sort_values(["trader", "symbol", "ts"], kind="stable").reset_index(drop=True)
    assert len(g) == len(P.g)
    for c in ("gstart", "gend", "signed", "notional", "vwap"):
        assert np.array_equal(g[c].to_numpy(), P.g[c].to_numpy(), equal_nan=True), c
    assert np.array_equal(g["ts"].to_numpy().astype("int64"), P.g_ts)
    pid = P.g["pid"].to_numpy()
    ts = P.g_ts
    gs, ge, vw = P.g["gstart"].to_numpy(), P.g["gend"].to_numpy(), P.g["vwap"].to_numpy()
    R.cash_g, R.fee_g, R.cpnl_g = g["cash"].to_numpy(), g["fee"].to_numpy(), g["closed_pnl"].to_numpy()
    R.g_pid, R.g_ts, R.g_gs, R.g_ge, R.g_vw = pid, ts, gs, ge, vw
    R.g_ntl = P.g["notional"].to_numpy()
    pr = P.pairs
    npair = len(pr)
    assert np.array_equal(pr["pid"].to_numpy(), np.arange(npair))
    r0, r1 = pr["r0"].to_numpy(), pr["r1"].to_numpy()
    si, pti = pr["sym_i"].to_numpy(), pr["ti"].to_numpy()
    gstart0 = pr["gstart0"].to_numpy()
    has_c = pr["carry_ns"].notna().to_numpy()
    cns = np.array([int(x) if h else -1 for x, h in zip(pr["carry_ns"].to_numpy(), has_c)], dtype=np.int64)
    R.pair_ti, R.pair_si, R.r0, R.r1 = pti, si, r0, r1
    R.first_ts = pr["first_ts"].to_numpy()
    R.gstart0, R.has_c, R.cns = gstart0, has_c, cns
    R.pair_start = np.where(has_c, cns, R.first_ts)

    # boundary prices PB[s, k] at W0 + k days (exposure grid close; k = 0 has no grid point)
    S = len(P.symbols)
    PB = np.full((S, ND + 1), np.nan)
    PB[:, 1:] = P.px[:, np.arange(1, ND + 1) * PPD - 1]
    R.PB = PB

    # boundary positions Q[pid, k]: gend of the last group with ts < W0 + k days; carried start before it
    gd = (ts - W0N) // DAY
    last = np.r_[(pid[1:] != pid[:-1]) | (gd[1:] != gd[:-1]), True]
    H = np.full((npair, ND + 1), -1, dtype=np.int64)
    H[pid[last], gd[last] + 1] = np.flatnonzero(last)
    H = np.maximum.accumulate(H, axis=1)
    Q = np.where(H >= 0, ge[np.maximum(H, 0)], 0.0)
    ck = np.where(has_c, (cns - W0N) // DAY, -1)
    for p in np.flatnonzero(has_c):
        Q[p, ck[p] + 1:gd[r0[p]] + 1] = gstart0[p]          # boundaries after carry_ns up to the first fill day
    R.Q = Q

    # cash and entries per pair-day
    cash = np.bincount(pid * ND + gd, weights=R.cash_g, minlength=npair * ND).reshape(npair, ND)
    E = np.zeros((npair, ND))
    Ebad = np.zeros((npair, ND), dtype=bool)
    pc = np.where(has_c, PB[si, np.maximum(ck, 0)], np.nan)
    cb = has_c & ~np.isfinite(pc)
    Ebad[np.flatnonzero(cb), ck[cb]] = True
    ok_c = has_c & ~cb
    np.add.at(E, (np.flatnonzero(ok_c), ck[ok_c]), gstart0[ok_c] * pc[ok_c])                  # (i)
    m2 = (gstart0 != 0) & ~has_c
    np.add.at(E, (np.flatnonzero(m2), gd[r0[m2]]), gstart0[m2] * vw[r0[m2]])                  # (ii)
    j = np.flatnonzero(pid[1:] == pid[:-1]) + 1
    jump = gs[j] - ge[j - 1]
    mj = np.abs(jump) > lib.POS_EPS
    jj = j[mj]
    np.add.at(E, (pid[jj], gd[jj]), jump[mj] * vw[jj])                                        # (iv)
    R.entries = dict(
        carry_i=dict(pairs=int(has_c.sum()), price_missing=int(cb.sum()),
                     notional=float(np.abs(gstart0[ok_c] * pc[ok_c]).sum())),
        first_fill_ii=dict(pairs=int(m2.sum()), notional=float(np.abs(gstart0[m2] * vw[r0[m2]]).sum())),
        jump_iv=dict(groups=int(mj.sum()), pairs=int(len(np.unique(pid[jj]))),
                     notional=float(np.abs(jump[mj] * vw[jj]).sum()),
                     max_abs_jump_coins=float(np.abs(jump[mj]).max()) if mj.any() else 0.0))
    R.jump_rows = jj
    R.E = E

    PBp = PB[si]
    v1, b1 = _qv(Q[:, 1:], PBp[:, 1:])
    v0, b0 = _qv(Q[:, :-1], PBp[:, :-1])
    R.mtm_pair = v1 - v0 + cash - E
    R.bad_pair = b1 | b0 | Ebad
    R.act_pair = (Q[:, 1:] != 0) | (Q[:, :-1] != 0)
    R.cash_pair = cash

    # funding-only pair-days (ledger.build_prep fday): szi_D * (P(d + 1d) - P(d))
    fd = P.fday
    R.fo_ti = fd["ti"].to_numpy()
    R.fo_day = fd["day_ns"].to_numpy().astype(np.int64)
    R.fo_D = (R.fo_day - W0N) // DAY
    R.fo_first = fd["pkg_first_ns"].to_numpy().astype(np.int64)
    fs, fz = fd["sym_i"].to_numpy(), fd["szi"].to_numpy()
    with np.errstate(invalid="ignore"):
        fm = fz * (PB[fs, R.fo_D + 1] - PB[fs, R.fo_D])
    R.fo_bad = ~np.isfinite(fm)
    R.fo_mtm = np.where(R.fo_bad, 0.0, fm)

    # per address-day: fees and fill counts, funding and record counts
    tmap = {t: i for i, t in enumerate(P.U)}
    p = inp["p"]
    fti = p["trader"].map(tmap).to_numpy().astype(np.int64)
    fD = (p["ts"].to_numpy().astype("int64") - W0N) // DAY
    R.FEE = np.bincount(fti * ND + fD, weights=p["fee"].to_numpy(), minlength=nU * ND).reshape(nU, ND)
    R.NFILL = np.bincount(fti * ND + fD, minlength=nU * ND).reshape(nU, ND)
    fu = inp["fund"][inp["fund"]["trader"].isin(tmap)]
    dn = fu["time"].dt.floor("D").to_numpy().astype("int64")
    inr = (dn >= W0N) & (dn < W1N)
    uti = fu["trader"].map(tmap).to_numpy().astype(np.int64)[inr]
    uD = (dn[inr] - W0N) // DAY
    R.FUND = np.bincount(uti * ND + uD, weights=fu["usdc"].to_numpy()[inr], minlength=nU * ND).reshape(nU, ND)
    R.NFUND = np.bincount(uti * ND + uD, minlength=nU * ND).reshape(nU, ND)
    R.fund_pre_w0 = dict(records=int((dn < W0N).sum()), usdc=float(fu["usdc"].to_numpy()[dn < W0N].sum()))
    R.hip3_lt, R.hip3_le = P.h3_lt, P.h3_le
    return R


# ---------------------------------------------------------------------------------------
# Daily series at one cutoff
# ---------------------------------------------------------------------------------------
def daily_at(R, C):
    """Daily table of every address at cutoff C on days 0 .. nC - 1 (d + 1d <= C): dict of (nU, nC) arrays."""
    P = R.P
    nU = P.nU
    nC = (C - W0N) // DAY
    assert (C - W0N) % DAY == 0 and 0 < nC <= ND
    inc = np.flatnonzero(R.first_ts < C)                 # package pairs: a fill before C
    mtm = np.zeros((nU, nC))
    nbad = np.zeros((nU, nC), dtype=np.int64)
    nact = np.zeros((nU, nC), dtype=np.int64)
    pti = R.pair_ti[inc]
    np.add.at(mtm, pti, R.mtm_pair[inc, :nC])
    np.add.at(nbad, pti, R.bad_pair[inc, :nC].astype(np.int64))
    np.add.at(nact, pti, R.act_pair[inc, :nC].astype(np.int64))
    k = (R.fo_first >= C) & (R.fo_day < C)               # funding-only pair-days: no fill before C
    np.add.at(mtm, (R.fo_ti[k], R.fo_D[k]), R.fo_mtm[k])
    np.add.at(nbad, (R.fo_ti[k], R.fo_D[k]), R.fo_bad[k].astype(np.int64))
    np.add.at(nact, (R.fo_ti[k], R.fo_D[k]), 1)
    start = np.full(nU, INT_MAX, dtype=np.int64)
    np.minimum.at(start, pti, R.pair_start[inc])
    np.minimum.at(start, R.fo_ti[k], R.fo_day[k])
    has = start < INT_MAX
    assert np.all(start[has] >= W0N)
    s0 = np.where(has, (np.where(has, start, W0N) - W0N) // DAY, nC)
    span = np.arange(nC)[None, :] >= s0[:, None]
    fee, fund = R.FEE[:, :nC], R.FUND[:, :nC]
    nfill, nfund = R.NFILL[:, :nC], R.NFUND[:, :nC]
    assert not ((nact > 0) | (nfill > 0))[~span].any(), "position or fill before the first appearance"
    act = span & ((nact > 0) | (nfill > 0) | (nfund > 0))
    inact = span & ~act
    assert np.all(mtm[inact] == 0) and np.all(fee[inact] == 0) and np.all(fund[inact] == 0)
    av = np.full((nU, nC), np.nan)
    wide = np.zeros((nU, nC), dtype=bool)
    q = W0N + np.arange(nC, dtype=np.int64) * DAY
    empty = (np.array([], dtype=np.int64), np.array([]))
    for ti in np.flatnonzero(has):
        sn_ts, sn_av = P.snaps.get(ti, empty)
        av[ti], wide[ti] = av_days(sn_ts, sn_av, C, q)
    fin = np.isfinite(av)
    with np.errstate(invalid="ignore"):
        reason = np.where(~span, -1, np.where(~act, 0, np.where(nbad > 0, 1, np.where(~fin, 2,
                                                                                     np.where(av < AV_MIN, 3, 0)))))
    valid = reason == 0
    pnl = np.where(inact, 0.0, mtm - fee + fund)
    r = np.full((nU, nC), np.nan)
    va = valid & act
    r[va] = pnl[va] / av[va]
    r[valid & ~act] = 0.0
    return dict(C=C, nC=nC, span=span, act=act, reason=reason, valid=valid, mtm=mtm, fee=fee, fund=fund, pnl=pnl,
                av=av, wide=wide, r=r, start=np.where(has, start, -1), nfund_outside=int(nfund[~span].sum()),
                fund_outside=float(R.FUND[:, :nC][~span].sum()))


# ---------------------------------------------------------------------------------------
# One window
# ---------------------------------------------------------------------------------------
def window_r(Dd, a, b, hip3):
    assert b == Dd["C"]
    ia, ib = (a - W0N) // DAY, (b - W0N) // DAY
    sl = slice(ia, ib)
    span, act, reason, valid = Dd["span"][:, sl], Dd["act"][:, sl], Dd["reason"][:, sl], Dd["valid"][:, sl]
    r, pnl, wide = Dd["r"][:, sl], Dd["pnl"][:, sl], Dd["wide"][:, sl]
    n = valid.sum(1)
    sup = {"hip3_asof": hip3, "n_span_days": span.sum(1), "n_ret_days": n, "n_act_days": act.sum(1),
           "n_act_noav": (reason == 2).sum(1), "n_act_lt100": (reason == 3).sum(1),
           "n_px_bad_days": (reason == 1).sum(1), "n_wide_av_days": (valid & act & wide).sum(1)}
    rv = np.where(valid, r, 0.0)
    pv = np.where(valid, pnl, 0.0)
    sup["n_gain_days"], sup["n_loss_days"] = (pv > 0).sum(1), (pv < 0).sum(1)
    sup["gain_usd"], sup["loss_usd"] = np.maximum(pv, 0.0).sum(1), np.maximum(-pv, 0.0).sum(1)
    sup["pnl_usd"] = pv.sum(1)
    sup["mtm_usd"] = np.where(valid, Dd["mtm"][:, sl], 0.0).sum(1)
    sup["fee_usd"] = np.where(valid, Dd["fee"][:, sl], 0.0).sum(1)
    sup["funding_usd"] = np.where(valid, Dd["fund"][:, sl], 0.0).sum(1)
    tot = rv.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / np.where(n > 0, n, 1)
        var = (np.where(valid, r - mean[:, None], 0.0) ** 2).sum(1) / np.where(n > 1, n - 1, 1)
        same = np.where(valid, r, -np.inf).max(1) == np.where(valid, r, np.inf).min(1)
        sd = np.where(same, 0.0, np.sqrt(var))
        sup["ret_sd"] = np.where(n >= 2, sd, np.nan)
        out = {"ret_sum": np.where(n > 0, tot, np.nan),
               "ret_mean_sd": np.where((n >= 2) & (sd > 0), mean / np.where(sd > 0, sd, 1.0), np.nan)}
        cs = np.cumsum(rv, axis=1)
        dd = (np.maximum.accumulate(np.maximum(cs, 0.0), axis=1) - cs).max(1) if ib > ia else np.zeros(len(n))
        out["max_dd"] = np.where(n > 0, dd, np.nan)
        G, L = sup["gain_usd"], sup["loss_usd"]
        out["gain_share"] = np.where(G + L > 0, G / np.where(G + L > 0, G + L, 1.0), np.nan)
    frame = pd.DataFrame({**out, **sup})
    for c in SUPPORT_R:
        if c.startswith("n_"):
            frame[c] = frame[c].astype(np.int64)
    frame["hip3_asof"] = frame["hip3_asof"].astype(bool)
    return frame[RCOLS + SUPPORT_R]


def run_windows(R, wt, verbose=True, keep_daily=()):
    """ledger_r / support_r for the windows of wt (one row per window id); daily tables of the cutoffs in
    keep_daily are returned too (key 'daily')."""
    t0 = time.time()
    U = R.P.U
    res, daily = {}, {}
    wins = sorted({(int(a.value), int(b.value)) for a, b in zip(wt["a"], wt["b"])})
    for C in sorted({b for _, b in wins}):
        Dd = daily_at(R, C)
        hip3 = (R.hip3_lt < C) | (R.hip3_le <= C)
        for a, b in wins:
            if b == C:
                res[(a, b)] = window_r(Dd, a, b, hip3)
        if C in keep_daily:
            daily[C] = Dd
    if verbose:
        log(f"  {len(res)} distinct windows, {len({b for _, b in wins})} cutoffs in {time.time() - t0:.1f}s")
    led, sup = [], []
    for r in wt.itertuples(index=False):
        fr = res[(r.a.value, r.b.value)]
        v = fr[RCOLS].copy()
        v.insert(0, "trader", U)
        lv = v.melt(id_vars="trader", var_name="column", value_name="value")
        lv.insert(1, "window_id", r.window_id)
        led.append(lv)
        s = fr[SUPPORT_R].copy()
        s.insert(0, "window_id", r.window_id)
        s.insert(0, "trader", U)
        sup.append(s)
    cat = lambda d, cols: d.astype({c: "category" for c in cols})  # noqa: E731
    return dict(ledger_r=cat(pd.concat(led, ignore_index=True), ["trader", "window_id", "column"]),
                support_r=pd.concat(sup, ignore_index=True), daily=daily)


def daily_frame(R, Dd):
    """daily_pnl.parquet rows: every span day of every address at the cutoff of Dd."""
    ti, D = np.nonzero(Dd["span"])
    rs = Dd["reason"][ti, D]
    bad = rs == 1
    mtm, pnl = Dd["mtm"][ti, D], Dd["pnl"][ti, D]
    return pd.DataFrame({"trader": pd.Categorical(R.P.U[ti]), "day": (W0N + D * DAY).astype("datetime64[ns]"),
                         "mtm": np.where(bad, np.nan, mtm), "fee": Dd["fee"][ti, D], "funding": Dd["fund"][ti, D],
                         "pnl": np.where(bad, np.nan, pnl), "av": Dd["av"][ti, D], "wide_av": Dd["wide"][ti, D],
                         "ret": Dd["r"][ti, D], "active": Dd["act"][ti, D], "valid": Dd["valid"][ti, D],
                         "reason": pd.Categorical([REASON[x] for x in rs])})


# ---------------------------------------------------------------------------------------
# Reconciliation diagnostic (full information as of W1)
# ---------------------------------------------------------------------------------------
def _grid_px(R, si, c):
    """Exposure-grid price of symbol si at grid closes c (NaN at or before W0)."""
    i = (c - W0N) // STEP - 1
    ok = (i >= 0) & (i < N)
    out = np.full(len(c), np.nan)
    out[ok] = R.P.px[si, i[ok]]
    return out


def recon(R, trs, fund):
    P = R.P
    tmap = {t: i for i, t in enumerate(P.U)}
    t = trs[trs["trader"].isin(tmap)]
    fo = P.fday[P.fday["pkg_first_ns"] >= W1N]           # funding-only as of W1
    fo_by = {}
    for (ti, coin), d in fo.groupby(["ti", "coin"]):
        fo_by.setdefault(ti, []).append((int(d["sym_i"].iloc[0]), dict(zip(d["day_ns"].astype("int64"), d["szi"]))))
    fu = fund[fund["trader"].isin(tmap)]
    hourly = fu["n_samples"].isna().to_numpy()
    fu_ti = fu["trader"].map(tmap).to_numpy()
    fu_t = fu["time"].to_numpy().astype("int64")
    fu_u = fu["usdc"].to_numpy()
    fh, fdl = {}, {}
    for ti in np.unique(fu_ti):
        m = fu_ti == ti
        o = np.argsort(fu_t[m & hourly], kind="stable")
        fh[ti] = (fu_t[m & hourly][o], np.r_[0.0, np.cumsum(fu_u[m & hourly][o])])
        fdl[ti] = (fu_t[m & ~hourly], fu_u[m & ~hourly])
    fill_ts = {ti: np.sort(d) for ti, d in pd.Series(P.p_ts).groupby(P.p["ti"].to_numpy())}
    rows = []
    for tr_, d in t.groupby("trader", sort=True):
        ti = tmap[tr_]
        ts = d["ts"].to_numpy().astype("int64")
        av = d["acct_val"].to_numpy()
        nd = d["net_deposit"].to_numpy()
        if len(ts) < 2:
            continue
        t1, t2 = ts[:-1], ts[1:]
        m = ((t2 - t1 <= RECON_GAP) & (nd[1:] == nd[:-1]) & (t1 >= W0N)
             & ~((P.h3_lt[ti] < t2) | (P.h3_le[ti] <= t2)))
        if not m.any():
            continue
        t1, t2, a1, a2 = t1[m], t2[m], av[:-1][m], av[1:][m]
        c1, c2 = t1 - (t1 - W0N) % STEP, t2 - (t2 - W0N) % STEP
        rep = np.zeros(len(t1))
        bad = np.zeros(len(t1), dtype=bool)
        for r in P.pairs_by_ti.get(ti, []):
            p = r.pid
            sl = slice(r.r0, r.r1)
            gts, gge, ggs, gvw = R.g_ts[sl], R.g_ge[sl], R.g_gs[sl], R.g_vw[sl]
            cc = np.r_[0.0, np.cumsum(R.cash_g[sl])]
            fc = np.r_[0.0, np.cumsum(R.fee_g[sl])]
            j1 = np.searchsorted(gts, c1, side="right") - 1
            j2 = np.searchsorted(gts, c2, side="right") - 1

            def q_at(c, jx):
                pre = np.where(R.has_c[p] & (c > R.cns[p]), R.gstart0[p], 0.0)
                return np.where(jx >= 0, gge[np.maximum(jx, 0)], pre)

            v1, b1 = _qv(q_at(c1, j1), _grid_px(R, r.sym_i, c1))
            v2, b2 = _qv(q_at(c2, j2), _grid_px(R, r.sym_i, c2))
            # entries in (c1, c2]: carry (position present at grid closes c > carry_ns), first fill, jumps
            ev_t, ev_v, ev_b = [], [], []
            if R.has_c[p]:
                pc = _grid_px(R, r.sym_i, np.array([R.cns[p]]))[0]
                ev_t.append(R.cns[p] + 1)
                ev_v.append(R.gstart0[p] * pc if np.isfinite(pc) else 0.0)
                ev_b.append(not np.isfinite(pc))
            elif R.gstart0[p] != 0:
                ev_t.append(gts[0])
                ev_v.append(R.gstart0[p] * gvw[0])
                ev_b.append(False)
            jmp = ggs[1:] - gge[:-1]
            for k in np.flatnonzero(np.abs(jmp) > lib.POS_EPS) + 1:
                ev_t.append(gts[k])
                ev_v.append(jmp[k - 1] * gvw[k])
                ev_b.append(False)
            ent = np.zeros(len(t1))
            for et, evv, eb in zip(ev_t, ev_v, ev_b):
                inn = (c1 < et) & (et <= c2)
                ent[inn] += evv
                bad |= inn & eb
            rep += v2 - v1 + (cc[j2 + 1] - cc[j1 + 1]) - (fc[j2 + 1] - fc[j1 + 1]) - ent
            bad |= b1 | b2
        for si, days in fo_by.get(ti, []):
            for i in range(len(t1)):
                b_ = [c1[i]] + list(range(int(c1[i] - c1[i] % DAY + DAY), int(c2[i]), DAY)) + [c2[i]]
                for s_, e_ in zip(b_[:-1], b_[1:]):
                    z = days.get(s_ - s_ % DAY)
                    if z is None or e_ <= s_:
                        continue
                    pe, ps = _grid_px(R, si, np.array([e_, s_]))
                    if np.isfinite(pe) and np.isfinite(ps):
                        rep[i] += z * (pe - ps)
                    else:
                        bad[i] = True
        if ti in fh:
            ht, hc = fh[ti]
            rep += hc[np.searchsorted(ht, t2, side="right")] - hc[np.searchsorted(ht, t1, side="right")]
            dt_, du = fdl[ti]
            for D_, u_ in zip(dt_, du):
                ov = np.clip(np.minimum(t2, D_ + DAY) - np.maximum(t1, D_), 0, None)
                rep += u_ * ov / DAY
        ft = fill_ts.get(ti, np.array([], dtype=np.int64))
        nf = np.searchsorted(ft, c2, side="right") - np.searchsorted(ft, c1, side="right")
        rep = np.where(bad, np.nan, rep)
        rows.append(pd.DataFrame({"trader": tr_, "t1": t1.astype("datetime64[ns]"), "t2": t2.astype("datetime64[ns]"),
                                  "d_av": a2 - a1, "replay_pnl": rep, "resid": (a2 - a1) - rep, "av1": a1,
                                  "has_fill": nf > 0, "n_fills": nf.astype(np.int64)}))
    out = pd.concat(rows, ignore_index=True)
    out["trader"] = out["trader"].astype("category")
    return out


def recon_summary(rc):
    fin = rc["resid"].notna()
    x = rc[fin]
    good = x["resid"].abs() <= np.maximum(5.0, 0.01 * x["av1"].abs())
    pos = x["av1"] > 0
    rel = (x["resid"].abs() / x["av1"])[pos]
    s = dict(intervals=len(rc), replay_undefined_px=int((~fin).sum()), assessed=len(x), within_tol=int(good.sum()),
             share_within_max5_1pct=float(good.mean()) if len(x) else np.nan,
             share_within_with_fill=float(good[x["has_fill"]].mean()), share_within_no_fill=float(good[~x["has_fill"]].mean()),
             n_with_fill=int(x["has_fill"].sum()),
             abs_resid_over_av1_quantiles={q: float(rel.quantile(q)) for q in (.1, .25, .5, .75, .9, .95, .99)},
             av1_nonpositive=int((~pos).sum()))
    b = x.assign(bad=~good).groupby("trader", observed=True).agg(n=("bad", "size"), n_bad=("bad", "sum"))
    b = b[b["n"] >= 20].assign(share_bad=lambda d: d["n_bad"] / d["n"]).sort_values(["share_bad", "n"],
                                                                                      ascending=[False, False])
    s["addresses_ge20_intervals"] = len(b)
    s["addresses_ge20_with_share_bad_gt_half"] = int((b["share_bad"] > 0.5).sum())
    return s, b.head(40)


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def coverage(sr):
    """Per L60 Monday: base-eligible addresses (ledger support.parquet) passing each component of the registered
    return-column gate (measures._r_col, kind 'L') and the whole gate."""
    import measures as MZ
    col = next(e for e in MZ.EXTENSION if e.name == "ret_sum")
    lsup = pd.read_parquet(OUT / "support.parquet", columns=["window_id", "trader", "base_eligible"])
    rows = []
    for M in MONDAYS:
        wid = f"L60_{M:%Y-%m-%d}"
        be = lsup[(lsup["window_id"] == wid) & lsup["base_eligible"]]["trader"].astype(str)
        s = sr[sr["window_id"] == wid].astype({"trader": str}).set_index("trader").reindex(be)
        parts = MZ.gate_parts(col, s, "L")
        allg = np.logical_and.reduce([v.to_numpy().astype(bool) for v in parts.values()])
        rows.append(dict(M=f"{M:%Y-%m-%d}", base=len(be), **{k: int(v.astype(bool).sum()) for k, v in parts.items()},
                         gate=int(allg.sum())))
    return pd.DataFrame(rows)


def build():
    global _logf
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "returns.log", "w")
    t0 = time.time()
    existing = {f"{n}.parquet": LG.sha256(OUT / f"{n}.parquet") for n in EXISTING if (OUT / f"{n}.parquet").exists()}
    inp = LG.load_inputs()
    U = sorted(inp["p"]["trader"].unique())
    R = build_r(inp, U)
    log(f"prepared in {time.time() - t0:.1f}s: U {len(U)}, pairs {len(R.first_ts):,}, groups {len(R.g_ts):,}, "
        f"funding-only pair-days {len(R.fo_D):,}")
    log(f"entries without a fill: {json.dumps(R.entries)}")
    log(f"funding records before W0 (never in a span): {R.fund_pre_w0}")
    wt = LG.window_table()
    tabs = run_windows(R, wt, keep_daily=(W1N,))
    for k in ("ledger_r", "support_r"):
        tabs[k].to_parquet(OUT / f"{k}.parquet", index=False)
        log(f"wrote {k}.parquet: {len(tabs[k]):,} rows x {tabs[k].shape[1]} cols")
    Dw = tabs["daily"][W1N]
    dp = daily_frame(R, Dw)
    dp.to_parquet(OUT / "daily_pnl.parquet", index=False)
    log(f"wrote daily_pnl.parquet: {len(dp):,} rows (span days as of W1); funding records on non-span days "
        f"{Dw['nfund_outside']:,} (usdc {Dw['fund_outside']:.2f})")
    rc_ = dp["reason"].astype(str).value_counts().to_dict()
    log(f"  as of W1: span days {len(dp):,}, active {int(dp['active'].sum()):,}, valid {int(dp['valid'].sum()):,}, "
        f"reasons {rc_}, wide_av valid active {int((dp['valid'] & dp['active'] & dp['wide_av']).sum()):,}")
    va = dp[dp["valid"] & dp["active"]]
    top = va.reindex(va["ret"].abs().sort_values(ascending=False).index).head(20)
    log("  20 largest |r_d| as of W1 (valid active days):")
    log(top[["trader", "day", "pnl", "mtm", "fee", "funding", "av", "wide_av", "ret"]].to_string(index=False))
    q = va["ret"].quantile([.001, .01, .05, .5, .95, .99, .999])
    log(f"  r_d quantiles (valid active, W1): {q.round(5).to_dict()}; |r| > 1: {int((va['ret'].abs() > 1).sum())}")

    sr = tabs["support_r"]
    cov = coverage(sr)
    log("\n=== per L60 window: base-eligible addresses passing each gate component (kind L) ===")
    log(cov.to_string(index=False))

    log("\n=== reconciliation diagnostic (full information as of W1) ===")
    trs = lib.load_traders()
    rc = recon(R, trs, inp["fund"])
    rc.to_parquet(OUT / "recon.parquet", index=False)
    rs, worst = recon_summary(rc)
    log(json.dumps(rs, indent=1))
    log("40 addresses (>= 20 intervals) with the largest share of intervals outside max($5, 1% of av1):")
    log(worst.reset_index().to_string(index=False))

    runtime = time.time() - t0
    srcs = [Path(__file__).resolve(), ROOT / "work/roll/code/ledger.py", ROOT / "work/v2dev/code/lib.py",
            ROOT / "work/profile/profile.py", ROOT / "work/profile2/code/exposure.py"]
    data = ROOT / "ArkStream笔试题_v2"
    files = [data / "笔试题数据包-fills", data / "笔试题数据包-traders.csv", LG.EXT / "funding.parquet",
             LG.EXT / "hl_candles_4h.parquet", LG.EXT / "symbol_map.csv"]
    man = dict(inputs={str(f.relative_to(ROOT)): LG.sha256(f) for f in files},
               klines_5m={f.name: LG.sha256(f) for f in sorted(LG.KLINES.glob("*.parquet"))},
               sources={str(f.relative_to(ROOT)): LG.sha256(f) for f in srcs},
               existing_build_tables_sha256=existing,
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             pyarrow=pyarrow.__version__, scipy=scipy.__version__),
               seed=SEED, run_utc=pd.Timestamp.now(tz="UTC").isoformat(), runtime_s=round(runtime, 1),
               rows=dict(ledger_r=len(tabs["ledger_r"]), support_r=len(sr), daily_pnl=len(dp), recon=len(rc)),
               av_rule=dict(bracket_h=168, narrow_h=48, edge_h=24, av_min=AV_MIN),
               reason_priority=["px_bad", "noav", "lt100"], entries_without_fill=R.entries,
               funding_pre_w0=R.fund_pre_w0, recon=rs, coverage_L60=cov.to_dict(orient="records"))
    (OUT / "manifest_r.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"\nruntime {runtime:.1f}s")


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def _norm(d, keys):
    d = d.astype({c: str for c in d.columns if isinstance(d[c].dtype, pd.CategoricalDtype)})
    return d.sort_values(keys, kind="stable").reset_index(drop=True)


def _ndiff(a, b, cols, tol=0.0):
    out = {}
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            x, y = x.to_numpy(float), y.to_numpy(float)
            same = (x == y) | (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol * np.maximum(np.abs(x), np.abs(y)))
        else:
            same = (x.astype(object).to_numpy() == y.astype(object).to_numpy())
        out[c] = int((~same).sum())
    return out


class Brute:
    """Independent, loop-based daily PnL / account value of one (trader, cutoff, day) from the raw inputs."""

    def __init__(self, inp):
        p = inp["p"]
        self.p_by = {t: d.sort_values(["ts", "tid"], kind="stable") for t, d in p.groupby("trader")}
        tape = p.sort_values(["symbol", "ts", "tid"], kind="stable")
        self.tape = {s: (d["ts"].to_numpy().astype("int64"), d["price"].to_numpy()) for s, d in tape.groupby("symbol")}
        self.kl = {}
        for s, k in inp["kl"].items():
            k = k[k["close"].notna()]
            self.kl[s] = (k["open_time"].to_numpy().astype("int64"), k["close"].to_numpy())
        h = inp["hl"][inp["hl"]["close"].notna()]
        self.hl = {s: ((d["open_time"] + pd.Timedelta(hours=4)).to_numpy().astype("int64"), d["close"].to_numpy())
                   for s, d in h.groupby("symbol")}
        self.f_by = {t: d for t, d in inp["fund"].groupby("trader")}
        self.s_by = {t: d.sort_values("ts") for t, d in inp["tr"].groupby("trader")}
        self.g_cache = {}

    def px(self, sym, t):
        if t <= W0N:
            return np.nan
        if sym in self.kl:
            o, c = self.kl[sym]
            i = np.searchsorted(o, t - STEP)
            if i < len(o) and o[i] == t - STEP and W0N <= o[i] < W1N:
                return float(c[i])
        if sym in self.hl:
            ct, c = self.hl[sym]
            best = None
            for x, v in zip(ct, c):
                if x <= t and (best is None or x >= best[0]):
                    best = (x, v)
            if best is not None:
                return float(best[1])
        if sym in self.tape:
            tt, pp = self.tape[sym]
            k = np.searchsorted(tt, t, side="right") - 1
            if k >= 0:
                return float(pp[k])
        return np.nan

    def groups(self, trader, sym):
        key = (trader, sym)
        if key not in self.g_cache:
            f = self.p_by[trader]
            self.g_cache[key] = lib.fill_groups(f[f["symbol"] == sym]).sort_values("ts").reset_index(drop=True)
        return self.g_cache[key]

    def carry(self, trader, sym, first_ts):
        f = self.f_by.get(trader)
        if f is None:
            return None
        days = set(f.loc[f["coin"] == sym, "time"].dt.floor("D").astype("int64"))
        d = first_ts - first_ts % DAY - DAY
        if d < W0N or d not in days:
            return None
        while d - DAY >= W0N and (d - DAY) in days:
            d -= DAY
        return d

    def day(self, trader, C, d):
        e = d + DAY
        f = self.p_by[trader]
        fts = f["ts"].to_numpy().astype("int64")
        fu = self.f_by.get(trader)
        mtm, bad, act, terms = 0.0, False, False, 0.0
        starts = []
        pkg = sorted(set(f.loc[fts < C, "symbol"]))
        for sym in pkg:
            g = self.groups(trader, sym)
            gts = g["ts"].to_numpy().astype("int64")
            first = int(gts[0])
            g0 = float(g["gstart"].iloc[0])
            cy = self.carry(trader, sym, first) if g0 != 0 else None
            starts.append(cy if cy is not None else first)

            def q(t):
                last = None
                for k in range(len(gts)):
                    if gts[k] < t:
                        last = k
                if last is not None:
                    return float(g["gend"].iloc[last])
                return g0 if (cy is not None and t > cy) else 0.0

            q0, q1 = q(d), q(e)
            act |= (q0 != 0) or (q1 != 0)
            for qq, tt, sg in ((q1, e, 1.0), (q0, d, -1.0)):
                if qq != 0:
                    pp = self.px(sym, tt)
                    if np.isnan(pp):
                        bad = True
                    else:
                        mtm += sg * qq * pp
                        terms += abs(qq * pp)
            ff = f[(f["symbol"] == sym) & (fts >= d) & (fts < e)]
            for _, row in ff.iterrows():
                mtm += -row["signed"] * row["price"]
                terms += abs(row["signed"] * row["price"])
            if cy is not None and cy == d:
                pp = self.px(sym, d)
                if np.isnan(pp):
                    bad = True
                else:
                    mtm -= g0 * pp
            if cy is None and g0 != 0 and d <= first < e:
                mtm -= g0 * float(g["vwap"].iloc[0])
            for k in range(1, len(gts)):
                if d <= gts[k] < e:
                    jmp = float(g["gstart"].iloc[k]) - float(g["gend"].iloc[k - 1])
                    if abs(jmp) > lib.POS_EPS:
                        mtm -= jmp * float(g["vwap"].iloc[k])
        if fu is not None:
            fw = fu[fu["time"] >= pd.Timestamp(W0N)]
            for coin in sorted(set(fw["coin"]) - set(pkg)):
                if ((f["symbol"] == coin) & (fts < C)).any():
                    continue
                fc = fw[fw["coin"] == coin].sort_values("time", kind="stable")
                fdays = {}
                for _, row in fc.iterrows():
                    fdays[int(row["time"].floor("D").value)] = row["szi"]
                fdays = {k: v for k, v in fdays.items() if abs(v) >= lib.POS_EPS}
                if not fdays or min(fdays) >= C:
                    continue
                starts.append(min(fdays))
                if d in fdays:
                    act = True
                    p0, p1 = self.px(coin, d), self.px(coin, e)
                    if np.isnan(p0) or np.isnan(p1):
                        bad = True
                    else:
                        mtm += fdays[d] * (p1 - p0)
                        terms += abs(fdays[d] * p1) + abs(fdays[d] * p0)
        fday = f[(fts >= d) & (fts < e)]
        fee = float(fday["fee"].sum())
        act |= len(fday) > 0
        fund = 0.0
        if fu is not None:
            fr = fu[fu["time"].dt.floor("D").astype("int64") == d]
            fund = float(fr["usdc"].sum())
            act |= len(fr) > 0
        span = bool(starts) and d >= min(starts) - min(starts) % DAY
        # account value: snapshots < C bracketing d
        sn = self.s_by.get(trader)
        prev = nxt = None
        if sn is not None:
            for t_, v_ in zip(sn["ts"].to_numpy().astype("int64"), sn["acct_val"].to_numpy()):
                if t_ >= C:
                    break
                if t_ <= d:
                    prev = (t_, v_)
                elif nxt is None:
                    nxt = (t_, v_)
        av, wide = np.nan, False
        if prev is not None and nxt is not None:
            if nxt[0] - prev[0] <= 168 * HOUR:
                av, wide = max(prev[1], nxt[1]), nxt[0] - prev[0] > 48 * HOUR
        elif prev is not None and d - prev[0] <= 24 * HOUR:
            av = prev[1]
        elif nxt is not None and nxt[0] - d <= 24 * HOUR:
            av = nxt[1]
        if not span:
            reason = -1
        elif not act:
            reason = 0
        elif bad:
            reason = 1
        elif not np.isfinite(av):
            reason = 2
        elif av < AV_MIN:
            reason = 3
        else:
            reason = 0
        pnl = 0.0 if (span and not act) else mtm - fee + fund
        r = (pnl / av if act else 0.0) if reason == 0 else np.nan
        return dict(pnl=pnl, mtm=mtm, fee=fee, fund=fund, av=av, wide=wide, r=r, reason=reason, act=act and span,
                    scale=terms + abs(fee) + abs(fund))


def _col_loop(r_list, p_list):
    """Plain-loop column formulas on one address-window's valid series (r_d, pnl_d in day order)."""
    n = len(r_list)
    if n == 0:
        return dict(ret_sum=np.nan, ret_mean_sd=np.nan, max_dd=np.nan, gain_share=np.nan)
    s = 0.0
    for x in r_list:
        s += x
    mean = s / n
    sm = np.nan
    if n >= 2 and max(r_list) != min(r_list):
        v = 0.0
        for x in r_list:
            v += (x - mean) ** 2
        sm = mean / math.sqrt(v / (n - 1))
    cum, peak, dd = 0.0, 0.0, 0.0
    for x in r_list:
        cum += x
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    G = sum(x for x in p_list if x > 0)
    L = sum(-x for x in p_list if x < 0)
    return dict(ret_sum=s, ret_mean_sd=sm, max_dd=dd, gain_share=G / (G + L) if G + L > 0 else np.nan)


def checks(before=None):
    global _logf
    _logf = open(LOGS / "checks_r.log", "w")
    t0 = time.time()
    res = {}
    rng = np.random.default_rng(SEED)
    wt = LG.window_table()
    inp = LG.load_inputs()
    U = sorted(inp["p"]["trader"].unique())
    R = build_r(inp, U)
    P = R.P
    led = pd.read_parquet(OUT / "ledger_r.parquet")
    sup = pd.read_parquet(OUT / "support_r.parquet")
    cuts = sorted({int(b.value) for b in wt["b"]})
    daily = {C: daily_at(R, C) for C in cuts}

    # ---------------- check 1: brute force ----------------
    log("=== check 1: brute-force daily pnl / av / r from the raw inputs ===")
    B = Brute(inp)
    ND_ = ND

    def pick_C(D, lo=None, hi=None):
        cs = [C for C in cuts if C >= W0N + (D + 1) * DAY and (lo is None or C > lo) and (hi is None or C <= hi)]
        return int(rng.choice(cs)) if cs else None

    cand = {k: [] for k in ("carried", "funding_only", "flip", "first_fill_gstart0", "jump", "snapshot_gap", "random")}
    for p in rng.permutation(np.flatnonzero(R.has_c))[:200]:
        ck = (R.cns[p] - W0N) // DAY
        fdD = (R.first_ts[p] - W0N) // DAY
        D = int(rng.integers(ck, fdD + 1))
        C = pick_C(D, lo=R.first_ts[p])
        if C is not None:
            cand["carried"].append((R.pair_ti[p], C, D))
    for i in rng.permutation(len(R.fo_D))[:2000]:
        C = pick_C(R.fo_D[i], hi=R.fo_first[i])
        if C is not None:
            cand["funding_only"].append((R.fo_ti[i], C, int(R.fo_D[i])))
    fl = np.flatnonzero((R.g_gs * R.g_ge) < 0)
    for j in rng.permutation(fl)[:200]:
        D = int((R.g_ts[j] - W0N) // DAY)
        C = pick_C(D)
        if C is not None:
            cand["flip"].append((R.pair_ti[R.g_pid[j]], C, D))
    m2 = np.flatnonzero((R.gstart0 != 0) & ~R.has_c)
    for p in rng.permutation(m2)[:200]:
        D = int((R.first_ts[p] - W0N) // DAY)
        C = pick_C(D)
        if C is not None:
            cand["first_fill_gstart0"].append((R.pair_ti[p], C, D))
    for j in rng.permutation(R.jump_rows)[:200]:
        D = int((R.g_ts[j] - W0N) // DAY)
        C = pick_C(D)
        if C is not None:
            cand["jump"].append((R.pair_ti[R.g_pid[j]], C, D))
    gD = [int((x.value - W0N) // DAY) for x in GAP_DAYS]
    Dw = daily[W1N]
    for _ in range(400):
        D = int(rng.choice(gD))
        tis = np.flatnonzero(Dw["act"][:, D])
        ti = int(rng.choice(tis))
        C = pick_C(D)
        cand["snapshot_gap"].append((ti, C, D))
    for _ in range(400):
        C = int(rng.choice(cuts))
        Dd = daily[C]
        ti, D = np.nonzero(Dd["span"])
        k = int(rng.integers(len(ti)))
        cand["random"].append((int(ti[k]), C, int(D[k])))
    c1 = {}
    worst = dict(pnl=0.0, av=0.0, r=0.0)
    fails = []
    for name, lst in cand.items():
        seen, picked = set(), []
        for x in lst:
            if x not in seen:
                seen.add(x)
                picked.append(x)
            if len(picked) == 60:
                break
        n_ok = 0
        for ti, C, D in picked:
            Dd = daily[C]
            b_ = B.day(P.U[ti], C, W0N + D * DAY)
            mine = dict(reason=int(Dd["reason"][ti, D]), pnl=Dd["pnl"][ti, D], av=Dd["av"][ti, D], r=Dd["r"][ti, D],
                        act=bool(Dd["act"][ti, D]), wide=bool(Dd["wide"][ti, D]))
            ok = mine["reason"] == b_["reason"] and mine["act"] == b_["act"]
            av_ok = (np.isnan(mine["av"]) and np.isnan(b_["av"])) or mine["av"] == b_["av"]
            ok &= av_ok and (mine["wide"] == b_["wide"] or not np.isfinite(b_["av"]))
            if b_["reason"] in (0, 2, 3):
                d_ = abs(mine["pnl"] - b_["pnl"])
                worst["pnl"] = max(worst["pnl"], d_ / max(b_["scale"], 1e-300) if d_ else 0.0)
                ok &= d_ <= 1e-9 * b_["scale"] + 1e-9
            if b_["reason"] == 0:
                rr = mine["r"], b_["r"]
                d_ = abs(rr[0] - rr[1])
                worst["r"] = max(worst["r"], d_ / max(abs(rr[1]), 1e-300) if d_ else 0.0)
                ok &= d_ <= 1e-9 * (b_["scale"] / max(abs(b_["av"]), 1e-300)) + 1e-12 if b_["act"] else rr[0] == 0
            n_ok += bool(ok)
            if not ok:
                fails.append(dict(stratum=name, trader=P.U[ti], C=str(pd.Timestamp(C)), day=str(pd.Timestamp(W0N + D * DAY)),
                                  mine=mine, brute={k: b_[k] for k in ("reason", "pnl", "av", "r", "act", "wide")}))
        c1[name] = dict(sampled=len(picked), agree=n_ok, reasons=pd.Series(
            [int(daily[C]["reason"][ti, D]) for ti, C, D in picked]).map(REASON).value_counts().to_dict())
    c1["total_sampled"] = sum(v["sampled"] for v in c1.values() if isinstance(v, dict))
    c1["max_rel_diff_pnl_over_scale"], c1["max_rel_diff_r"] = worst["pnl"], worst["r"]
    c1["failures"] = fails[:20]
    c1["ok"] = not fails and c1["total_sampled"] >= 300
    log(json.dumps(c1, indent=1, default=str))
    res["check1_brute"] = c1

    # ---------------- check 2: telescoping per package pair (as of W1) ----------------
    log("\n=== check 2: telescoping per package pair (as of W1) ===")
    p = inp["p"]
    tot_cash = p.groupby(["trader", "symbol"])["cash"].sum()
    turn = p.groupby(["trader", "symbol"])["notional"].sum()
    pr = P.pairs
    key = list(zip(pr["trader"], pr["symbol"]))
    tc = tot_cash.reindex(key).to_numpy()
    tv = turn.reindex(key).to_numpy()
    q_last = R.g_ge[R.r1 - 1]
    p_last = R.PB[R.pair_si, ND_]
    ent = np.zeros(len(pr))
    for i in range(len(pr)):
        r0_, r1_ = R.r0[i], R.r1[i]
        if R.has_c[i]:
            ent[i] += R.gstart0[i] * R.PB[R.pair_si[i], (R.cns[i] - W0N) // DAY]
        elif R.gstart0[i] != 0:
            ent[i] += R.gstart0[i] * R.g_vw[r0_]
        for k in range(r0_ + 1, r1_):
            jmp = R.g_gs[k] - R.g_ge[k - 1]
            if abs(jmp) > lib.POS_EPS:
                ent[i] += jmp * R.g_vw[k]
    with np.errstate(invalid="ignore"):
        rhs = np.where(q_last != 0, q_last * p_last, 0.0) - ent + tc
    lhs = R.mtm_pair.sum(1)
    anybad = R.bad_pair.any(1)
    d2 = np.abs(lhs - rhs)
    tol = 1e-9 * (tv + np.abs(np.nan_to_num(q_last * p_last)) + np.abs(ent)) + 1e-6
    okp = ~anybad
    c2 = dict(pairs=len(pr), pairs_with_missing_price_day=int(anybad.sum()), tested=int(okp.sum()),
              violations=int((d2[okp] > tol[okp]).sum()), max_abs_diff=float(d2[okp].max()),
              max_diff_over_turnover=float((d2[okp] / tv[okp]).max()))
    c2["ok"] = c2["violations"] == 0
    log(json.dumps(c2, indent=1))
    res["check2_telescoping"] = c2

    # ---------------- check 3: vs Hyperliquid closed_pnl ----------------
    log("\n=== check 3: sum of daily mtm vs sum of closed_pnl (clean flat pairs) ===")
    hasjump = np.zeros(len(pr), dtype=bool)
    hasjump[R.g_pid[R.jump_rows]] = True
    clean = ~R.has_c & (R.gstart0 == 0) & ~hasjump & (q_last == 0) & ~anybad
    cp = np.bincount(R.g_pid, weights=R.cpnl_g, minlength=len(pr))
    fe = np.bincount(R.g_pid, weights=R.fee_g, minlength=len(pr))
    diff = (lhs - cp)[clean]
    tvc = tv[clean]
    within = np.abs(diff) <= np.maximum(1.0, 1e-3 * tvc)
    c3 = dict(pairs=int(clean.sum()), share_within_max1_0p1pct_turnover=float(within.mean()),
              n_outside=int((~within).sum()), sum_mtm=float(lhs[clean].sum()), sum_closed_pnl=float(cp[clean].sum()),
              sum_diff=float(diff.sum()), sum_fees_same_pairs=float(fe[clean].sum()),
              diff_quantiles={q: float(np.quantile(diff, q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
              abs_diff_over_turnover_quantiles={q: float(np.quantile(np.abs(diff) / tvc, q)) for q in (.5, .9, .99, 1)},
              share_exact_1e6=float((np.abs(diff) <= 1e-6).mean()))
    w = np.flatnonzero(clean)[np.argsort(-np.abs(diff))[:10]]
    c3["largest_abs_diff"] = [dict(trader=pr["trader"].iloc[i], symbol=pr["symbol"].iloc[i], mtm=float(lhs[i]),
                                   closed_pnl=float(cp[i]), fee=float(fe[i]), turnover=float(tv[i])) for i in w]
    c3["ok"] = True                                   # diagnostic: differences are reported, not gated
    log(json.dumps(c3, indent=1, default=str))
    res["check3_closed_pnl"] = c3

    # ---------------- check 4: prefix ----------------
    log("\n=== check 4: prefix equality (all windows ending at C; inputs truncated at C) ===")
    c4 = {}
    for M in (pd.Timestamp("2025-10-13"), pd.Timestamp("2025-11-24")):
        ms = f"{M:%Y-%m-%d}"
        tin, removed = LG.truncate_inputs(inp, M)
        Rt = build_r(tin, U)
        sw = wt[wt["b"] == M].reset_index(drop=True)
        tt = run_windows(Rt, sw, verbose=False)
        cm = {"removed": removed, "windows": list(sw["window_id"])}
        tot = dict(ledger_cells=0, ledger_exact_differ=0, ledger_1e12_differ=0, support_cells=0, support_exact_differ=0,
                   support_1e12_differ=0)
        for wid in sw["window_id"]:
            a_ = _norm(led[led["window_id"] == wid], ["trader", "column"])
            b_ = _norm(tt["ledger_r"][tt["ledger_r"]["window_id"] == wid], ["trader", "column"])
            assert len(a_) == len(b_) and (a_[["trader", "column"]].to_numpy() == b_[["trader", "column"]].to_numpy()).all()
            tot["ledger_cells"] += len(a_)
            tot["ledger_exact_differ"] += sum(_ndiff(a_, b_, ["value"]).values())
            tot["ledger_1e12_differ"] += sum(_ndiff(a_, b_, ["value"], 1e-12).values())
            a_ = _norm(sup[sup["window_id"] == wid], ["trader"])
            b_ = _norm(tt["support_r"][tt["support_r"]["window_id"] == wid], ["trader"])
            assert (a_["trader"].to_numpy() == b_["trader"].to_numpy()).all()
            tot["support_cells"] += len(a_) * len(SUPPORT_R)
            tot["support_exact_differ"] += sum(_ndiff(a_, b_, SUPPORT_R).values())
            tot["support_1e12_differ"] += sum(_ndiff(a_, b_, SUPPORT_R, 1e-12).values())
        cm.update(tot)
        log(f"{ms}: {cm}")
        c4[ms] = cm
    c4["ok"] = all(v["ledger_1e12_differ"] == 0 and v["support_1e12_differ"] == 0 for v in c4.values()
                   if isinstance(v, dict))
    res["check4_prefix"] = c4

    # ---------------- check 5: column formulas ----------------
    log("\n=== check 5: column formulas by plain loops (sampled address-windows) ===")
    lx = led.astype({"trader": str, "window_id": str, "column": str}).set_index(["window_id", "trader", "column"])["value"]
    sx = sup.set_index(["window_id", "trader"])
    has_days = sx[sx["n_span_days"] > 0].index.to_numpy()
    pick = rng.choice(len(has_days), size=240, replace=False)
    wmap = {r.window_id: (int(r.a.value), int(r.b.value)) for r in wt.itertuples(index=False)}
    tmap = {t: i for i, t in enumerate(P.U)}
    worst5, bad5, n_sd = {c: 0.0 for c in RCOLS}, 0, 0
    for i in pick:
        wid, tr_ = has_days[i]
        a, b = wmap[wid]
        Dd = daily[b]
        ti = tmap[tr_]
        rl, pl = [], []
        for D in range((a - W0N) // DAY, (b - W0N) // DAY):
            if Dd["valid"][ti, D]:
                rl.append(float(Dd["r"][ti, D]))
                pl.append(float(Dd["pnl"][ti, D]))
        want = _col_loop(rl, pl)
        n_sd += np.isfinite(want["ret_mean_sd"])
        okw = len(rl) == sx.loc[(wid, tr_), "n_ret_days"]
        sabs = sum(abs(x) for x in rl)
        for c in RCOLS:
            x, y = lx[(wid, tr_, c)], want[c]
            if np.isnan(x) and np.isnan(y):
                continue
            d_ = abs(x - y)
            rel = d_ / max(abs(y), 1e-300) if d_ else 0.0
            worst5[c] = max(worst5[c], rel)
            # summation order differs (numpy pairwise vs sequential): tolerance on the scale of the summed terms
            okw &= d_ <= 1e-9 * max(abs(y), sabs if c in ("ret_sum", "max_dd") else 1.0)
        bad5 += not okw
    c5 = dict(sampled=len(pick), failures=bad5, with_finite_ret_mean_sd=int(n_sd), max_rel_diff=worst5, ok=bad5 == 0)
    log(json.dumps(c5, indent=1))
    res["check5_formulas"] = c5

    # ---------------- check 6: identities ----------------
    log("\n=== check 6: identities ===")
    s = sup
    scale = s["mtm_usd"].abs() + s["fee_usd"].abs() + s["funding_usd"].abs()
    lsup = pd.read_parquet(OUT / "support.parquet", columns=["window_id", "trader", "hip3_asof"]) \
        .astype({"window_id": str, "trader": str})
    hj = s[["window_id", "trader", "hip3_asof"]].astype({"window_id": str, "trader": str}).merge(
        lsup, on=["window_id", "trader"], suffixes=("", "_ledger"))
    inact_bad, inact_n = 0, 0
    for C, Dd in daily.items():
        ia = Dd["span"] & ~Dd["act"]
        inact_n += int(ia.sum())
        inact_bad += int((ia & ~(Dd["valid"] & (Dd["r"] == 0) & (Dd["pnl"] == 0))).sum())
    c6 = dict(rows=len(s),
              days_partition_violations=int((s["n_ret_days"] + s["n_act_noav"] + s["n_act_lt100"] + s["n_px_bad_days"]
                                             != s["n_span_days"]).sum()),
              act_le_span_violations=int((s["n_act_days"] > s["n_span_days"]).sum()),
              inactive_days_checked=inact_n, inactive_not_valid_zero=inact_bad,
              pnl_identity_violations=int((np.abs(s["pnl_usd"] - (s["mtm_usd"] - s["fee_usd"] + s["funding_usd"]))
                                           > 1e-9 * scale + 1e-6).sum()),
              gain_loss_identity_violations=int((np.abs(s["gain_usd"] - s["loss_usd"] - s["pnl_usd"])
                                                 > 1e-9 * (s["gain_usd"] + s["loss_usd"]) + 1e-6).sum()),
              gain_loss_day_counts_le_ret_days=int((s["n_gain_days"] + s["n_loss_days"] > s["n_ret_days"]).sum()),
              max_dd_negative=int((led.loc[led["column"] == "max_dd", "value"] < 0).sum()),
              gain_share_outside_0_1=int(((led["column"] == "gain_share") & ((led["value"] < 0) | (led["value"] > 1))).sum()),
              hip3_rows_joined=len(hj), hip3_differs_from_ledger_support=int((hj["hip3_asof"] != hj["hip3_asof_ledger"]).sum()),
              ledger_rows=len(led), ledger_rows_expected=len(U) * len(wt) * len(RCOLS))
    c6["ok"] = (c6["days_partition_violations"] == 0 and c6["act_le_span_violations"] == 0 and inact_bad == 0
                and c6["pnl_identity_violations"] == 0 and c6["gain_loss_identity_violations"] == 0
                and c6["gain_loss_day_counts_le_ret_days"] == 0 and c6["max_dd_negative"] == 0
                and c6["gain_share_outside_0_1"] == 0 and c6["hip3_differs_from_ledger_support"] == 0
                and c6["hip3_rows_joined"] == len(s) and c6["ledger_rows"] == c6["ledger_rows_expected"])
    log(json.dumps(c6, indent=1))
    res["check6_identities"] = c6

    # ---------------- check 7: regression ----------------
    log("\n=== check 7: regression (earlier overview rows, existing build tables) ===")
    c7 = {}
    man = json.loads((OUT / "manifest_r.json").read_text())
    cur = {f: LG.sha256(OUT / f) for f in man["existing_build_tables_sha256"]}
    c7["existing_tables_vs_manifest_r"] = {f: cur[f] == h for f, h in man["existing_build_tables_sha256"].items()}
    if before is not None:
        before = Path(before)
        for name, path in (("overview", OUT / "measures/overview.csv"), ("len_overview", OUT / "lengths/len_overview.csv")):
            a_ = pd.read_csv(before / path.name, dtype=str, keep_default_na=False)
            b_ = pd.read_csv(path, dtype=str, keep_default_na=False)
            old = b_[b_["column"].isin(a_["column"])].reset_index(drop=True)
            same_cols = list(a_.columns) == list(b_.columns)
            eq = same_cols and len(old) == len(a_) and bool((old.to_numpy() == a_.to_numpy()).all())
            c7[name] = dict(rows_before=len(a_), rows_now=len(b_), new_rows=sorted(set(b_["column"]) - set(a_["column"])),
                            same_columns=same_cols, earlier_rows_identical=eq)
        hf = before / "hashes.txt"
        if hf.exists():
            hb = {}
            for line in hf.read_text().splitlines():
                h, f = line.split(maxsplit=1)
                hb[f.strip()] = h
            # lengths/len_* are the length-sensitivity result tables: they gain the new columns' rows by design
            # (their earlier rows are covered by the len_overview comparison above)
            grow = sorted(f for f in hb if f.startswith("lengths/len_"))
            hb = {f: h for f, h in hb.items() if f not in grow}
            hc = {f: (LG.sha256(OUT / f) if (OUT / f).exists() else None) for f in hb}
            c7["build_hashes_vs_before"] = dict(files=len(hb), identical=sum(hc[f] == h for f, h in hb.items()),
                                                differing=[f for f, h in hb.items() if hc[f] != h],
                                                not_compared_expected_to_grow=grow)
    c7["ok"] = (all(c7["existing_tables_vs_manifest_r"].values())
                and all(c7[k]["earlier_rows_identical"] for k in ("overview", "len_overview") if k in c7)
                and ("build_hashes_vs_before" not in c7 or not c7["build_hashes_vs_before"]["differing"]))
    c7["ran_with_before_dir"] = before is not None
    log(json.dumps(c7, indent=1))
    res["check7_regression"] = c7

    res["all_ok"] = all(v.get("ok", True) for v in res.values() if isinstance(v, dict))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "checks_r.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"\nall_ok {res['all_ok']}; checks done in {res['runtime_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--before", default=None, help="directory with the pre-change overview.csv, len_overview.csv, "
                                                   "hashes.txt (regression check)")
    args = ap.parse_args()
    checks(args.before) if args.checks else build()
