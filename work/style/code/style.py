"""Per-address entry-alignment score from Binance 5-minute bars, with stability of three baselines.

Opening decisions (steps 1-2) follow the "Revised implementable definition" in work/codex6/REVIEW.md:
episode-aware 60-second clusters of opening-leg notional. Prices (steps 3-4) come from 5-minute bar
closes in work/extdata/build/klines_5m. Each decision gets z_h = s * r_h / sigma_h, and three variants
are aggregated per address x sample x class x horizon: raw, A (same-4h-block volatility-matched grid
baseline) and B (coin's own mean over the sample). The stability table compares pre/post and even/odd
week rank correlations across addresses.

Inputs (read-only): lib.prep_perp_fills(), v2dev/build/groups.parquet, extdata/build/klines_5m,
codex6/decisions.parquet (reconciliation only). Outputs: work/style/build/{decisions.parquet,
address_style.csv, stability.csv, summary.json}, log work/style/logs/style.log. No network.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/xc/Documents/arkstream_quest")
STYLE = ROOT / "work/style"
OUT = STYLE / "build"
LOGS = STYLE / "logs"
os.environ["ARK_DATA_DIR"] = str(ROOT / "ArkStream笔试题_v2")
# lib mkdirs its build/tables dirs on import; point both at our own build dir
os.environ["ARK_BUILD_DIR"] = str(OUT)
os.environ["ARK_TABLES_DIR"] = str(OUT)
sys.path.insert(0, str(ROOT / "work/v2dev/code"))
import lib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

KLINES = ROOT / "work/extdata/build/klines_5m"
GROUPS = ROOT / "work/v2dev/build/groups.parquet"
CODEX_DEC = ROOT / "work/codex6/decisions.parquet"

STEP = pd.Timedelta(minutes=5)
G0 = pd.Timestamp("2025-07-01 00:05:00")           # close time of the first bar of any file
G_END = pd.Timestamp("2026-01-05 00:00:00")        # close time of the last bar (open 2026-01-04 23:55)
NGRID = int((G_END - G0) / STEP) + 1
HORIZONS = {"1h": 1, "4h": 4, "1d": 24, "7d": 168}  # hours
WIN = pd.Timestamp("2025-07-27 00:00:00")          # window start, week-index anchor
SPLIT = pd.Timestamp("2025-11-03 00:00:00")
CUTOFF = pd.Timestamp("2026-01-05 00:00:00")
GAP_S = 60
MAX_DUR_S = 300
VOL_BARS = 288
A_MIN = 5
A_LO, A_HI = 2 / 3, 1.5
GATE_N, GATE_DAYS = 30, 10
SEED = 20261119
N_BOOT = 2000
SAMPLES = ["pre", "post", "full", "even", "odd"]
CLASSES = {"taker": "w_taker", "maker": "w_maker", "all": "w"}
VARIANTS = {"raw": "score_raw", "A": "score_A", "B": "score_B"}

_LOG = None


def log(msg):
    print(msg, flush=True)
    _LOG.write(msg + "\n")
    _LOG.flush()


# ---------------------------------------------------------------------------------------
# Steps 1-2: opening decisions
# ---------------------------------------------------------------------------------------
def build_clusters():
    """Episode-aware 60 s opening clusters. Returns (clusters, fill-level counts for the summary)."""
    p = lib.prep_perp_fills()
    g = pd.read_parquet(GROUPS).sort_values(["trader", "symbol", "ts"], kind="stable").reset_index(drop=True)

    # episodes: new trader x symbol pair, flat start, or sign flip
    new_pair = g["trader"].ne(g["trader"].shift()) | g["symbol"].ne(g["symbol"].shift())
    flip = (g["gstart"] * g["gend"]) < 0
    starts_flat_or_flip = g["gstart"].eq(0) | flip
    g["episode"] = (new_pair | starts_flat_or_flip).cumsum()
    # observed opening timestamp of the episode; left-censored episodes (pair starts in a position) have none
    g["ep_open_ts"] = g["ts"].where(starts_flat_or_flip)
    ep_open = g.groupby("episode")["ep_open_ts"].min()

    p = p.merge(g[["trader", "symbol", "ts", "episode", "mixed"]], on=["trader", "symbol", "ts"],
                how="left", validate="many_to_one")
    assert p["episode"].notna().all()

    # opening quantity per fill; a flip contributes only the residual new position
    q = p["start_position"].to_numpy().copy()
    q[np.abs(q) < lib.POS_EPS] = 0.0
    d = p["signed"].to_numpy()
    open_q = np.where(q * d >= 0, np.abs(d), np.maximum(np.abs(d) - np.abs(q), 0.0))
    open_q[open_q < lib.POS_EPS] = 0.0
    p["open_ntl"] = open_q * p["price"].to_numpy()

    # all orders (closing and forced included) take part in clustering
    keys = ["trader", "symbol", "oid"]
    o = p.groupby(keys).agg(first_ts=("ts", "min"), last_ts=("ts", "max"), side=("side", "first"),
                            nside=("side", "nunique"), nep=("episode", "nunique"),
                            episode=("episode", "first")).reset_index()
    o = o.sort_values(["trader", "symbol", "first_ts", "last_ts", "oid"], kind="stable").reset_index(drop=True)
    prev = o.shift()
    single = o["nside"].eq(1) & o["nep"].eq(1)
    join = (o["trader"].eq(prev["trader"]) & o["symbol"].eq(prev["symbol"])
            & single & single.shift(fill_value=False)
            & o["side"].eq(prev["side"]) & o["episode"].eq(prev["episode"])
            & ((o["first_ts"] - prev["last_ts"]).dt.total_seconds() <= GAP_S))
    o["cid"] = (~join).cumsum()
    p = p.merge(o[keys + ["cid", "nep"]], on=keys, how="left", validate="many_to_one")

    opening = p["open_ntl"] > 0
    ambiguous = p["mixed"] | p["nep"].ne(1)
    eligible = opening & ~p["is_forced"] & ~p["is_hip3"] & ~ambiguous
    counts = dict(
        fills=int(len(p)),
        open_ntl_nonforced_all=float(p.loc[opening & ~p["is_forced"], "open_ntl"].sum()),
        open_ntl_forced=float(p.loc[opening & p["is_forced"], "open_ntl"].sum()),
        open_ntl_hip3_nonforced=float(p.loc[opening & ~p["is_forced"] & p["is_hip3"], "open_ntl"].sum()),
        open_rows_ambiguous=int((opening & ~p["is_forced"] & ~p["is_hip3"] & ambiguous).sum()),
        open_ntl_ambiguous=float(p.loc[opening & ~p["is_forced"] & ~p["is_hip3"] & ambiguous, "open_ntl"].sum()),
        open_ntl_ambiguous_mixed_group=float(p.loc[opening & ~p["is_forced"] & ~p["is_hip3"] & p["mixed"],
                                                   "open_ntl"].sum()),
        open_ntl_ambiguous_multi_episode_order=float(p.loc[opening & ~p["is_forced"] & ~p["is_hip3"]
                                                           & p["nep"].ne(1), "open_ntl"].sum()),
        orders=int(len(o)), multi_episode_orders=int(o["nep"].gt(1).sum()), mixed_groups=int(g["mixed"].sum()),
    )

    op = p[eligible].copy()
    op["maker_ntl"] = np.where(op["is_maker"], op["open_ntl"], 0.0)
    op["taker_ntl"] = np.where(op["is_maker"], 0.0, op["open_ntl"])
    c = op.groupby("cid").agg(trader=("trader", "first"), symbol=("symbol", "first"),
                              episode=("episode", "first"), t=("ts", "min"), last=("ts", "max"),
                              s=("sgn", "first"), nsgn=("sgn", "nunique"), w=("open_ntl", "sum"),
                              w_maker=("maker_ntl", "sum"), w_taker=("taker_ntl", "sum"),
                              n_fills=("tid", "size")).reset_index()
    assert c["nsgn"].eq(1).all(), "cluster with both buy and sell opening legs"
    c = c.drop(columns="nsgn")
    c = c[c["w"] > 0].reset_index(drop=True)
    c["is_first"] = c["t"].eq(c["episode"].map(ep_open))
    c["duration_s"] = (c["last"] - c["t"]).dt.total_seconds()
    c = c.sort_values(["trader", "symbol", "t", "cid"], kind="stable").reset_index(drop=True)
    return c, counts


# ---------------------------------------------------------------------------------------
# Steps 3-4: 5-minute grid, lookback returns, volatility scale
# ---------------------------------------------------------------------------------------
def grid_arrays(symbol):
    """On the global close-time grid: r_h, sigma_h, q_h = r_h / sigma_h for every horizon, or None."""
    f = KLINES / f"{symbol}.parquet"
    if not f.exists():
        return None
    k = pd.read_parquet(f, columns=["open_time", "close"])
    idx = ((k["open_time"] + STEP - G0) / STEP).to_numpy()
    assert np.all(idx == np.round(idx)) and idx.min() >= 0 and idx.max() < NGRID
    lp = np.full(NGRID, np.nan)
    lp[idx.astype(np.int64)] = np.log(k["close"].to_numpy())
    ret = np.full(NGRID, np.nan)
    ret[1:] = lp[1:] - lp[:-1]
    # V at grid i: squared 5m returns of bars i-287..i; needs closes i-288..i (NaN propagates otherwise)
    V = np.full(NGRID, np.nan)
    V[VOL_BARS:] = (sliding_window_view(ret[1:], VOL_BARS) ** 2).sum(axis=1)
    n_zero_v = int(np.sum(V == 0))
    V[~(V > 0)] = np.nan                       # zero variance gives no usable scale
    out = {"V": V, "n_zero_v": n_zero_v}
    for h, hh in HORIZONS.items():
        L = hh * 12
        r = np.full(NGRID, np.nan)
        r[L:] = lp[L:] - lp[:-L]
        sig = np.sqrt(V * hh / 24)
        out[f"r_{h}"] = r
        out[f"sigma_{h}"] = sig
        out[f"q_{h}"] = r / sig
    return out


def grid_times():
    return G0 + STEP * np.arange(NGRID)


def sample_masks():
    """Grid-point time sets of each sample (for the B baseline)."""
    c = pd.DatetimeIndex(grid_times())
    full = (c >= WIN) & (c < CUTOFF)
    wk = ((c - WIN) // pd.Timedelta(days=7)).to_numpy()
    return {"pre": (c >= WIN) & (c < SPLIT), "post": (c >= SPLIT) & (c < CUTOFF), "full": full,
            "even": full & (wk % 2 == 0), "odd": full & (wk % 2 == 1)}


def score_decisions(c):
    """Per horizon r, sigma, z, A baseline. Returns (c with columns, B means per symbol x sample x horizon)."""
    c["excl"] = np.where(c["duration_s"] > MAX_DUR_S, "duration_gt_300s", None)
    masks = sample_masks()
    ci = ((c["t"].dt.floor("5min") - G0) / STEP).to_numpy().astype(np.int64)   # c(t): last close <= t
    bi = ((c["t"].dt.floor("4h") - G0) / STEP).to_numpy().astype(np.int64)     # first grid point of 4h block
    assert ci.min() >= 0 and (bi + 47).max() < NGRID
    s = c["s"].to_numpy()
    for h in HORIZONS:
        for col in ("r", "sigma", "z", "srbp", "bA"):
            c[f"{col}_{h}"] = np.nan
        c[f"nA_{h}"] = 0
    no_file = []
    m_rows = []
    n_zero_v = 0
    for sym, ix in c.groupby("symbol", sort=True).indices.items():
        ga = grid_arrays(sym)
        if ga is None:
            no_file.append(sym)
            rows = c.index[ix]
            c.loc[rows[c.loc[rows, "excl"].isna().to_numpy()], "excl"] = "no_kline_file"
            continue
        n_zero_v += ga["n_zero_v"]
        for smp, msk in masks.items():
            for h in HORIZONS:
                qv = ga[f"q_{h}"][msk]
                ok = np.isfinite(qv)
                m_rows.append(dict(symbol=sym, sample=smp, h=h, m=float(qv[ok].mean()) if ok.any() else np.nan,
                                   n_grid=int(ok.sum())))
        i_t, i_b = ci[ix], bi[ix]
        cand = i_b[:, None] + np.arange(48)[None, :]
        for h in HORIZONS:
            r = ga[f"r_{h}"][i_t]
            sig = ga[f"sigma_{h}"][i_t]
            q = ga[f"q_{h}"][i_t]
            cq = ga[f"q_{h}"][cand]
            ratio = ga[f"sigma_{h}"][cand] / sig[:, None]
            ok = np.isfinite(cq) & (ratio >= A_LO) & (ratio <= A_HI)
            nA = ok.sum(axis=1)
            meanA = np.where(ok, cq, 0.0).sum(axis=1) / np.maximum(nA, 1)
            bA = np.where((nA >= A_MIN) & np.isfinite(q), s[ix] * meanA, np.nan)
            rows = c.index[ix]
            c.loc[rows, f"r_{h}"] = r
            c.loc[rows, f"sigma_{h}"] = sig
            c.loc[rows, f"z_{h}"] = s[ix] * q
            c.loc[rows, f"srbp_{h}"] = s[ix] * r * 1e4
            c.loc[rows, f"bA_{h}"] = bA
            c.loc[rows, f"nA_{h}"] = np.where(np.isfinite(q), nA, 0)
    # z and the baseline are reported only for decisions that pass the cluster-level gates
    scored = c["excl"].isna()
    for h in HORIZONS:
        c.loc[~scored, [f"z_{h}", f"bA_{h}"]] = np.nan
        c.loc[~scored, f"nA_{h}"] = 0
        reason = np.where(~scored, "cluster_excluded",
                          np.where(c[f"r_{h}"].isna(), "missing_return",
                                   np.where(c[f"sigma_{h}"].isna(), "missing_sigma", None)))
        c[f"excl_{h}"] = reason
        assert (c[f"z_{h}"].notna() == pd.isna(c[f"excl_{h}"])).all()
    m = pd.DataFrame(m_rows)
    return c, m, sorted(no_file), n_zero_v


# ---------------------------------------------------------------------------------------
# Steps 6-7: samples and aggregation
# ---------------------------------------------------------------------------------------
def decision_samples(c):
    wk = ((c["t"] - WIN) // pd.Timedelta(days=7)).to_numpy()
    return {"pre": (c["t"] < SPLIT).to_numpy(), "post": (c["t"] >= SPLIT).to_numpy(),
            "full": np.ones(len(c), dtype=bool), "even": wk % 2 == 0, "odd": wk % 2 == 1}


def wavg(df, val, wcol, by):
    num = (df[val] * df[wcol]).groupby(df[by]).sum()
    return num / df[wcol].groupby(df[by]).sum()


def aggregate(c, m):
    sc = c[c["excl"].isna()].copy()
    sc["day"] = sc["t"].dt.floor("D")
    smp = decision_samples(sc)
    m_idx = m.set_index(["symbol", "sample", "h"])["m"]
    rows = []
    for sample in SAMPLES:
        ds = sc[smp[sample]]
        for h in HORIZONS:
            d = ds[ds[f"z_{h}"].notna()].copy()
            if d.empty:
                continue
            d["z"] = d[f"z_{h}"]
            d["xA"] = d["z"] - d[f"bA_{h}"]
            mh = m_idx.reindex(pd.MultiIndex.from_arrays([d["symbol"], np.full(len(d), sample),
                                                          np.full(len(d), h)])).to_numpy()
            d["xB"] = d["z"] - d["s"] * mh
            d["srbp"] = d[f"srbp_{h}"]
            d["pos"] = (d["srbp"] > 0).astype(float)
            d["hasA"] = d["xA"].notna()
            assert d["xB"].notna().all()
            for cls, wcol in CLASSES.items():
                e = d[d[wcol] > 0].copy()
                if e.empty:
                    continue
                e["wt"] = e[wcol]
                eA = e[e["hasA"]]
                gt = e.groupby("trader")
                a = pd.DataFrame({
                    "n_dec": gt.size(), "n_days": gt["day"].nunique(), "w_sum": gt["wt"].sum(),
                    "score_raw": wavg(e, "z", "wt", "trader"),
                    "n_A": eA.groupby("trader").size(), "n_days_A": eA.groupby("trader")["day"].nunique(),
                    "w_sum_A": eA.groupby("trader")["wt"].sum(),
                    "score_A": wavg(eA, "xA", "wt", "trader"),
                    "score_B": wavg(e, "xB", "wt", "trader"),
                    "raw_bp": wavg(e, "srbp", "wt", "trader"),
                    "pos_share": gt["pos"].mean(),
                })
                a["n_A"] = a["n_A"].fillna(0).astype(int)
                a["n_days_A"] = a["n_days_A"].fillna(0).astype(int)
                a["w_sum_A"] = a["w_sum_A"].fillna(0.0)
                a["reportable"] = (a["n_dec"] >= GATE_N) & (a["n_days"] >= GATE_DAYS)
                a = a.reset_index()
                a.insert(1, "sample", sample)
                a.insert(2, "cls", cls)
                a.insert(3, "h", h)
                rows.append(a)
    out = pd.concat(rows, ignore_index=True)
    out["sample"] = pd.Categorical(out["sample"], SAMPLES, ordered=True)
    out["cls"] = pd.Categorical(out["cls"], list(CLASSES), ordered=True)
    out["h"] = pd.Categorical(out["h"], list(HORIZONS), ordered=True)
    out = out.sort_values(["trader", "sample", "cls", "h"], kind="stable").reset_index(drop=True)
    for col in ("sample", "cls", "h"):
        out[col] = out[col].astype(str)
    return out


# ---------------------------------------------------------------------------------------
# Step 8: stability
# ---------------------------------------------------------------------------------------
def boot_spearman(x, y, rng):
    """Point Spearman and percentile 95% interval from address resampling."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    ii = rng.integers(0, n, size=(N_BOOT, n))
    rx = rankdata(x[ii], axis=1)
    ry = rankdata(y[ii], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (rx * ry).sum(axis=1) / den
    rb = rb[np.isfinite(rb)]          # a draw of constant values has no rank correlation
    lo, hi = np.percentile(rb, [2.5, 97.5]) if len(rb) else (np.nan, np.nan)
    return rho, float(lo), float(hi)


def stability(a):
    rows = []
    for h in HORIZONS:
        for cls in CLASSES:
            cell = a[(a["h"] == h) & (a["cls"] == cls)]
            for var, col in VARIANTS.items():
                row = dict(h=h, cls=cls, variant=var)
                for tag, (s1, s2) in (("pp", ("pre", "post")), ("eo", ("even", "odd"))):
                    l = cell[(cell["sample"] == s1) & cell["reportable"]].set_index("trader")[col]
                    r = cell[(cell["sample"] == s2) & cell["reportable"]].set_index("trader")[col]
                    j = pd.concat([l.rename("x"), r.rename("y")], axis=1, join="inner").dropna().sort_index()
                    # fresh generator per cell: every cell is reproducible on its own
                    rho, lo, hi = boot_spearman(j["x"].to_numpy(), j["y"].to_numpy(), np.random.default_rng(SEED))
                    row.update({f"{tag}_n": len(j), f"{tag}_rho": rho, f"{tag}_lo": lo, f"{tag}_hi": hi})
                pre = cell[(cell["sample"] == "pre") & cell["reportable"]][col].dropna()
                row.update(pre_n=len(pre), pre_median=pre.median(), pre_p10=pre.quantile(0.1),
                           pre_p90=pre.quantile(0.9), pre_share_neg=float((pre < 0).mean()) if len(pre) else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Reconciliation with codex6
# ---------------------------------------------------------------------------------------
def reconcile(c):
    x = pd.read_parquet(CODEX_DEC)
    mine = c[c["trader"].isin(set(x["trader"]))]
    k = ["trader", "symbol", "t", "last"]
    xm = x.rename(columns={"side": "s"})[k + ["s", "w", "maker_w", "role"]]
    j = mine[k + ["s", "w", "w_maker", "is_first"]].merge(xm, on=k + ["s"], how="outer", indicator=True,
                                                         suffixes=("", "_cx"))
    both = j[j["_merge"] == "both"]
    return dict(
        codex_clusters=int(len(x)), codex_ntl=float(x["w"].sum()), codex_traders=int(x["trader"].nunique()),
        mine_clusters=int(len(mine)), mine_ntl=float(mine["w"].sum()), mine_traders=int(mine["trader"].nunique()),
        mine_clusters_all_traders=int(len(c)), mine_traders_all=int(c["trader"].nunique()),
        matched_on_trader_symbol_t_last_s=int(len(both)),
        only_mine=int((j["_merge"] == "left_only").sum()), only_codex=int((j["_merge"] == "right_only").sum()),
        dup_keys_mine=int(mine.duplicated(k + ["s"]).sum()), dup_keys_codex=int(xm.duplicated(k + ["s"]).sum()),
        max_abs_w_diff=float((both["w"] - both["w_cx"]).abs().max()),
        max_abs_maker_diff=float((both["w_maker"] - both["maker_w"]).abs().max()),
        first_flag_disagree=int((both["is_first"] != both["role"].eq("first")).sum()),
    )


# ---------------------------------------------------------------------------------------
def main():
    global _LOG
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _LOG = open(LOGS / "style.log", "w")
    t0 = time.time()

    c, counts = build_clusters()
    log(f"[{time.time() - t0:.1f}s] clusters {len(c):,}, opening notional {c['w'].sum():,.2f}, "
        f"traders {c['trader'].nunique()}, symbols {c['symbol'].nunique()}")
    rec = reconcile(c)
    log("reconciliation vs codex6: " + json.dumps(rec))

    c, m, no_file, n_zero_v = score_decisions(c)
    log(f"[{time.time() - t0:.1f}s] scored; symbols without kline file {len(no_file)}; zero-variance grid points {n_zero_v}")

    keep = ["trader", "symbol", "cid", "episode", "t", "last", "duration_s", "s", "w", "w_maker", "w_taker",
            "n_fills", "is_first", "excl"]
    for h in HORIZONS:
        keep += [f"r_{h}", f"sigma_{h}", f"z_{h}", f"srbp_{h}", f"bA_{h}", f"nA_{h}", f"excl_{h}"]
    c[keep].to_parquet(OUT / "decisions.parquet", index=False)

    a = aggregate(c, m)
    a.to_csv(OUT / "address_style.csv", index=False)
    log(f"[{time.time() - t0:.1f}s] address_style rows {len(a):,}")

    st = stability(a)
    st.to_csv(OUT / "stability.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 100):
        log(st.round(3).to_string(index=False))

    # summary
    cand_ntl = float(c["w"].sum())
    excl = {}
    for reason, d in c.groupby(c["excl"].fillna("none")):
        excl[reason] = dict(n=int(len(d)), ntl=float(d["w"].sum()))
    both_excl = c[(c["duration_s"] > MAX_DUR_S) & ~c["symbol"].map(lambda s: (KLINES / f"{s}.parquet").exists())]
    per_h = {}
    for h in HORIZONS:
        r = c[f"excl_{h}"].fillna("scored")
        per_h[h] = {k: dict(n=int((r == k).sum()), ntl=float(c.loc[r == k, "w"].sum()),
                            ntl_share_of_candidate=float(c.loc[r == k, "w"].sum() / cand_ntl))
                    for k in ["scored", "cluster_excluded", "missing_return", "missing_sigma"]}
        hasA = c[f"bA_{h}"].notna()
        per_h[h]["scored_with_A"] = dict(n=int(hasA.sum()), ntl=float(c.loc[hasA, "w"].sum()),
                                         ntl_share_of_candidate=float(c.loc[hasA, "w"].sum() / cand_ntl))
        per_h[h]["scored_ntl_share_incl_ambiguous"] = float(per_h[h]["scored"]["ntl"]
                                                            / (cand_ntl + counts["open_ntl_ambiguous"]))
    rep = a[a["reportable"]].groupby(["sample", "cls", "h"]).size()
    summary = dict(
        fill_level=counts,
        clusters_total=int(len(c)), clusters_ntl=cand_ntl,
        clusters_first=int(c["is_first"].sum()),
        cluster_exclusions=excl,
        clusters_dur_gt_300s_and_no_kline_file=dict(n=int(len(both_excl)), ntl=float(both_excl["w"].sum())),
        symbols_with_clusters=int(c["symbol"].nunique()), symbols_without_kline_file=no_file,
        zero_variance_grid_points=n_zero_v,
        per_horizon=per_h,
        reportable_cells={f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in rep.items()},
        reconciliation_codex6=rec,
        runtime_s=round(time.time() - t0, 1),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
