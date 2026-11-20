"""Walk-forward version of the direction label: does a portfolio formed on a rolling label keep its exposure?  (Not a return test.)

Every Monday T the label (direction.scores / classify, >= MIN_W in-market days) is recomputed on the trader-days with
T - x <= day < T (x in X_DAYS) of the full build; a K-address portfolio is formed by a rule on the scaled units and its
exposure over T <= day < T + H is measured.  The replay is causal, so filtering the full build by day is the point-in-time view.
Scale s_i (lookback in-market days): vol = sd (ddof=1) of daily PnL, gross = mean daily gross; s_i missing or <= 0 -> dropped.
Rules (weight 1/K each): random, long_only, short_only, balanced (K/2 + K/2), N_DRAWS draws each; neutral_top = the K smallest |A|.
Readings known at T: rd_corr (lookback Spearman with BTC), rd_ng (lookback net/gross), rd_last (net/gross on day T-1).
Outcome: fw_ng (forward-week sum net / sum gross: a signed weekly aggregate, opposite days cancel), fw_dmax (largest
|net/gross| over the H forward daily closes; fw_dmax < NB = every daily close inside the band), idle (share of the K with
no forward row), forward daily PnL (stitched in S2).
S3 also gives a PnL-based outcome per reading bucket: `pooled_fw_pnl_corr` = Spearman correlation, pooled over every
(portfolio, forward day) of the bucket, between the portfolio's forward daily PnL signed by sign(reading) and the BTC return;
`pooled_fw_pnl_corr_signlast` is the same signed by sign(rd_last) for every reading (same sign convention across readings).

Output: direction_wf.csv (section, x, scheme, dates, rule, metric, stat, value).
Run   : <venv python> -W ignore direction_wf.py > direction_wf.log 2>&1
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from direction import rows, trader_day, btc_close, scores, classify, FULL, OUT
from direction_balance import part_data, metrics, colcorr

X_DAYS, H, K, N_DRAWS, MIN_W, SEED, NB = (56, 84), 7, 20, 500, 20, 20251103, 0.3
CUT_CORR, CUT_LAST = 0.2, 0.2   # joint table in S3; picked after seeing the single-reading buckets (in-sample choice)
FIRST, END = pd.Timestamp("2025-07-28"), pd.Timestamp("2026-01-05")
SCHEMES, RULES = ("vol", "gross"), ("random", "long_only", "short_only", "balanced", "neutral_top")
CLASSES, D1 = ["short-like", "neutral", "long-like"], pd.Timedelta(days=1)
np.seterr(divide="ignore", invalid="ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def load():
    td = trader_day(rows(FULL))
    ret = btc_close(FULL).pct_change().dropna()
    ret.index = ret.index - D1  # label = start of the UTC PnL day
    ret = ret.rename("btc").rename_axis("day")
    return td.merge(ret.reset_index(), on="day", how="inner"), ret


def decision_dates(x):
    return [T for T in pd.date_range(FIRST, END, freq="W-MON")
            if T - pd.Timedelta(days=x) >= FIRST and T + pd.Timedelta(days=H) <= END]


def window(td, ret, T, x):
    lo, hi = T - pd.Timedelta(days=x), T + pd.Timedelta(days=H)
    L, Fw = td[(td["day"] >= lo) & (td["day"] < T)], td[(td["day"] >= T) & (td["day"] < hi)]
    rL, rF = ret[(ret.index >= lo) & (ret.index < T)], ret[(ret.index >= T) & (ret.index < hi)]
    assert len(rL) == x and len(rF) == H
    # self-check 2: no lookback row at or after T, no forward row outside [T, T + H)
    assert (L["day"] < T).all() and (L["day"] >= lo).all() and (Fw["day"] >= T).all() and (Fw["day"] < hi).all()
    S = scores(L, MIN_W)
    S = S[S["eligible"]]
    assert S["A"].notna().all()
    return L, Fw, rL, rF, S.assign(cls=classify(S["A"]))


def pick(rng, pool, k, n):
    """N_DRAWS rows, each with weight 1/K on k distinct addresses drawn from pool."""
    picks = pool[np.argsort(rng.random((N_DRAWS, len(pool))), axis=1)[:, :k]]
    W = np.zeros((N_DRAWS, n))
    W[np.arange(N_DRAWS)[:, None], picks] = 1 / K
    return W


def slow_check(L, Fw, rL, rF, T, s, mem, got, scheme):
    """Self-check 1: one portfolio's readings and fw_ng in plain pandas (no part_data / metrics)."""
    def part(td, r):
        d = td[td["trader"].isin(mem)].copy()
        for c in ("pnl", "net", "gross"):
            d[c] = d[c] / d["trader"].map(s) / K
        return d, d.groupby("day")["pnl"].sum().reindex(r.index, fill_value=0.0)
    (dL, pL), (dF, _) = part(L, rL), part(Fw, rF)
    last = dL[dL["day"] == T - D1]
    want = dict(rd_corr=pL.corr(rL, method="spearman"), rd_ng=dL["net"].sum() / dL["gross"].sum(),
                rd_last=last["net"].sum() / last["gross"].sum() if last["gross"].sum() > 0 else np.nan,
                fw_ng=dF["net"].sum() / dF["gross"].sum() if dF["gross"].sum() > 0 else np.nan)
    for k, v in want.items():
        assert np.isclose(v, got[k], rtol=1e-9, atol=1e-12, equal_nan=True), (k, v, got[k])
    print(f"  self-check 1 OK ({scheme}, {T.date()}, balanced draw 0): " + ", ".join(f"{k} {v:.6f}" for k, v in want.items()))


def run_date(rng, L, Fw, rL, rF, S, T, scheme, skips, check):
    g = L[L["trader"].isin(S.index)].groupby("trader")
    s = (g["pnl"].std(ddof=1) if scheme == "vol" else g["gross"].mean()).reindex(S.index)
    s = s[s > 0]
    n, lab = len(s), S["cls"].reindex(s.index).values
    DL, DF = part_data(L, rL, s), part_data(Fw, rF, s)
    last = L[L["day"] == T - D1].set_index("trader").reindex(s.index).fillna(0.0)
    nl, gl = last["net"].values / s.values, last["gross"].values / s.values
    no_fw = (~s.index.isin(Fw["trader"])).astype(float)   # float: a boolean matmul would give "any", not a count
    piv = lambda c: Fw.pivot(index="day", columns="trader", values=c).reindex(index=rF.index, columns=s.index).fillna(0.0).values / s.values
    NF, GF = piv("net"), piv("gross")                     # H x addresses, forward daily net / gross in scaled units
    assert np.allclose(NF.sum(0), DF["Nsum"]) and np.allclose(GF.sum(0), DF["Gsum"])
    iL, iS = np.flatnonzero(lab == "long-like"), np.flatnonzero(lab == "short-like")
    Ws = {}
    for rule, pool in (("random", np.arange(n)), ("long_only", iL), ("short_only", iS)):
        if len(pool) >= K: Ws[rule] = pick(rng, pool, K, n)
        else: skips[rule] += 1
    if min(len(iL), len(iS)) >= K // 2:
        Ws["balanced"] = pick(rng, iL, K // 2, n) + pick(rng, iS, K // 2, n)
        # self-check 3 (balanced part): exactly K/2 from each class
        assert ((Ws["balanced"][:, iL] > 0).sum(1) == K // 2).all() and ((Ws["balanced"][:, iS] > 0).sum(1) == K // 2).all()
    else: skips["balanced"] += 1
    assert n >= K
    top = pd.DataFrame({"a": S["A"].reindex(s.index).abs().values, "t": s.index}).sort_values(["a", "t"]).index[:K]
    Ws["neutral_top"] = np.zeros((1, n))
    Ws["neutral_top"][0, top] = 1 / K
    out = {}
    for rule, W in Ws.items():
        assert ((W > 0).sum(1) == K).all()  # self-check 3: exactly K distinct addresses per draw
        ML, MF = metrics(W, DL), metrics(W, DF)
        den = W @ gl
        dmax = np.nanmax(np.abs((NF @ W.T) / (GF @ W.T)), axis=0)   # largest |net/gross| over the forward daily closes
        out[rule] = dict(rd_corr=ML["corr_s"], rd_ng=ML["net_gross"], rd_last=np.where(den > 0, (W @ nl) / den, np.nan),
                         fw_ng=MF["net_gross"], fw_dmax=dmax, idle=(W > 0) @ no_fw / K, P=DF["X"] @ W.T, r=DF["r"],
                         mem=frozenset(s.index[W[0] > 0]))
    if check and "balanced" in out:
        slow_check(L, Fw, rL, rF, T, s, out["balanced"]["mem"], {k: out["balanced"][k][0] for k in ("rd_corr", "rd_ng", "rd_last", "fw_ng")}, scheme)
    return n, out


rec = []
def put(sec, x, scheme, dates, rule, metric, stat, value):
    rec.append((sec, x, scheme, dates, rule, metric, stat, float(value)))


def show(sec, index, cond=None):
    R = pd.DataFrame(rec, columns=["section", "x", "scheme", "dates", "rule", "metric", "stat", "value"])
    R = R[R["section"] == sec] if cond is None else R[(R["section"] == sec) & cond(R)]
    cols = [c for c in ("metric", "stat") if c not in index]
    t = R.pivot_table(index=index, columns=cols, values="value", aggfunc="first", sort=False)
    if len(cols) == 2: t.columns = [f"{m}:{st}" for m, st in t.columns]
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    td, ret = load()
    res, univ, skips, labels, dsets = {}, {}, {}, {}, {}
    print("=== setup ===")
    for x in X_DAYS:
        ds = decision_dates(x)
        assert len(ds) == {56: 15, 84: 11}[x]
        print(f"x={x}: {len(ds)} decision dates {ds[0].date()} .. {ds[-1].date()}")
        win = {T: window(td, ret, T, x) for T in ds}
        labels[x] = {T: w[4]["cls"] for T, w in win.items()}
        for scheme in SCHEMES:
            rng, skips[x, scheme] = np.random.default_rng(SEED), dict.fromkeys(RULES, 0)
            for i, T in enumerate(ds):
                univ[x, scheme, T], res[x, scheme, T] = run_date(rng, *win[T], T, scheme, skips[x, scheme], check=(i == 0))
        tab = pd.DataFrame({T: {"eligible": len(labels[x][T]), **labels[x][T].value_counts().reindex(CLASSES, fill_value=0),
                                **{f"universe_{sc}": univ[x, sc, T] for sc in SCHEMES}} for T in ds}).T
        tab.index = [T.date() for T in ds]
        print(tab.to_string())
        put("setup", x, "", "all", "", "n_dates", "count", len(ds))
        for c in tab.columns:
            sc = c.split("_")[1] if c.startswith("universe_") else ""
            for st in ("mean", "min", "max"):
                put("setup", x, sc, "all", "", "universe_size" if sc else c, st, getattr(tab[c].astype(float), st)())
        for sc in SCHEMES:
            print(f"  rule skips, scheme {sc}: {skips[x, sc]}")
            for rule, v in skips[x, sc].items():
                put("setup", x, sc, "all", rule, "skips", "count", v)
        dsets[x] = {"all": ds} if x == 84 else {"all": ds, "common": decision_dates(84)}

    for x in X_DAYS:
        for scheme in SCHEMES:
            for dn, ds in dsets[x].items():
                for rule in RULES:
                    rs = [res[x, scheme, T][rule] for T in ds if rule in res[x, scheme, T]]
                    if not rs: continue
                    cat = lambda k: np.concatenate([r[k] for r in rs])
                    f = cat("fw_ng"); a, fin = np.abs(f), np.isfinite(f)
                    p = lambda m, st, v: put("S1", x, scheme, dn, rule, m, st, v)
                    p("n", "count", len(f)); p("n_dates", "count", len(rs))
                    for st, q in (("median", .5), ("p25", .25), ("p75", .75)): p("abs_fw_ng", st, np.nanquantile(a, q))
                    p("abs_fw_ng", "share_lt_NB", (a[fin] < NB).mean()); p("fw_ng", "median", np.nanmedian(f))
                    p("fw_ng", "share_nan", 1 - fin.mean()); p("idle", "mean", cat("idle").mean())
                    dm = cat("fw_dmax")
                    p("fw_dmax", "median", np.nanmedian(dm)); p("fw_dmax", "share_lt_NB", (dm[np.isfinite(dm)] < NB).mean())
                    p("abs_rd_ng", "median", np.nanmedian(np.abs(cat("rd_ng")))); p("abs_rd_corr", "median", np.nanmedian(np.abs(cat("rd_corr"))))
                    # S2: path j = draw j at every date, forward daily PnL stitched in time order
                    c = colcorr(rankdata(np.vstack([r["P"] for r in rs]), axis=0), rankdata(np.concatenate([r["r"] for r in rs])))
                    c = c[np.isfinite(c)]
                    p = lambda m, st, v: put("S2", x, scheme, dn, rule, m, st, v)
                    p("n_dates", "count", len(rs)); p("n_paths", "count", len(c)); p("abs_corr", "median", np.median(np.abs(c)))
                    for st, q in (("p25", .25), ("p75", .75)): p("abs_corr", st, np.quantile(np.abs(c), q))
                    p("corr", "median", np.median(c)); p("abs_corr", "share_lt_NB", (np.abs(c) < NB).mean())
            # S3: does a reading known at T predict the forward exposure? (all dates)
            for rule in ("random", "balanced"):
                rs = [res[x, scheme, T][rule] for T in dsets[x]["all"] if rule in res[x, scheme, T]]
                f, dm = np.concatenate([r["fw_ng"] for r in rs]), np.concatenate([r["fw_dmax"] for r in rs])
                vc, vl = np.concatenate([r["rd_corr"] for r in rs]), np.concatenate([r["rd_last"] for r in rs])
                Pm = np.hstack([r["P"] for r in rs])                                       # H x portfolios, forward daily PnL
                Rm = np.hstack([np.repeat(r["r"][:, None], r["P"].shape[1], axis=1) for r in rs])   # same shape, BTC return
                for rd in ("rd_corr", "rd_ng", "rd_last"):
                    v = np.concatenate([r[rd] for r in rs])
                    ok = np.isfinite(v) & np.isfinite(f)
                    v, g, dx = v[ok], f[ok], dm[ok]
                    Ps, Rs = np.sign(v) * Pm[:, ok], Rm[:, ok]                             # PnL signed by the reading's direction
                    Pl = np.sign(vl[ok]) * Pm[:, ok]                                       # PnL signed by rd_last (NaN where rd_last is)
                    put("S3", x, scheme, "all", rule, rd, "n_both_finite", ok.sum())
                    put("S3", x, scheme, "all", rule, rd, "spearman_signed", pd.Series(v).corr(pd.Series(g), method="spearman"))
                    put("S3", x, scheme, "all", rule, rd, "spearman_abs", pd.Series(np.abs(v)).corr(pd.Series(np.abs(g)), method="spearman"))
                    put("S3", x, scheme, "all", rule, rd, "spearman_abs_dmax", pd.Series(np.abs(v)).corr(pd.Series(dx), method="spearman"))
                    b = pd.cut(np.abs(v), [0, .1, .2, .3, .5, np.inf], right=False)
                    B = pd.DataFrame({"b": b, "a": np.abs(g), "dm": dx, "same": np.sign(v) == np.sign(g)}).groupby("b", observed=False)
                    for lab, grp in B:
                        m = f"{rd} |.| in {lab}"
                        # PnL-based outcome: pooled over (portfolio, forward day) of the bucket, Spearman of reading-signed PnL with BTC
                        ix = grp.index.values
                        put("S3", x, scheme, "all", rule, m, "pooled_fw_pnl_corr", pd.Series(Ps[:, ix].ravel()).corr(
                            pd.Series(Rs[:, ix].ravel()), method="spearman") if len(ix) else np.nan)
                        put("S3", x, scheme, "all", rule, m, "pooled_fw_pnl_corr_signlast", pd.Series(Pl[:, ix].ravel()).corr(
                            pd.Series(Rs[:, ix].ravel()), method="spearman") if len(ix) else np.nan)
                        put("S3", x, scheme, "all", rule, m, "n", len(grp))
                        put("S3", x, scheme, "all", rule, m, "median_fw_dmax", grp["dm"].median())
                        put("S3", x, scheme, "all", rule, m, "share_fw_dmax_lt_NB", (grp["dm"] < NB).mean())
                        put("S3", x, scheme, "all", rule, m, "median_abs_fw_ng", grp["a"].median())
                        put("S3", x, scheme, "all", rule, m, "share_abs_fw_ng_lt_NB", (grp["a"] < NB).mean())
                        put("S3", x, scheme, "all", rule, m, "share_same_sign", grp["same"].mean())
                # S3 joint: both readings against one cut each (CUT_CORR, CUT_LAST were chosen AFTER looking at the buckets above)
                ok = np.isfinite(vc) & np.isfinite(vl) & np.isfinite(f)
                for lc in (True, False):
                    for ll in (True, False):
                        ix = np.flatnonzero(ok & ((np.abs(vc) < CUT_CORR) == lc) & ((np.abs(vl) < CUT_LAST) == ll))
                        m = f"joint |rd_corr|{'<' if lc else '>='}{CUT_CORR} & |rd_last|{'<' if ll else '>='}{CUT_LAST}"
                        put("S3", x, scheme, "all", rule, m, "pooled_fw_pnl_corr", pd.Series((np.sign(vl[ix]) * Pm[:, ix]).ravel()).corr(
                            pd.Series(Rm[:, ix].ravel()), method="spearman") if len(ix) else np.nan)   # signed by rd_last
                        put("S3", x, scheme, "all", rule, m, "n", len(ix))
                        put("S3", x, scheme, "all", rule, m, "median_fw_dmax", np.median(dm[ix]) if len(ix) else np.nan)
                        put("S3", x, scheme, "all", rule, m, "share_fw_dmax_lt_NB", (dm[ix] < NB).mean() if len(ix) else np.nan)
                        put("S3", x, scheme, "all", rule, m, "median_abs_fw_ng", np.median(np.abs(f[ix])) if len(ix) else np.nan)
                        put("S3", x, scheme, "all", rule, m, "share_abs_fw_ng_lt_NB", (np.abs(f[ix]) < NB).mean() if len(ix) else np.nan)
            # S5: turnover of neutral_top over consecutive dates
            ds = dsets[x]["all"]
            to = np.array([1 - len(res[x, scheme, a]["neutral_top"]["mem"] & res[x, scheme, b]["neutral_top"]["mem"]) / K
                           for a, b in zip(ds[:-1], ds[1:])])
            for st in ("mean", "min", "max"): put("S5", x, scheme, "all", "neutral_top", "share_replaced", st, getattr(to, st)())
            put("S5", x, scheme, "all", "neutral_top", "share_replaced", "n_pairs", len(to))

        # S4: label churn (eligibility from scores only, independent of scheme)
        ds = dsets[x]["all"]
        for kind, pairs in (("consecutive", list(zip(ds[:-1], ds[1:]))),
                            ("nonoverlap", [(T, T + pd.Timedelta(days=x)) for T in ds if T + pd.Timedelta(days=x) in ds])):
            put("S4", x, "", "all", "", kind, "n_pairs", len(pairs))
            for c in ["all"] + CLASSES:
                rows_ = []
                for a, b in pairs:
                    A_ = labels[x][a] if c == "all" else labels[x][a][labels[x][a] == c]
                    nb = labels[x][b].reindex(A_.index)
                    rows_.append(dict(n=len(A_), not_eligible=nb.isna().mean(), same_class=(nb == A_).mean(),
                                      different_class=(nb.notna() & (nb != A_)).mean()))
                D = pd.DataFrame(rows_, columns=["n", "not_eligible", "same_class", "different_class"])
                for col in D.columns:
                    put("S4", x, "", "all", "", f"{kind}:{c}", f"{col}_mean", D[col].mean())
                if c == "all":
                    put("S4", x, "", "all", "", f"{kind}:{c}", "same_class_min", D["same_class"].min())
                    put("S4", x, "", "all", "", f"{kind}:{c}", "same_class_max", D["same_class"].max())

    for sc in SCHEMES:
        print(f"\n=== S1 forward exposure, scheme {sc} (pooled date x draw; NB = {NB}; share_lt_NB over finite fw_ng) ===")
        show("S1", ["x", "dates", "rule"], lambda R: R["scheme"] == sc)
        print(f"\n=== S2 stitched forward correlation with BTC, scheme {sc} (over paths) ===")
        show("S2", ["x", "dates", "rule"], lambda R: R["scheme"] == sc)
        for x in X_DAYS:
            print(f"\n=== S3 reading at T vs forward fw_ng, x={x}, scheme {sc} (all dates) ===")
            show("S3", ["rule", "metric"], lambda R: (R["scheme"] == sc) & (R["x"] == x))
    print("\n=== S4 label churn (averaged over date pairs) ===")
    show("S4", ["x", "metric"])
    print("\n=== S5 neutral_top turnover (share of K replaced, consecutive dates) ===")
    show("S5", ["x", "scheme"])
    pd.DataFrame(rec, columns=["section", "x", "scheme", "dates", "rule", "metric", "stat", "value"]).to_csv(
        f"{OUT}/direction_wf.csv", index=False)


if __name__ == "__main__":
    main()
