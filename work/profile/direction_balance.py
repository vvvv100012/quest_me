"""Does grouping by the FORMATION direction class reduce a portfolio's market exposure after the split?  (Not a return test.)

Universe: every formation-eligible address (direction.scores on formation data, class from A).  Addresses with no
validation rows stay in and contribute zeros.  Each address is scaled by a formation-only s_i:
  vol   : sd (ddof=1) of daily PnL over its formation in-market days (primary)
  gross : mean daily gross exposure over its formation in-market days (s_i == 0 excluded)
  vol_oracle : sd of daily PnL over its VALIDATION in-market days.  LOOK-AHEAD, diagnostic only: it shows how much of the
               validation result is due to position scale changing after the split (addresses without it are excluded)
Also printed: the ratio of validation to formation daily-PnL sd per address (all return days, zero-filled), and for the
draws `top1_var_share` = the largest single address's share of sum_i w_i^2 var(X_i) in the part.
Day x address matrices over all return days of a part (zero-filled): X = pnl/s, N = net/s, G = gross/s.
Portfolio metrics for non-negative weights w on the scaled units, per part:
  corr_s / corr_p : Spearman / Pearson of P_t = sum_i w_i X_it with the BTC return
  net_gross       : sum w N / sum w G
  beta_offset     : beta(P) / sum_i w_i |beta_i|   (OLS slopes with intercept on the BTC return)
Portfolios: whole-group (all, long_only, short_only, neutral_only, balanced_count, balanced_beta) and N_DRAWS random
draws of k addresses (random, the same per class, balanced_count, balanced_beta on the same draws as balanced_count).
balanced_beta multiplies the short side by c = -beta_F(long side mean) / beta_F(short side mean), formation slopes;
c not finite or <= 0 -> 1 (flagged), clipped to [0.2, 5] (flagged).

Output: direction_balance.csv (kind, scheme, k, portfolio, part, metric, stat, value).
Run   : <venv python> -W ignore direction_balance.py
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from direction import load, scores, classify, SPLIT, MIN_F, OUT

SEED, N_DRAWS, KS, C_LO, C_HI = 20251103, 2000, (10, 20), 0.2, 5.0
PARTS = ("formation", "validation")
np.seterr(divide="ignore", invalid="ignore")


def colcorr(P, r):
    """Pearson correlation of every column of P with the vector r (NaN for a constant column)."""
    Pc, rc = P - P.mean(0), r - r.mean()
    return (rc @ Pc) / np.sqrt((Pc ** 2).sum(0) * (rc ** 2).sum())


def part_data(td, r, s):
    piv = lambda c: td.pivot(index="day", columns="trader", values=c).reindex(index=r.index, columns=s.index).fillna(0.0).values / s.values
    X, N, G = piv("pnl"), piv("net"), piv("gross")
    rv = r.values
    rc = rv - rv.mean()
    beta = (rc @ (X - X.mean(0))) / (rc @ rc)
    return dict(X=X, Nsum=N.sum(0), Gsum=G.sum(0), r=rv, beta=beta, var=X.var(0, ddof=1))


def metrics(W, d):
    """W: draws x addresses weight matrix -> dict of metric arrays (one value per row of W)."""
    P = d["X"] @ W.T
    return dict(corr_s=colcorr(rankdata(P, axis=0), rankdata(d["r"])), corr_p=colcorr(P, d["r"]),
                net_gross=(W @ d["Nsum"]) / (W @ d["Gsum"]), beta_offset=(W @ d["beta"]) / (W @ np.abs(d["beta"])))


def c_factor(bL, bS):
    c = -np.asarray(bL, float) / np.asarray(bS, float)
    fb = ~np.isfinite(c) | (c <= 0)
    c = np.where(fb, 1.0, c)
    cl = (c < C_LO) | (c > C_HI)
    return np.clip(c, C_LO, C_HI), fb, cl


def side_w(mask):
    return mask / mask.sum()


def pick_w(rng, pool, k, n, total=1.0):
    """N_DRAWS rows, each with weight total/k on k distinct addresses drawn from pool."""
    picks = pool[np.argsort(rng.random((N_DRAWS, len(pool))), axis=1)[:, :k]]
    W = np.zeros((N_DRAWS, n))
    W[np.arange(N_DRAWS)[:, None], picks] = total / k
    return W


def main():
    tdF, tdV, ret = load()
    F = scores(tdF, MIN_F)
    F = F[F["eligible"]]
    lab_all = classify(F["A"])
    rF, rV = ret[ret.index < SPLIT], ret[ret.index >= SPLIT]
    print(f"universe (formation-eligible) {len(F)}; with zero validation rows {(~F.index.isin(tdV['trader'])).sum()}")
    g = tdF[tdF["trader"].isin(F.index)].groupby("trader")
    scale = {"vol": g["pnl"].std(ddof=1), "gross": g["gross"].mean(),
             "vol_oracle": tdV[tdV["trader"].isin(F.index)].groupby("trader")["pnl"].std(ddof=1)}  # look-ahead, diagnostic
    out = []
    # how much does an address's PnL scale change across the split?  (all return days of each part, zero-filled)
    sd = [td.pivot(index="day", columns="trader", values="pnl").reindex(index=r.index, columns=F.index).fillna(0.0).std(ddof=1)
          for td, r in ((tdF, rF), (tdV, rV))]
    ratio = sd[1] / sd[0]          # all of the universe; an address that stopped trading has ratio 0
    qs = {"p05": .05, "p25": .25, "median": .5, "p75": .75, "p95": .95}
    out += [("scale", "vol", np.nan, "universe", "validation/formation", "sd_ratio", st, ratio.quantile(q)) for st, q in qs.items()]
    out += [("scale", "vol", np.nan, "universe", "validation/formation", "sd_ratio", "n_zero_validation_sd", float((sd[1] == 0).sum())),
            ("scale", "vol", np.nan, "universe", "validation/formation", "sd", "spearman", sd[0].rank().corr(sd[1].rank()))]
    print("validation sd / formation sd of daily PnL per address: " + ", ".join(f"{st} {ratio.quantile(q):.3f}" for st, q in qs.items())
          + f"; zero validation sd {(sd[1] == 0).sum()}; Spearman of the two sds {sd[0].rank().corr(sd[1].rank()):.3f}")
    for scheme, s in scale.items():
        s = s.reindex(F.index)
        print(f"\n######## scheme {scheme}: excluded (s_i missing or 0): {(~(s > 0)).sum()}; s_i min/median {s.min() / s.median():.2e},"
              f" p01/median {s.quantile(0.01) / s.median():.3f}, max/median {s.max() / s.median():.1f}")
        s = s[s > 0]
        lab = lab_all.reindex(s.index).values
        n = len(s)
        D = {"formation": part_data(tdF, rF, s), "validation": part_data(tdV, rV, s)}
        for p in PARTS:
            ab = np.sort(np.abs(D[p]["beta"]))[::-1]
            print(f"  {p}: top-5 addresses' share of sum_i |beta_i| {ab[:5].sum() / ab.sum():.3f}")
        L, S, Nn = lab == "long-like", lab == "short-like", lab == "neutral"
        bF = D["formation"]["beta"]

        # whole-group portfolios
        c, fb, cl = c_factor(side_w(L) @ bF, side_w(S) @ bF)
        print(f"  whole-group balanced_beta: c = {float(c):.3f} (fallback {bool(fb)}, clipped {bool(cl)})")
        out += [("whole", scheme, np.nan, "balanced_beta", "formation", m, "value", float(v))
                for m, v in (("c", c), ("c_fallback", fb), ("c_clipped", cl))]
        Wg = {"all": np.full(n, 1 / n), "long_only": side_w(L), "short_only": side_w(S), "neutral_only": side_w(Nn),
              "balanced_count": 0.5 * side_w(L) + 0.5 * side_w(S), "balanced_beta": 0.5 * side_w(L) + 0.5 * float(c) * side_w(S)}
        for name, w in Wg.items():
            for p in PARTS:
                out += [("whole", scheme, np.nan, name, p, m, "value", float(v[0])) for m, v in metrics(w[None, :], D[p]).items()]

        # random draws
        rng = np.random.default_rng(SEED)
        iU, iL, iS, iN = np.arange(n), np.flatnonzero(L), np.flatnonzero(S), np.flatnonzero(Nn)
        for k in KS:
            Wd = {"random": pick_w(rng, iU, k, n), "long_only": pick_w(rng, iL, k, n),
                  "short_only": pick_w(rng, iS, k, n), "neutral_only": pick_w(rng, iN, k, n)}
            WL, WS = pick_w(rng, iL, k // 2, n, 0.5), pick_w(rng, iS, k // 2, n, 0.5)
            c, fb, cl = c_factor(WL @ bF, WS @ bF)
            Wd["balanced_count"], Wd["balanced_beta"] = WL + WS, WL + c[:, None] * WS
            out += [("draws", scheme, k, "balanced_beta", "formation", "c", "share_fallback", fb.mean()),
                    ("draws", scheme, k, "balanced_beta", "formation", "c", "share_clipped", cl.mean()),
                    ("draws", scheme, k, "balanced_beta", "formation", "c", "median", np.median(c))]
            for name, W in Wd.items():
                for p in PARTS:
                    M = metrics(W, D[p])
                    for m in ("corr_s", "net_gross", "beta_offset"):
                        a = np.abs(M[m])
                        out += [("draws", scheme, k, name, p, f"abs_{m}", st, np.nanquantile(a, q))
                                for st, q in (("median", .5), ("p05", .05), ("p25", .25), ("p75", .75), ("p95", .95))]
                    wv = W ** 2 * D[p]["var"]
                    out.append(("draws", scheme, k, name, p, "top1_var_share", "median", np.nanmedian(wv.max(1) / wv.sum(1))))
                    cs = M["corr_s"][np.isfinite(M["corr_s"])]
                    out += [("draws", scheme, k, name, p, "corr_s", "median", np.median(cs)),
                            ("draws", scheme, k, name, p, "abs_corr_s", "share_lt_0.3", (np.abs(cs) < 0.3).mean()),
                            ("draws", scheme, k, name, p, "corr_s", "n_undefined", float(N_DRAWS - len(cs)))]

    R = pd.DataFrame(out, columns=["kind", "scheme", "k", "portfolio", "part", "metric", "stat", "value"])
    R.to_csv(f"{OUT}/direction_balance.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    fmt = lambda v: f"{v:.3f}"
    for scheme in scale:
        print(f"\n=== whole-group portfolios, scheme {scheme} (formation = in-sample) ===")
        w = R[(R.kind == "whole") & (R.scheme == scheme) & (R.metric != "c") & ~R.metric.str.startswith("c_")]
        t = w.pivot_table(index="portfolio", columns=["metric", "part"], values="value", sort=False)
        print(t[["corr_s", "corr_p", "net_gross", "beta_offset"]].to_string(float_format=fmt))
        for k in KS:
            print(f"\n=== draws, scheme {scheme}, k = {k} ({N_DRAWS} draws) ===")
            d = R[(R.kind == "draws") & (R.scheme == scheme) & (R.k == k) & (R.metric != "c")]
            t = d.pivot_table(index=["portfolio", "part"], columns=["metric", "stat"], values="value", sort=False)
            cols = [(m, st) for m in ("abs_corr_s", "abs_net_gross", "abs_beta_offset")
                    for st in ("median", "p05", "p25", "p75", "p95")] + [("corr_s", "median"), ("abs_corr_s", "share_lt_0.3"),
                                                                         ("top1_var_share", "median"), ("corr_s", "n_undefined")]
            t = t[cols]
            t.columns = [f"{m}:{st}" for m, st in cols]
            print(t.to_string(float_format=fmt))
            c = R[(R.kind == "draws") & (R.scheme == scheme) & (R.k == k) & (R.metric == "c")].set_index("stat")["value"]
            print(f"  balanced_beta c: median {c['median']:.3f}, share fallback {c['share_fallback']:.4f}, share clipped {c['share_clipped']:.4f}")


if __name__ == "__main__":
    main()
