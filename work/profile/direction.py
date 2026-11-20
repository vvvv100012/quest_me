"""The 'market direction' profile dimension and its stability across the 2025-11-03 split.

Definition (per address, over its in-market days that have a BTC return):
  A = Spearman correlation of daily net PnL (pnl_gross_obs - fee, summed over symbols) with the same-day BTC return.
  B = sum of net end-of-day exposure / sum of gross end-of-day exposure (position-based comparator).
  class (from A, threshold TH, strict): long-like > TH, short-like < -TH, neutral otherwise.
Formation = days < SPLIT, taken from the formation-only build (raw records >= SPLIT physically dropped), eligible
with >= MIN_F in-market days.  Validation = days >= SPLIT from the full build, eligible with >= MIN_V days.

Inputs : work/v2dev/build/{daily_ts,sym_daily}.parquet (full), work/profile/build_formation/... (formation-only;
         rebuilt via 02_replay.py if missing).
Outputs: direction_scores.csv (one row per address), direction_stability.csv (section, metric, value).
Run    : <venv python> -W ignore direction.py [--selftest]
"""
import os, subprocess, sys
import numpy as np, pandas as pd

ROOT = "/Users/xc/Documents/arkstream_quest"
FULL, FORM = f"{ROOT}/work/v2dev/build", f"{ROOT}/work/profile/build_formation"
OUT = f"{ROOT}/work/profile"
SPLIT = pd.Timestamp("2025-11-03")
MIN_F, MIN_V, MIN_H = 30, 20, 15
TH = 0.3
EVENT_DAYS = pd.to_datetime(["2025-10-10", "2025-10-11"])
CLASSES = ["short-like", "neutral", "long-like"]


def ensure_formation_build():
    if all(os.path.exists(f"{FORM}/{f}.parquet") for f in ("daily_ts", "sym_daily")):
        return
    env = dict(os.environ, ARK_DATA_DIR=f"{ROOT}/ArkStream笔试题_v2", ARK_ASOF="2025-11-03 00:00:00",
               ARK_BUILD_DIR=FORM, ARK_TABLES_DIR=f"{FORM}/tables")
    subprocess.run([sys.executable, "-W", "ignore", "02_replay.py"], cwd=f"{ROOT}/work/v2dev/code", env=env,
                   check=True, stdout=subprocess.DEVNULL)


def rows(build):
    d = pd.read_parquet(f"{build}/daily_ts.parquet",
                        columns=["trader", "day", "pnl_gross_obs", "fee", "exposure_obs", "is_hip3", "pre_first_fill"])
    d = d[~d["pre_first_fill"]]
    return d.assign(pnl=d["pnl_gross_obs"] - d["fee"], gross=d["exposure_obs"].abs())


def trader_day(d):
    return d.groupby(["trader", "day"]).agg(pnl=("pnl", "sum"), net=("exposure_obs", "sum"),
                                            gross=("gross", "sum")).reset_index()


def btc_close(build):
    s = pd.read_parquet(f"{build}/sym_daily.parquet")
    return s[s["symbol"] == "BTC"].set_index("day_end")["close"].sort_index()


def load():
    ensure_formation_build()
    full, form = trader_day(rows(FULL)), trader_day(rows(FORM))
    tdF, full_pre = form[form["day"] < SPLIT], full[full["day"] < SPLIT]
    tdV = full[full["day"] >= SPLIT]
    # (a) the formation-only build reproduces the full build before the split
    m = full_pre.merge(tdF, on=["trader", "day"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "trader-day keys differ between builds before the split"
    for c in ("pnl", "net", "gross"):
        assert np.isclose(m[c + "_x"], m[c + "_y"], rtol=1e-9, atol=1e-9).all(), f"{c} differs between builds"
    print(f"check OK: formation build equals full build before the split ({len(m)} trader-days)")
    # (b) BTC closes up to the split are the same in both builds
    cF, cA = btc_close(FORM), btc_close(FULL)
    cF, cA = cF[cF.index <= SPLIT], cA[cA.index <= SPLIT]
    assert cF.index.equals(cA.index) and (cF.values == cA.values).all(), "BTC closes differ before the split"
    print(f"check OK: BTC closes with day_end <= split equal in both builds ({len(cA)} closes)")
    c = btc_close(FULL)
    ret = c.pct_change().dropna()
    ret.index = ret.index - pd.Timedelta(days=1)  # label = start of the UTC PnL day
    ret = ret.rename("btc").rename_axis("day")
    add = lambda td: td.merge(ret.reset_index(), on="day", how="inner")
    return add(tdF), add(tdV), ret


def scores(td, min_days):
    g = td.groupby("trader")
    out = pd.DataFrame({"n_days": g.size()})
    out["eligible"] = (out["n_days"] >= min_days) & (g["pnl"].nunique() > 1)
    out["A"] = g[["pnl", "btc"]].apply(lambda x: x["pnl"].rank().corr(x["btc"].rank()))
    gs = g["gross"].sum()
    out["B"] = (g["net"].sum() / gs).where(gs > 0)
    out.loc[~out["eligible"], ["A", "B"]] = np.nan
    return out[["n_days", "A", "B", "eligible"]]


def classify(s, th=TH):
    s = pd.Series(s)
    return pd.Series(np.select([s > th, s < -th], ["long-like", "short-like"], "neutral"), index=s.index).where(s.notna())


def agree(x, y, th=TH):
    """Agreement of two scores on the addresses where both are defined."""
    m = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    st = m[m["x"].abs() > th]
    cx, cy = classify(st["x"], th), classify(st["y"], th)
    return dict(n=len(m), spearman=m["x"].rank().corr(m["y"].rank()), pearson=m["x"].corr(m["y"]),
                n_strong=len(st), same_sign=(np.sign(st["x"]) == np.sign(st["y"])).mean(),
                same_class=(cx == cy).mean(), to_opposite=int((np.sign(st["x"]) * st["y"] < -th).sum()))


def selftest():
    days = pd.date_range("2025-08-01", periods=40)
    btc = np.random.default_rng(0).normal(0, 0.02, 40)
    mk = lambda t, sgn, n: pd.DataFrame(dict(trader=t, day=days[:n], btc=btc[:n], pnl=sgn * 1000 * btc[:n],
                                             net=sgn * 1000.0, gross=1000.0))
    s = scores(pd.concat([mk("X", 1, 40), mk("Y", -1, 40), mk("Z", 1, 10)]), MIN_F)
    c = classify(s["A"])
    assert np.isclose(s.loc["X", "A"], 1) and np.isclose(s.loc["X", "B"], 1) and c["X"] == "long-like"
    assert np.isclose(s.loc["Y", "A"], -1) and np.isclose(s.loc["Y", "B"], -1) and c["Y"] == "short-like"
    assert not s.loc["Z", "eligible"] and pd.isna(s.loc["Z", "A"]) and pd.isna(c["Z"]) and s.loc["Z", "n_days"] == 10
    k = classify(pd.Series([-0.3, 0.3, 0.31, -0.31, np.nan]))
    assert k[:4].tolist() == ["neutral", "neutral", "long-like", "short-like"] and pd.isna(k[4])
    print("selftest OK")


def main():
    tdF, tdV, ret = load()
    rec = []
    def put(sec, metric, val):
        rec.append((sec, metric, val))
        print(f"  {metric:<58} {val:.4f}" if isinstance(val, float) else f"  {metric:<58} {val}")

    print("\n=== 1. setup ===")
    rF, rV = ret[ret.index < SPLIT], ret[ret.index >= SPLIT]
    hF, hV = rows(FORM), rows(FULL)
    hF, hV = hF[hF["day"].isin(rF.index)], hV[hV["day"].isin(rV.index)]
    for p, r, h in (("formation", rF, hF), ("validation", rV, hV)):
        put("setup", f"{p} return days", len(r))
        put("setup", f"{p} BTC cumulative return", float((1 + r).prod() - 1))
        put("setup", f"{p} BTC daily return sd", float(r.std()))
        put("setup", f"{p} share of |row PnL| on is_hip3 rows", float(h["pnl"].abs()[h["is_hip3"]].sum() / h["pnl"].abs().sum()))

    F, V = scores(tdF, MIN_F), scores(tdV, MIN_V)
    F["cls"], V["cls"] = classify(F["A"]), classify(V["A"])
    S = F.add_prefix("f_").join(V.add_prefix("v_"), how="outer")
    for p in "fv":
        S[f"{p}_n_days"] = S[f"{p}_n_days"].fillna(0).astype(int)
        S[f"{p}_eligible"] = S[f"{p}_eligible"].fillna(False).astype(bool)
    S = S.rename(columns={"f_cls": "f_class", "v_cls": "v_class"}).rename_axis("trader").reset_index()
    S[["trader"] + [f"{p}_{c}" for p in "fv" for c in ("n_days", "A", "B", "class", "eligible")]].to_csv(
        f"{OUT}/direction_scores.csv", index=False)

    print("\n=== 2. counts ===")
    fe, ve = F.index[F["eligible"]], V.index[V["eligible"]]
    both = fe.intersection(ve)
    put("counts", "formation eligible", len(fe)); put("counts", "validation eligible", len(ve))
    put("counts", "both eligible", len(both))
    for p, X, e in (("formation", F, fe), ("validation", V, ve)):
        for c in CLASSES:
            put("counts", f"{p} {c}", int((X.loc[e, "cls"] == c).sum()))

    print("\n=== 3. attrition: formation class -> validation class (all formation-eligible) ===")
    fc, vc = F.loc[fe, "cls"], V["cls"].reindex(fe)
    opp = {"long-like": "short-like", "short-like": "long-like"}
    att = {}
    for c in CLASSES:
        sel = fc == c
        att[c] = dict(n=sel.sum(), no_validation_score=vc[sel].isna().sum(), same_class=(vc[sel] == c).sum(),
                      to_neutral=(vc[sel] == "neutral").sum() if c != "neutral" else np.nan,
                      to_directional=vc[sel].isin(list(opp)).sum() if c == "neutral" else np.nan,
                      to_opposite=(vc[sel] == opp[c]).sum() if c != "neutral" else np.nan)
    T = pd.DataFrame(att).T
    T.loc["directional"] = T.loc[["long-like", "short-like"]].sum(min_count=1)
    T.loc["all"] = T.loc[CLASSES].sum(min_count=1)
    T = T.astype("Int64")
    T["retention_all"] = T["same_class"] / T["n"]
    T["retention_scored"] = T["same_class"] / (T["n"] - T["no_validation_score"])
    T["upper_bound"] = (T["same_class"] + T["no_validation_score"]) / T["n"]
    print("  retention_all = same/n (observed, all formation-eligible); retention_scored = same/(n - no_validation_score);"
          "\n  upper_bound = (same + no_validation_score)/n (if every missing one had kept its class)")
    print("  " + T.to_string(float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))
    for c, r in T.iterrows():
        rec += [("attrition", f"{c}/{k}", float(v)) for k, v in r.items() if pd.notna(v)]

    print("\n=== 4. across-split agreement (both eligible) ===")
    for k in ("A", "B"):
        for m, v in agree(F.loc[both, k], V.loc[both, k]).items():
            put("across_split", f"{k} {m}", v)
    tab = pd.crosstab(pd.Categorical(F.loc[both, "cls"], CLASSES), pd.Categorical(V.loc[both, "cls"], CLASSES),
                      rownames=["formation"], colnames=["validation"], dropna=False)
    print("  crosstab A (rows formation class, cols validation class)\n  " + tab.to_string().replace("\n", "\n  "))
    rec += [("crosstab_A", f"{a}->{b}", int(tab.loc[a, b])) for a in CLASSES for b in CLASSES]

    print("\n=== 5. reference comparisons inside one part (each side >= %d in-market days) ===" % MIN_H)
    wkF, wkV = tdF["day"].dt.isocalendar().week % 2, tdV["day"].dt.isocalendar().week % 2
    mid = rF.index[len(rF) // 2]
    refs = {"formation even vs odd ISO week": (tdF[wkF == 0], tdF[wkF == 1]),
            "validation even vs odd ISO week": (tdV[wkV == 0], tdV[wkV == 1]),
            "formation first vs second half": (tdF[tdF["day"] < mid], tdF[tdF["day"] >= mid])}
    for name, (a, b) in refs.items():
        sa, sb = scores(a, MIN_H), scores(b, MIN_H)
        for k in ("A", "B"):
            g = agree(sa[k], sb[k])
            for m in ("n", "spearman", "same_sign", "same_class"):
                put("reference", f"{name} {k} {m}", g[m])

    print("\n=== 6. A vs B in formation ===")
    m = F.loc[fe, ["A", "B"]].dropna()
    put("A_vs_B", "n (eligible, both defined)", len(m))
    put("A_vs_B", "Spearman", float(m["A"].rank().corr(m["B"].rank())))
    put("A_vs_B", "opposite signs", int((np.sign(m["A"]) * np.sign(m["B"]) < 0).sum()))
    put("A_vs_B", "different classes", int((classify(m["A"]) != classify(m["B"])).sum()))
    put("A_vs_B", "B undefined", int(F.loc[fe, "B"].isna().sum()))
    put("A_vs_B", "share |B| > 0.999 (B defined)", float((F.loc[fe, "B"].dropna().abs() > 0.999).mean()))

    print("\n=== 7. sensitivities for A across the split ===")
    # minimum lowered by the number of removed days, so no address of the cohort drops out for losing those days
    Fx = scores(tdF[~tdF["day"].isin(EVENT_DAYS)], MIN_F - len(EVENT_DAYS))
    keep = both.intersection(Fx.index[Fx["eligible"]])
    put("sensitivity", "no event days: cohort addresses losing eligibility", len(both) - len(keep))
    for k, v in agree(Fx.loc[keep, "A"], V.loc[keep, "A"]).items():
        put("sensitivity", f"no event days: A {k}", v)
    for th in (0.2, 0.4):
        g = agree(F.loc[both, "A"], V.loc[both, "A"], th)
        for k in ("n_strong", "same_sign", "same_class"):
            put("sensitivity", f"th={th}: A {k}", g[k])
    pd.DataFrame(rec, columns=["section", "metric", "value"]).to_csv(f"{OUT}/direction_stability.csv", index=False)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
