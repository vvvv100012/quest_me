"""Lookback-length sensitivity of the rolling profile evaluation (work/codex10/REVIEW.md section 2.9, the
"Keep 60 as the operating window" bullets).

Step 1, profiles. L30_M = [M-30d, M) for all 11 Mondays and L90_M = [M-90d, M) only for the seven Mondays
2025-10-27 .. 2025-12-08 (the other four would start before the first package fill day 2025-07-27; L90 is never padded
or shortened), each at its own cutoff C = M, through ledger.py (load_inputs, build_prep, profile_windows, to_tables),
excursion.py (load_hilo, build_x, run_windows) and returns.py (build_r on the same ledger prep, run_windows). L60 of two
Mondays goes through the same pass and must equal the existing build rows exactly (then dropped).

Step 2, measures, with the forward window always the existing F28_M (values, supports, F gates unchanged). For length
l: B_lM = base-eligible on Ll_M (>= 30 orders on >= 10 fill days inside Ll_M); A_ljM = B_lM, the column's L gate
(kind "L", the same section 2.3 L-support counts at every length) and a defined Ll value; C_ljM = A_ljM, the F28 gate
and a defined F28 value.
  native panels   per (l, M, column): |B|, |A|, |C|, rho, same-tier fraction with p0 and discriminating flag,
                  level summary; top symbol: p_same, p0, chance-adjusted
  common cohorts  c30_60_90 = C_30 & C_60 & C_90 on the seven shared dates; c30_60 = C_30 & C_60 on all 11 dates;
                  ranks recomputed on the cohort for each length
  aggregation     measures.aggregate rules (equal-date mean, median of per-date medians, K, min, max, omitted)
                  over all11 (30, 60) and shared7 (30, 60, 90). No winner, no ranking of lengths, no flags.

Statistics, gates, registry and aggregation are imported from measures.py (not modified).
Inputs (read-only): as ledger.py / excursion.py / returns.py; work/roll/build/{ledger, support, top_symbol, ledger_x,
support_x, ledger_r, support_r, decisions, episodes, clusters, address_sums}.parquet and windows.csv (through
measures.load); build/measures/{rank_date, report_date}.parquet (checks only).
Outputs: work/roll/build/lengths/ (ledger, support, top_symbol, ledger_x, support_x, ledger_r, support_r for the L30 /
L90 windows,
windows.csv, len_native_date, len_native_agg, len_common_date, len_common_agg, len_overview.csv, manifest.json);
log work/roll/logs/lengths.log. `--checks` writes build/lengths/checks.json and logs/lengths_checks.log.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/Users/xc/Documents/arkstream_quest")
ROLL = ROOT / "work/roll"
BUILD = ROLL / "build"
OUT = BUILD / "lengths"
LOGS = ROLL / "logs"
sys.path.insert(0, str(ROLL / "code"))
import measures as MZ  # noqa: E402  (imports ledger)
import excursion as EX  # noqa: E402
import returns as RT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

LED = MZ.LED
MS = MZ.MS
SHARED = MS[4:]                                       # 2025-10-27 .. 2025-12-08
LENGTHS = (30, 60, 90)
SANITY = ("2025-09-29", "2025-12-08")                 # L60 recomputed through this path and compared
PREFIX_M = "2025-11-24"
GROUPS = {"all11": MS, "shared7": SHARED}
COHORTS = {"c30_60": ((30, 60), "all11"), "c30_60_90": ((30, 60, 90), "shared7")}
NEW_TABLES = ("ledger", "support", "top_symbol", "ledger_x", "support_x", "ledger_r", "support_r")
EXT_SUPPORT = ("support_r", "support_x")              # joined in this order, as measures.load (sorted file names)

_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


# ---------------------------------------------------------------------------------------
# Step 1: profiles of the extra lookback windows
# ---------------------------------------------------------------------------------------
def window_table(sanity=True, mondays=MS):
    """window_id, kind, M, a, b (Timestamps, as ledger.window_table) for L30, L90 (shared dates) and the sanity L60."""
    d = pd.Timedelta(days=1)
    rows = []
    for M in LED.MONDAYS:
        ms = f"{M:%Y-%m-%d}"
        if ms not in mondays:
            continue
        rows.append((f"L30_{ms}", "L30", M, M - 30 * d, M))
        if ms in SHARED:
            rows.append((f"L90_{ms}", "L90", M, M - 90 * d, M))
        else:                                  # L90 would start before the package: excluded, never shortened
            assert M - 90 * d < LED.W0, ms
        if sanity and ms in SANITY:
            rows.append((f"L60_{ms}", "L60", M, M - 60 * d, M))
    wt = pd.DataFrame(rows, columns=["window_id", "kind", "M", "a", "b"])
    assert (wt["a"] >= LED.W0).all() and (wt["b"] <= LED.W1).all()
    return wt


def profiles(inp, hilo, U, wt):
    """Ledger, excursion and return tables of the windows of wt. build_x builds the ledger prep with
    ledger.build_prep(inp, U) (X.P) and only reads it; the ledger pass and returns.build_r use that same prep."""
    X = EX.build_x(inp, hilo, U)
    res = LED.profile_windows(X.P, list(zip(wt["a"], wt["b"])), verbose=False)
    tabs = LED.to_tables(X.P, res, wt)
    tx = EX.run_windows(X, wt, verbose=False)
    tr = RT.run_windows(RT.build_r(inp, U, P=X.P), wt, verbose=False)
    return dict(ledger=tabs["ledger"], support=tabs["support"], top_symbol=tabs["top_symbol"],
                ledger_x=tx["ledger_x"], support_x=tx["support_x"], ledger_r=tr["ledger_r"], support_r=tr["support_r"])


KEYS = {"ledger": ["trader", "column"], "ledger_x": ["trader", "column"], "top_symbol": ["trader"],
        "support": ["trader"], "support_x": ["trader"], "ledger_r": ["trader", "column"], "support_r": ["trader"]}


def _norm(d, keys):
    d = d.astype({c: str for c in d.columns if isinstance(d[c].dtype, pd.CategoricalDtype)})
    return d.sort_values(keys, kind="stable").reset_index(drop=True)


def compare(new, old, wids):
    """Exact comparison (NaN == NaN) of the tables of `new` and `old` (dicts of frames) on the window ids."""
    out = {}
    for name, keys in KEYS.items():
        r = {}
        for w in wids:
            a = _norm(new[name][new[name]["window_id"].astype(str) == w], keys)
            b = _norm(old[name][old[name]["window_id"].astype(str) == w], keys)
            if len(a) != len(b) or not (a[keys].to_numpy() == b[keys].to_numpy()).all():
                r[w] = dict(rows=[len(a), len(b)], keys_differ=True)
                continue
            cols = [c for c in a.columns if c not in ["window_id"] + keys]
            d = LED._eq_frames(a, b, cols)
            r[w] = dict(rows=len(a), cells=len(a) * len(cols), cells_differing=int(sum(d.values())),
                        differing_cols={k: v for k, v in d.items() if v})
        out[name] = r
    return out


def _ok(cmp):
    return all(not v.get("keys_differ") and v["cells_differing"] == 0 for r in cmp.values() for v in r.values())


def _cells(cmp):
    return {name: sum(v.get("cells", 0) for v in r.values()) for name, r in cmp.items()}


def read_build(names, wids):
    out = {}
    for n in names:
        d = pd.read_parquet(BUILD / f"{n}.parquet")
        out[n] = d[d["window_id"].astype(str).isin(wids)]
    return out


# ---------------------------------------------------------------------------------------
# Step 2: measures
# ---------------------------------------------------------------------------------------
def load(reg):
    """measures.load (L60, F28 and the rest of the existing build) plus the L30 / L90 windows of build/lengths, pivoted
    and joined exactly as measures.load does (support_r / support_x columns colliding with earlier ones are ignored)."""
    data = MZ.load(reg)
    U = data["U"]
    sup = pd.read_parquet(OUT / "support.parquet").astype({"window_id": str, "trader": str})
    for f in EXT_SUPPORT:
        sx = pd.read_parquet(OUT / f"{f}.parquet").astype({"window_id": str, "trader": str})
        new = [c for c in sx.columns if c not in sup.columns]
        sup = sup.merge(sx[["window_id", "trader"] + new], on=["window_id", "trader"], how="left")
    wide = MZ._pivot_long(pd.read_parquet(OUT / "ledger.parquet"))
    for f in ("ledger_r", "ledger_x"):
        lx = MZ._pivot_long(pd.read_parquet(OUT / f"{f}.parquet"))
        wide = wide.join(lx[[e.name for e in reg if e.source == f]], how="left")
    top = pd.read_parquet(OUT / "top_symbol.parquet").astype({"window_id": str, "trader": str}) \
        .set_index(["window_id", "trader"])["top_symbol"]
    for w, d in sup.groupby("window_id"):
        assert w not in data["sup"], w
        data["sup"][w] = d.set_index("trader").reindex(U)
        v = wide.loc[w].reindex(U)
        v["top_symbol"] = top.loc[w].reindex(U)
        data["vals"][w] = v
    return data


def stats_on(col, xl, xf):
    """Headline statistics on one paired cohort."""
    if col.level_type == "category":
        cs, _ = MZ.category_stats(xl, xf)
        return dict(n=cs["n"], p_same=cs["p_same"], cat_p0=cs["p0"], chance_adj=cs["chance_adj"])
    rho, why = MZ.rank_corr(xl, xf)
    rt = MZ.rel_tiers(xl, xf)
    lv = MZ.level_stats(xl, xf, col.level_type, col.name)
    return dict(n=len(xl), rho=rho, reason=why, small_cohort=bool(len(xl) < MZ.SMALL), same_tier=rt["same_tier"],
                p0=rt["p0"], discriminating=rt["discriminating"], level_med=lv["level_med"],
                level_frac=lv["level_frac"], level_unit=lv["unit"])


def compute(reg, data):
    """Native and common-cohort date tables, plus the cohort masks (for the checks)."""
    nat, com, masks = [], [], {}
    for M in MS:
        Ls = [30, 60] + ([90] if M in SHARED else [])
        sF = data["sup"][f"F28_{M}"]
        for col in reg:
            xF, okF = MZ.values(data, f"F28_{M}", col)
            gF = MZ.gate(col, sF, "F").to_numpy()
            C, X = {}, {}
            for ln in Ls:
                wid = f"L{ln}_{M}"
                sL = data["sup"][wid]
                B = sL["base_eligible"].to_numpy().astype(bool)
                xL, okL = MZ.values(data, wid, col)
                A = B & MZ.gate(col, sL, "L").to_numpy() & okL
                C[ln], X[ln] = A & gF & okF, xL
                nat.append(dict(length=ln, M=M, column=col.name, level_type=col.level_type, B=int(B.sum()),
                                A=int(A.sum()), C=int(C[ln].sum()), **stats_on(col, xL[C[ln]], xF[C[ln]])))
            for name, (ls, _) in COHORTS.items():
                if not set(ls) <= set(Ls):
                    continue
                Cc = np.logical_and.reduce([C[ln] for ln in ls])
                masks[(name, M, col.name)] = (Cc, {ln: C[ln] for ln in ls})
                for ln in ls:
                    com.append(dict(cohort=name, length=ln, M=M, column=col.name, level_type=col.level_type,
                                    **stats_on(col, X[ln][Cc], xF[Cc])))
        log(f"  {M} done")
    nat, com = pd.DataFrame(nat), pd.DataFrame(com)
    nat = nat.rename(columns={"n": "n_cohort"})
    assert (nat["n_cohort"] == nat["C"]).all()
    return nat.drop(columns="n_cohort"), com, masks


RULES = {"rho": "mean", "same_tier": "mean", "p0": "mean", "discriminating_int": "mean", "level_med": "median",
         "level_frac": "mean", "p_same": "mean", "cat_p0": "mean", "chance_adj": "mean"}


def aggregate(df, keys, rules, groups, warn):
    """measures.aggregate with these groups of Mondays (its GROUPS is a module global, swapped for the call)."""
    saved = MZ.GROUPS
    MZ.GROUPS = groups
    try:
        return MZ.aggregate(df, keys, rules, warn)
    finally:
        MZ.GROUPS = saved


def _prep_agg(d):
    """Same-tier fraction and p0 aggregated over discriminating dates only, as in measures.py."""
    disc = d["discriminating"].fillna(False).astype(bool)
    d = MZ.masked(d, disc, ["same_tier", "p0"])
    d["discriminating_int"] = disc.astype(float).where(d["level_type"] != "category")
    return d


def aggregates(nat, com, reg):
    warn = {e.name: e.warn for e in reg}
    n_rules = dict(RULES, B="count", A="count", C="count")
    parts = []
    for ln in LENGTHS:
        g = {k: v for k, v in GROUPS.items() if ln != 90 or k == "shared7"}
        parts.append(aggregate(_prep_agg(nat[nat["length"] == ln]), ["length", "column"], n_rules, g, warn))
    nagg = pd.concat(parts, ignore_index=True)
    parts = []
    for name, (_, grp) in COHORTS.items():
        parts.append(aggregate(_prep_agg(com[com["cohort"] == name]), ["cohort", "length", "column"],
                               dict(RULES, n="count"), {grp: GROUPS[grp]}, warn))
    return nagg, pd.concat(parts, ignore_index=True)


def overview(nagg, cagg, reg):
    def g(agg, stat, field="value", **k):
        m = agg["stat"] == stat
        for a, b in k.items():
            m &= agg[a] == b
        s = agg.loc[m, field]
        return s.iloc[0] if len(s) else np.nan

    rows = []
    for e in reg:
        c = e.name
        cat = e.level_type == "category"
        st = "p_same" if cat else "rho"
        o = dict(column=c, level_type=e.level_type, headline_stat=st)
        for grp, ls in (("all11", (30, 60)), ("shared7", (30, 60, 90))):
            for ln in ls:
                o[f"native_{st}_{ln}_{grp}"] = g(nagg, st, length=ln, column=c, group=grp)
                o[f"native_K_{ln}_{grp}"] = g(nagg, st, "K", length=ln, column=c, group=grp)
                o[f"native_medC_{ln}_{grp}"] = g(nagg, "C", length=ln, column=c, group=grp)
        for name, (ls, grp) in COHORTS.items():
            for ln in ls:
                o[f"common_{st}_{ln}_{name}_{grp}"] = g(cagg, st, cohort=name, length=ln, column=c, group=grp)
                o[f"common_K_{ln}_{name}_{grp}"] = g(cagg, st, "K", cohort=name, length=ln, column=c, group=grp)
            o[f"common_median_n_{name}_{grp}"] = g(cagg, "n", cohort=name, length=ls[0], column=c, group=grp)
        o["warn"] = e.warn
        rows.append(o)
    return pd.DataFrame(rows)        # numeric columns carry rho, top_symbol p_same (named by headline_stat)


def manifest(tables, runtime):
    data = ROOT / "ArkStream笔试题_v2"
    raw = [data / "笔试题数据包-fills", data / "笔试题数据包-traders.csv", LED.EXT / "funding.parquet",
           LED.EXT / "hl_candles_4h.parquet", LED.EXT / "symbol_map.csv"]
    built = [BUILD / f"{n}.parquet" for n in ("ledger", "support", "top_symbol", "ledger_x", "support_x", "ledger_r",
                                              "support_r", "decisions", "episodes", "clusters", "address_sums")] \
        + [BUILD / "windows.csv"]
    src = [Path(__file__).resolve(), ROLL / "code/measures.py", ROLL / "code/ledger.py", ROLL / "code/excursion.py",
           ROLL / "code/returns.py", ROOT / "work/v2dev/code/lib.py", ROOT / "work/profile/profile.py", ROOT / "work/profile2/code/exposure.py",
           MZ.SPEC]
    return dict(spec=str(MZ.SPEC), spec_section="2.9 lookback-length sensitivity (L30 / L60 / L90)",
                inputs={MZ._rel(f): LED.sha256(f) for f in raw + built},
                klines_5m={f.name: LED.sha256(f) for f in sorted(LED.KLINES.glob("*.parquet"))},
                sources={MZ._rel(f): LED.sha256(f) for f in src},
                outputs={MZ._rel(OUT / f"{n}.parquet"): LED.sha256(OUT / f"{n}.parquet") for n in NEW_TABLES},
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              pyarrow=pyarrow.__version__, scipy=scipy.__version__),
                mondays=MS, shared7=SHARED, groups=GROUPS, cohorts={k: list(v[0]) for k, v in COHORTS.items()},
                sanity_l60_mondays=list(SANITY), registry=MZ.reg_dump(MZ.registry()),
                conventions=dict(base="B_lM = base_eligible of Ll_M (>= 30 orders on >= 10 fill days in Ll_M)",
                                 L_gate="kind 'L' of measures.gate at every length", F="F28_M values, supports, gates",
                                 aggregation="measures.aggregate rules; same-tier / p0 over discriminating dates only",
                                 no_selection="no winner, no ranking of lengths, no flags"),
                tables={k: int(len(v)) for k, v in tables.items()},
                run_utc=pd.Timestamp.now(tz="UTC").isoformat(), runtime_s=round(runtime, 1))


def build():
    global _logf
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "lengths.log", "w")
    t0 = time.time()
    wt = window_table()
    inp = LED.load_inputs()
    hilo = EX.load_hilo()
    U = sorted(inp["p"]["trader"].unique())
    log(f"loaded in {time.time() - t0:.1f}s; windows: {wt['kind'].value_counts().to_dict()}")
    tabs = profiles(inp, hilo, U, wt)
    log(f"profiles done at {time.time() - t0:.1f}s")
    sw = [f"L60_{m}" for m in SANITY]
    cmp = compare(tabs, read_build(NEW_TABLES, sw), sw)
    log(f"L60 sanity (cells compared {_cells(cmp)}): {json.dumps(cmp)}")
    assert _ok(cmp), "L60 through the lengths path differs from the existing build"
    keep = wt.loc[wt["kind"] != "L60", "window_id"].tolist()
    for n in NEW_TABLES:
        d = tabs[n][tabs[n]["window_id"].astype(str).isin(keep)]
        if isinstance(d["window_id"].dtype, pd.CategoricalDtype):
            d = d.assign(window_id=d["window_id"].cat.remove_unused_categories())
        d.to_parquet(OUT / f"{n}.parquet", index=False)
        log(f"wrote {n}.parquet: {len(d):,} rows x {d.shape[1]} cols")
    w = wt[wt["kind"] != "L60"]
    w.assign(M=w["M"].dt.strftime("%Y-%m-%d"), a=w["a"].dt.strftime("%Y-%m-%d"),
             b=w["b"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "windows.csv", index=False)
    del tabs, inp, hilo

    reg = MZ.registry()
    data = load(reg)
    nat, com, _ = compute(reg, data)
    nagg, cagg = aggregates(nat, com, reg)
    ov = overview(nagg, cagg, reg)
    tables = dict(len_native_date=nat, len_native_agg=nagg, len_common_date=com, len_common_agg=cagg)
    for k, d in tables.items():
        d.to_parquet(OUT / f"{k}.parquet", index=False)
        d.to_csv(OUT / f"{k}.csv", index=False)
        log(f"wrote {k}: {len(d):,} rows x {d.shape[1]} cols")
    ov.to_csv(OUT / "len_overview.csv", index=False)
    log(f"wrote len_overview.csv: {len(ov)} rows x {ov.shape[1]} cols")
    runtime = time.time() - t0
    man = manifest(dict(tables, len_overview=ov), runtime)
    man["l60_sanity"] = dict(cells_compared=_cells(cmp), all_equal=_ok(cmp))
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"runtime {runtime:.1f}s")


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def _same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def independent(M, col):
    """Common-cohort rhos written separately: pandas on the parquet files, inline gates, scipy.spearmanr."""
    def frame(w):
        root = OUT if w.startswith(("L30", "L90")) else BUILD
        s = pd.read_parquet(root / "support.parquet")
        s = s[s.window_id.astype(str) == w].assign(trader=lambda d: d.trader.astype(str)).set_index("trader")
        x = pd.read_parquet(root / "support_x.parquet")
        x = x[x.window_id.astype(str) == w].assign(trader=lambda d: d.trader.astype(str)).set_index("trader")
        s = s.join(x[["n_px", "d_px"]])
        x = pd.read_parquet(root / "support_r.parquet")
        x = x[x.window_id.astype(str) == w].assign(trader=lambda d: d.trader.astype(str)).set_index("trader")
        s = s.join(x[["n_ret_days", "n_act_days", "n_act_noav", "n_act_lt100", "n_px_bad_days"]])
        led = pd.read_parquet(root / {"mae_px_med": "ledger_x.parquet", "ret_mean_sd": "ledger_r.parquet"}.get(
            col, "ledger.parquet"))
        v = led[(led.window_id.astype(str) == w) & (led.column.astype(str) == col)]
        s["v"] = v.assign(trader=v.trader.astype(str)).set_index("trader")["value"]
        return s

    def gate(s, side):
        if col == "maker_share":
            return pd.Series(True, index=s.index) if side == "L" else (s.n_orders >= 15) & (s.n_fill_days >= 5)
        if col == "hold_med_h":
            return s.n_complete >= (10 if side == "L" else 5)
        if col == "ret_mean_sd":
            return ((s.n_ret_days >= 14) & ~s.hip3_asof.astype(bool)
                    & (s.n_act_noav + s.n_act_lt100 + s.n_px_bad_days <= 0.1 * s.n_act_days))
        return (s.n_px >= (10 if side == "L" else 5)) & (s.d_px >= (3 if side == "L" else 2))

    f = frame(f"F28_{M}")
    fok = gate(f, "F") & f.v.notna()
    L, C = {}, {}
    for ln in (30, 60, 90):
        s = frame(f"L{ln}_{M}").reindex(f.index)
        L[ln] = s.v
        C[ln] = s.base_eligible.astype(bool) & gate(s, "L") & s.v.notna() & fok
    out = {}
    for name, ls in (("c30_60", (30, 60)), ("c30_60_90", (30, 60, 90))):
        cc = np.logical_and.reduce([C[ln] for ln in ls])
        out[name] = dict(n=int(cc.sum()), rho={ln: float(stats.spearmanr(L[ln][cc], f.v[cc])[0]) for ln in ls})
    return out


def checks():
    global _logf
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "lengths_checks.log", "w")
    t0 = time.time()
    res = {}
    nat = pd.read_parquet(OUT / "len_native_date.parquet")
    com = pd.read_parquet(OUT / "len_common_date.parquet")

    # ---- 1: L60 through the lengths path equals the existing build ----
    log("=== check 1: L60 (two Mondays) through the lengths path vs build/ ===")
    inp = LED.load_inputs()
    hilo = EX.load_hilo()
    U = sorted(inp["p"]["trader"].unique())
    wt = window_table()
    sw = [f"L60_{m}" for m in SANITY]
    tabs = profiles(inp, hilo, U, wt)               # the same joint pass as the build (L30, L90 and L60 together)
    cmp = compare(tabs, read_build(NEW_TABLES, sw), sw)
    # and the written L30 / L90 tables equal this rerun
    keep = wt.loc[wt["kind"] != "L60", "window_id"].tolist()
    built = {n: pd.read_parquet(OUT / f"{n}.parquet") for n in NEW_TABLES}
    cmp_b = compare(tabs, built, keep)
    res["check1_l60_sanity"] = dict(mondays=list(SANITY), cells_compared=_cells(cmp), comparison=cmp,
                                    rerun_equals_written_L30_L90=_ok(cmp_b), written_cells=_cells(cmp_b),
                                    ok=_ok(cmp) and _ok(cmp_b))
    log(json.dumps({k: v for k, v in res["check1_l60_sanity"].items() if k != "comparison"}, default=str))
    del tabs, built

    # ---- 2: the 60-day native rows reproduce build/measures ----
    log("\n=== check 2: 60-day native rows vs build/measures rank_date / report_date (F28) ===")
    rk = pd.read_parquet(BUILD / "measures/rank_date.parquet")
    rp = pd.read_parquet(BUILD / "measures/report_date.parquet")
    n60 = nat[nat["length"] == 60]
    j = n60.merge(rk[rk.panel == "F28"][["M", "column", "rho", "n"]], on=["M", "column"], how="left",
                  suffixes=("", "_m"))
    num = j[j["level_type"] != "category"]
    j2 = n60.merge(rp[rp.panel == "F28"][["M", "column", "B", "A", "C"]], on=["M", "column"], how="left",
                   suffixes=("", "_m"))
    c2 = dict(rows_60=len(n60), rank_rows_matched=int(num["n"].notna().sum()), rank_rows=len(num),
              rho_differ=int((~_same(num["rho"], num["rho_m"])).sum()),
              n_differ=int((num["C"] != num["n"]).sum()),
              report_rows_matched=int(j2["B_m"].notna().sum()),
              B_differ=int((j2["B"] != j2["B_m"]).sum()), A_differ=int((j2["A"] != j2["A_m"]).sum()),
              C_differ=int((j2["C"] != j2["C_m"]).sum()))
    c2["ok"] = (c2["rank_rows_matched"] == c2["rank_rows"] == int((rk.panel == "F28").sum())
                and c2["report_rows_matched"] == len(n60) == int((rp.panel == "F28").sum())
                and not any(c2[k] for k in ("rho_differ", "n_differ", "B_differ", "A_differ", "C_differ")))
    res["check2_l60_reproduces_measures"] = c2
    log(json.dumps(c2))

    # ---- 3: cohorts ----
    log("\n=== check 3: common cohorts within each native C; sizes; L90 dates ===")
    reg = MZ.registry()
    data = load(reg)
    nat2, com2, masks = compute(reg, data)
    sub = sum(int((Cc & ~Cn[ln]).sum()) for Cc, Cn in masks.values() for ln in Cn)
    nn = nat.set_index(["length", "M", "column"])["C"]
    size_bad = 0
    for (name, M, c), (Cc, Cn) in masks.items():
        rows = com[(com.cohort == name) & (com.M == M) & (com.column == c)]
        size_bad += int(len(rows) != len(Cn) or (rows["n"] != int(Cc.sum())).any())
        size_bad += int(any(int(Cc.sum()) > nn[(ln, M, c)] for ln in Cn))
        size_bad += int(any(int(Cn[ln].sum()) != nn[(ln, M, c)] for ln in Cn))
    l90_led = pd.read_parquet(OUT / "ledger.parquet", columns=["window_id"])["window_id"].astype(str).unique()
    l90_dates = sorted({w[4:] for w in l90_led if w.startswith("L90_")})
    c3 = dict(cohort_masks=len(masks), addresses_outside_native_C=sub, size_mismatches=size_bad,
              L90_native_dates=sorted(nat.loc[nat.length == 90, "M"].unique().tolist()),
              L90_common_dates=sorted(com.loc[com.length == 90, "M"].unique().tolist()), L90_ledger_dates=l90_dates,
              c30_60_90_dates=sorted(com.loc[com.cohort == "c30_60_90", "M"].unique().tolist()),
              tables_equal_rerun=bool(_frames_equal(nat, nat2) and _frames_equal(com, com2)),
              L30_dates=int(nat.loc[nat.length == 30, "M"].nunique()))
    c3["ok"] = (sub == 0 and size_bad == 0 and c3["L90_native_dates"] == SHARED == c3["L90_common_dates"]
                == c3["L90_ledger_dates"] == c3["c30_60_90_dates"] and c3["tables_equal_rerun"] and c3["L30_dates"] == 11)
    res["check3_cohorts"] = c3
    log(json.dumps(c3))
    del data

    # ---- 4: prefix sanity at M = 2025-11-24 ----
    log(f"\n=== check 4: L30 / L90 of {PREFIX_M} from inputs truncated at M ===")
    M = pd.Timestamp(PREFIX_M)
    tin, removed = LED.truncate_inputs(inp, M)
    th = EX.truncate_hilo(hilo, M)
    wp = window_table(sanity=False, mondays=[PREFIX_M])
    tt = profiles(tin, th, U, wp)
    built = {n: pd.read_parquet(OUT / f"{n}.parquet") for n in NEW_TABLES}
    cmp4 = compare(tt, built, list(wp["window_id"]))
    res["check4_prefix"] = dict(windows=list(wp["window_id"]), removed=dict(removed, hilo_bars=sum(
        len(hilo[s]) - len(th[s]) for s in hilo)), cells_compared=_cells(cmp4), comparison=cmp4, ok=_ok(cmp4))
    log(json.dumps({k: v for k, v in res["check4_prefix"].items() if k != "comparison"}, default=str))
    log(json.dumps(cmp4, default=str))

    # ---- 5: independent common-cohort rhos ----
    log("\n=== check 5: independent recomputation of the common-cohort rhos ===")
    c5 = {}
    for M in ("2025-10-27", "2025-12-08"):
        for c in ("maker_share", "hold_med_h", "mae_px_med", "ret_mean_sd"):
            ind = independent(M, c)
            ok = True
            for name, r in ind.items():
                for ln, rho in r["rho"].items():
                    m = com[(com.cohort == name) & (com.M == M) & (com.column == c) & (com.length == ln)].iloc[0]
                    r.setdefault("mine", {})[ln] = float(m["rho"])
                    ok &= int(m["n"]) == r["n"] and bool(np.isclose(m["rho"], rho, rtol=1e-12, atol=1e-14))
            c5[f"{M}|{c}"] = dict(ind, ok=bool(ok))
            log(f"  {M}|{c}: {c5[f'{M}|{c}']}")
    res["check5_independent"] = dict(cases=c5, ok=all(v["ok"] for v in c5.values()))

    res["all_ok"] = all(v.get("ok", True) for v in res.values() if isinstance(v, dict))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"\nall_ok {res['all_ok']}; checks done in {res['runtime_s']}s")


def _frames_equal(a, b):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            if not _same(x, y).all():
                return False
        elif not ((x.astype(object) == y.astype(object)) | (x.isna() & y.isna())).all():
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="run the acceptance checks on the built outputs")
    args = ap.parse_args()
    checks() if args.checks else build()
