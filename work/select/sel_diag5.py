"""Throwaway: 60-day direction label (report definition, old v2dev build) per Monday, joined to the qualified addresses;
quota trial 4 long / 4 short / 2 free with the label; label retention at a 63-day gap."""
import sys, io, contextlib
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/profile")
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import direction_wf as W
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 100)
td, ret = W.load()
P = pd.read_parquet("sel_panel2.parquet"); p = P[P.ret_mean_sd.notna()].copy()
sc = p.order_ntl_med * np.minimum(1.0, 2500 / p.av_med)
ok = ((p.av_med >= 5000) & (sc >= 10) & ~(p.hold_med_h < 1) & (p.maker_share <= 0.8) & (p.r28_days >= 7)
      & ~(p.lev_p95 > 8) & (p.max_dd <= 0.6) & (p.liq <= 0) & ~(p.mae_av_med > 0.03) & ~(p.add_loser_share > 0.9))
s = p[ok].copy()
lab, allM = [], [T for T in pd.date_range("2025-09-29", "2025-12-29", freq="W-MON")]
L = {}
for T in allM:
    S = W.window(td, ret, T, 60)[4]
    L[T] = S["cls"]
    lab.append(pd.DataFrame({"trader": S.index, "M": T.strftime("%Y-%m-%d"), "label": S["cls"].values, "A": S["A"].values}))
lab = pd.concat(lab)
s = s.merge(lab, on=["trader", "M"], how="left")
s["label"] = s["label"].fillna("none")
s["nb"] = np.where(s.net_bias > 0.3, "long-like", np.where(s.net_bias < -0.3, "short-like", "neutral"))
print("label vs net_bias class (qualified, labelled):"); print(pd.crosstab(s.label, s.nb))
t = s.groupby(["M", "label"]).agg(n=("A", "size"), pos=("ret_mean_sd", lambda x: (x > 0).sum())).unstack("label").fillna(0).astype(int)
t.columns = [f"{b}_{a}" for a, b in t.columns]; t.insert(0, "qualified", s.groupby("M").size())
print(t.to_string())
s["side"] = s.label.map({"long-like": "long", "short-like": "short"}).fillna("neutral")
s["pct_side"] = s.groupby(["M", "side"])["ret_mean_sd"].rank(pct=True, ascending=False)
Ms = sorted(s.M.unique())
def run(N, repeat, quota):
    held, rows = [], []
    for M in Ms:
        d = s[s.M == M].sort_values("ret_mean_sd", ascending=False)
        busy = set().union(*held[-3:]) if (held and not repeat) else set()
        d = d[~d.trader.isin(busy) & (d.ret_mean_sd > 0)]
        parts, used = [], set()
        for side in ("long", "short"):
            x = d[d.side == side].head(quota[side]); parts.append(x); used |= set(x.trader)
        parts.append(d[~d.trader.isin(used)].head(quota["free"]))
        pk = pd.concat(parts); held.append(set(pk.trader))
        allh = [a for h in held[-4:] for a in h]
        g = lambda sd, c, f: getattr(pk.loc[pk.side == sd, c], f)()
        rows.append(dict(M=M, n=len(pk), long=(pk.side == "long").sum(), short=(pk.side == "short").sum(), neutral=(pk.side == "neutral").sum(),
                         long_min=g("long", "ret_mean_sd", "min"), long_pct=g("long", "pct_side", "max"),
                         short_min=g("short", "ret_mean_sd", "min"), short_pct=g("short", "pct_side", "max"),
                         distinct4=len(set(allh)), max_sleeves=pd.Series(allh).value_counts().max()))
    return pd.DataFrame(rows).set_index("M")
for rep in (False, True):
    print(f"\n=== 4/4/2, repeat={rep} ==="); print(run(10, rep, {"long": 4, "short": 4, "free": 2}).round(3).to_string())
# label retention at a 63-day gap (nearest non-overlapping spacing on Mondays)
rows = []
for a in allM:
    b = a + pd.Timedelta(days=63)
    if b in L:
        for c in ["all"] + W.CLASSES:
            A_ = L[a] if c == "all" else L[a][L[a] == c]
            nb = L[b].reindex(A_.index)
            rows.append(dict(c=c, same=(nb == A_).mean(), gone=nb.isna().mean()))
r = pd.DataFrame(rows); print("\nretention at 63-day gap, pairs:", len(r) // 4); print(r.groupby("c").mean().round(3))
