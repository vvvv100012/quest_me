"""Throwaway: supply check for 10 addresses per sleeve (40 in total), with / without repeats, and by direction side.
Lookback data only. Direction side = net_bias (> .3 long, < -.3 short) as a stand-in for the 60-day label."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 100)
P = pd.read_parquet("sel_panel2.parquet")
p = P[P.ret_mean_sd.notna()].copy()
ALLOC = 2500.0
scaled = p.order_ntl_med * np.minimum(1.0, ALLOC / p.av_med)
print("scaled order < 10 at 2,500 per address (AV>=5000 rows):", (scaled[p.av_med >= 5000] < 10).mean().round(4))
adm = (p.av_med >= 5000) & (scaled >= 10) & ~(p.hold_med_h < 1) & (p.maker_share <= 0.8) & (p.r28_days >= 7)
risk = ~(p.lev_p95 > 8) & (p.max_dd <= 0.6) & (p.liq <= 0) & ~(p.mae_av_med > 0.03) & ~(p.add_loser_share > 0.9)
s = p[adm & risk].copy()
s["side"] = np.where(s.net_bias > 0.3, "long", np.where(s.net_bias < -0.3, "short", "neutral"))
s["pct"] = s.groupby("M")["ret_mean_sd"].rank(pct=True, ascending=False)
s["pct_side"] = s.groupby(["M", "side"])["ret_mean_sd"].rank(pct=True, ascending=False)
Ms = sorted(s.M.unique())
t = s.groupby(["M", "side"]).agg(n=("pct", "size"), pos=("ret_mean_sd", lambda x: (x > 0).sum()),
                                 pos10=("ret_mean_sd", lambda x: (x > 0.1).sum())).unstack("side")
t.columns = [f"{b}_{a}" for a, b in t.columns]; t["survivors"] = s.groupby("M").size()
print(t[["survivors", "long_n", "long_pos", "long_pos10", "short_n", "short_pos", "short_pos10", "neutral_n", "neutral_pos"]].to_string())

def run(N, repeat, quota=None):
    held, rows = [], []
    for M in Ms:
        d = s[s.M == M].sort_values("ret_mean_sd", ascending=False)
        busy = set().union(*held[-3:]) if (held and not repeat) else set()
        d = d[~d.trader.isin(busy)]
        if quota is None:
            pk = d.head(N)
        else:
            parts, used = [], set()
            for side, k in quota.items():
                if side == "free": continue
                x = d[(d.side == side) & (d.ret_mean_sd > 0)].head(k); parts.append(x); used |= set(x.trader)
            x = d[~d.trader.isin(used) & (d.ret_mean_sd > 0)].head(quota["free"]); parts.append(x)
            pk = pd.concat(parts)
        held.append(set(pk.trader))
        act = held[-4:]; allh = [a for h in act for a in h]
        r = dict(M=M, n=len(pk), worst_pct=pk.pct.max(), min_score=pk.ret_mean_sd.min(), distinct4=len(set(allh)),
                 slots4=len(allh), max_sleeves=pd.Series(allh).value_counts().max(),
                 long=(pk.side == "long").sum(), short=(pk.side == "short").sum(), neutral=(pk.side == "neutral").sum())
        if quota is not None:
            r["long_worst_pct_side"] = pk.loc[pk.side == "long", "pct_side"].max()
            r["long_min_score"] = pk.loc[pk.side == "long", "ret_mean_sd"].min()
        rows.append(r)
    return pd.DataFrame(rows).set_index("M")

for N in (5, 10):
    for rep in (False, True):
        print(f"\n=== N={N} repeat={rep} no direction quota ==="); print(run(N, rep).round(3).to_string())
for N, q in ((5, {"long": 2, "short": 2, "free": 1}), (10, {"long": 4, "short": 4, "free": 2})):
    for rep in (False, True):
        print(f"\n=== N={N} repeat={rep} quota {q} ==="); print(run(N, rep, q).round(3).to_string())
