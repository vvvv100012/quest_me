import sys
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/v2dev/code")
import numpy as np, pandas as pd
import lib
pd.set_option('display.width',250,'display.max_columns',40,'display.max_rows',100)
P = pd.read_parquet("sel_panel.parquet")
f = lib.load_fills()
f["ntl"] = f["size"] * f["price"]
fo = f[f["is_forced"]].copy()
fo["kind"] = np.where(fo["dir"].str.startswith("Liquidated"), "liq", np.where(fo["dir"].str.startswith("Auto"), "adl", "settle"))
print(fo.groupby(["kind"]).agg(n=("ntl", "size"), ntl=("ntl", "sum"), traders=("trader", "nunique")))
print(fo["dir"].value_counts().head(12))
out = []
for M in sorted(P["M"].unique()):
    T = pd.Timestamp(M); a = T - pd.Timedelta(days=60)
    w = fo[(fo["ts"] >= a) & (fo["ts"] < T)]
    g = w.pivot_table(index="trader", columns="kind", values="ntl", aggfunc="sum").fillna(0.0)
    n = w[w["kind"] == "liq"].groupby("trader")["ts"].apply(lambda s: s.dt.floor("D").nunique()).rename("liq_days")
    g = g.join(n).reset_index(); g["M"] = M
    out.append(g)
G = pd.concat(out).fillna(0.0)
for c in ("liq", "adl", "settle", "liq_days"):
    if c not in G: G[c] = 0.0
P = P.merge(G[["trader", "M", "liq", "adl", "settle", "liq_days"]], on=["trader", "M"], how="left").fillna({"liq": 0, "adl": 0, "settle": 0, "liq_days": 0})
P["liq_share"] = P["liq"] / P["turnover"]
P.to_parquet("sel_panel2.parquet")
pool = P[P.ret_mean_sd.notna()]
print("pool rows", len(pool))
print("any liq", (pool.liq > 0).mean(), "any adl", (pool.adl > 0).mean(), "any settle", (pool.settle > 0).mean())
print("liq_share quantiles among liq>0:", pool.loc[pool.liq > 0, "liq_share"].quantile([.1, .25, .5, .75, .9]).round(4).to_dict())
print("liq_share >=1%:", (pool.liq_share >= .01).mean(), ">=5%:", (pool.liq_share >= .05).mean(), "liq_days>=2:", (pool.liq_days >= 2).mean())
