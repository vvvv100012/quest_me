import numpy as np, pandas as pd
pd.set_option('display.width',250,'display.max_columns',40,'display.max_rows',100)
P = pd.read_parquet("sel_panel2.parquet")
pool = P[P.ret_mean_sd.notna()].copy()
ALLOC = 5000.0
pool["ratio"] = np.minimum(1.0, ALLOC / pool["av_med"])
pool["scaled_order"] = pool["order_ntl_med"] * ALLOC / pool["av_med"]
print("scaled median order quantiles:", pool.scaled_order.quantile([.02,.05,.1,.2,.5]).round(1).to_dict())
print("order/av quantiles:", (pool.order_ntl_med/pool.av_med).quantile([.02,.05,.1,.2,.5,.8]).round(4).to_dict())
# recency vs forward activity (B rows, all)
P["f_active"] = (P.f28_orders >= 15) & (P.f28_days >= 5)
P["f_any"] = P.f28_orders > 0
b = pd.cut(P.r28_days, [-1, 0, 2, 4, 6, 9, 14, 28])
print(P.groupby(b).agg(n=("f_active", "size"), f_active=("f_active", "mean"), f_any=("f_any", "mean")).round(3))
pool["f_active"] = (pool.f28_orders >= 15) & (pool.f28_days >= 5)
# last-14-day activity not available; use r28
rules = {
 "A1 av>=5000": pool.av_med >= 5000,
 "A1b av>=2000": pool.av_med >= 2000,
 "A1c av>=10000": pool.av_med >= 10000,
 "A2 scaled order>=10": pool.scaled_order >= 10,
 "A2b scaled order>=20": pool.scaled_order >= 20,
 "A3 hold>=1h|na": ~(pool.hold_med_h < 1),
 "A3b hold>=0.5h|na": ~(pool.hold_med_h < 0.5),
 "A3c hold>=4h|na": ~(pool.hold_med_h < 4),
 "A4 maker<=0.8": pool.maker_share <= 0.8,
 "A4b maker<=0.6": pool.maker_share <= 0.6,
 "A5 r28days>=5": pool.r28_days >= 5,
 "A5b r28days>=7": pool.r28_days >= 7,
 "A5c r28days>=10": pool.r28_days >= 10,
 "R1 lev95<=8|na": ~(pool.lev_p95 > 8),
 "R1b lev95<=5|na": ~(pool.lev_p95 > 5),
 "R1c lev95<=10|na": ~(pool.lev_p95 > 10),
 "R2 dd<=0.6": pool.max_dd <= 0.6,
 "R2b dd<=0.5": pool.max_dd <= 0.5,
 "R2c dd<=0.3": pool.max_dd <= 0.3,
 "R3 no liquidation": pool.liq <= 0,
 "R4 mae<=3%|na": ~(pool.mae_av_med > 0.03),
 "R4b mae<=2%|na": ~(pool.mae_av_med > 0.02),
 "R5 addloser<=0.8|na": ~(pool.add_loser_share > 0.8),
 "R5b addloser<=0.9|na": ~(pool.add_loser_share > 0.9),
}
R = pd.DataFrame(rules)
t = pd.DataFrame({"pass_share": R.mean(), "min_wk": R.groupby(pool.M).mean().min(), "max_wk": R.groupby(pool.M).mean().max(),
                  "f_active_pass": [pool.f_active[R[c]].mean() for c in R], "f_active_fail": [pool.f_active[~R[c]].mean() for c in R]})
print(t.round(3).to_string())
adm = R["A1 av>=5000"] & R["A2 scaled order>=10"] & R["A3 hold>=1h|na"] & R["A4 maker<=0.8"] & R["A5b r28days>=7"]
risk = R["R1 lev95<=8|na"] & R["R2 dd<=0.6"] & R["R3 no liquidation"] & R["R4 mae<=3%|na"] & R["R5b addloser<=0.9|na"]
pool["adm"], pool["risk"] = adm, risk
fn = pool.groupby("M").agg(pool=("adm", "size"), adm=("adm", "sum"))
fn["adm_risk"] = (adm & risk).groupby(pool.M).sum()
surv = pool[adm & risk].copy()
surv["rk"] = surv.groupby("M")["ret_mean_sd"].rank(pct=True, ascending=False)
fn["band20"] = (surv.rk <= 0.2).groupby(surv.M).sum()
fn["band_min_score"] = surv[surv.rk <= 0.2].groupby("M")["ret_mean_sd"].min().round(3)
fn["band_min_retsum"] = surv[surv.rk <= 0.2].groupby("M")["ret_sum"].min().round(3)
fn["band_long"] = (surv[surv.rk <= 0.2].net_bias > 0.3).groupby(surv.M).sum()
fn["band_short"] = (surv[surv.rk <= 0.2].net_bias < -0.3).groupby(surv.M).sum()
print(fn.to_string()); print(fn.mean().round(1).to_dict())
# incremental removals inside admission and risk
seq = ["A1 av>=5000", "A2 scaled order>=10", "A3 hold>=1h|na", "A4 maker<=0.8", "A5b r28days>=7", "R1 lev95<=8|na", "R2 dd<=0.6", "R3 no liquidation", "R4 mae<=3%|na", "R5b addloser<=0.9|na"]
m = pd.Series(True, index=pool.index); nW = pool.M.nunique()
for s in seq:
    m2 = m & R[s]; print(f"{s:24s} removes {(m.sum()-m2.sum())/nW:6.1f}/wk  left {m2.sum()/nW:6.1f}"); m = m2
cols = ["lev_p95", "max_dd", "mae_av_med", "add_loser_share", "top1_share", "exit_rate_ratio", "ret_mean_sd", "ret_sum", "av_med", "hold_med_h", "maker_share"]
print(pool[cols].corr(method="spearman").round(2).to_string())
# repeat presence in band on consecutive Mondays
b = surv[surv.rk <= 0.2]
Ms = sorted(b.M.unique()); 
for i in range(1, len(Ms)):
    a0, a1 = set(b[b.M == Ms[i-1]].trader), set(b[b.M == Ms[i]].trader)
    print(Ms[i], len(a1), "also in previous band:", len(a0 & a1))
