"""Throwaway: lookback (L60) distributions of the shortlisted columns in the weekly candidate pool. No forward data."""
import sys, pickle
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/roll/code")
import numpy as np, pandas as pd
import measures as M
M.log = lambda *a: None
reg = M.registry(); R = {c.name: c for c in reg}
data = M.load(reg)
COLS = ["av_med", "hold_med_h", "hold_turnover_h", "maker_share", "order_ntl_med", "orders_per_active_day_med",
        "active_day_share", "lev_p95", "lev_med", "max_dd", "mae_av_med", "add_loser_share", "forced_ntl_share",
        "top1_share", "exit_rate_ratio", "ret_sum", "ret_mean_sd", "gain_share", "net_bias", "n_symbols"]
rows = []
for Mo in M.MS:
    sL = data["sup"][f"L60_{Mo}"]; sR = data["sup"][f"R28_{Mo}"] if f"R28_{Mo}" in data["sup"] else None
    B = sL["base_eligible"].to_numpy().astype(bool)
    d = pd.DataFrame({"trader": data["U"], "M": Mo, "B": B})
    for c in COLS:
        x, ok = M.values(data, f"L60_{Mo}", R[c])
        g = M.gate(R[c], sL, "L").to_numpy()
        d[c] = np.where(g & ok, x, np.nan)
    d["hip3"] = sL["hip3_asof"].to_numpy().astype(bool)
    for k in ("n_orders", "n_fill_days", "forced_ntl", "turnover", "n_complete"):
        d[k] = sL[k].to_numpy()
    if sR is not None:
        d["r28_orders"] = sR["n_orders"].to_numpy(); d["r28_days"] = sR["n_fill_days"].to_numpy()
    sF = data["sup"][f"F28_{Mo}"]
    d["f28_orders"] = sF["n_orders"].to_numpy(); d["f28_days"] = sF["n_fill_days"].to_numpy()
    for c in ("ret_sum", "max_dd", "ret_mean_sd"):
        x, ok = M.values(data, f"F28_{Mo}", R[c]); g = M.gate(R[c], sF, "F").to_numpy()
        d["F_" + c] = np.where(g & ok, x, np.nan)
    rows.append(d[d["B"]])
P = pd.concat(rows, ignore_index=True)
P.to_parquet("sel_panel.parquet")
print(P.shape, sorted(data["wt"]["kind"].unique()))
