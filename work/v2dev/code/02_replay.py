"""Position / PnL replay from fills only (no account data, no external prices).

For every (trader, symbol) the fill chain gives exact positions; a pooled price tape
(all perp fills of all addresses in the package) gives marks. Value process
    V(t) = cumulative trade cash(t) + position(t) * mark(t)
so PnL over any interval is V(b) - V(a) (gross of fees; fees tracked separately).
Funding is NOT in the package and is not reconstructed.

A position that pre-dates a pair's first fill (`pos0`, from that fill's start_position) is
only *revealed* by that fill. Rows are therefore kept at (trader, symbol) level with the
pair's first-fill time, and two PnL columns are written:
    pnl_gross      hindsight view: pos0 is held from the start of the fill window
    pnl_gross_obs  observable view: nothing before the pair's first fill
Every as-of-T consumer filters pairs with first fill < T or uses the *_obs columns.

Outputs (build/):
  tape.parquet             pooled prints, one row per (symbol, ts): size-weighted price (deterministic)
  groups.parquet           fill groups (trader, symbol, ts) with start/end position, vwap
  forced.parquet           liquidation / ADL / settlement fills
  daily_ts.parquet         per trader x symbol x UTC day
  daily_trader.parquet     per trader x UTC day (hindsight view; used by the audit only)
  snap_value_pair.parquet  V per (trader, symbol) at each account snapshot time (reconciliation input)
  sym_daily.parquet        per-symbol daily closes from the tape (risk weights)
"""
import numpy as np
import pandas as pd
from lib import (BUILD, CUTOFF, FILL_WINDOW_START, fill_groups, load_traders, prep_perp_fills)


def main():
    p = prep_perp_fills()
    # one print per (symbol, ts): size-weighted price. Several fills can share a timestamp at
    # different prices; a deterministic consolidation keeps as-of marks independent of row order.
    tape = p.groupby(["symbol", "ts"], sort=True).agg(n=("notional", "sum"), q=("size", "sum")).reset_index()
    tape["price"] = tape["n"] / tape["q"]
    tape = tape[["symbol", "ts", "price"]]
    tape.to_parquet(BUILD / "tape.parquet")

    forced = p[p["is_forced"]][["trader", "symbol", "ts", "dir", "side", "price", "size", "notional", "closed_pnl", "tid"]]
    forced.to_parquet(BUILD / "forced.parquet")
    g = fill_groups(p)
    g["ccash"] = g.groupby(["trader", "symbol"])["cash"].cumsum()
    g["cfee"] = g.groupby(["trader", "symbol"])["fee"].cumsum()
    g.to_parquet(BUILD / "groups.parquet")

    first = g.groupby(["trader", "symbol"]).head(1)
    pairs = first[["trader", "symbol", "ts", "gstart", "gend", "vwap", "notional", "is_hip3"]].rename(columns={
        "ts": "first_ts", "gstart": "pos0", "gend": "first_gend", "vwap": "first_px", "notional": "first_notional"})

    tp = tape.rename(columns={"ts": "tape_ts"}).assign(ts=lambda d: d["tape_ts"]).sort_values("ts", kind="stable")

    def value_at(query: pd.DataFrame) -> pd.DataFrame:
        """query: trader, symbol, ts  ->  pos, ccash, cfee, mark, mark_age_h, value, exposure (per pair)."""
        q = pd.merge_asof(query.sort_values("ts", kind="stable"), g[["trader", "symbol", "ts", "gend", "ccash", "cfee"]].sort_values("ts", kind="stable"),
                          on="ts", by=["trader", "symbol"], direction="backward")
        q = q.merge(pairs[["trader", "symbol", "pos0"]], on=["trader", "symbol"], how="left")
        q["pos"] = q["gend"].fillna(q["pos0"])          # before the first fill: pre-existing position (hindsight)
        q["ccash"] = q["ccash"].fillna(0.0)
        q["cfee"] = q["cfee"].fillna(0.0)
        q = pd.merge_asof(q.sort_values("ts", kind="stable"), tp, on="ts", by="symbol", direction="backward").reset_index(drop=True)
        miss = q["price"].isna()
        if miss.any():  # no print yet for this symbol: take the first later print (flat mark -> zero PnL until then)
            m = q.loc[miss, ["symbol", "ts"]].assign(_row=lambda x: x.index)
            fwd = pd.merge_asof(m.sort_values("ts", kind="stable"), tp, on="ts", by="symbol", direction="forward").set_index("_row")
            q.loc[fwd.index, "price"] = fwd["price"]
            q.loc[fwd.index, "tape_ts"] = fwd["tape_ts"]
        q["mark_age_h"] = (q["ts"] - q["tape_ts"]).dt.total_seconds() / 3600.0
        q["value"] = q["ccash"] + q["pos"] * q["price"]
        q["exposure"] = q["pos"] * q["price"]
        return q.drop(columns=["gend", "pos0"])

    # ---- UTC-midnight grid ----
    days = pd.date_range(FILL_WINDOW_START, CUTOFF, freq="D")
    grid = pairs[["trader", "symbol"]].merge(pd.DataFrame({"ts": days}), how="cross")
    v = value_at(grid).sort_values(["trader", "symbol", "ts"], kind="stable").reset_index(drop=True)
    grp = v.groupby(["trader", "symbol"])
    v["pnl_gross"] = grp["value"].diff()
    v["fee"] = grp["cfee"].diff()
    v["pos_prev"] = grp["pos"].shift()
    v["exp_prev"] = grp["exposure"].shift()
    v["px_prev"] = grp["price"].shift()
    v["day"] = v["ts"] - pd.Timedelta(days=1)              # label = start of the UTC day
    d = v.dropna(subset=["pnl_gross"]).copy()
    g["day"] = g["ts"].dt.floor("D")
    g["notl_after"] = g["gend"].abs() * g["vwap"]
    act = g.groupby(["trader", "symbol", "day"]).agg(
        turnover=("notional", "sum"), maker_notional=("maker_notional", "sum"),
        forced_notional=("forced_notional", "sum"), n_groups=("n", "size"), n_fills=("n", "sum"),
        closed_pnl=("closed_pnl", "sum"), peak_intraday=("notl_after", "max")).reset_index()
    d = d.merge(act, on=["trader", "symbol", "day"], how="left")
    for c in ["turnover", "maker_notional", "forced_notional", "n_groups", "n_fills", "closed_pnl", "peak_intraday"]:
        d[c] = d[c].fillna(0.0)
    d = d.merge(pairs, on=["trader", "symbol"])
    d["first_day"] = d["first_ts"].dt.floor("D")
    d["pre_first_fill"] = d["day"] < d["first_day"]       # whole day lies before the pair's first fill
    d["peak_abs_exp"] = np.maximum(np.maximum(d["exp_prev"].abs(), d["exposure"].abs()), d["peak_intraday"])
    # observable view: nothing before the first fill; on the first-fill day remove the pos0 leg
    # (pos0 marked from the day's opening mark to the first fill price) and replace the first
    # group's turnover by the trade a follower needs to reach the revealed position.
    is_first = d["day"] == d["first_day"]
    d["pnl_gross_obs"] = np.where(d["pre_first_fill"], 0.0,
                                  np.where(is_first, d["pnl_gross"] - d["pos0"] * (d["first_px"] - d["px_prev"]), d["pnl_gross"]))
    d["turnover_obs"] = np.where(d["pre_first_fill"], 0.0,
                                 np.where(is_first, d["turnover"] - d["first_notional"] + d["first_gend"].abs() * d["first_px"], d["turnover"]))
    d["exposure_obs"] = np.where(d["pre_first_fill"], 0.0, d["exposure"])
    d["peak_abs_exp_obs"] = np.where(d["pre_first_fill"], 0.0, np.where(is_first, np.maximum(d["exposure"].abs(), d["peak_intraday"]), d["peak_abs_exp"]))
    keep = (d["turnover"] > 0) | (d["pos"].abs() > 0) | (d["pos_prev"].abs() > 0)
    d = d.loc[keep, ["trader", "symbol", "day", "pnl_gross", "pnl_gross_obs", "fee", "turnover", "turnover_obs", "maker_notional",
                     "forced_notional", "n_groups", "n_fills", "closed_pnl", "pos", "exposure", "exposure_obs", "exp_prev",
                     "peak_abs_exp", "peak_abs_exp_obs", "mark_age_h", "is_hip3", "pre_first_fill", "first_ts"]]
    d = d.rename(columns={"first_ts": "pair_first_ts"})
    d.to_parquet(BUILD / "daily_ts.parquet")

    # ---- per trader-day aggregate (hindsight view; audit only) ----
    agg = d.assign(abs_exp=d["exposure"].abs()).groupby(["trader", "day"]).agg(
        pnl_gross=("pnl_gross", "sum"), fee=("fee", "sum"), turnover=("turnover", "sum"),
        maker_notional=("maker_notional", "sum"), n_fills=("n_fills", "sum"),
        gross_exp_eod=("abs_exp", "sum"), net_exp_eod=("exposure", "sum")).reset_index()
    agg.to_parquet(BUILD / "daily_trader.parquet")

    # ---- per-symbol daily closes from the pooled tape (risk weights) ----
    dq = pd.DataFrame({"symbol": tape["symbol"].unique()}).merge(pd.DataFrame({"ts": days}), how="cross")
    dc = pd.merge_asof(dq.sort_values("ts", kind="stable"), tp, on="ts", by="symbol", direction="backward").dropna(subset=["price"])
    dc["age_h"] = (dc["ts"] - dc["tape_ts"]).dt.total_seconds() / 3600.0
    dc = dc.rename(columns={"ts": "day_end", "price": "close"})[["symbol", "day_end", "close", "age_h"]]
    dc.to_parquet(BUILD / "sym_daily.parquet")

    # ---- values at account snapshot times, per pair (reconciliation input) ----
    t = load_traders()
    t = t[t["trader"].isin(pairs["trader"].unique()) & (t["ts"] >= FILL_WINDOW_START)]
    q = t[["trader", "ts"]].merge(pairs[["trader", "symbol"]], on="trader")
    sv = value_at(q)[["trader", "symbol", "ts", "value", "cfee", "exposure", "pos", "mark_age_h"]]
    sv = sv.merge(pairs[["trader", "symbol", "first_ts"]].rename(columns={"first_ts": "pair_first_ts"}), on=["trader", "symbol"])
    sv.to_parquet(BUILD / "snap_value_pair.parquet")
    print("pairs", len(pairs), "daily_ts rows", len(d), "trader-days", len(agg), "snapshot-pair rows", len(sv))
    print("mark age (h) on open exposure, quantiles:",
          d.loc[d["pos"].abs() > 0, "mark_age_h"].quantile([.5, .9, .99]).round(3).to_dict())


if __name__ == "__main__":
    main()
