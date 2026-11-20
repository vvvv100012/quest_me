"""Prepare point-in-time exposures and forward dollar profit for the selection experiment.

Run with the project Python environment from the repository root. Prices, positions,
funding, and profit use the reviewed rolling-ledger construction. Direction labels
retain the definition and data series of the existing direction study.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/select"
sys.path.insert(0, str(ROOT / "work/roll/code"))
sys.path.insert(0, str(ROOT / "work/profile"))
import ledger as ledger
import returns as returns
import direction


def main():
    inp = ledger.load_inputs()
    universe = sorted(inp["p"]["trader"].unique())
    state = returns.build_r(inp, universe)
    dates = pd.date_range("2025-09-29", "2026-01-05", freq="D")
    index = {trader: i for i, trader in enumerate(universe)}
    panel = pd.read_parquet(OUT / "sel_panel2.parquet")
    mondays = sorted(pd.to_datetime(panel["M"].unique()))
    prices = state.PB[state.pair_si]
    net = np.zeros((len(dates), len(universe)))
    gross = np.zeros_like(net)
    bad = np.zeros_like(net, dtype=bool)
    for k, day in enumerate(dates):
        cutoff = day.value
        boundary = (cutoff - ledger.W0N) // ledger.DAY
        # A pair's first fill must already be known at the observation boundary.
        known = state.first_ts < cutoff
        position = state.Q[:, boundary]
        relevant = known & (position != 0)
        values = position[relevant] * prices[relevant, boundary]
        ti = state.pair_ti[relevant]
        np.add.at(net[k], ti, np.nan_to_num(values))
        np.add.at(gross[k], ti, np.nan_to_num(np.abs(values)))
        bad[k, ti[~np.isfinite(values)]] = True
        fd = state.P.fday
        fm = (fd["pkg_first_ns"] >= cutoff) & (fd["day_ns"] == cutoff - ledger.DAY)
        f = fd.loc[fm]
        v = f["szi"].to_numpy() * state.PB[f["sym_i"].to_numpy(), boundary]
        ti = f["ti"].to_numpy()
        np.add.at(net[k], ti, np.nan_to_num(v))
        np.add.at(gross[k], ti, np.nan_to_num(np.abs(v)))
        bad[k, ti[~np.isfinite(v)]] = True
    # Independently compare all Monday openings to the existing position ledger.
    op = pd.read_parquet(ROOT / "work/roll/build/open_positions.parquet")
    for day in mondays:
        o = op[op.cutoff == day].copy()
        assert o.ntl.notna().all(), f"unpriced open position on {day}"
        o["signed"] = o.sign * o.ntl
        g = o.groupby("trader")[["signed", "ntl"]].sum()
        ids = np.array([index[t] for t in g.index])
        i = dates.get_loc(day)
        assert np.allclose(net[i, ids], g.signed, atol=1e-7, rtol=1e-10)
        assert np.allclose(gross[i, ids], g.ntl, atol=1e-7, rtol=1e-10)
        extra = set(np.flatnonzero(gross[i])) - set(ids)
        assert not extra, f"{day}: extra addresses {[(universe[j], gross[i,j]) for j in list(extra)[:5]]}"
    net[bad] = np.nan
    gross[bad] = np.nan
    sd_rows = []
    for day in mondays:
        daily = returns.daily_at(state, day.value)
        end = daily["nC"]
        lo = end - 60
        pnl = daily["pnl"][:, lo:end]
        active = daily["act"][:, lo:end]
        # Dollar profit standard deviation on in-market days, including inactive
        # observations only when positions or funding make the day active.
        usable = active & (daily["reason"][:, lo:end] != 1) & np.isfinite(pnl)
        n = usable.sum(axis=1)
        mean = np.divide(np.where(usable, pnl, 0).sum(axis=1), n,
                         out=np.zeros(len(universe)), where=n > 0)
        sse = np.where(usable, (pnl - mean[:, None]) ** 2, 0).sum(axis=1)
        sd = np.sqrt(np.divide(sse, n - 1, out=np.full(len(universe), np.nan), where=n > 1))
        sd_rows.append(pd.DataFrame({"trader": universe, "M": day, "pnl_sd": sd,
                                     "sd_active_days": n}))
    outcome_pnl = np.zeros((len(dates) - 1, len(universe)))
    for i, day in enumerate(dates[:-1]):
        outcome = returns.daily_at(state, (day + pd.Timedelta(days=1)).value)
        end = outcome["nC"]
        values = outcome["pnl"][:, end - 1].copy()
        reason = outcome["reason"][:, end - 1]
        values[reason == 1] = np.nan
        values[~outcome["span"][:, end - 1]] = 0.0
        outcome_pnl[i] = values
    np.savez_compressed(OUT / "simulation_outcomes.npz", pnl=outcome_pnl)
    pd.concat(sd_rows, ignore_index=True).to_parquet(OUT / "simulation_volatility.parquet", index=False)
    np.savez_compressed(OUT / "simulation_exposure.npz", dates=dates.to_numpy(), traders=np.array(universe),
                        net=net, gross=gross)
    td = direction.trader_day(direction.rows(direction.FULL))
    btc = direction.btc_close(direction.FULL).pct_change().dropna()
    btc.index -= pd.Timedelta(days=1)
    td = td.merge(btc.rename("btc").rename_axis("day").reset_index(), on="day")
    labels = []
    for day in mondays:
        sample = td[(td.day >= day - pd.Timedelta(days=60)) & (td.day < day)]
        scores = direction.scores(sample, 20)
        scores = scores[scores.eligible]
        label = direction.classify(scores.A)
        labels.append(pd.DataFrame({"M": day, "trader": label.index, "direction": label.values}))
    pd.concat(labels, ignore_index=True).to_parquet(OUT / "simulation_labels.parquet", index=False)
    print(f"prepared {len(universe)} addresses, {len(dates)} daily closes, {len(mondays)} Mondays")
    print("Monday exposure matched open_positions.parquet, including funding-only positions")
    print(f"unpriced exposure cells: {bad.sum()}")


if __name__ == "__main__":
    main()
