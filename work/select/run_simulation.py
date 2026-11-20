"""Rolling four-sleeve address-selection experiment.

Inputs are frozen rolling-ledger results and prepare_simulation.py outputs.
Each sleeve is funded with USD 25,000, $2,500 per slot, without compounding.
Profit and exposures are multiplied by the formation-time copy multiplier.
Forward profit includes the source trader's fees and funding; it does not model
copying latency or replacement of source fees by the follower's fees.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "work/select"
SEED = 20261120
DRAWS = 1000
SLOT = 2500.0
SLEEVE = 25000.0
TOTAL = 100000.0
HOLD = 28


def annualized_sharpe(daily_returns):
    """Zero risk-free rate, 365 calendar days, sample standard deviation."""
    if len(daily_returns) < 2 or not np.isfinite(daily_returns).all() or np.ptp(daily_returns) == 0:
        return np.nan
    return np.sqrt(365) * daily_returns.mean() / daily_returns.std(ddof=1)


def maximum_drawdown(pnl, capital):
    """Linear profit relative to original capital, including the starting zero."""
    if not np.isfinite(pnl).all():
        return np.nan
    cumulative = np.r_[0.0, np.cumsum(pnl)] / capital
    return float(np.max(np.maximum.accumulate(cumulative) - cumulative))


def ratio(net, gross):
    if not np.isfinite(net).all() or not np.isfinite(gross).all():
        return np.nan
    denominator = gross.sum()
    return float(net.sum() / denominator) if denominator > 0 else np.nan


def trim(ids, weights, old_ids, old_weights, net, gross, threshold):
    """Remove the new member yielding the lowest absolute net/gross, down to six."""
    ids = list(ids)
    weights = list(weights)
    def reading():
        all_ids = np.asarray(old_ids + ids, dtype=int)
        all_weights = np.asarray(old_weights + weights)
        return ratio(net[all_ids] * all_weights, gross[all_ids] * all_weights)
    value = reading()
    while np.isfinite(value) and abs(value) >= threshold and len(ids) > 6:
        options = []
        for i in range(len(ids)):
            j = old_ids + ids[:i] + ids[i + 1:]
            w = np.asarray(old_weights + weights[:i] + weights[i + 1:])
            value_after = ratio(net[j] * w, gross[j] * w)
            options.append(abs(value_after) if np.isfinite(value_after) else np.inf)
        remove = int(np.argmin(options))
        ids.pop(remove)
        weights.pop(remove)
        value = reading()
    return ids, weights, value


def balance_weights(ids, weights, old_ids, old_weights, net, gross, threshold):
    """Scale down positions whose current net contribution has the majority sign."""
    ids = np.asarray(ids, dtype=int)
    weights = np.asarray(weights, dtype=float)
    all_ids = np.r_[np.asarray(old_ids, dtype=int), ids]
    all_weights = np.r_[old_weights, weights]
    value = ratio(net[all_ids] * all_weights, gross[all_ids] * all_weights)
    if not np.isfinite(value) or abs(value) < threshold:
        return ids.tolist(), weights.tolist(), value
    sign = np.sign(value)
    side = sign * net[ids] > 0
    if not side.any():
        return ids.tolist(), weights.tolist(), value
    # The signed constraint is linear in the retained fraction of that side.
    n = net[all_ids] * all_weights
    g = gross[all_ids] * all_weights
    ns = (net[ids[side]] * weights[side]).sum()
    gs = (gross[ids[side]] * weights[side]).sum()
    target = threshold - 1e-10
    numerator = sign * n.sum() - target * g.sum()
    denominator = sign * ns - target * gs
    if denominator <= 0:
        return ids.tolist(), weights.tolist(), value
    reduction = min(1.0, max(0.0, numerator / denominator))
    weights[side] *= 1 - reduction
    active = weights > 0
    ids = ids[active]
    weights = weights[active]
    all_ids = np.r_[np.asarray(old_ids, dtype=int), ids]
    all_weights = np.r_[old_weights, weights]
    value = ratio(net[all_ids] * all_weights, gross[all_ids] * all_weights)
    return ids.tolist(), weights.tolist(), value


def get_data():
    raw = np.load(HERE / "simulation_exposure.npz")
    traders = raw["traders"].astype(str)
    exposure_dates = pd.DatetimeIndex(raw["dates"])
    dates = pd.date_range("2025-09-29", "2026-01-04")
    close_indices = exposure_dates.get_indexer(dates + pd.Timedelta(days=1))
    monday_indices = exposure_dates.get_indexer(pd.date_range("2025-09-29", periods=11, freq="W-MON"))
    assert (close_indices >= 0).all() and (monday_indices >= 0).all()
    profits = np.load(HERE / "simulation_outcomes.npz")["pnl"]
    assert profits.shape == (len(dates), len(traders))
    panel = pd.read_parquet(HERE / "sel_panel2.parquet")
    panel = panel[panel.ret_mean_sd.notna()].copy()
    panel["M"] = pd.to_datetime(panel.M)
    panel = panel.merge(pd.read_parquet(HERE / "simulation_labels.parquet"), on=["M", "trader"], how="left")
    panel = panel.merge(pd.read_parquet(HERE / "simulation_volatility.parquet"), on=["M", "trader"], how="left")
    panel["direction"] = panel.direction.fillna("neutral")
    panel["id"] = pd.Index(traders).get_indexer(panel.trader)
    assert (panel.id >= 0).all() and not panel.duplicated(["M", "trader"]).any()
    admission = ((panel.av_med >= 5000) & (panel.order_ntl_med * SLOT / panel.av_med >= 10)
                 & ~(panel.hold_med_h < 1) & (panel.maker_share <= .8) & (panel.r28_days >= 7))
    risk = (~(panel.lev_p95 > 8) & (panel.max_dd <= .6) & (panel.liq <= 0)
            & ~(panel.mae_av_med > .03) & ~(panel.add_loser_share > .9))
    panel["admission"] = admission
    panel["qualified"] = admission & risk
    return dict(panel=panel, traders=traders, pnl=profits, net=raw["net"][close_indices],
                gross=raw["gross"][close_indices], monday_net=raw["net"][monday_indices],
                monday_gross=raw["gross"][monday_indices], dates=dates)


def simulate(data, scenario, draw, rng):
    dates = sorted(data["panel"].M.unique())
    held = []
    groups = []
    allocations = []
    combined_pnl = np.zeros(len(data["dates"]))
    combined_net = np.zeros(len(data["dates"]))
    combined_gross = np.zeros(len(data["dates"]))
    combined_allocation = np.zeros(len(data["dates"]))
    combined_pnl_bad = np.zeros(len(data["dates"]), dtype=bool)
    combined_exp_bad = np.zeros(len(data["dates"]), dtype=bool)
    for week, date in enumerate(dates):
        candidates = data["panel"][data["panel"].M == date]
        if scenario == "1":
            candidates = candidates[candidates.admission]
        elif scenario not in ("0",):
            candidates = candidates[candidates.qualified]
        old = [h for h in held if h["week"] > week - 4]
        old_ids = [int(i) for h in old for i in h["ids"]]
        old_weights = [float(i) for h in old for i in h["weights"]]
        if scenario != "4a":
            candidates = candidates[~candidates.id.isin(old_ids)]
        candidates = candidates.sort_values(["ret_mean_sd", "trader"], ascending=[False, True])
        if scenario in ("0", "1", "2"):
            count = min(10, len(candidates))
            chosen = candidates.iloc[rng.choice(len(candidates), size=count, replace=False)]
        elif scenario == "3":
            chosen = candidates.head(10)
        else:
            candidates = candidates[candidates.ret_mean_sd > 0]
            a = candidates[candidates.direction == "long-like"].head(4)
            b = candidates[candidates.direction == "short-like"].head(4)
            used = set(a.id) | set(b.id)
            c = candidates[~candidates.id.isin(used)].head(2)
            chosen = pd.concat([a, b, c])
        ids = chosen.id.astype(int).tolist()
        av = chosen.av_med.to_numpy(dtype=float)
        weights = (SLOT / av).tolist()
        reason = ""
        if scenario == "4b":
            sd = chosen.pnl_sd.to_numpy(dtype=float)
            if not np.all(np.isfinite(sd) & (sd > 0)):
                reason = "missing_formation_volatility"
            elif len(ids):
                inv = 1 / sd
                weights = (len(ids) * SLOT * inv / np.dot(inv, av)).tolist()
        threshold = .3 if scenario == "4d" else .2
        proposed_value = np.nan
        if scenario.startswith("4") and not reason:
            args = (ids, weights, old_ids, old_weights, data["monday_net"][week],
                    data["monday_gross"][week], threshold)
            ids, weights, proposed_value = balance_weights(*args) if scenario == "4c" else trim(*args)
            if len(ids) < 6:
                reason = "fewer_than_six"
            elif not np.isfinite(proposed_value):
                reason = "exposure_undefined"
            elif abs(proposed_value) >= threshold:
                reason = "exposure_limit"
        proposed_ids, proposed_weights = ids.copy(), weights.copy()
        if reason:
            ids, weights = [], []
        held.append(dict(week=week, ids=ids, weights=weights))
        start = week * 7
        end = start + HOLD
        idx = np.asarray(ids, dtype=int)
        w = np.asarray(weights)
        p = data["pnl"][start:end, idx] @ w if len(idx) else np.zeros(HOLD)
        n = data["net"][start:end, idx] @ w if len(idx) else np.zeros(HOLD)
        g = data["gross"][start:end, idx] @ w if len(idx) else np.zeros(HOLD)
        badp = ~np.isfinite(p)
        bade = ~np.isfinite(n) | ~np.isfinite(g)
        combined_pnl[start:end] += np.nan_to_num(p)
        combined_net[start:end] += np.nan_to_num(n)
        combined_gross[start:end] += np.nan_to_num(g)
        combined_pnl_bad[start:end] |= badp
        combined_exp_bad[start:end] |= bade
        alloc = float(np.dot(chosen.set_index("id").loc[idx, "av_med"], w)) if len(idx) else 0.0
        combined_allocation[start:end] += alloc
        allocations.append(alloc)
        group_net_ratio = np.divide(n, g, out=np.full_like(n, np.nan), where=(g > 0) & ~bade)
        groups.append(dict(scenario=scenario, draw=draw, week=str(pd.Timestamp(date).date()),
                           members=len(ids), proposed_members=len(proposed_ids), manual_reason=reason,
                           proposed_net_gross=proposed_value, allocated=alloc,
                           funded_fraction=alloc / SLEEVE,
                           return_28d=p.sum() / SLEEVE if not badp.any() else np.nan,
                           annualized_sharpe_28d=annualized_sharpe(p / SLEEVE),
                           max_drawdown_28d=maximum_drawdown(p, SLEEVE),
                           missing_profit_days=int(badp.sum()),
                           gross_exposure_zero_days=int((g == 0).sum()),
                           median_abs_daily_net_gross=np.nanmedian(np.abs(group_net_ratio)) if np.isfinite(group_net_ratio).any() else np.nan,
                           proposed_traders="|".join(data["traders"][proposed_ids]),
                           proposed_weights="|".join(f"{x:.12g}" for x in proposed_weights),
                           traders="|".join(data["traders"][idx]),
                           weights="|".join(f"{x:.12g}" for x in weights)))
    combined_pnl[combined_pnl_bad] = np.nan
    combined_net[combined_exp_bad] = np.nan
    combined_gross[combined_exp_bad] = np.nan
    daily_ratio = np.divide(combined_net, combined_gross, out=np.full_like(combined_net, np.nan),
                            where=np.isfinite(combined_gross) & (combined_gross > 0))
    # Full-four-sleeve interval starts at the fourth allocation, ends with the
    # last scheduled rebalancing period. The last 21 days have no new sleeves.
    window = slice(21, len(dates) * 7)
    path = dict(scenario=scenario, draw=draw,
                manual_weeks=sum(bool(x["manual_reason"]) for x in groups),
                missing_pnl_days=int(combined_pnl_bad.sum()),
                portfolio_return=combined_pnl.sum() / TOTAL if np.isfinite(combined_pnl).all() else np.nan,
                full_period_annualized_sharpe=annualized_sharpe(combined_pnl / TOTAL),
                full_period_daily_volatility=np.std(combined_pnl / TOTAL, ddof=1) if np.isfinite(combined_pnl).all() else np.nan,
                full_period_maximum_drawdown=maximum_drawdown(combined_pnl, TOTAL),
                daily_volatility=np.std(combined_pnl[window] / TOTAL, ddof=1) if np.isfinite(combined_pnl[window]).all() else np.nan,
                maximum_drawdown=maximum_drawdown(combined_pnl[window], TOTAL),
                median_abs_net_gross=np.nanmedian(np.abs(daily_ratio[window])) if np.isfinite(daily_ratio[window]).any() else np.nan,
                mean_funded_fraction=float(np.mean(combined_allocation[window] / TOTAL)))
    daily = pd.DataFrame(dict(scenario=scenario, draw=draw, day=data["dates"], pnl=combined_pnl,
                              net=combined_net, gross=combined_gross, net_gross=daily_ratio,
                              allocated=combined_allocation, funded_fraction=combined_allocation / TOTAL))
    return groups, path, daily


def checks():
    ids, w, x = trim([0, 1, 2, 3, 4, 5, 6], [1.] * 7, [], [],
                     np.array([10., -1., 0, 0, 0, 0, 0]), np.array([10., 1., 1, 1, 1, 1, 1]), .2)
    assert len(ids) == 6 and abs(x) < .2 and 0 not in ids
    assert np.isclose(maximum_drawdown(np.array([1., -3., 1.]), 10), .3)
    _, w, x = balance_weights([0, 1], [1., 1.], [], [], np.array([10., -5.]), np.array([10., 5.]), .2)
    assert np.isclose(x, .2) and np.isclose(w[1], 1) and w[0] < 1


def main():
    checks()
    d = get_data()
    all_groups, paths, daily = [], [], []
    scenarios = ("0", "1", "2", "3", "4", "4a", "4b", "4c", "4d")
    for scenario in scenarios:
        rng = np.random.default_rng(SEED)
        count = DRAWS if scenario in ("0", "1", "2") else 1
        for draw in range(count):
            groups, path, day = simulate(d, scenario, draw, rng)
            all_groups.extend(groups)
            paths.append(path)
            if draw == 0:
                daily.append(day)
        print(f"{scenario}: {count} paths finished", flush=True)
    groups = pd.DataFrame(all_groups)
    paths = pd.DataFrame(paths)
    groups.to_parquet(HERE / "simulation_groups.parquet", index=False)
    paths.to_csv(HERE / "simulation_paths.csv", index=False)
    pd.concat(daily).to_csv(HERE / "simulation_daily_first_path.csv", index=False)
    with open(HERE / "simulation_settings.json", "w") as f:
        json.dump(dict(seed=SEED, draws=DRAWS, slot=SLOT, sleeve=SLEEVE, total=TOTAL,
                       start="2025-09-29", last_formation="2025-12-08",
                       end_exclusive="2026-01-05", volatility_window="four-sleeve interval 2025-10-20 to 2025-12-15 exclusive",
                       price_missing="mark dependent group and path metrics unknown",
                       manual="new sleeve is cash, existing sleeves continue"), f, indent=2)
    print("group counts", len(groups), "paths", len(paths))
    print(paths.groupby("scenario")[["manual_weeks", "missing_pnl_days", "portfolio_return",
                                      "daily_volatility", "maximum_drawdown", "median_abs_net_gross"]].median().to_string())


if __name__ == "__main__":
    main()
