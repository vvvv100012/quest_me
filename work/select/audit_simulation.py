"""Independent row-wise reconstruction of saved simulation paths and decisions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "work/select"


def pairs(row):
    if not row.traders:
        return []
    return list(zip(row.traders.split("|"), map(float, row.weights.split("|"))))


def main():
    groups = pd.read_parquet(HERE / "simulation_groups.parquet")
    daily = pd.read_csv(HERE / "simulation_daily_first_path.csv", parse_dates=["day"])
    paths = pd.read_csv(HERE / "simulation_paths.csv")
    pnl = pd.read_parquet(ROOT / "work/roll/build/daily_pnl.parquet")
    pnl["trader"] = pnl.trader.astype(str)
    pnl = pnl.set_index(["day", "trader"])
    assert pnl.index.is_unique
    full_profit = pnl["pnl"].to_dict()
    raw = np.load(HERE / "simulation_exposure.npz")
    available_pnl = np.load(HERE / "simulation_outcomes.npz")["pnl"]
    traders = pd.Index(raw["traders"].astype(str))
    dates = pd.DatetimeIndex(raw["dates"])
    panel = pd.read_parquet(HERE / "sel_panel2.parquet").set_index(["M", "trader"])
    count = 0
    maxerr = 0.0
    cutoff_differences = {}
    for scenario in daily.scenario.unique():
        selection = groups[(groups.scenario == scenario) & (groups.draw == 0)].copy()
        selection["start"] = pd.to_datetime(selection.week)
        output = daily[daily.scenario == scenario].set_index("day")
        actual_total = 0.0
        full_total = 0.0
        for day, row in output.iterrows():
            active = selection[(selection.start <= day) & (day < selection.start + pd.Timedelta(days=28))]
            expected_pnl, expected_net, expected_gross, expected_alloc = 0., 0., 0., 0.
            exposure_day = dates.get_loc(day + pd.Timedelta(days=1))
            for r in active.itertuples(index=False):
                for trader, weight in pairs(r):
                    i = traders.get_loc(trader)
                    value = available_pnl[dates.get_loc(day), i]
                    expected_pnl += weight * value
                    actual_total += weight * value
                    full_total += weight * full_profit.get((day, trader), 0.0)
                    expected_net += weight * raw["net"][exposure_day, i]
                    expected_gross += weight * raw["gross"][exposure_day, i]
                    expected_alloc += weight * panel.loc[(r.week, trader), "av_med"]
                    count += 1
            for name, want in (("pnl", expected_pnl), ("net", expected_net), ("gross", expected_gross),
                               ("allocated", expected_alloc)):
                got = row[name]
                assert np.isclose(got, want, rtol=2e-11, atol=1e-7, equal_nan=True), (scenario, day, name, got, want)
                if np.isfinite(want):
                    maxerr = max(maxerr, abs(got - want))
        selected_path = paths[(paths.scenario == scenario) & (paths.draw == 0)].iloc[0]
        filled = output.loc["2025-10-20":"2025-12-14", "funded_fraction"].mean()
        assert np.isclose(filled, selected_path.mean_funded_fraction)
        cutoff_differences[scenario] = (actual_total - full_total) / 100000
    no_repeat = groups[groups.scenario.isin(["0", "1", "2", "3", "4", "4b", "4c", "4d"])]
    for (_, _), g in no_repeat.groupby(["scenario", "draw"]):
        previous = []
        for r in g.itertuples(index=False):
            current = set(t for t, _ in pairs(r))
            assert not current.intersection(set().union(*previous[-3:])) if previous else True
            previous.append(current)
            if r.manual_reason:
                assert not current
    print(f"independent daily reconstruction passed: {count} member-days, max absolute error {maxerr:.3g}")
    print(f"no-repeat and manual-cash checks passed on {len(no_repeat)} sleeve records")
    print("return difference between next-day and full-cutoff accounting:", cutoff_differences)


if __name__ == "__main__":
    main()
