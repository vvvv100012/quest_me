"""Generate report tables from the simulated groups and paths."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
base = root / "work/select"
paths = pd.read_csv(base / "simulation_paths.csv")
groups = pd.read_parquet(base / "simulation_groups.parquet")
path_metrics = ["portfolio_return", "full_period_daily_volatility", "full_period_maximum_drawdown",
                "full_period_annualized_sharpe", "median_abs_net_gross", "mean_funded_fraction"]
paths.groupby("scenario")[path_metrics].median().to_csv(base / "summary_portfolios.csv")
by_path = groups.groupby(["scenario", "draw"])
summary = by_path.agg(return_median=("return_28d", "median"),
                      drawdown_median=("max_drawdown_28d", "median"),
                      sharpe_median=("annualized_sharpe_28d", "median"),
                      losing_fraction=("return_28d", lambda x: (x < 0).mean()),
                      worst_return=("return_28d", "min"),
                      manual_weeks=("manual_reason", lambda x: (x != "").sum()))
summary.groupby("scenario").median().to_csv(base / "summary_groups.csv")
groups[(groups.scenario == "4") & (groups.draw == 0)][
    ["week", "members", "proposed_members", "manual_reason", "proposed_net_gross",
     "allocated", "return_28d", "max_drawdown_28d", "annualized_sharpe_28d"]
].to_csv(base / "summary_main_weekly.csv", index=False)
panel = pd.read_parquet(base / "sel_panel2.parquet")
panel = panel[panel.ret_mean_sd.notna()]
admit = ((panel.av_med >= 5000) & (panel.order_ntl_med * 2500 / panel.av_med >= 10)
         & ~(panel.hold_med_h < 1) & (panel.maker_share <= .8) & (panel.r28_days >= 7))
risk = (~(panel.lev_p95 > 8) & (panel.max_dd <= .6) & (panel.liq <= 0)
        & ~(panel.mae_av_med > .03) & ~(panel.add_loser_share > .9))
funnel = panel.assign(admission=admit, qualified=admit & risk).groupby("M").agg(
    candidates=("trader", "size"), admitted=("admission", "sum"), qualified=("qualified", "sum"))
funnel.to_csv(base / "summary_funnel.csv")
raw = np.load(base / "simulation_exposure.npz")
traders = pd.Index(raw["traders"].astype(str))
outcomes = np.load(base / "simulation_outcomes.npz")["pnl"]
main = groups[groups.scenario == "4"].set_index("week").sort_index()
comparison = groups[groups.scenario == "3"].sort_values("week")
profit = np.zeros(len(outcomes))
for i, row in enumerate(comparison.itertuples()):
    indices = traders.get_indexer(row.traders.split("|"))
    weights = np.array([float(w) for w in row.weights.split("|")])
    profit[i * 7:i * 7 + 28] += (outcomes[i * 7:i * 7 + 28, indices] @ weights
                                    * main.loc[row.week, "allocated"] / row.allocated)
matched = profit[21:77].std(ddof=1) / 100000
r = paths.set_index("scenario")
pd.DataFrame([{
    "unscaled_top_score_daily_volatility": r.loc["3", "daily_volatility"],
    "matched_funding_top_score_daily_volatility": matched,
    "main_daily_volatility": r.loc["4", "daily_volatility"],
}]).to_csv(base / "summary_matched_funding.csv", index=False)
print("Report tables: summary_portfolios.csv, summary_groups.csv, summary_main_weekly.csv,")
print("summary_funnel.csv, summary_matched_funding.csv")
print(paths[paths.scenario == "4"][path_metrics].to_string(index=False))
