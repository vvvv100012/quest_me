"""Shared loading / cleaning helpers. Every script reads raw CSVs through here so
that the cleaning rules are stated exactly once.

Raw files (never modified):
  traders: 笔试题数据包-traders.csv   daily-ish account snapshots
  fills:   笔试题数据包-fills          fill-level records (CSV without extension)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ARK_DATA_DIR", ROOT.parent / "ArkStream笔试题_v2"))
BUILD = Path(os.environ.get("ARK_BUILD_DIR", ROOT / "build"))
TABLES = Path(os.environ.get("ARK_TABLES_DIR", ROOT / "tables"))
BUILD.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
# Prefix mode (test_prefix_invariance.py): physically drop every raw record with timestamp >= ARK_ASOF.
_ASOF_ENV = pd.Timestamp(os.environ["ARK_ASOF"]) if os.environ.get("ARK_ASOF") else None

TRADERS_CSV = DATA_DIR / "笔试题数据包-traders.csv"
FILLS_CSV = DATA_DIR / "笔试题数据包-fills"

# Decision time. The package ships no README stating the cutoff, so the default is the first
# UTC midnight after the last record in either file (last fill 2026-01-04 23:59:42.020 UTC,
# last snapshot 2026-01-04 23:59:31.671). An earlier cutoff is a one-variable change:
#     ARK_CUTOFF="2025-12-22 00:00:00" ./run_all.sh
# Every raw record at or after the cutoff is dropped by the loaders, and the historical
# decision dates are derived from it (decision_dates below).
CUTOFF = pd.Timestamp(os.environ.get("ARK_CUTOFF", "2026-01-05 00:00:00"))
FILL_WINDOW_START = pd.Timestamp("2025-07-27 00:00:00")  # first fill 2025-07-27 08:52:47 UTC
ASOF = CUTOFF if _ASOF_ENV is None else min(_ASOF_ENV, CUTOFF)   # loaders keep ts < ASOF


def decision_dates(formation_days: int = 56, horizon_days: int = 28) -> pd.DatetimeIndex:
    """Historical decision times for the walk-forward: every Monday 00:00 UTC that has a full
    formation window of fills behind it and a full forward window before the cutoff."""
    first = FILL_WINDOW_START + pd.Timedelta(days=formation_days)
    last = CUTOFF - pd.Timedelta(days=horizon_days)
    d = pd.date_range(first.normalize(), last.normalize(), freq="D")
    return d[d.dayofweek == 0]

SPOT_DIRS = {"Buy", "Sell", "Spot Dust Conversion"}
POS_EPS = 1e-9             # coins; far below any lot size, far above float-sum residue (max seen ~1e-11)
FORCED_DIRS_PREFIX = ("Liquidated", "Auto-Deleveraging", "Settlement")


def load_traders() -> pd.DataFrame:
    """Account snapshots. Adds:
      ts        parsed timestamp (naive; shown to be UTC by 03_reconcile offset test)
      acct_val  pnl + net_deposit. ASSUMPTION (report A14): `pnl` is all-time PnL of the whole
                perp account, so pnl + net_deposit is its total value, while `equity`
                (== cross_margin_summary_account_value) covers cross margin only. The package
                has no field that settles this; evidence is in tables/audit_summary.csv.
      iso_gap   acct_val - equity  (>0 when isolated positions / non-cross value exist)
    """
    t = pd.read_csv(TRADERS_CSV)
    assert (t["user"] == t["trader"]).all()
    assert np.allclose(t["equity"], t["cross_margin_summary_account_value"])
    t["ts"] = pd.to_datetime(t["timestamp"], format="mixed")
    assert (t["date"] == t["ts"].dt.strftime("%Y-%m-%d")).all()
    t = t[t["ts"] < ASOF]
    t = t.sort_values(["trader", "ts"]).reset_index(drop=True)
    t["acct_val"] = t["pnl"] + t["net_deposit"]
    t["iso_gap"] = t["acct_val"] - t["equity"]
    return t[["trader", "ts", "date", "equity", "net_deposit", "pnl", "acct_val", "iso_gap"]]


def load_fills(perp_only: bool = True) -> pd.DataFrame:
    """Fill records. Adds ts (UTC, naive), signed size, end_pos, flags.
    perp_only drops spot rows (dir in SPOT_DIRS); spot never enters position/PnL replay.
    """
    f = pd.read_csv(FILLS_CSV)
    f["ts"] = pd.to_datetime(f["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    f = f[f["ts"] < ASOF].copy()
    f["is_spot"] = f["dir"].isin(SPOT_DIRS)
    f["is_hip3"] = f["symbol"].str.contains(":", regex=False)
    f["is_forced"] = f["dir"].str.startswith(FORCED_DIRS_PREFIX)
    f["sgn"] = np.where(f["side"] == "buy", 1.0, -1.0)
    f["signed"] = f["sgn"] * f["size"]
    f["end_pos"] = f["start_position"] + f["signed"]
    if perp_only:
        f = f[~f["is_spot"]]
    return f.sort_values(["trader", "symbol", "ts", "tid"]).reset_index(drop=True)


def fill_groups(p: pd.DataFrame) -> pd.DataFrame:
    """Collapse fills sharing (trader, symbol, ts) into one event.

    Several fills of one order share a timestamp and their row order is arbitrary,
    so position continuity is only defined at the group level: for an all-buy group
    the group's start is min(start_position) and its end is max(end_pos) (mirror for
    all-sell). Mixed-side groups (3 in the whole file) take start from the row whose
    start_position is not any other row's end_pos; they are kept, flagged `mixed`.
    """
    g = p.groupby(["trader", "symbol", "ts"], sort=True).agg(
        n=("tid", "size"), nbuy=("sgn", lambda s: int((s > 0).sum())),
        smin=("start_position", "min"), smax=("start_position", "max"),
        emin=("end_pos", "min"), emax=("end_pos", "max"),
        signed=("signed", "sum"), notional=("notional", "sum"), abs_size=("size", "sum"),
        cash=("cash", "sum"), fee=("fee", "sum"), closed_pnl=("closed_pnl", "sum"),
        maker_notional=("maker_notional", "sum"), forced_notional=("forced_notional", "sum"),
        px=("price", "last"), is_hip3=("is_hip3", "first"),
    ).reset_index()
    allbuy = g["nbuy"] == g["n"]
    allsell = g["nbuy"] == 0
    g["mixed"] = ~(allbuy | allsell)
    g["gstart"] = np.where(allbuy, g["smin"], np.where(allsell, g["smax"], np.nan))
    g["gend"] = g["gstart"] + g["signed"]
    # mixed groups: net position change is still exact; start is recovered from the
    # previous group's end where one exists (filled below).
    prev_end = g.groupby(["trader", "symbol"])["gend"].shift()
    m = g["mixed"]
    g.loc[m, "gstart"] = prev_end[m]
    g.loc[m, "gend"] = g.loc[m, "gstart"] + g.loc[m, "signed"]
    # start_position + signed size is a float sum: a closed position can come out as ~1e-13
    # instead of 0. Downstream code tests "is there a position" with > 0, so snap residues to 0.
    for c in ("gstart", "gend"):
        g.loc[g[c].abs() < POS_EPS, c] = 0.0
    g["vwap"] = g["notional"] / g["abs_size"]          # true VWAP of the group (also for mixed-side groups)
    return g


def prep_perp_fills() -> pd.DataFrame:
    p = load_fills(perp_only=True)
    p["cash"] = -p["signed"] * p["price"]          # cash flow of the trade, before fee
    p["maker_notional"] = np.where(p["is_maker"], p["notional"], 0.0)
    p["forced_notional"] = np.where(p["is_forced"], p["notional"], 0.0)
    return p
