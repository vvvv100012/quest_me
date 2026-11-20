"""Cross-sectional stability measures of the rolling per-address profile ledger (work/roll/code/ledger.py),
per work/codex10/REVIEW.md sections 2.4-2.11 (with the gates of section 2.3).

For every registry column j and Monday M: B_M = base-eligible on L60_M; A_jM = B_M with the column's L60 gate and a
defined L60 value; C_jM = A_jM with the F28 gate and a defined F28 value (weeks: the Fk7 gate, R28: the F28 gate).

  2.4  rank agreement (Pearson of average ranks on C) and top-symbol category agreement
  2.5  level change on C (diff / ratio / account-value rules)
  2.6  conditional relative-rank retention on C, and frozen-L tiers on all A
  2.7  reportability, activity fractions on B, one primary reason per address of A\\C and B\\A, overlapping flags
  2.8  native pooled and equal-address tier forward summaries with concentration and spreads
  2.9  forward weeks F1..F4 (native gated cohorts and the common cohort C_all4), matched-length R28 diagnostic,
       four sleeve phases. The L30 / L90 length sensitivity is NOT implemented here.
  2.3  fixed-gate sensitivity (F28 order-derived columns at >=30 orders / 10 days, alignment at >=30 decisions /
       10 days)
  2.10 equal-date aggregation over all dates, phases and the early / bridge / late groups. No address-resampling
       bands and no confidence intervals of any kind.
  2.11 descriptive flags; never a "stable" label.

Everything is driven by REGISTRY (+ EXTENSION, empty for now): an extension entry may name its own long ledger file
(source="ledger_x") and support file (support="support_x") and give its gate as gate(sup, kind) with kind in
"L" (L60, without B_M), "F" (F28 / R28) or "W" (a forward week); those files are read only when an entry
references them and they exist.

Inputs (read-only): work/roll/build/{ledger, top_symbol, support, decisions, episodes, clusters,
address_sums}.parquet, windows.csv, gate_counts.csv. Outputs: work/roll/build/measures/ (date and aggregate tables,
overview.csv, manifest.json); log work/roll/logs/measures.log. `--checks` runs the acceptance checks on the built
outputs (build/measures/checks.json, logs/measures_checks.log).
"""
import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

sys.dont_write_bytecode = True
ROOT = Path("/Users/xc/Documents/arkstream_quest")
ROLL = ROOT / "work/roll"
BUILD = ROLL / "build"
OUT = BUILD / "measures"
LOGS = ROLL / "logs"
SPEC = ROOT / "work/codex10/REVIEW.md"
sys.path.insert(0, str(ROLL / "code"))
import ledger as LED  # noqa: E402  (gate(), FAMILY, VALUE_COLS; its import only touches work/roll/build)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

MS = [f"{m:%Y-%m-%d}" for m in LED.MONDAYS]
PHASES = {"phase0": ["2025-09-29", "2025-10-27", "2025-11-24"], "phase1": ["2025-10-06", "2025-11-03", "2025-12-01"],
          "phase2": ["2025-10-13", "2025-11-10", "2025-12-08"], "phase3": ["2025-10-20", "2025-11-17"]}
GROUPS = {"all": MS, **PHASES, "early": MS[:5], "bridge": MS[5:8], "late": MS[8:]}
WEEKS = ["F1", "F2", "F3", "F4"]
FWD_PANELS = ["F28"] + WEEKS
DIFF_THR = 0.2
SMALL = 30

# ---------------------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------------------
WARN_EP = ("duration/opportunity: window-contained complete-episode statistic; its observation filter differs "
           "between 60-, 28- and 7-day windows, so a change is not a holding-tendency change without qualification")
WARN_SYM = "opportunity: distinct-symbol count over windows of different lengths"


@dataclass(frozen=True)
class Col:
    name: str
    family: str                      # support family (ledger.FAMILY), "category" family is "fill"; "ext" otherwise
    level_type: str                  # diff | ratio | av | category
    source: str = "ledger"           # ledger | top_symbol | ledger_x | ledger_r
    support: str = "support"         # support | support_x | support_r
    tier_pool: Optional[str] = None  # section 2.8 pooling rule
    warn: str = ""
    gate: Optional[Callable] = None  # extension: gate(sup, kind) -> bool Series or {name: bool Series}
    obs: Optional[Callable] = None   # extension: obs(sup, kind) -> bool Series "has relevant observations"


def _c(name, level_type, tier_pool=None, warn=""):
    return Col(name, LED.FAMILY[name], level_type, tier_pool=tier_pool, warn=warn)


ALIGN = [f"align_{k}" for k in LED.ALIGN_CELLS]
REGISTRY = (
    [_c(c, "diff") for c in ("active_day_share", "top1_share", "sym_hhi", "btc_eth_share", "long_open_share",
                             "forced_ntl_share", "add_ntl_share", "time_in_market", "net_bias", "long_time_share")]
    + [_c("maker_share", "diff", "maker_ratio"), _c("add_loser_share", "diff", "loss_add_ratio")]
    + [_c(c, "diff", warn=WARN_EP) for c in ("single_shot_share", "flip_share")]
    + [_c(c, "diff", "align") for c in ALIGN]
    + [_c(c, "ratio") for c in ("orders_per_active_day_med", "clusters_per_active_day_med", "orders_per_cluster",
                                "fills_per_order", "order_ntl_med", "turnover_per_active_day_med", "gross_exp_med",
                                "gross_exp_p95", "n_pos_med", "hold_turnover_h", "lev_med", "lev_p95")]
    + [_c("cluster_ntl_med", "ratio", "cluster_ntl_median"), _c("n_symbols", "ratio", warn=WARN_SYM),
       _c("hold_med_h", "ratio", "hold_median", WARN_EP), _c("hold_wavg_h", "ratio", "hold_wmean", WARN_EP),
       _c("adds_per_ep_med", "ratio", warn=WARN_EP)]
    + [_c("av_med", "av")]
    + [Col("top_symbol", "fill", "category", source="top_symbol")]
)

# ---- Extension columns (ledger_x.parquet / support_x.parquet). Empty for now. Example entry:
# Col("x_col", "ext", "ratio", source="ledger_x", support="support_x",
#     gate=lambda s, k: s["n_scored_ep"] >= {"L": 10, "F": 5, "W": 3}[k],
#     obs=lambda s, k: s["n_scored_ep"] > 0)
#
# Columns of work/roll/code/excursion.py (v2). Gates (work/codex12/REVIEW.md part 2, item 8):
#   excursion medians   >= 10 / 5 / 3 path-class episodes finite in that unit on >= 3 / 2 / 2 opening days (L / F / W)
#   excursion p90       >= 30 such episodes on every horizon, same day floors
#   *_av_*              additionally no HIP-3 evidence as of the cutoff
#   loss / win medians  >= 5 / 3 / 2 losing / winning complete episodes; the ratio needs both
#   exit rates          >= 5 / 3 / 2 exits in the state and positive time in it; the ratio needs both states
#   open-loss shares    >= 3 known-entry open positions at the cutoff
#
# Columns of work/roll/code/returns.py (ledger_r.parquet / support_r.parquet):
#   return columns      >= 14 / 14 / 5 valid days, no HIP-3 evidence as of the cutoff, and at most 10% of active days
#                       dropped for a missing price or account value
WARN_X = ("duration/opportunity: path observed from the open to the close or the window cutoff, on episodes opened "
          "inside the window; window length changes the observed part; read the matched R28->F28 comparison")
WARN_HR = ("duration/opportunity: complete episodes inside the window only; long holds and positions never closed "
           "are missing; read the matched R28->F28 comparison")
WARN_ER = ("exits per hour spent in the state, path-class episodes opened inside the window; the level depends on "
           "the window length (compare addresses within one length); read the matched R28->F28 comparison")
WARN_OPEN = "point-in-time at the window cutoff (L60 and R28 share the cutoff M, so the matched comparison is trivial)"
X_MED_N, X_MED_D, X_P90_N = {"L": 10, "F": 5, "W": 3}, {"L": 3, "F": 2, "W": 2}, 30
X_WL = {"L": 5, "F": 3, "W": 2}
X_OPEN = 3
WARN_R = ("daily mark-to-market PnL net of fees and funding over the account-value proxy; sums are not compounded; "
          "performance persistence, not a behaviour trait; the level scales with the window length")
R_DAYS = {"L": 14, "F": 14, "W": 5}


def _x_gate(unit, p90):
    def g(s, k):
        p = {"episodes": s[f"n_{unit}"] >= (X_P90_N if p90 else X_MED_N[k]), "episode_days": s[f"d_{unit}"] >= X_MED_D[k]}
        if unit == "av":
            p["no_hip3"] = ~s["hip3_asof"].astype(bool)
        return p
    return g


def _x_col(name, unit, p90):
    return Col(name, "ext", "ratio", source="ledger_x", support="support_x", warn=WARN_X, gate=_x_gate(unit, p90),
               obs=lambda s, k: s[f"n_{unit}"] > 0)


def _wl_col(name, sides):
    return Col(name, "ext", "ratio", source="ledger_x", support="support_x", warn=WARN_HR,
               gate=lambda s, k: {f"n_{x}": s[f"n_{x}"] >= X_WL[k] for x in sides},
               obs=lambda s, k: sum(s[f"n_{x}"] for x in sides) > 0)


def _er_col(name, sides):
    """Exit rate in a state (exits per hour in loss / in gain) and their ratio: >= 5 / 3 / 2 exits in every state
    the column uses (the X_WL floors) and positive observed time in that state."""
    return Col(name, "ext", "ratio", source="ledger_x", support="support_x", warn=WARN_ER,
               gate=lambda s, k: {**{f"n_exit_{x}": s[f"n_exit_{x}"] >= X_WL[k] for x in sides},
                                  **{f"t_{x}_h": s[f"t_{x}_h"] > 0 for x in sides}},
               obs=lambda s, k: sum(s[f"n_exit_{x}"] for x in sides) > 0)


def _r_col(name):
    return Col(name, "ext", "diff", source="ledger_r", support="support_r", warn=WARN_R,
               gate=lambda s, k: {"ret_days": s["n_ret_days"] >= R_DAYS[k],
                                  "no_hip3": ~s["hip3_asof"].astype(bool),
                                  "day_coverage": (s["n_act_noav"] + s["n_act_lt100"] + s["n_px_bad_days"])
                                                  <= 0.1 * s["n_act_days"]},
               obs=lambda s, k: s["n_span_days"] > 0)


EXTENSION: list = (
    [_x_col(f"{kind}_{unit}_{st}", unit, st == "p90")
     for kind, units in (("mae", ("px", "vol", "rt", "av")), ("mfe", ("px", "vol", "rt")))
     for unit in units for st in ("med", "p90")]
    + [_wl_col("hold_med_loss_h", ("loss",)), _wl_col("hold_med_win_h", ("win",)),
       _wl_col("hold_loss_win_ratio", ("loss", "win"))]
    + [_er_col("exit_rate_loss", ("loss",)), _er_col("exit_rate_gain", ("gain",)),
       _er_col("exit_rate_ratio", ("loss", "gain"))]
    + [Col(c, "ext", "diff", source="ledger_x", support="support_x", warn=WARN_OPEN,
           gate=lambda s, k: {"open_known": s["n_open_known"] >= X_OPEN}, obs=lambda s, k: s["n_open_known"] > 0)
       for c in ("open_loss_share_n", "open_loss_share_ntl")]
    + [_r_col(c) for c in ("ret_sum", "ret_mean_sd", "max_dd", "gain_share")]
)


def registry():
    """REGISTRY + EXTENSION entries whose ledger / support files exist (others are dropped with a log line)."""
    out = list(REGISTRY)
    for e in EXTENSION:
        missing = [f for f in {e.source, e.support} if f not in ("ledger", "support", "top_symbol")
                   and not (BUILD / f"{f}.parquet").exists()]
        if missing:
            log(f"extension column {e.name}: {missing} not found, skipped")
            continue
        out.append(e)
    names = [e.name for e in out]
    assert len(names) == len(set(names)), "duplicate registry names"
    return out


_logf = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


# ---------------------------------------------------------------------------------------
# Gates (section 2.3)
# ---------------------------------------------------------------------------------------
GRID_DAYS = {"L": 14, "F": 14, "W": 5}
INM_DAYS = {"L": 10, "F": 5, "W": 2}
COMPLETE = {"L": 10, "F": 5, "W": 3}
EVENTS = {"L": (30, 10), "F": (15, 5), "W": (5, 2), "S": (30, 10)}     # groups / decisions / orders and days
ORDERS = {"F": (15, 5), "W": (5, 2), "S": (30, 10)}
STRICT_FAMILIES = ("fill", "turnover", "align")                        # fixed-gate sensitivity (kind "S")


def gate_parts(col, sup, kind):
    """Named pass/fail components of the column's gate on a support frame; the gate is their conjunction.
    kind: 'L' (L60, without B_M), 'F' (F28 and R28), 'W' (forward week), 'S' (F28 fixed-gate sensitivity)."""
    if col.gate is not None:
        g = col.gate(sup, kind)
        return dict(g) if isinstance(g, dict) else {"ext_gate": g}
    fam = col.family
    if kind == "S" and fam not in STRICT_FAMILIES:
        raise ValueError(f"no stricter gate for family {fam}")
    p = {}

    def orders():
        n, d = ORDERS[kind]
        p["orders"] = sup["n_orders"] >= n
        p["fill_days"] = sup["n_fill_days"] >= d

    def grid():
        p["grid_days"] = sup["valid_grid_days"] >= GRID_DAYS["F" if kind == "S" else kind]

    if fam == "fill":
        if kind != "L":
            orders()
    elif fam == "tim":
        grid()
    elif fam == "grid_q":
        grid()
        p["inm_days"] = sup["n_inm_days"] >= INM_DAYS[kind]
    elif fam == "lev":
        grid()
        p["lev_days"] = sup["n_lev_days"] >= INM_DAYS[kind]
        p["no_hip3"] = ~sup["hip3_asof"].astype(bool)
    elif fam == "av":
        p["snap_days"] = sup["n_snap_days"] >= INM_DAYS[kind]
        p["no_hip3"] = ~sup["hip3_asof"].astype(bool)
    elif fam == "turnover":
        grid()
        if kind == "L":
            p["turnover_pos"] = sup["turnover"] > 0
        else:
            orders()
    elif fam == "complete":
        p["complete"] = sup["n_complete"] >= COMPLETE[kind]
    elif fam in ("add", "loss_add", "align"):
        n, d = EVENTS[kind]
        pre = {"add": "n_inc_", "loss_add": "n_later_add_"}.get(fam)
        cn, cd = (f"{pre}groups", f"{pre}days") if pre else (f"{col.name[6:]}_n", f"{col.name[6:]}_days")
        p["events"] = sup[cn] >= n
        p["event_days"] = sup[cd] >= d
    else:
        raise ValueError(fam)
    return p


def gate(col, sup, kind):
    g = pd.Series(True, index=sup.index)
    for v in gate_parts(col, sup, kind).values():
        g &= v.astype(bool)
    return g


def has_obs(col, sup, kind):
    """Relevant observations exist in the window (section 2.7 reason 2). Built-in families do not depend on kind."""
    if col.gate is not None:
        return col.obs(sup, kind).astype(bool) if col.obs is not None else pd.Series(True, index=sup.index)
    fam = col.family
    if fam == "fill":
        return sup["n_fills"] > 0
    if fam in ("tim", "grid_q", "lev"):
        return sup["n_valid_points"] > 0
    if fam == "turnover":                     # needs both the grid and fills (turnover denominator)
        return (sup["n_valid_points"] > 0) & (sup["n_fills"] > 0)
    if fam == "av":
        return sup["n_snap"] > 0
    if fam == "complete":
        return sup["n_complete"] > 0
    if fam == "add":
        return sup["n_inc_groups"] > 0
    if fam == "loss_add":
        return sup["n_later_add_groups"] > 0
    return sup[f"{col.name[6:]}_n"] > 0       # alignment


def diag_flags(col, sup):
    """Overlapping diagnostic flags (section 2.7) of a window's support rows."""
    f = {"no_fills": sup["n_fills"] == 0, "missing_prices": sup["n_missing_px_points"] > 0,
         "av_lt100": sup["n_inm_av_lt100"] > 0, "av_invalid_bracket": sup["n_inm_av_missing"] > 0,
         "no_opening_groups": sup["n_inc_groups"] == 0, "no_later_adds": sup["n_later_add_groups"] == 0,
         "hip3_asof": sup["hip3_asof"].astype(bool)}
    if col.family == "align":
        k = col.name[6:]
        f["align_coverage_failure"] = (sup[f"cand_w_{k.split('_')[0]}"] > 0) & (sup[f"{k}_n"] == 0)
    return f


# ---------------------------------------------------------------------------------------
# Core statistics (pure functions on arrays; exercised by the fixtures)
# ---------------------------------------------------------------------------------------
def tiers_of(x):
    """Mid-percentile tiers of a finite sample (section 2.6): u = (average rank - 0.5) / n; 1 if u <= 1/3,
    2 if u <= 2/3, else 3. Exact integer arithmetic on 2 * rank (u = (2r - 1) / 2n), so ties are never split."""
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = 3 * (np.rint(2 * stats.rankdata(x, method="average")).astype(np.int64) - 1)
    return np.where(k <= 2 * n, 1, np.where(k <= 4 * n, 2, 3))


def frozen_bins(ref, v):
    """Bin of forward values v in the frozen L distribution ref: u_L(v) = (#(ref < v) + 0.5 #(ref = v)) / |ref|,
    same thresholds as tiers_of (and identical to it for v in ref)."""
    s = np.sort(ref)
    n = len(s)
    lt, le = np.searchsorted(s, v, "left"), np.searchsorted(s, v, "right")
    k = 3 * (2 * lt + (le - lt))
    return np.where(k <= 2 * n, 1, np.where(k <= 4 * n, 2, 3))


def rank_corr(x, y, sides=("L", "F")):
    """Pearson correlation of average ranks; (rho, reason). NA if n < 3 or ranks constant on a side."""
    n = len(x)
    if n < 3:
        return np.nan, "n<3"
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    cx, cy = bool(np.all(rx == rx[0])), bool(np.all(ry == ry[0]))
    if cx or cy:
        return np.nan, ("constant ranks on both sides" if cx and cy else f"constant {sides[0] if cx else sides[1]} ranks")
    return float(np.corrcoef(rx, ry)[0, 1]), ""


def xquantile(x, p):
    """Type-7 quantile with extended-real arithmetic (section 2.5): x may contain +inf; if the interpolation puts
    positive weight on +inf the result is +inf, an exact finite order statistic stays finite. The position
    (n-1)p is computed exactly as a rational, so an exact order statistic is never perturbed by rounding."""
    s = np.sort(np.asarray(x, dtype=float))
    n = len(s)
    if n == 0:
        return np.nan
    fr = Fraction(p).limit_denominator(10**6) * (n - 1)
    lo = fr.numerator // fr.denominator
    g = float(fr - lo)
    if g == 0 or lo >= n - 1:
        return float(s[min(lo, n - 1)])
    if np.isinf(s[lo + 1]):
        return np.inf
    return float(s[lo] + g * (s[lo + 1] - s[lo]))


def ratio_of(L, F):
    """r = F / L with the zero conventions of section 2.5 (L = F = 0 -> 1, L = 0 < F -> inf, L > 0 = F -> 0)."""
    L, F = np.asarray(L, float), np.asarray(F, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((L == 0) & (F == 0), 1.0, np.where(L == 0, np.inf, F / L))
    r[(L < 0) | (F < 0)] = np.nan                         # nonnegative columns only; counted as n_negative
    return r


def _ratio_summary(r, pre):
    ok = ~np.isnan(r)
    r = r[ok]
    return {f"{pre}med_r": xquantile(r, .5), f"{pre}p10_r": xquantile(r, .1), f"{pre}p90_r": xquantile(r, .9),
            f"{pre}frac_r_gt2_or_lt_half": float(np.mean((r > 2) | (r < .5))) if len(r) else np.nan}


def level_stats(L, F, level_type, name=""):
    """Section 2.5 on a paired cohort. Headline fields: level_med (median d, median r, or AV median dollar d) and
    level_frac (fraction |d| > 0.2, fraction of factor-of-two changes; AV: on the positive subset)."""
    L, F = np.asarray(L, float), np.asarray(F, float)
    n = len(L)
    d = F - L
    o = {"n": n, "med_abs_change": xquantile(np.abs(d), .5)}
    if level_type == "diff":
        unit = ("normalized-return units" if name.startswith("align_")
                else "daily mean/SD units (0.2 = arbitrary diagnostic threshold)" if name == "ret_mean_sd"
                else "fraction units (0.2 = 20 pp)")
        o.update(unit=unit,
                 med_d=xquantile(d, .5), p10_d=xquantile(d, .1), p90_d=xquantile(d, .9),
                 frac_abs_d_gt_thr=float(np.mean(np.abs(d) > DIFF_THR)) if n else np.nan)
        o.update(level_med=o["med_d"], level_frac=o["frac_abs_d_gt_thr"])
    elif level_type == "ratio":
        r = ratio_of(L, F)
        o.update(unit="ratio F/L", n_negative=int(((L < 0) | (F < 0)).sum()),
                 n_both_zero=int(((L == 0) & (F == 0)).sum()), n_L0_Fpos=int(((L == 0) & (F > 0)).sum()),
                 n_Lpos_F0=int(((L > 0) & (F == 0)).sum()), **_ratio_summary(r, ""))
        o.update(level_med=o["med_r"], level_frac=o["frac_r_gt2_or_lt_half"])
    elif level_type == "av":
        pos = (L > 0) & (F > 0)
        o.update(unit="USD difference on all C; ratio on C_positive", med_d=xquantile(d, .5), p10_d=xquantile(d, .1),
                 p90_d=xquantile(d, .9), n_pos_subset=int(pos.sum()), n_nonpos_excluded=int(n - pos.sum()),
                 **_ratio_summary(F[pos] / L[pos], "pos_"))
        o.update(level_med=o["med_d"], level_frac=o["pos_frac_r_gt2_or_lt_half"])
    else:
        raise ValueError(level_type)
    return o


def _matrix(a, b):
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (a - 1, b - 1), 1)
    return m


def rel_tiers(L, F):
    """Conditional relative-rank retention (section 2.6 panel 1): tiers recomputed on the paired cohort."""
    n = len(L)
    m = _matrix(tiers_of(L), tiers_of(F))
    o = {"n": n}
    for t in range(3):
        for s in range(3):
            o[f"m{t + 1}{s + 1}"] = int(m[t, s])
    if n == 0:
        o.update(discriminating=False, same_tier=np.nan, p0=np.nan, excess=np.nan, chance_adj=np.nan,
                 extreme=np.nan, e0=np.nan)
        return o
    f = m / n
    pL, pF = m.sum(1) / n, m.sum(0) / n
    for t in range(3):
        o[f"pL{t + 1}"], o[f"pF{t + 1}"] = pL[t], pF[t]
        for s in range(3):
            o[f"f{t + 1}{s + 1}"] = f[t, s]
            o[f"cond{t + 1}{s + 1}"] = m[t, s] / m[t].sum() if m[t].sum() else np.nan
    same, p0 = float(np.trace(f)), float(pL @ pF)
    o.update(discriminating=bool((m.sum(1) > 0).all() and (m.sum(0) > 0).all()), same_tier=same, p0=p0,
             excess=same - p0, chance_adj=(same - p0) / (1 - p0) if p0 < 1 else np.nan,
             extreme=float(f[0, 2] + f[2, 0]), e0=float(pL[0] * pF[2] + pL[2] * pF[0]),
             top_to_bottom=o["cond31"], bottom_to_top=o["cond13"])
    return o


def frozen_panel(LA, inC, FC):
    """Frozen-L tiers on all A (section 2.6 panel 2). LA: L values on A; inC: bool over A (reportable in the
    forward window); FC: forward values of those members, in A order."""
    N, G = len(LA), int(inC.sum())
    tA = tiers_of(LA)
    bins = frozen_bins(LA, FC)
    m = _matrix(tA[inC], bins)
    S = int(np.trace(m))
    o = {"N": N, "G": G, "S": S, "S_over_N": S / N if N else np.nan, "S_over_G": S / G if G else np.nan,
         "bound_lo": S / N if N else np.nan, "bound_hi": (S + N - G) / N if N else np.nan}
    for t in range(3):
        size = int((tA == t + 1).sum())
        o[f"tier{t + 1}_n"] = size
        for s in range(3):
            o[f"m{t + 1}{s + 1}"] = int(m[t, s])
            o[f"out{t + 1}_bin{s + 1}"] = m[t, s] / size if size else np.nan
        o[f"tier{t + 1}_unreportable"] = size - int(m[t].sum())
        o[f"out{t + 1}_unrep"] = o[f"tier{t + 1}_unreportable"] / size if size else np.nan
    return o


def category_stats(l, f):
    """Top-symbol agreement (section 2.4): p_same, independent-marginal baseline p0, chance-adjusted value."""
    l, f = np.asarray(l, dtype=object), np.asarray(f, dtype=object)
    n = len(l)
    tr = pd.DataFrame({"sym_L": l, "sym_F": f}).value_counts().rename("count").reset_index()
    if n == 0:
        return {"n": 0, "p_same": np.nan, "p0": np.nan, "chance_adj": np.nan, "n_sym_L": 0, "n_sym_F": 0}, tr
    pl, pf = pd.Series(l).value_counts() / n, pd.Series(f).value_counts() / n
    p0 = float((pl * pf.reindex(pl.index).fillna(0.0)).sum())
    ps = float(np.mean(l == f))
    return {"n": n, "p_same": ps, "p0": p0, "chance_adj": (ps - p0) / (1 - p0) if p0 < 1 else np.nan,
            "n_sym_L": int(pl.size), "n_sym_F": int(pf.size)}, tr


def order_status(P):
    """Strict ordering of three tier values: preserved / tied / not preserved / not assessable."""
    if any(not np.isfinite(x) for x in P):
        return "not assessable"
    if P[0] < P[1] < P[2]:
        return "preserved"
    if P[0] > P[1] or P[1] > P[2]:
        return "not preserved"
    return "tied"


def spread_summary(PL, PF):
    """Native spreads Delta = P3 - P1, adjacent differences, ordering and the spread ratio rule (section 2.8)."""
    o = {}
    for side, P in (("L", PL), ("F", PF)):
        o[f"delta_{side}"] = P[2] - P[0]
        o[f"adj12_{side}"], o[f"adj23_{side}"] = P[1] - P[0], P[2] - P[1]
        o[f"order_{side}"] = order_status(P)
    dL, dF = o["delta_L"], o["delta_F"]
    if not np.isfinite(dL) or not np.isfinite(dF):
        o["spread_ratio"], o["ratio_status"] = np.nan, "not assessable"
    elif dL <= 0:
        o["spread_ratio"], o["ratio_status"] = np.nan, "not interpretable"
    else:
        o["spread_ratio"], o["ratio_status"] = dF / dL, "reported"
    return o


# ---------------------------------------------------------------------------------------
# Pooling records (section 2.8)
# ---------------------------------------------------------------------------------------
@dataclass
class Pool:
    """Records of one pooled column in one window. mode 'median': P = median of v; 'wmean': P = sum(num)/sum(w).
    Per-address arrays (length |U|): native pooling weight, records, record days, notional."""
    mode: str
    ai: np.ndarray
    v: np.ndarray
    w: np.ndarray
    num: np.ndarray
    addr_w: np.ndarray
    addr_n: np.ndarray
    addr_days: np.ndarray
    addr_ntl: np.ndarray


def _addr_days(ai, day, nU):
    if len(ai) == 0:
        return np.zeros(nU, dtype=np.int64)
    u = pd.DataFrame({"a": ai, "d": day}).drop_duplicates()
    return np.bincount(u["a"].to_numpy(), minlength=nU)


def make_pool(mode, ai, v, w, ntl, day, nU, addr_n=None, addr_days=None):
    ai = np.asarray(ai, dtype=np.int64)
    v, w = np.asarray(v, float), np.asarray(w, float)
    num = w * v if mode == "wmean" else np.zeros(len(v))
    return Pool(mode, ai, v, w, num,
                addr_w=np.bincount(ai, weights=w, minlength=nU) if mode == "wmean" else np.bincount(ai, minlength=nU)
                .astype(float),
                addr_n=np.bincount(ai, minlength=nU) if addr_n is None else np.asarray(addr_n),
                addr_days=_addr_days(ai, day, nU) if addr_days is None else np.asarray(addr_days),
                addr_ntl=np.bincount(ai, weights=np.nan_to_num(np.asarray(ntl, float)), minlength=nU))


def build_pools(reg, data, wids):
    """{(window_id, column): Pool} for the tier-pool columns on the given windows."""
    nU, sup = data["nU"], data["sup"]
    rec = data["records"]
    out = {}
    for wid in wids:
        dec, ep, cl, sm = (rec[k].get(wid) for k in ("decisions", "episodes", "clusters", "address_sums"))
        for col in reg:
            tp = col.tier_pool
            if tp is None:
                continue
            if tp == "align":
                cls, h, var = col.name[6:].split("_")
                w = dec[f"w_{cls}"].to_numpy()
                val = dec[f"z_{h}" if var == "raw" else f"xA_{h}"].to_numpy()
                k = (w > 0) & np.isfinite(val)
                d = dec[k]
                out[(wid, col.name)] = make_pool("wmean", d["ai"], val[k], w[k], w[k], d["day"], nU)
            elif tp == "hold_median":
                out[(wid, col.name)] = make_pool("median", ep["ai"], ep["dur_h"], np.ones(len(ep)), ep["open_ntl"],
                                                 ep["day"], nU)
            elif tp == "hold_wmean":
                d = ep[ep["open_ntl"].notna()]
                out[(wid, col.name)] = make_pool("wmean", d["ai"], d["dur_h"], d["open_ntl"], d["open_ntl"], d["day"], nU)
            elif tp == "cluster_ntl_median":
                out[(wid, col.name)] = make_pool("median", cl["ai"], cl["ntl"], np.ones(len(cl)), cl["ntl"], cl["day"], nU)
            elif tp in ("maker_ratio", "loss_add_ratio"):
                num_c, den_c, n_c, d_c = (("maker_ntl", "turnover", "n_fills", "n_fill_days") if tp == "maker_ratio"
                                          else ("loser_add_ntl", "later_add_ntl", "n_later_add_groups",
                                                "n_later_add_days"))
                den, num = sm[den_c].to_numpy(), sm[num_c].to_numpy()
                k = den > 0
                ai = np.flatnonzero(k)
                s = sup[wid]
                out[(wid, col.name)] = make_pool("wmean", ai, num[k] / den[k], den[k], den[k], np.zeros(len(ai)), nU,
                                                 addr_n=s[n_c].to_numpy(), addr_days=s[d_c].to_numpy())
            else:
                raise ValueError(tp)
    return out


def pool_tier(pool, member):
    """Native pooled value and concentration of one tier (member: bool over U)."""
    sel = member[pool.ai]
    if pool.mode == "median":
        P = float(np.median(pool.v[sel])) if sel.any() else np.nan
    else:
        W = pool.w[sel].sum()
        P = float(pool.num[sel].sum() / W) if W > 0 else np.nan
    aw = pool.addr_w[member]
    tw = aw.sum()
    return {"P": P, "contrib": int((aw > 0).sum()), "records": int(pool.addr_n[member].sum()),
            "days": int(pool.addr_days[member].sum()), "ntl": float(pool.addr_ntl[member].sum()),
            "weight": float(tw), "max_share": float(aw.max() / tw) if tw > 0 else np.nan,
            "kish": float(tw ** 2 / (aw ** 2).sum()) if tw > 0 else np.nan}


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------
def _pivot_long(d):
    d = d.astype({"window_id": str, "trader": str, "column": str})
    return d.set_index(["window_id", "trader", "column"])["value"].unstack("column")


def load(reg):
    t0 = time.time()
    wt = pd.read_csv(BUILD / "windows.csv")
    wt = wt[wt["kind"] != "ref"].reset_index(drop=True)
    assert sorted(wt["M"].unique()) == MS
    wids = list(wt["window_id"])
    sup = pd.read_parquet(BUILD / "support.parquet")
    sup = sup[sup["window_id"].isin(wids)].astype({"window_id": str, "trader": str})
    U = np.array(sorted(sup["trader"].unique()))
    nU = len(U)
    exts = {e.support for e in reg} - {"support"}
    for f in sorted(exts):
        x = pd.read_parquet(BUILD / f"{f}.parquet").astype({"window_id": str, "trader": str})
        new = [c for c in x.columns if c not in sup.columns]
        log(f"  {f}: joined {len(new)} columns; {len(x.columns) - 2 - len(new)} name collisions ignored")
        sup = sup.merge(x[["window_id", "trader"] + new], on=["window_id", "trader"], how="left")
    sup_w = {w: d.set_index("trader").reindex(U) for w, d in sup.groupby("window_id")}

    led = pd.read_parquet(BUILD / "ledger.parquet")
    wide = _pivot_long(led[led["window_id"].isin(wids)])
    for f in sorted({e.source for e in reg} - {"ledger", "top_symbol"}):
        x = pd.read_parquet(BUILD / f"{f}.parquet")
        x = _pivot_long(x) if {"column", "value"} <= set(x.columns) else \
            x.astype({"window_id": str, "trader": str}).set_index(["window_id", "trader"])
        keep = [e.name for e in reg if e.source == f]
        wide = wide.join(x[keep], how="left")
    top = pd.read_parquet(BUILD / "top_symbol.parquet").astype({"window_id": str, "trader": str})
    top = top[top["window_id"].isin(wids)].set_index(["window_id", "trader"])["top_symbol"]
    vals = {}
    for w in wids:
        v = wide.loc[w].reindex(U)
        v["top_symbol"] = top.loc[w].reindex(U)
        vals[w] = v

    # pooling records on the L60 / F28 / weekly windows
    pw = list(wt.loc[wt["kind"].isin(["L60"] + FWD_PANELS), "window_id"])
    records = {}
    spec = {"decisions": ["trader", "window_id", "t", "w_taker", "w_maker"] + [f"{p}_{h}" for p in ("z", "xA")
                                                                             for h in LED.HORIZONS],
            "episodes": ["trader", "window_id", "open_ts", "dur_h", "open_ntl"],
            "clusters": ["trader", "window_id", "first_ts", "ntl"],
            "address_sums": None}
    tcol = {"decisions": "t", "episodes": "open_ts", "clusters": "first_ts"}
    for k, cols in spec.items():
        d = pd.read_parquet(BUILD / f"{k}.parquet", columns=cols)
        d = d[d["window_id"].isin(pw)].astype({"window_id": str})
        d["ai"] = pd.Categorical(d["trader"].astype(str), categories=U).codes.astype(np.int64)
        assert (d["ai"] >= 0).all()
        if k in tcol:
            ns = d[tcol[k]].to_numpy().astype("datetime64[ns]").astype(np.int64)
            d["day"] = ns // LED.DAY
        if k == "address_sums":
            records[k] = {w: g.set_index("ai").reindex(np.arange(nU)).fillna(0.0) for w, g in d.groupby("window_id")}
        else:
            records[k] = {w: g.reset_index(drop=True) for w, g in d.groupby("window_id")}
    log(f"loaded in {time.time() - t0:.1f}s: U {nU}, windows {len(wids)}, value columns {wide.shape[1]}")
    return dict(wt=wt, U=U, nU=nU, sup=sup_w, vals=vals, records=records)


def values(data, wid, col):
    v = data["vals"][wid][col.name]
    if col.level_type == "category":
        x = v.to_numpy(dtype=object)
        return x, v.notna().to_numpy()
    x = v.to_numpy(dtype=float)
    return x, np.isfinite(x)


# ---------------------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------------------
def reasons(col, sup, ok_val, members, kind, panel, M, setname, hip3_label):
    """One primary reason per address in `members` (section 2.7 order) and the overlapping flag counts."""
    g = gate(col, sup, kind).to_numpy()
    obs = has_obs(col, sup, kind).to_numpy()
    hip_scope = col.family in ("lev", "av") or (col.gate is not None and "no_hip3" in gate_parts(col, sup, kind))
    hip = hip_scope & sup["hip3_asof"].astype(bool).to_numpy()
    prim = {hip3_label: members & hip}
    left = members & ~hip
    prim["no_relevant_observations"] = left & ~obs
    left &= obs
    prim["insufficient_support"] = left & ~g
    left &= g
    prim["undefined_value"] = left & ~ok_val
    assert not (left & ok_val).any(), "member of the reason set with gate and value"
    rows = [dict(panel=panel, M=M, column=col.name, set=setname, type="primary", name=k, count=int(v.sum()))
            for k, v in prim.items()]
    for k, v in gate_parts(col, sup, kind).items():
        rows.append(dict(panel=panel, M=M, column=col.name, set=setname, type="gate_shortfall", name=k,
                         count=int((members & ~v.to_numpy().astype(bool)).sum())))
    for k, v in diag_flags(col, sup).items():
        rows.append(dict(panel=panel, M=M, column=col.name, set=setname, type="flag", name=k,
                         count=int((members & v.to_numpy().astype(bool)).sum())))
    return rows


def numeric_measures(name, level_type, xL, xF):
    rho, why = rank_corr(xL, xF)
    return rho, why, level_stats(xL, xF, level_type, name), rel_tiers(xL, xF)


def compute(reg, data):
    T = {k: [] for k in ("rank", "category", "category_trans", "level", "tier_rel", "tier_frozen", "report",
                         "reasons", "activity", "pool", "week", "matched", "gate_sens")}
    sup = data["sup"]
    pools = build_pools(reg, data, [f"{k}_{M}" for M in MS for k in ["L60"] + FWD_PANELS])
    for M in MS:
        sL = sup[f"L60_{M}"]
        B = sL["base_eligible"].to_numpy().astype(bool)
        # activity fractions on all B (column-independent)
        for panel in FWD_PANELS:
            s = sup[f"{panel}_{M}"]
            nf = s["n_fills"].to_numpy()[B]
            no, nd = s["n_orders"].to_numpy()[B], s["n_fill_days"].to_numpy()[B]
            r = dict(panel=panel, M=M, B=int(B.sum()), no_fills=int((nf == 0).sum()), any_fills=int((nf > 0).sum()))
            gates = [(15, 5), (30, 10)] if panel == "F28" else [(5, 2)]
            for a, b in gates:
                r[f"orders{a}_days{b}"] = int(((no >= a) & (nd >= b)).sum())
            for k in [c for c in r if c not in ("panel", "M", "B")]:
                r[f"frac_{k}"] = r[k] / r["B"] if r["B"] else np.nan
            T["activity"].append(r)

        for col in reg:
            cat = col.level_type == "category"
            base = dict(M=M, column=col.name)
            xL, okL = values(data, f"L60_{M}", col)
            gL = gate(col, sL, "L").to_numpy()
            A = B & gL & okL
            iA = np.flatnonzero(A)
            tA = None if cat else tiers_of(xL[A])
            T["reasons"] += reasons(col, sL, okL, B & ~A, "L", "L60", M, "B\\A", "hip3_preexisting_scope")
            Cs = {}
            for panel in FWD_PANELS:
                kind = "F" if panel == "F28" else "W"
                sF = sup[f"{panel}_{M}"]
                xF, okF = values(data, f"{panel}_{M}", col)
                gF = gate(col, sF, kind).to_numpy()
                C = A & gF & okF
                Cs[panel] = C
                p = dict(panel=panel, **base)
                rep = dict(p, B=int(B.sum()), L_gate=int((B & gL).sum()), A=int(A.sum()),
                           F_gate_among_A=int((A & gF).sum()), C=int(C.sum()))
                rep.update(A_over_B=rep["A"] / rep["B"] if rep["B"] else np.nan,
                           C_over_A=rep["C"] / rep["A"] if rep["A"] else np.nan)
                if not cat:
                    for t in (1, 2, 3):
                        m = tA == t
                        rep[f"tier{t}_n"] = int(m.sum())
                        rep[f"tier{t}_C"] = int(C[iA][m].sum())
                        rep[f"tier{t}_frac"] = rep[f"tier{t}_C"] / rep[f"tier{t}_n"] if m.any() else np.nan
                T["report"].append(rep)
                T["reasons"] += reasons(col, sF, okF, A & ~C, kind, panel, M, "A\\C", "hip3_scope_change")
                if cat:
                    cs, tr = category_stats(xL[C], xF[C])
                    T["category"].append(dict(p, **cs))
                    T["category_trans"].append(tr.assign(panel=panel, M=M, column=col.name))
                    continue
                rho, why = rank_corr(xL[C], xF[C])
                T["rank"].append(dict(p, n=int(C.sum()), rho=rho, reason=why,
                                      small_cohort=bool(C.sum() < SMALL)))
                if panel == "F28":
                    T["level"].append(dict(p, level_type=col.level_type, **level_stats(xL[C], xF[C], col.level_type,
                                                                                        col.name)))
                    T["tier_rel"].append(dict(p, **rel_tiers(xL[C], xF[C])))
                    T["tier_frozen"].append(dict(p, **frozen_panel(xL[A], C[iA], xF[C])))
                if col.tier_pool is not None:
                    T["pool"] += pool_rows(col, data, pools, M, panel, A, iA, tA)

            if cat:
                # fixed-gate sensitivity for the order-family category
                sF = sup[f"F28_{M}"]
                xF, okF = values(data, f"F28_{M}", col)
                Cst = A & gate(col, sF, "S").to_numpy() & okF
                for coh, C in (("C_primary", Cs["F28"]), ("C_strict", Cst)):
                    cs, _ = category_stats(xL[C], xF[C])
                    T["gate_sens"].append(dict(cohort=coh, **base, **cs))
                continue

            # weeks: common cohort C_all4, ranks afresh on it
            Call = A & Cs["F1"] & Cs["F2"] & Cs["F3"] & Cs["F4"]
            wr = dict(base, A=int(A.sum()), n_all4=int(Call.sum()), small_cohort_all4=bool(Call.sum() < SMALL),
                      **{f"n_native_{k}": int(Cs[k].sum()) for k in WEEKS})
            for k in WEEKS:
                xk, _ = values(data, f"{k}_{M}", col)
                wr[f"rho_{k}"], wr[f"reason_{k}"] = rank_corr(xL[Call], xk[Call])
            wr["rho_F4_minus_F1"] = wr["rho_F4"] - wr["rho_F1"]
            wr["seven_day_contained"] = col.family == "complete" or col.name in (
                "hold_med_loss_h", "hold_med_win_h", "hold_loss_win_ratio",
                "exit_rate_loss", "exit_rate_gain", "exit_rate_ratio",
                "ret_sum", "ret_mean_sd", "max_dd", "gain_share")
            T["week"].append(wr)

            # matched-length diagnostic: R28 with the F28 gates, on A, passing both R and F gates
            sR = sup[f"R28_{M}"]
            xR, okR = values(data, f"R28_{M}", col)
            xF, _ = values(data, f"F28_{M}", col)
            Cm = Cs["F28"] & gate(col, sR, "F").to_numpy() & okR
            for pair, xS in (("R28->F28", xR), ("L60->F28", xL)):
                rho, why, lv, rt = numeric_measures(col.name, col.level_type, xS[Cm], xF[Cm])
                T["matched"].append(dict(base, pair=pair, A=int(A.sum()), C=int(Cs["F28"].sum()), n=int(Cm.sum()),
                                         rho=rho, reason=why, small_cohort=bool(Cm.sum() < SMALL),
                                         **{f"lvl_{k}": v for k, v in lv.items() if k != "n"},
                                         **{f"rel_{k}": v for k, v in rt.items() if k != "n"}))

            # fixed-gate sensitivity (F28 order-derived columns and alignment)
            if col.gate is None and col.family in STRICT_FAMILIES:
                sF = sup[f"F28_{M}"]
                _, okF = values(data, f"F28_{M}", col)
                Cst = A & gate(col, sF, "S").to_numpy() & okF
                for coh, C in (("C_primary", Cs["F28"]), ("C_strict", Cst)):
                    rho, why, lv, rt = numeric_measures(col.name, col.level_type, xL[C], xF[C])
                    fz = frozen_panel(xL[A], C[iA], xF[C])
                    T["gate_sens"].append(dict(cohort=coh, **base, A=int(A.sum()), n=int(C.sum()), rho=rho,
                                               reason=why, small_cohort=bool(C.sum() < SMALL),
                                               level_med=lv["level_med"], level_frac=lv["level_frac"],
                                               med_abs_change=lv["med_abs_change"],
                                               discriminating=rt["discriminating"], same_tier=rt["same_tier"],
                                               p0=rt["p0"], excess=rt["excess"], extreme=rt["extreme"],
                                               S_over_N=fz["S_over_N"], S_over_G=fz["S_over_G"]))
        log(f"  {M} done")
    out = {k: pd.DataFrame(v) for k, v in T.items() if k != "category_trans"}
    out["category_trans"] = pd.concat(T["category_trans"], ignore_index=True)[
        ["panel", "M", "column", "sym_L", "sym_F", "count"]]
    return out


def pool_rows(col, data, pools, M, panel, A, iA, tA):
    """Section 2.8 rows (P native pooled, E equal-address) for deployed L tiers on A, forward window `panel`."""
    nU = data["nU"]
    mem = []
    for t in (1, 2, 3):
        m = np.zeros(nU, dtype=bool)
        m[iA[tA == t]] = True
        mem.append(m)
    rows = []
    base = dict(panel=panel, M=M, column=col.name, A=int(A.sum()), **{f"tier{t + 1}_n": int(m.sum())
                                                                      for t, m in enumerate(mem)})
    sides = {"L": f"L60_{M}", "F": f"{panel}_{M}"}
    rp, re = dict(base, summary="P", mode=pools[(sides["L"], col.name)].mode), dict(base, summary="E")
    PP, EE = {}, {}
    for side, wid in sides.items():
        pool = pools[(wid, col.name)]
        x = data["vals"][wid][col.name].to_numpy(dtype=float)
        PP[side], EE[side] = [], []
        for t, m in enumerate(mem, 1):
            st = pool_tier(pool, m)
            PP[side].append(st.pop("P"))
            for k, v in st.items():
                rp[f"{k}{t}_{side}"] = v
            xm = x[m & np.isfinite(x)]
            EE[side].append(float(xm.mean()) if len(xm) else np.nan)
            re[f"contrib{t}_{side}"] = int(len(xm))
        for t in range(3):
            rp[f"P{t + 1}_{side}"], re[f"P{t + 1}_{side}"] = PP[side][t], EE[side][t]
    rp.update(spread_summary(PP["L"], PP["F"]))
    re.update(spread_summary(EE["L"], EE["F"]))
    return [rp, re]


# ---------------------------------------------------------------------------------------
# Aggregation (section 2.10)
# ---------------------------------------------------------------------------------------
def aggregate(df, keys, rules, warn=None):
    """Equal-date aggregation of every stat over each group of Mondays. rule 'mean' (probabilities, correlations,
    fractions): value = arithmetic mean of defined per-date values; 'median' (level medians): median of per-date
    medians; 'count': median of the per-date counts. Always with median, min, max, K and the omitted dates."""
    out = []
    df = df.copy()
    df["_k"] = df[keys].astype(str).agg("|".join, axis=1)
    kf = df.drop_duplicates("_k").set_index("_k")[keys]
    for stat, rule in rules.items():
        if stat not in df.columns:
            continue
        w = df.pivot(index="_k", columns="M", values=stat).reindex(columns=MS)
        x = w.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        for g, dates in GROUPS.items():
            idx = [MS.index(d) for d in dates]
            xs = x[:, idx]
            ok = ~np.isnan(xs)
            K = ok.sum(1)
            with warnings.catch_warnings():              # all-NaN rows give NaN (K = 0)
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, med = np.nanmean(xs, 1), np.nanmedian(xs, 1)
                mn, mx = np.nanmin(xs, 1), np.nanmax(xs, 1)
            val = med if rule in ("median", "count") else mean
            r = kf.loc[w.index].reset_index(drop=True)
            r = r.assign(group=g, stat=stat, rule=rule, value=val, mean=mean, median=med, min=mn, max=mx, K=K,
                         n_dates=len(dates),
                         omitted=[",".join(d for d, o in zip(dates, row) if not o) for row in ok])
            out.append(r)
    res = pd.concat(out, ignore_index=True)
    if warn is not None and "column" in keys:
        res["warn"] = res["column"].map(warn).fillna("")
    return res


def masked(df, cond, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].where(cond)
    return d


def pool_agg(pool):
    """Across dates (section 2.8): equal-date mean P_t and spreads on dates with all three tiers defined on both
    sides, ratio mean Delta_F / mean Delta_L only if mean Delta_L > 0; never an average of ratios."""
    rows = []
    for (panel, col, summ), d in pool.groupby(["panel", "column", "summary"], sort=False):
        d = d.set_index("M")
        for g, dates in GROUPS.items():
            s = d.reindex(dates)
            pc = [f"P{t}_{side}" for side in "LF" for t in (1, 2, 3)]
            full = s[pc].notna().all(axis=1)
            sf = s[full]
            r = dict(panel=panel, column=col, summary=summ, group=g, n_dates=len(dates), K_all3=int(full.sum()),
                     omitted=",".join(s.index[~full]))
            for c in pc + ["delta_L", "delta_F"]:
                r[f"mean_{c}"] = float(sf[c].mean()) if len(sf) else np.nan
            dl, dfw = r["mean_delta_L"], r["mean_delta_F"]
            if not np.isfinite(dl):
                r["spread_ratio"], r["ratio_status"] = np.nan, "not assessable"
            elif dl <= 0:
                r["spread_ratio"], r["ratio_status"] = np.nan, "not interpretable"
            else:
                r["spread_ratio"], r["ratio_status"] = dfw / dl, "reported"
            st = s["order_F"]
            ass = st.notna() & (st != "not assessable")
            r["n_assessable_F"] = int(ass.sum())
            r["frac_strict_F_order"] = float((st[ass] == "preserved").mean()) if ass.any() else np.nan
            r["n_tied_F"] = int((st == "tied").sum())
            r["n_empty_or_undefined_F"] = int((st == "not assessable").sum())
            r["n_nonpos_delta_L"] = int((s["delta_L"] <= 0).sum())
            if summ == "P":
                for t in (1, 2, 3):
                    for side in "LF":
                        for k in ("max_share", "kish"):
                            r[f"mean_{k}{t}_{side}"] = float(s[f"{k}{t}_{side}"].mean())
                        r[f"mean_contrib_frac{t}_{side}"] = float((s[f"contrib{t}_{side}"] / s[f"tier{t}_n"]
                                                                   .where(s[f"tier{t}_n"] > 0)).mean())
            rows.append(r)
    return pd.DataFrame(rows)


REL_HEAD = ["same_tier", "p0", "excess", "chance_adj", "extreme", "e0", "top_to_bottom", "bottom_to_top"]


def aggregates(t, reg):
    warn = {e.name: e.warn for e in reg}
    A = {}
    A["rank_agg"] = aggregate(t["rank"], ["panel", "column"], {"rho": "mean", "n": "count"}, warn)
    A["category_agg"] = aggregate(t["category"], ["panel", "column"],
                                  {"p_same": "mean", "p0": "mean", "chance_adj": "mean", "n": "count"}, warn)
    lr = {"level_med": "median", "level_frac": "mean", "med_d": "median", "med_r": "median",
          "frac_abs_d_gt_thr": "mean", "frac_r_gt2_or_lt_half": "mean", "med_abs_change": "median",
          "pos_med_r": "median", "pos_frac_r_gt2_or_lt_half": "mean", "n": "count", "n_pos_subset": "count",
          "n_both_zero": "count", "n_L0_Fpos": "count", "n_Lpos_F0": "count"}
    A["level_agg"] = aggregate(t["level"], ["panel", "column", "level_type"], lr, warn)
    # headline retention statistics on discriminating dates only; matrix and margins on dates with n > 0
    tr = masked(t["tier_rel"], t["tier_rel"]["discriminating"].astype(bool), REL_HEAD)
    tr["discriminating_int"] = t["tier_rel"]["discriminating"].astype(int)
    rr = {**{k: "mean" for k in REL_HEAD}, "n": "count", "discriminating_int": "mean",
          **{f"f{a}{b}": "mean" for a in (1, 2, 3) for b in (1, 2, 3)},
          **{f"p{s}{a}": "mean" for s in "LF" for a in (1, 2, 3)}}
    A["tier_rel_agg"] = aggregate(tr, ["panel", "column"], rr, warn)
    fr = {"S_over_N": "mean", "S_over_G": "mean", "bound_lo": "mean", "bound_hi": "mean", "N": "count", "G": "count",
          **{f"out{a}_{b}": "mean" for a in (1, 2, 3) for b in ("bin1", "bin2", "bin3", "unrep")}}
    A["tier_frozen_agg"] = aggregate(t["tier_frozen"], ["panel", "column"], fr, warn)
    rp = {"B": "count", "A": "count", "C": "count", "A_over_B": "mean", "C_over_A": "mean",
          **{f"tier{a}_frac": "mean" for a in (1, 2, 3)}}
    A["report_agg"] = aggregate(t["report"], ["panel", "column"], rp, warn)
    a = t["activity"]
    A["activity_agg"] = aggregate(a, ["panel"], {c: ("mean" if c.startswith("frac_") else "count")
                                                 for c in a.columns if c not in ("panel", "M")})
    rs = t["reasons"]
    rs = rs.merge(t["report"][["panel", "M", "column", "A"]], on=["panel", "M", "column"], how="left")
    rB = t["report"][t["report"]["panel"] == "F28"][["M", "column", "B"]]
    rs = rs.merge(rB, on=["M", "column"], how="left")
    rs["denom"] = np.where(rs["set"] == "B\\A", rs["B"], rs["A"])
    rs["frac"] = rs["count"] / rs["denom"].where(rs["denom"] > 0)
    A["reasons_agg"] = aggregate(rs[["panel", "M", "column", "set", "type", "name", "frac", "count"]],
                                 ["panel", "column", "set", "type", "name"], {"frac": "mean", "count": "count"}, warn)
    A["pool_agg"] = pool_agg(t["pool"])
    A["pool_agg"]["warn"] = A["pool_agg"]["column"].map(warn).fillna("")
    wk = {f"rho_{k}": "mean" for k in WEEKS}
    wk.update(rho_F4_minus_F1="mean", n_all4="count", **{f"n_native_{k}": "count" for k in WEEKS})
    A["week_agg"] = aggregate(t["week"], ["column"], wk, warn)
    mrule = {"rho": "mean", "n": "count", "lvl_level_med": "median", "lvl_level_frac": "mean",
             **{f"rel_{k}": "mean" for k in REL_HEAD}}
    mt = masked(t["matched"], t["matched"]["rel_discriminating"].astype(bool), [f"rel_{k}" for k in REL_HEAD])
    A["matched_agg"] = aggregate(mt, ["column", "pair"], mrule, warn)
    gs = t["gate_sens"]
    gs = masked(gs, gs["discriminating"].fillna(False).astype(bool), ["same_tier", "p0", "excess", "extreme"]) \
        if "discriminating" in gs.columns else gs
    A["gate_sens_agg"] = aggregate(gs, ["column", "cohort"],
                                   {"rho": "mean", "n": "count", "level_med": "median", "level_frac": "mean",
                                    "same_tier": "mean", "p0": "mean", "excess": "mean", "extreme": "mean",
                                    "S_over_N": "mean", "S_over_G": "mean", "p_same": "mean", "chance_adj": "mean"},
                                   warn)
    return A


# ---------------------------------------------------------------------------------------
# Flags and overview (section 2.11)
# ---------------------------------------------------------------------------------------
def _get(agg, stat, group="all", field="value", **keys):
    m = (agg["stat"] == stat) & (agg["group"] == group)
    for k, v in keys.items():
        m &= agg[k] == v
    s = agg.loc[m, field]
    return s.iloc[0] if len(s) else np.nan


def flags_and_overview(t, A, reg):
    fl, ov = [], []
    rk, tr, lv, fz = A["rank_agg"], A["tier_rel_agg"], A["level_agg"], A["tier_frozen_agg"]
    rp, wk, mt, ct = A["report_agg"], A["week_agg"], A["matched_agg"], A["category_agg"]
    for e in reg:
        c = e.name
        k = dict(panel="F28", column=c)
        o = dict(column=c, level_type=e.level_type, family=e.family, tier_pool=e.tier_pool or "",
                 median_B=_get(rp, "B", **k), median_A=_get(rp, "A", **k), median_C=_get(rp, "C", **k),
                 mean_C_over_A=_get(rp, "C_over_A", **k), mean_A_over_B=_get(rp, "A_over_B", **k))
        if e.level_type == "category":
            o.update(cat_mean_p_same=_get(ct, "p_same", **k), cat_mean_p0=_get(ct, "p0", **k),
                     cat_mean_chance_adj=_get(ct, "chance_adj", **k), cat_K=_get(ct, "p_same", field="K", **k))
            for g in ("early", "bridge", "late", *PHASES):
                o[f"cat_p_same_{g}"] = _get(ct, "p_same", g, **k)
            o.update(flag="no numeric flag (category)", warn=e.warn)
            fl.append(dict(column=c, rank_flag="", retention_flag="", flag=o["flag"], K_rho=np.nan, K_disc=np.nan,
                           median_C=o["median_C"]))
            ov.append(o)
            continue
        o.update(mean_rho=_get(rk, "rho", **k), min_rho=_get(rk, "rho", field="min", **k),
                 max_rho=_get(rk, "rho", field="max", **k), K_rho=_get(rk, "rho", field="K", **k))
        for g in ("early", "bridge", "late", *PHASES):
            o[f"rho_{g}"] = _get(rk, "rho", g, **k)
        o.update(K_disc=_get(tr, "same_tier", field="K", **k), mean_same_tier=_get(tr, "same_tier", **k),
                 mean_p0=_get(tr, "p0", **k), mean_extreme=_get(tr, "extreme", **k), mean_e0=_get(tr, "e0", **k),
                 mean_frozen_S_over_N=_get(fz, "S_over_N", **k), mean_frozen_S_over_G=_get(fz, "S_over_G", **k),
                 level_unit=t["level"].loc[t["level"]["column"] == c, "unit"].iloc[0],
                 level_median_of_date_medians=_get(lv, "level_med", **k),
                 level_mean_frac_beyond=_get(lv, "level_frac", **k), zero_cases="")
        if e.level_type == "ratio":            # L=F=0 / L=0<F / L>0=F, address-date counts summed for display only
            lt = t["level"][(t["level"]["column"] == c)]
            o["zero_cases"] = (f"{int(lt['n_both_zero'].sum())}/{int(lt['n_L0_Fpos'].sum())}/"
                               f"{int(lt['n_Lpos_F0'].sum())}")
        o.update(matched_rho_R28=_get(mt, "rho", column=c, pair="R28->F28"),
                 matched_rho_L60=_get(mt, "rho", column=c, pair="L60->F28"),
                 matched_K=_get(mt, "rho", field="K", column=c, pair="R28->F28"),
                 matched_median_n=_get(mt, "n", column=c, pair="R28->F28"))
        for w in WEEKS:
            o[f"week_native_rho_{w}"] = _get(rk, "rho", panel=w, column=c)
        for w in WEEKS:
            o[f"all4_rho_{w}"] = _get(wk, f"rho_{w}", column=c)
        o["all4_rho_F4_minus_F1"] = _get(wk, "rho_F4_minus_F1", column=c)
        o["all4_median_n"] = _get(wk, "n_all4", column=c)
        # flags
        if not (o["K_rho"] > 0):
            rf = "insufficient discrimination or support"
        else:
            rf = "low observed conditional rank agreement" if o["mean_rho"] < .5 else ""
        if not (o["K_disc"] > 0):
            tf = "insufficient discrimination or support"
        else:
            tf = "low observed conditional retention" if o["mean_same_tier"] < .5 else ""
        flag = "; ".join(dict.fromkeys(x for x in (rf, tf) if x))
        o.update(flag=flag, warn=e.warn)
        fl.append(dict(column=c, rank_flag=rf, retention_flag=tf, flag=flag, K_rho=o["K_rho"], mean_rho=o["mean_rho"],
                       K_disc=o["K_disc"], mean_same_tier=o["mean_same_tier"], median_C=o["median_C"],
                       min_C=_get(rp, "C", field="min", **k), max_C=_get(rp, "C", field="max", **k),
                       note="descriptive conventions (rho < .5, same-tercile < .5), not calibrated failure tests; "
                            "an unflagged column is not labelled stable"))
        ov.append(o)
    return pd.DataFrame(fl), pd.DataFrame(ov)


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p):
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def reg_dump(reg):
    out = []
    for e in reg:
        d = dict(name=e.name, family=e.family, level_type=e.level_type, source=e.source, support=e.support,
                 tier_pool=e.tier_pool, warn=e.warn)
        for k in ("gate", "obs"):
            f = getattr(e, k)
            try:
                d[k] = None if f is None else inspect.getsource(f).strip()
            except OSError:                              # defined without a source file
                d[k] = repr(f)
        out.append(d)
    return out


def gate_table():
    return dict(L60=dict(base="n_orders>=30 & n_fill_days>=10 (B_M)", grid_days=GRID_DAYS["L"], inm_lev_snap_days=INM_DAYS["L"],
                         complete=COMPLETE["L"], events_days=EVENTS["L"], fill="defined value",
                         turnover="grid & turnover>0"),
                F28=dict(orders_days=ORDERS["F"], grid_days=GRID_DAYS["F"], inm_lev_snap_days=INM_DAYS["F"],
                         complete=COMPLETE["F"], events_days=EVENTS["F"]),
                R28="F28 gates",
                Fk7=dict(orders_days=ORDERS["W"], grid_days=GRID_DAYS["W"], inm_lev_snap_days=INM_DAYS["W"],
                         complete=COMPLETE["W"], events_days=EVENTS["W"]),
                F28_strict=dict(orders_days=ORDERS["S"], events_days=EVENTS["S"], families=STRICT_FAMILIES),
                hip3="AV / leverage require no HIP-3 evidence as of the window cutoff")


def build():
    global _logf
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "measures.log", "w")
    t0 = time.time()
    reg = registry()
    names = {e.name for e in REGISTRY}
    assert names == set(LED.VALUE_COLS) | {"top_symbol"}, "registry does not cover the ledger columns"
    assert all(e.family == LED.FAMILY[e.name] for e in REGISTRY if e.name in LED.FAMILY)
    data = load(reg)
    t = compute(reg, data)
    log(f"date-level tables done at {time.time() - t0:.1f}s")
    A = aggregates(t, reg)
    fl, ov = flags_and_overview(t, A, reg)
    log(f"aggregates done at {time.time() - t0:.1f}s")
    names = {"rank": "rank_date", "category": "category_date", "category_trans": "category_trans_date",
             "level": "level_date", "tier_rel": "tier_rel_date", "tier_frozen": "tier_frozen_date",
             "report": "report_date", "reasons": "reasons_date", "activity": "activity_date", "pool": "pool_date",
             "week": "week_date", "matched": "matched_date", "gate_sens": "gate_sens_date"}
    tables = {names[k]: v for k, v in t.items()}
    tables.update(A)
    tables["flags"] = fl
    for k, d in tables.items():
        d.to_parquet(OUT / f"{k}.parquet", index=False)
        d.to_csv(OUT / f"{k}.csv", index=False)
        log(f"wrote {k}: {len(d):,} rows x {d.shape[1]} cols")
    ov.to_csv(OUT / "overview.csv", index=False)
    log(f"wrote overview.csv: {len(ov)} rows x {ov.shape[1]} cols")
    ins = ["ledger", "top_symbol", "support", "decisions", "episodes", "clusters", "address_sums"]
    ins += sorted({e.source for e in reg} | {e.support for e in reg} - set(ins))
    files = [BUILD / f"{f}.parquet" for f in ins] + [BUILD / "windows.csv", BUILD / "gate_counts.csv"]
    runtime = time.time() - t0
    man = dict(spec=str(SPEC), spec_sections="2.3 gates (incl. Fk7, fixed-gate sensitivity), 2.4-2.8, 2.9 (weeks, "
                                             "matched R28, sleeve phases; no L30/L90), 2.10 (no resampling bands), 2.11",
               inputs={_rel(f): sha256(f) for f in files},
               sources={_rel(p): sha256(p) for p in (Path(__file__).resolve(), ROLL / "code/ledger.py",
                                                                       SPEC)},
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             pyarrow=pyarrow.__version__, scipy=scipy.__version__),
               mondays=MS, groups=GROUPS, gates=gate_table(), diff_threshold=DIFF_THR, small_cohort=SMALL,
               registry=reg_dump(reg), extension=reg_dump([e for e in reg if e in EXTENSION]),
               tables={k: int(len(v)) for k, v in tables.items()},
               conventions=dict(
                   aggregation="equal-date mean of defined per-date values (rule mean); median of per-date medians "
                               "(rule median); median of counts (rule count); no address-date concatenation",
                   retention_headline="same-tier / p0 / excess / extreme aggregated over discriminating dates only",
                   gate_sensitivity="primary statistic on the stricter cohort equals the stricter-gate native result "
                                    "(forward values do not depend on the gate); C_primary rows shown for reference",
                   no_intervals="no confidence intervals, p-values or resampling bands"),
               run_utc=pd.Timestamp.now(tz="UTC").isoformat(), runtime_s=round(runtime, 1))
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"runtime {runtime:.1f}s")


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def _close(a, b, rtol=1e-12, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore"):                  # inf - inf
        return bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b) | (np.abs(a - b) <= atol + rtol * np.abs(b))))


def fixtures():
    """Synthetic frames with hand-computed answers."""
    r = {}
    # ties: [1,1,1,2,3,4] -> avg ranks 2,2,2,4,5,6, u = 1.5/6, .., 3.5/6, 4.5/6, 5.5/6 -> tiers 1,1,1,3? u=3.5/6=.583->2
    t = tiers_of(np.array([1, 1, 1, 2, 3, 4.]))
    r["ties_tiers"] = dict(got=t.tolist(), want=[1, 1, 1, 2, 3, 3], ok=t.tolist() == [1, 1, 1, 2, 3, 3])
    # boundary tie: n=6, two-way tie at ranks 2,3 -> avg 2.5, u = 2/6 = 1/3 exactly -> tier 1 (never split)
    t = tiers_of(np.array([0, 1, 1, 5, 6, 7.]))
    r["tie_on_boundary"] = dict(got=t.tolist(), want=[1, 1, 1, 2, 3, 3], ok=t.tolist() == [1, 1, 1, 2, 3, 3])
    t = tiers_of(np.full(7, 3.0))
    rho, why = rank_corr(np.full(7, 3.0), np.arange(7.))
    r["constant_column"] = dict(tiers=t.tolist(), rho=rho, reason=why,
                                ok=bool((t == 2).all() and np.isnan(rho) and why == "constant L ranks"))
    rho, why = rank_corr(np.array([1., 2]), np.array([2., 1]))
    r["n_lt_3"] = dict(rho=rho, reason=why, ok=bool(np.isnan(rho) and why == "n<3"))
    rho, _ = rank_corr(np.array([1., 2, 2, 3]), np.array([1., 3, 2, 4]))
    r["rho_with_ties"] = dict(got=rho, want=float(stats.spearmanr([1, 2, 2, 3], [1, 3, 2, 4])[0]),
                              ok=_close(rho, stats.spearmanr([1, 2, 2, 3], [1, 3, 2, 4])[0]))
    # ratio zero cases and infinite quantile: L=[0,0,1,2], F=[0,1,0,4] -> r=[1,inf,0,2], sorted [0,1,2,inf]
    lv = level_stats(np.array([0, 0, 1, 2.]), np.array([0, 1, 0, 4.]), "ratio")
    want = dict(n_both_zero=1, n_L0_Fpos=1, n_Lpos_F0=1, med_r=1.5, p10_r=0.3, p90_r=np.inf,
                frac_r_gt2_or_lt_half=0.5, med_abs_change=1.0)
    r["ratio_zero_cases"] = dict(got={k: lv[k] for k in want}, want=want,
                                 ok=all(_close(lv[k], v) for k, v in want.items()))
    q = xquantile(np.array([0, 1, 2, np.inf]), 2 / 3)     # exact order statistic 2, finite although next is inf
    q2 = xquantile(np.array([0, 1, 2, np.inf]), 0.7)      # weight .1 on inf -> inf
    r["xquantile_exact_finite"] = dict(got=[q, q2], want=[2.0, np.inf], ok=q == 2.0 and q2 == np.inf)
    lv = level_stats(np.array([.1, .5, .9]), np.array([.4, .5, .6]), "diff")
    r["diff_threshold"] = dict(got=[lv["med_d"], lv["frac_abs_d_gt_thr"]], want=[0.0, 2 / 3],
                               ok=_close(lv["med_d"], 0.0) and _close(lv["frac_abs_d_gt_thr"], 2 / 3))
    lv = level_stats(np.array([-5., 10, 20, 0]), np.array([5., 30, 10, 7]), "av")
    r["av_positive_subset"] = dict(got=[lv["med_d"], lv["n_pos_subset"], lv["pos_med_r"]], want=[8.5, 2, 1.75],
                                   ok=_close(lv["med_d"], 8.5) and lv["n_pos_subset"] == 2 and _close(lv["pos_med_r"], 1.75))
    # category baseline: L=[a,a,b,b], F=[a,b,b,b] -> p_same=.75, p0=.5*.25+.5*.75=.5, chance-adjusted .5
    cs, tr = category_stats(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    r["category_p0"] = dict(got=[cs["p_same"], cs["p0"], cs["chance_adj"], int(tr["count"].sum())],
                            want=[.75, .5, .5, 4],
                            ok=_close([cs["p_same"], cs["p0"], cs["chance_adj"]], [.75, .5, .5]) and tr["count"].sum() == 4)
    # empty tier: L=[1,1,1,1,2,2] -> u = 2/6 (tier 1) and 5/6 (tier 3), tier 2 empty -> non-discriminating
    rt = rel_tiers(np.array([1, 1, 1, 1, 2, 2.]), np.arange(6.))
    r["empty_tier_non_discriminating"] = dict(pL=[rt["pL1"], rt["pL2"], rt["pL3"]], discriminating=rt["discriminating"],
                                              ok=(not rt["discriminating"]) and rt["pL2"] == 0)
    rt = rel_tiers(np.arange(9.), np.arange(9.)[::-1])
    r["reversal_tiers"] = dict(same=rt["same_tier"], extreme=rt["extreme"], p0=rt["p0"], e0=rt["e0"],
                               ok=_close([rt["same_tier"], rt["extreme"], rt["p0"], rt["e0"]], [1 / 3, 2 / 3, 1 / 3, 2 / 9]))
    # frozen: A L = [1..6], tiers 1,1,2,2,3,3; C = first five with F = [10,0,3.5,3,1]: u_L = 1, 0, 3/6, 2.5/6, 0.5/6
    # -> bins 3,1,2,2,1; same bin for the 2nd, 3rd, 4th -> S = 3, bounds [3/6, (3+6-5)/6]
    fz = frozen_panel(np.arange(1, 7.), np.array([1, 1, 1, 1, 1, 0], bool), np.array([10, 0, 3.5, 3, 1.]))
    r["frozen_counts"] = dict(S=fz["S"], N=fz["N"], G=fz["G"], hi=fz["bound_hi"],
                              unrep=[fz[f"tier{t}_unreportable"] for t in (1, 2, 3)],
                              ok=fz["S"] == 3 and fz["N"] == 6 and fz["G"] == 5 and _close(fz["bound_hi"], 4 / 6)
                              and [fz[f"tier{t}_unreportable"] for t in (1, 2, 3)] == [0, 0, 1])
    s = spread_summary([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    r["spread_ratio_not_interpretable"] = dict(got=[s["delta_L"], s["ratio_status"], s["order_L"], s["order_F"]],
                                               ok=s["ratio_status"] == "not interpretable" and np.isnan(s["spread_ratio"])
                                               and s["delta_L"] == -2.0 and s["order_F"] == "preserved")
    s = spread_summary([1.0, 1.0, 1.0], [1.0, 1.0, 2.0])
    r["spread_zero_L"] = dict(got=[s["ratio_status"], s["order_F"]],
                              ok=s["ratio_status"] == "not interpretable" and s["order_F"] == "tied")
    s = spread_summary([1.0, 2.0, 3.0], [1.0, np.nan, 2.0])
    r["spread_empty_tier"] = dict(got=[s["spread_ratio"], s["order_F"]],
                                  ok=s["spread_ratio"] == 0.5 and s["order_F"] == "not assessable")
    # pooling: two addresses, scores +1 / -1 at weights 1 / 100 -> -99/101; Kish = 101^2 / 10001
    pl = make_pool("wmean", [0, 1], [1., -1.], [1., 100.], [1., 100.], [0, 0], 2)
    st = pool_tier(pl, np.array([True, True]))
    r["pool_weighted"] = dict(P=st["P"], kish=st["kish"], max_share=st["max_share"],
                              ok=_close([st["P"], st["kish"], st["max_share"]], [-99 / 101, 101 ** 2 / 10001, 100 / 101]))
    # aggregation: equal-date mean of defined values, K, omitted dates
    df = pd.DataFrame({"panel": "F28", "column": "x", "M": MS, "rho": [np.nan] + [.1 * i for i in range(1, 11)]})
    ag = aggregate(df, ["panel", "column"], {"rho": "mean"})
    a_all = ag[ag["group"] == "all"].iloc[0]
    a_p0 = ag[ag["group"] == "phase0"].iloc[0]
    r["aggregate_rule"] = dict(all=[a_all["value"], a_all["K"], a_all["omitted"]], phase0=[a_p0["value"], a_p0["K"]],
                               ok=_close(a_all["value"], .55) and a_all["K"] == 10 and a_all["omitted"] == MS[0]
                               and _close(a_p0["value"], (.4 + .8) / 2) and a_p0["K"] == 2)
    for v in r.values():
        v["ok"] = bool(v["ok"])
    return r


def independent(data_dir, mine):
    """Straightforward second implementation for three columns and two Mondays (pandas + scipy.spearmanr)."""
    led = pd.read_parquet(data_dir / "ledger.parquet")
    sup = pd.read_parquet(data_dir / "support.parquet")
    res = {}
    gates = {"maker_share": (lambda s: pd.Series(True, index=s.index), lambda s: (s.n_orders >= 15) & (s.n_fill_days >= 5)),
             "hold_med_h": (lambda s: s.n_complete >= 10, lambda s: s.n_complete >= 5),
             "align_taker_1d_raw": (lambda s: (s.taker_1d_raw_n >= 30) & (s.taker_1d_raw_days >= 10),
                                    lambda s: (s.taker_1d_raw_n >= 15) & (s.taker_1d_raw_days >= 5))}
    dec = pd.read_parquet(data_dir / "decisions.parquet", columns=["trader", "window_id", "w_taker", "z_1d"])
    eps = pd.read_parquet(data_dir / "episodes.parquet", columns=["trader", "window_id", "dur_h"])
    sums = pd.read_parquet(data_dir / "address_sums.parquet")
    for M in ("2025-10-13", "2025-12-01"):
        for col, (gl, gf) in gates.items():
            def side(w):
                s = sup[sup.window_id == w].set_index("trader")
                v = led[(led.window_id == w) & (led.column == col)].set_index("trader")["value"]
                v.index = v.index.astype(str)
                s.index = s.index.astype(str)
                return s, v.reindex(s.index)
            sl, vl = side(f"L60_{M}")
            sf, vf = side(f"F28_{M}")
            sf, vf = sf.reindex(sl.index), vf.reindex(sl.index)
            A = sl.base_eligible & gl(sl) & vl.notna()
            C = A & gf(sf) & vf.notna()
            L, F = vl[C], vf[C]
            rho = stats.spearmanr(L, F)[0]

            def tier(x):
                u = (x.rank(method="average") - 0.5) / len(x)
                return pd.cut(u, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=[1, 2, 3]).astype(int)
            same = float((tier(L) == tier(F)).mean())
            if col == "hold_med_h":
                rr = []
                for a, b in zip(L, F):
                    rr.append(1.0 if a == 0 and b == 0 else (np.inf if a == 0 else b / a))
                rr = np.array(rr)
                lev = (float(np.median(rr)), float(np.mean((rr > 2) | (rr < .5))))
            else:
                d = (F - L).to_numpy()
                lev = (float(np.median(d)), float(np.mean(np.abs(d) > .2)))
            tA = tier(vl[A])
            pools = {}
            for side_, w in (("L", f"L60_{M}"), ("F", f"F28_{M}")):
                for t in (1, 2, 3):
                    mem = set(tA.index[tA == t])
                    if col == "maker_share":
                        s = sums[(sums.window_id == w) & sums.trader.isin(mem)]
                        pv = s.maker_ntl.sum() / s.turnover.sum()
                    elif col == "hold_med_h":
                        e = eps[(eps.window_id == w) & eps.trader.astype(str).isin(mem)]
                        pv = float(np.median(e.dur_h)) if len(e) else np.nan
                    else:
                        e = dec[(dec.window_id == w) & dec.trader.astype(str).isin(mem) & (dec.w_taker > 0)
                                & dec.z_1d.notna()]
                        pv = (e.w_taker * e.z_1d).sum() / e.w_taker.sum()
                    pools[f"P{t}_{side_}"] = float(pv)
            # mine
            rk = mine["rank_date"]
            rk = rk[(rk.panel == "F28") & (rk.M == M) & (rk.column == col)].iloc[0]
            tr = mine["tier_rel_date"]
            tr = tr[(tr.panel == "F28") & (tr.M == M) & (tr.column == col)].iloc[0]
            lv = mine["level_date"]
            lv = lv[(lv.panel == "F28") & (lv.M == M) & (lv.column == col)].iloc[0]
            pl = mine["pool_date"]
            pl = pl[(pl.panel == "F28") & (pl.M == M) & (pl.column == col) & (pl.summary == "P")].iloc[0]
            mine_pools = {k: float(pl[k]) for k in pools}
            ok = (int(rk.n) == int(C.sum()) and _close(rk.rho, rho) and _close(tr.same_tier, same)
                  and _close([lv.level_med, lv.level_frac], list(lev)) and _close(list(mine_pools.values()),
                                                                                list(pools.values()), rtol=1e-10))
            res[f"{M}|{col}"] = dict(n=int(C.sum()), rho=[float(rk.rho), float(rho)], same_tier=[float(tr.same_tier), same],
                                     level=[[float(lv.level_med), float(lv.level_frac)], list(lev)],
                                     pools_mine=mine_pools, pools_indep=pools, ok=ok)
    # three aggregate rows from the date tables
    rk, ra = mine["rank_date"], mine["rank_agg"]
    x = rk[(rk.panel == "F28") & (rk.column == "maker_share")]["rho"]
    a1 = ra[(ra.panel == "F28") & (ra.column == "maker_share") & (ra.group == "all") & (ra.stat == "rho")].iloc[0]
    tr, ta = mine["tier_rel_date"], mine["tier_rel_agg"]
    t2 = tr[(tr.panel == "F28") & (tr.column == "hold_med_h") & tr.M.isin(GROUPS["early"]) & tr.discriminating]
    a2 = ta[(ta.panel == "F28") & (ta.column == "hold_med_h") & (ta.group == "early") & (ta.stat == "same_tier")].iloc[0]
    lv, la = mine["level_date"], mine["level_agg"]
    t3 = lv[(lv.panel == "F28") & (lv.column == "hold_med_h") & lv.M.isin(GROUPS["late"])]
    a3 = la[(la.panel == "F28") & (la.column == "hold_med_h") & (la.group == "late") & (la.stat == "med_r")].iloc[0]
    res["agg_rank_mean_all"] = dict(indep=float(x.mean()), mine=float(a1.value), K=[int(x.notna().sum()), int(a1.K)],
                                    ok=_close(a1.value, x.mean()) and int(a1.K) == int(x.notna().sum()))
    res["agg_same_tier_early"] = dict(indep=float(t2.same_tier.mean()), mine=float(a2.value),
                                      K=[len(t2), int(a2.K)], ok=_close(a2.value, t2.same_tier.mean()) and len(t2) == a2.K)
    res["agg_level_median_late"] = dict(indep=float(np.median(t3.med_r)), mine=float(a3.value),
                                        ok=_close(a3.value, np.median(t3.med_r)))
    for v in res.values():
        v["ok"] = bool(v["ok"])
    return res


def checks():
    global _logf
    LOGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOGS / "measures_checks.log", "w")
    t0 = time.time()
    res = {}
    reg = registry()
    mine = {p.stem: pd.read_parquet(p) for p in OUT.glob("*.parquet")}

    # ---- 1: gate counts and gate function ----
    log("=== check 1: A / C counts vs gate_counts.csv; gate() vs ledger.gate() ===")
    gc = pd.read_csv(BUILD / "gate_counts.csv")
    rp = mine["report_date"]
    rp = rp[rp.panel == "F28"]
    j = gc.merge(rp, on=["M", "column"], how="left", suffixes=("_csv", ""))
    cnt = {c: int((j[f"{c}_csv"] != j[c]).sum()) for c in ("B", "L_gate", "A", "F_gate_among_A", "C")}
    res["check1_gate_counts"] = dict(rows_csv=len(gc), rows_matched=int(j["A"].notna().sum()), mismatches=cnt,
                                     ok=len(gc) == 528 and j["A"].notna().all() and not any(cnt.values()))
    sup = pd.read_parquet(BUILD / "support.parquet")
    sup = sup[~sup.window_id.astype(str).str.startswith("ref_")]
    gm = {}
    for e in REGISTRY:
        if e.name not in LED.FAMILY:
            continue
        for kind in ("L", "F"):
            a, b = gate(e, sup, kind).to_numpy(), LED.gate(sup, e.name, kind).to_numpy().astype(bool)
            gm[f"{e.name}|{kind}"] = int((a != b).sum())
    res["check1_gate_function"] = dict(rows=len(sup), columns=len(gm) // 2, mismatching_rows=sum(gm.values()),
                                       ok=sum(gm.values()) == 0)
    log(json.dumps({k: v for k, v in res.items()}, default=str, indent=1))

    # ---- 2: reconciliation ----
    log("\n=== check 2: reconciliation ===")
    c2 = {}
    rs = mine["reasons_date"]
    pr = rs[rs.type == "primary"].groupby(["panel", "M", "column", "set"])["count"].sum().reset_index()
    rep = mine["report_date"]
    a_c = pr[pr.set == "A\\C"].merge(rep, on=["panel", "M", "column"])
    c2["reasons_A_minus_C"] = dict(rows=len(a_c), bad=int((a_c["count"] != a_c["A"] - a_c["C"]).sum()))
    b_a = pr[pr.set == "B\\A"].merge(rep[rep.panel == "F28"], on=["M", "column"])
    c2["reasons_B_minus_A"] = dict(rows=len(b_a), bad=int((b_a["count"] != b_a["B"] - b_a["A"]).sum()))
    tr = mine["tier_rel_date"]
    msum = tr[[f"m{a}{b}" for a in (1, 2, 3) for b in (1, 2, 3)]].sum(axis=1)
    c2["rel_matrix_sums"] = dict(rows=len(tr), bad=int((msum != tr["n"]).sum()))
    fz = mine["tier_frozen_date"]
    fsum = fz[[f"m{a}{b}" for a in (1, 2, 3) for b in (1, 2, 3)]].sum(axis=1)
    four = sum(fz[f"m{t}{s}"] for t in (1, 2, 3) for s in (1, 2, 3)) + sum(fz[f"tier{t}_unreportable"] for t in (1, 2, 3))
    tiers_ok = all(((fz[[f"m{t}{s}" for s in (1, 2, 3)]].sum(axis=1) + fz[f"tier{t}_unreportable"])
                    == fz[f"tier{t}_n"]).all() for t in (1, 2, 3))
    c2["frozen"] = dict(rows=len(fz), matrix_ne_G=int((fsum != fz["G"]).sum()), outcomes_ne_N=int((four != fz["N"]).sum()),
                        per_tier_outcomes_sum_to_tier_size=bool(tiers_ok),
                        tier_sizes_ne_N=int((fz[["tier1_n", "tier2_n", "tier3_n"]].sum(axis=1) != fz["N"]).sum()))
    fr_ = fz.merge(rep[rep.panel == "F28"], on=["panel", "M", "column"])
    c2["frozen_vs_report"] = dict(bad=int(((fr_["N"] != fr_["A"]) | (fr_["G"] != fr_["C"])).sum()))
    ct = mine["category_trans_date"].groupby(["panel", "M", "column"])["count"].sum().reset_index()
    ct = ct.merge(mine["category_date"], on=["panel", "M", "column"])
    c2["category_trans_sums"] = dict(rows=len(ct), bad=int((ct["count"] != ct["n"]).sum()))
    rp4 = rep[rep.column != "top_symbol"]
    c2["report_tiers"] = dict(bad=int(((rp4[["tier1_n", "tier2_n", "tier3_n"]].sum(axis=1) != rp4["A"])
                                       | (rp4[["tier1_C", "tier2_C", "tier3_C"]].sum(axis=1) != rp4["C"])).sum()))
    ac = mine["activity_date"]
    sup_ = sup.astype({"window_id": str})
    bad_act = 0
    for r in ac.itertuples(index=False):
        sl = sup_[sup_.window_id == f"L60_{r.M}"].set_index("trader")
        sf = sup_[sup_.window_id == f"{r.panel}_{r.M}"].set_index("trader").reindex(sl.index)
        B = sl.base_eligible
        bad_act += int(r.no_fills + r.any_fills != r.B)
        bad_act += int(r.any_fills != int((B & (sf.n_fills > 0)).sum()))
        if r.panel == "F28":
            bad_act += int(not (r.orders30_days10 <= r.orders15_days5 <= r.any_fills))
            bad_act += int(r.orders15_days5 != int((B & (sf.n_orders >= 15) & (sf.n_fill_days >= 5)).sum()))
        else:
            bad_act += int(not (r.orders5_days2 <= r.any_fills))
    c2["activity"] = dict(rows=len(ac), bad=bad_act,
                          B_equals_report=bool((ac[ac.panel == "F28"].set_index("M")["B"]
                                                == rep[(rep.panel == "F28") & (rep.column == "maker_share")]
                                                .set_index("M")["B"]).all()))
    mt = mine["matched_date"]
    c2["matched_cohort_le_C"] = dict(bad=int((mt["n"] > mt["C"]).sum()))
    gs = mine["gate_sens_date"]
    gp = gs.pivot_table(index=["M", "column"], columns="cohort", values="n", aggfunc="first")
    c2["strict_cohort_le_primary"] = dict(bad=int((gp["C_strict"] > gp["C_primary"]).sum()))
    # pooling records reproduce the per-address ledger values
    data = load(reg)
    pools = build_pools(reg, data, [f"{k}_{M}" for M in ("2025-10-13", "2025-12-01") for k in ("L60", "F28", "F2")])
    pc = {}
    for (wid, col), pl in pools.items():
        x = data["vals"][wid][col].to_numpy(dtype=float)
        if pl.mode == "wmean":
            num = np.bincount(pl.ai, weights=pl.num, minlength=data["nU"])
            y = np.where(pl.addr_w > 0, num / np.where(pl.addr_w > 0, pl.addr_w, 1), np.nan)
        else:
            y = pd.Series(pl.v).groupby(pl.ai).median().reindex(range(data["nU"])).to_numpy()
        pc[f"{wid}|{col}"] = int((~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.abs(x) + 1e-12))).sum())
    c2["pool_records_vs_ledger_values"] = dict(pairs=len(pc), addresses_differing=sum(pc.values()),
                                               worst={k: v for k, v in pc.items() if v})
    c2["ok"] = (all(v.get("bad", 0) == 0 for v in c2.values() if isinstance(v, dict))
                and c2["frozen"]["matrix_ne_G"] == 0 and c2["frozen"]["outcomes_ne_N"] == 0
                and c2["frozen"]["per_tier_outcomes_sum_to_tier_size"] and c2["frozen"]["tier_sizes_ne_N"] == 0
                and c2["activity"]["B_equals_report"] and sum(pc.values()) == 0)
    res["check2_reconciliation"] = c2
    log(json.dumps(c2, default=str, indent=1))

    # ---- 3: fixtures ----
    log("\n=== check 3: fixtures ===")
    fx = fixtures()
    res["check3_fixtures"] = dict(cases=fx, ok=all(v["ok"] for v in fx.values()))
    for k, v in fx.items():
        log(f"  {k}: {v}")

    # ---- 4: independent recomputation ----
    log("\n=== check 4: independent recomputation ===")
    ind = independent(BUILD, mine)
    res["check4_independent"] = dict(cases=ind, ok=all(v["ok"] for v in ind.values()))
    for k, v in ind.items():
        log(f"  {k}: {v}")

    # ---- 5: duplicate-window identity F28_M == R28_{M+28d} ----
    log("\n=== check 5: F28 of M equals R28 of M+28d ===")
    c5 = {}
    for i, M in enumerate(MS):
        M2 = f"{pd.Timestamp(M) + pd.Timedelta(days=28):%Y-%m-%d}"
        if M2 not in MS:
            continue
        a, b = data["sup"][f"F28_{M}"], data["sup"][f"R28_{M2}"]
        sd = 0
        for c in a.columns:
            if c == "window_id":
                continue
            x, y = a[c], b[c]
            same = (x.to_numpy() == y.to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
            sd += int((~same).sum())
        va, vb = data["vals"][f"F28_{M}"], data["vals"][f"R28_{M2}"]
        vd = 0
        for c in va.columns:
            x, y = va[c], vb[c]
            same = (x.to_numpy() == y.to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
            vd += int((~same).sum())
        c5[f"F28_{M}=R28_{M2}"] = dict(support_cells_differing=sd, value_cells_differing=vd)
    res["check5_duplicate_windows"] = dict(pairs=c5, ok=len(c5) == 7 and all(
        v["support_cells_differing"] == 0 and v["value_cells_differing"] == 0 for v in c5.values()))
    log(json.dumps(c5, indent=1))

    res["all_ok"] = all(v.get("ok", True) for v in res.values() if isinstance(v, dict))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"\nall_ok {res['all_ok']}; checks done in {res['runtime_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="run the acceptance checks on the built outputs")
    args = ap.parse_args()
    checks() if args.checks else build()
