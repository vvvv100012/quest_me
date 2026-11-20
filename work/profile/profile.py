"""Per-address trading profile table for the Hyperliquid perp fills dataset.

One row per address with at least one perp fill (1,173). Dimensions:
  D1 frequency   active days, orders per day, decision clusters, maker share
  D2 holding     position episodes from fill groups, turnover-implied holding time
  D3 size        order / cluster notional, synthetic account value, leverage
  D4 behaviour   symbol concentration, long bias, adds / reductions, adding to losers, flips

Inputs (read-only): lib.prep_perp_fills(), lib.load_traders(), v2dev/build/groups.parquet,
v2dev/build/snap_value_pair.parquet. Outputs (this directory): address_profile.csv,
profile_quantiles.csv. The script prints the build checks at the end.
"""
import os
import sys
import time
import warnings

os.environ.setdefault("ARK_DATA_DIR", "/Users/xc/Documents/arkstream_quest/ArkStream笔试题_v2")
sys.path.insert(0, "/Users/xc/Documents/arkstream_quest/work/v2dev/code")
import lib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.simplefilter("ignore", FutureWarning)

OUT = os.path.dirname(os.path.abspath(__file__))
BUILD = "/Users/xc/Documents/arkstream_quest/work/v2dev/build"
CUTOFF = lib.CUTOFF
CLUSTER_GAP = pd.Timedelta(seconds=60)
EXP_EPS = 1e-6
FLIP_DIRS = {"Long > Short", "Short > Long"}
H = 3600e9  # ns per hour


def q(s, p):
    """Quantile that returns NaN on empty input."""
    s = s.dropna()
    return s.quantile(p) if len(s) else np.nan


def safe_div(a, b):
    a = pd.Series(a, dtype=float) if not isinstance(a, pd.Series) else a.astype(float)
    b = b.astype(float)
    return a / b.where(b != 0)


# ---------------------------------------------------------------------------------------
# Episodes from fill groups
# ---------------------------------------------------------------------------------------
def build_episodes(g):
    """Walk every (trader, symbol) in time order. Returns (episodes, per-group increasing notional).

    Episode fields: open_ts / close_ts (NaT when censored), sign, open_ntl (position-increasing
    notional), end_by_flip, and, for episodes that are not left-censored, the later-add
    notional and the part of it added to a loser (vs the running average entry price).
    """
    pid = g["pid"].to_numpy()
    ts = g["ts"].to_numpy().astype("int64")
    gs = g["gstart"].to_numpy()
    ge = g["gend"].to_numpy()
    sg = g["signed"].to_numpy()
    nt = g["notional"].to_numpy()
    vw = g["vwap"].to_numpy()
    n = len(g)
    NAT = np.iinfo("int64").min
    rows = []            # finished episodes
    cur = None           # [pid, open_ts, sign, open_ntl, left_cens, later_add, later_loser, avg_entry]

    def close(c, t, flip):
        rows.append((c[0], c[1], t, c[2], c[3], c[4], t == NAT, flip, c[5], c[6]))

    for i in range(n):
        if i == 0 or pid[i] != pid[i - 1]:
            if cur is not None:
                close(cur, NAT, False)              # right-censored
            cur = None
            if gs[i] != 0:                          # position pre-dates the pair's first group
                cur = [pid[i], NAT, np.sign(gs[i]), 0.0, True, 0.0, 0.0, np.nan]
        s0, s1 = gs[i], ge[i]
        if s0 == 0:
            if s1 != 0:                             # open from flat
                cur = [pid[i], ts[i], np.sign(s1), nt[i], False, 0.0, 0.0, vw[i]]
            # s0 == s1 == 0: same-timestamp round trip, no episode
        elif s1 == 0:                               # close to flat
            close(cur, ts[i], False)
            cur = None
        elif np.sign(s0) != np.sign(s1):            # flip: close old, open new, split pro rata
            close(cur, ts[i], True)
            cur = [pid[i], ts[i], np.sign(s1), nt[i] * abs(s1) / abs(sg[i]), False, 0.0, 0.0, vw[i]]
        elif abs(s1) > abs(s0):                     # add to an existing position
            cur[3] += nt[i]
            if not cur[4]:
                cur[5] += nt[i]
                avg = cur[7]
                if (cur[2] > 0 and vw[i] < avg) or (cur[2] < 0 and vw[i] > avg):
                    cur[6] += nt[i]
                cur[7] = (avg * abs(s0) + vw[i] * (abs(s1) - abs(s0))) / abs(s1)
        # else: reduction (or equal-size mixed group), average entry unchanged
    if cur is not None:
        close(cur, NAT, False)

    ep = pd.DataFrame(rows, columns=["pid", "open_ns", "close_ns", "sign", "open_ntl", "left_cens",
                                     "right_cens", "end_by_flip", "later_add_ntl", "later_loser_ntl"])
    for a, b in (("open_ns", "open_ts"), ("close_ns", "close_ts")):
        # int64 min is NaT in datetime64[ns]; a view keeps exact ns (a float round trip would not)
        ep[b] = ep[a].to_numpy(dtype="int64").view("datetime64[ns]")
    ep = ep.drop(columns=["open_ns", "close_ns"])
    ep["complete"] = ~ep["left_cens"] & ~ep["right_cens"]
    ep["dur_h"] = (ep["close_ts"] - ep["open_ts"]).dt.total_seconds() / 3600.0
    ep["eid"] = np.arange(len(ep))

    # direct (vectorised) position-increasing notional per group, for the conservation check
    a0, a1 = np.abs(gs), np.abs(ge)
    flip = (gs != 0) & (ge != 0) & (np.sign(gs) != np.sign(ge))
    inc_full = ((gs == 0) & (ge != 0)) | ((gs != 0) & ~flip & (a1 > a0))
    inc = np.where(inc_full, nt, 0.0)
    inc = np.where(flip, nt * a1 / np.where(sg == 0, np.nan, np.abs(sg)), inc)
    return ep, inc, flip


def assign_fills(p, ep):
    """Map raw fill rows to episodes by (trader, symbol, ts in [open, close]) with dir roles:
    Open* increases, Close* and forced dirs reduce, flip dirs reduce the old episode and
    increase the new one. Returns distinct-oid counts of adds / reductions per episode."""
    lo, hi = pd.Timestamp("1970-01-01"), pd.Timestamp("2100-01-01")
    e = ep[["trader", "symbol", "eid", "open_ts", "close_ts"]].copy()
    e["o"] = e["open_ts"].fillna(lo)
    e["c"] = e["close_ts"].fillna(hi)
    f = p[["trader", "symbol", "ts", "oid", "dir", "is_forced"]].copy()
    is_flip = f["dir"].isin(FLIP_DIRS)
    is_inc = f["dir"].str.startswith("Open") | is_flip
    is_red = f["dir"].str.startswith("Close") | f["is_forced"] | is_flip
    out = {}
    # increasing role: latest episode with open <= ts, keep if ts <= close
    fi = f[is_inc].sort_values("ts")
    m = pd.merge_asof(fi, e.sort_values("o"), left_on="ts", right_on="o", by=["trader", "symbol"],
                      direction="backward")
    m = m[m["eid"].notna() & (m["ts"] <= m["c"])]
    out["add"] = m.groupby("eid")["oid"].nunique()
    out["n_inc_rows"] = (int(is_inc.sum()), len(m))
    # reducing role: earliest episode with close >= ts, keep if open <= ts
    fr = f[is_red].sort_values("ts")
    m = pd.merge_asof(fr, e.sort_values("c"), left_on="ts", right_on="c", by=["trader", "symbol"],
                      direction="forward")
    m = m[m["eid"].notna() & (m["ts"] >= m["o"])]
    out["red"] = m.groupby("eid")["oid"].nunique()
    out["n_red_rows"] = (int(is_red.sum()), len(m))
    return out


def main():
    t0 = time.time()
    p = lib.prep_perp_fills()
    g = pd.read_parquet(f"{BUILD}/groups.parquet").sort_values(["trader", "symbol", "ts"]).reset_index(drop=True)
    sv = pd.read_parquet(f"{BUILD}/snap_value_pair.parquet")
    tr = lib.load_traders()
    print(f"loaded in {time.time() - t0:.1f}s: fills {len(p):,}, groups {len(g):,}, snap pairs {len(sv):,}")
    print("dir values:", p["dir"].value_counts().to_dict())

    p["day"] = p["ts"].dt.floor("D")
    traders = pd.Index(sorted(p["trader"].unique()), name="trader")
    prof = pd.DataFrame(index=traders)

    # ---------------- identity / sample ----------------
    gt = p.groupby("trader")
    prof["first_fill"] = gt["ts"].min()
    prof["last_fill"] = gt["ts"].max()
    prof["n_fills"] = gt.size()
    orders = p.groupby(["trader", "symbol", "oid"]).agg(
        first_ts=("ts", "min"), last_ts=("ts", "max"), ntl=("notional", "sum"),
        nside=("side", "nunique"), side=("side", "first")).reset_index()
    prof["n_orders"] = orders.groupby("trader").size()
    prof["n_groups"] = g.groupby("trader").size()
    prof["turnover"] = gt["notional"].sum()

    # ---------------- D1 frequency ----------------
    act = p[["trader", "day"]].drop_duplicates()
    prof["active_days"] = act.groupby("trader").size()
    span = (CUTOFF - prof["first_fill"].dt.floor("D")).dt.days     # first-fill day .. 2026-01-04 inclusive
    prof["active_day_share"] = prof["active_days"] / span
    prof["enough"] = (prof["active_days"] >= 10) & (prof["n_orders"] >= 30)
    prof["fills_per_order"] = prof["n_fills"] / prof["n_orders"]

    orders["day"] = orders["first_ts"].dt.floor("D")
    od = act.merge(orders.groupby(["trader", "day"]).size().rename("k").reset_index(),
                   on=["trader", "day"], how="left").fillna({"k": 0})
    prof["orders_per_active_day_med"] = od.groupby("trader")["k"].median()
    prof["orders_per_active_day_p90"] = od.groupby("trader")["k"].quantile(0.9)

    # decision clusters within (trader, symbol)
    orders = orders.sort_values(["trader", "symbol", "first_ts", "last_ts", "oid"]).reset_index(drop=True)
    same_pair = (orders["trader"].eq(orders["trader"].shift()) & orders["symbol"].eq(orders["symbol"].shift()))
    mixed = orders["nside"] > 1
    joins = (same_pair & orders["side"].eq(orders["side"].shift()) & ~mixed & ~mixed.shift(fill_value=False)
             & ((orders["first_ts"] - orders["last_ts"].shift()) <= CLUSTER_GAP))
    orders["cid"] = (~joins).cumsum()
    cl = orders.groupby("cid").agg(trader=("trader", "first"), first_ts=("first_ts", "min"),
                                   ntl=("ntl", "sum"), k=("oid", "size"))
    cl["day"] = cl["first_ts"].dt.floor("D")
    cd = act.merge(cl.groupby(["trader", "day"]).size().rename("k").reset_index(),
                   on=["trader", "day"], how="left").fillna({"k": 0})
    prof["clusters_per_active_day_med"] = cd.groupby("trader")["k"].median()
    n_clusters = cl.groupby("trader").size()
    prof["orders_per_cluster"] = prof["n_orders"] / n_clusters
    prof["maker_share"] = safe_div(gt["maker_notional"].sum(), prof["turnover"])

    # ---------------- D2 holding period ----------------
    pairs = g[["trader", "symbol"]].drop_duplicates().reset_index(drop=True)
    pairs["pid"] = np.arange(len(pairs))
    g = g.merge(pairs, on=["trader", "symbol"], how="left")
    ep, inc_direct, flip_grp = build_episodes(g)
    g["inc_direct"] = inc_direct
    g["is_flip"] = flip_grp
    ep = ep.merge(pairs, on="pid", how="left")
    ce = ep[ep["complete"]].copy()

    gp = ce.groupby("trader")
    prof["n_episodes"] = gp.size().reindex(traders).fillna(0).astype(int)
    prof["n_left_cens"] = ep[ep["left_cens"]].groupby("trader").size().reindex(traders).fillna(0).astype(int)
    prof["n_right_cens"] = ep[ep["right_cens"]].groupby("trader").size().reindex(traders).fillna(0).astype(int)
    last_g = g.groupby("pid").tail(1)
    last_g = last_g[last_g["gend"] != 0]
    prof["open_exposure_end"] = (last_g["gend"].abs() * last_g["vwap"]).groupby(last_g["trader"]).sum() \
        .reindex(traders).fillna(0.0)
    hq = gp["dur_h"].quantile([0.25, 0.5, 0.75]).unstack()
    prof["hold_med_h"] = hq[0.5]
    prof["hold_p25_h"] = hq[0.25]
    prof["hold_p75_h"] = hq[0.75]
    ce["w_dur"] = ce["open_ntl"] * ce["dur_h"]
    prof["hold_wavg_h"] = safe_div(gp["w_dur"].sum(), gp["open_ntl"].sum())
    ntl_tot = gp["open_ntl"].sum()
    for name, mask in (("lt1m", ce["dur_h"] < 1 / 60), ("lt5m", ce["dur_h"] < 5 / 60),
                       ("lt1h", ce["dur_h"] < 1.0), ("gt1d", ce["dur_h"] > 24.0)):
        prof[f"ep_{name}_cnt"] = mask.groupby(ce["trader"]).mean()
        prof[f"ep_{name}_ntl"] = safe_div(ce["open_ntl"].where(mask, 0.0).groupby(ce["trader"]).sum(), ntl_tot)

    # turnover-implied holding time
    nxt = g.groupby("pid")["ts"].shift(-1).fillna(CUTOFF)
    held = g["gend"].abs() * g["vwap"] * (nxt - g["ts"]).dt.total_seconds() / 3600.0
    prof["hold_turnover_h"] = safe_div(held.groupby(g["trader"]).sum().reindex(traders), prof["turnover"] / 2)

    # ---------------- D3 size ----------------
    og = orders.groupby("trader")["ntl"]
    prof["order_ntl_med"] = og.median()
    prof["order_ntl_p90"] = og.quantile(0.9)
    prof["order_ntl_p99"] = og.quantile(0.99)
    prof["cluster_ntl_med"] = cl.groupby("trader")["ntl"].median()
    dt_ = p.groupby(["trader", "day"])["notional"].sum()
    prof["turnover_per_active_day_med"] = dt_.groupby("trader").median()

    # synthetic account value at snapshot times
    sv = sv[sv["trader"].isin(traders)].copy()
    sv["exposure"] = sv["exposure"].where(sv["exposure"].abs() >= EXP_EPS, 0.0)
    sv["r"] = sv["value"] - sv["cfee"]
    sv["aexp"] = sv["exposure"].abs()
    sv["hip3_aexp"] = np.where(sv["symbol"].str.contains(":", regex=False), sv["aexp"], 0.0)
    sv["has_pos"] = sv["exposure"] != 0
    sn = sv.groupby(["trader", "ts"]).agg(R=("r", "sum"), hip3_exp=("hip3_aexp", "sum"), gross_exp=("aexp", "sum"),
                                          net_exp=("exposure", "sum"), n_pos=("has_pos", "sum")).reset_index()
    n_sn = len(sn)
    sn = sn.merge(tr[["trader", "ts", "pnl", "net_deposit", "acct_val"]], on=["trader", "ts"], how="left")
    n_unmatched = int(sn["pnl"].isna().sum())
    sn = sn.sort_values(["trader", "ts"]).reset_index(drop=True)
    first_clean = sn[sn["hip3_exp"] == 0].groupby("trader").head(1).set_index("trader")
    first_any = sn.groupby("trader").head(1).set_index("trader")
    anchor = first_any.copy()
    anchor.loc[first_clean.index] = first_clean
    anchor["anchor_has_hip3"] = ~anchor.index.isin(first_clean.index)
    sn = sn.merge(anchor[["ts", "pnl", "R"]].rename(columns={"ts": "a_ts", "pnl": "a_pnl", "R": "a_R"}),
                  left_on="trader", right_index=True, how="left")
    sn["av_synth"] = sn["net_deposit"] + sn["a_pnl"] - sn["a_R"] + sn["R"]
    sn["av_snap"] = sn["acct_val"]
    sn["inm"] = sn["gross_exp"] > 0
    # snapshot statistics describe the address while it is observably trading: drop snapshots
    # taken before its first fill (the anchor above may still be one of them)
    sn = sn[sn["ts"] >= sn["trader"].map(prof["first_fill"])].reset_index(drop=True)

    gs_ = sn.groupby("trader")
    prof["n_snaps"] = gs_.size().reindex(traders).fillna(0).astype(int)
    prof["time_in_market"] = gs_["inm"].mean()
    inm = sn[sn["inm"]]
    gi = inm.groupby("trader")
    prof["gross_exp_med"] = gi["gross_exp"].median()
    prof["gross_exp_p95"] = gi["gross_exp"].quantile(0.95)
    prof["av_synth_med"] = gs_["av_synth"].median()
    prof["av_snap_med"] = gs_["av_snap"].median()
    prof["av_synth_last"] = gs_["av_synth"].last()
    prof["av_snap_last"] = gs_["av_snap"].last()
    prof["av_bad_share"] = (sn["av_synth"] < 100).groupby(sn["trader"]).mean()
    prof["anchor_ts"] = anchor["ts"]
    prof["anchor_has_hip3"] = anchor["anchor_has_hip3"]
    lv = inm[inm["av_synth"] >= 100].copy()
    lv["lev"] = lv["gross_exp"] / lv["av_synth"]
    gl = lv.groupby("trader")["lev"]
    prof["lev_med"] = gl.median()
    prof["lev_p95"] = gl.quantile(0.95)
    prof["n_lev_obs"] = gl.size().reindex(traders).fillna(0).astype(int)
    last_snap = gs_["ts"].max()
    sn_last = sn[sn["ts"] > sn["trader"].map(last_snap) - pd.Timedelta(days=28)]
    av_ref = sn_last.groupby("trader")["av_synth"].median().reindex(traders)
    av_ref = av_ref.where(av_ref >= 100)
    prof["av_ref"] = av_ref
    fail = orders["ntl"] * 1000 / orders["trader"].map(av_ref) < 10
    prof["min_order_fail_share"] = fail.groupby(orders["trader"]).mean().where(av_ref.notna())

    # ---------------- D4 behaviour and exposure ----------------
    st = p.groupby(["trader", "symbol"])["notional"].sum()
    sh = st / st.groupby("trader").transform("sum")
    prof["n_symbols"] = st.groupby("trader").size()
    top = sh.groupby("trader").idxmax()
    prof["top1_symbol"] = top.map(lambda x: x[1])
    prof["top1_share"] = sh.groupby("trader").max()
    prof["sym_hhi"] = (sh ** 2).groupby("trader").sum()
    prof["hip3_share"] = safe_div(p["notional"].where(p["is_hip3"], 0.0).groupby(p["trader"]).sum(), prof["turnover"])
    prof["btc_eth_share"] = safe_div(p["notional"].where(p["symbol"].isin(["BTC", "ETH"]), 0.0)
                                     .groupby(p["trader"]).sum(), prof["turnover"])
    ol = p["notional"].where(p["dir"] == "Open Long", 0.0).groupby(p["trader"]).sum()
    oo = p["notional"].where(p["dir"].isin(["Open Long", "Open Short"]), 0.0).groupby(p["trader"]).sum()
    prof["long_open_share"] = safe_div(ol, oo)
    prof["net_bias"] = safe_div(gs_["net_exp"].sum(), gs_["gross_exp"].sum())
    prof["long_time_share"] = (inm["net_exp"] > 0).groupby(inm["trader"]).mean()
    prof["n_pos_med"] = gi["n_pos"].median()
    prof["n_pos_p95"] = gi["n_pos"].quantile(0.95)

    af = assign_fills(p, ep)
    ce["n_add_orders"] = ce["eid"].map(af["add"]).fillna(0).astype(int)
    ce["n_red_orders"] = ce["eid"].map(af["red"]).fillna(0).astype(int)
    gp = ce.groupby("trader")
    prof["adds_per_ep_med"] = gp["n_add_orders"].median()
    prof["reds_per_ep_med"] = gp["n_red_orders"].median()
    prof["single_shot_share"] = ((ce["n_add_orders"] == 1) & (ce["n_red_orders"] == 1)).groupby(ce["trader"]).mean()

    nl = ep[~ep["left_cens"]].groupby("trader")      # average entry undefined in left-censored episodes
    later = nl["later_add_ntl"].sum().reindex(traders)
    prof["add_ntl_share"] = safe_div(later, nl["open_ntl"].sum().reindex(traders))
    prof["add_loser_share"] = safe_div(nl["later_loser_ntl"].sum().reindex(traders), later)
    prof["n_flips"] = g.groupby("trader")["is_flip"].sum().astype(int)
    prof["flip_share"] = gp["end_by_flip"].mean()
    prof["n_forced_fills"] = gt["is_forced"].sum().astype(int)
    prof["forced_ntl_share"] = safe_div(gt["forced_notional"].sum(), prof["turnover"])

    prof = prof.reset_index()
    prof.to_csv(f"{OUT}/address_profile.csv", index=False)

    # ---------------- quantile table ----------------
    num = [c for c in prof.columns if pd.api.types.is_numeric_dtype(prof[c]) and not pd.api.types.is_bool_dtype(prof[c])]
    qrows = []
    for sample, d in (("all", prof), ("enough", prof[prof["enough"]])):
        for c in num:
            s = d[c].astype(float)
            r = {"sample": sample, "column": c, "count": int(s.notna().sum()), "mean": s.mean()}
            for pp in (5, 25, 50, 75, 95):
                r[f"p{pp:02d}"] = q(s, pp / 100)
            qrows.append(r)
    pd.DataFrame(qrows).to_csv(f"{OUT}/profile_quantiles.csv", index=False)

    # ---------------- checks ----------------
    print("\n=== check 1: totals ===")
    print(f"rows {len(prof):,} (expect 1,173); sum n_fills {prof['n_fills'].sum():,} (expect 864,623); "
          f"sum n_orders {prof['n_orders'].sum():,} (expect 224,024)")
    print(f"sum turnover {prof['turnover'].sum():,.4f} vs sum fill notional {p['notional'].sum():,.4f}; "
          f"diff {prof['turnover'].sum() - p['notional'].sum():.3e}")
    print(f"sum n_groups {prof['n_groups'].sum():,} (groups.parquet {len(g):,}); "
          f"group notional - fill notional {g['notional'].sum() - p['notional'].sum():.3e}")
    print(f"enough addresses: {int(prof['enough'].sum())}; mixed-side orders: {int(mixed.sum())}; "
          f"clusters: {len(cl):,}")
    print(f"snapshot rows {n_sn:,}, unmatched to load_traders {n_unmatched}; "
          f"addresses with snapshots {sn['trader'].nunique()}; anchor_has_hip3 {int(anchor['anchor_has_hip3'].sum())}")

    print("\n=== check 2: open_ntl conservation per pair ===")
    a = ep.groupby("pid")["open_ntl"].sum()
    b = g.groupby("pid")["inc_direct"].sum()
    diff = (a.reindex(b.index).fillna(0) - b).abs()
    print(f"pairs {len(b):,}; max abs diff {diff.max():.3e}; max rel diff {(diff / b.where(b > 0)).max():.3e}")
    print(f"episodes: total {len(ep):,}, complete {int(ep['complete'].sum()):,}, left-censored "
          f"{int(ep['left_cens'].sum()):,}, right-censored {int(ep['right_cens'].sum()):,}, both "
          f"{int((ep['left_cens'] & ep['right_cens']).sum()):,}")
    zz = int(((g["gstart"] == 0) & (g["gend"] == 0)).sum())
    print(f"zero-to-zero groups (no episode): {zz}")
    cont = (g["gstart"] - g.groupby("pid")["gend"].shift()).abs()
    print(f"group continuity: max |gstart - prev gend| {cont.max():.3e}; "
          f"zero/nonzero mismatches {int(((g['gstart'] == 0) != (g.groupby('pid')['gend'].shift() == 0))[cont.notna()].sum())}")
    print(f"fill->episode assignment: increasing rows {af['n_inc_rows'][0]:,} assigned {af['n_inc_rows'][1]:,}; "
          f"reducing rows {af['n_red_rows'][0]:,} assigned {af['n_red_rows'][1]:,}")
    print(f"complete episodes with 0 add orders {int((ce['n_add_orders'] == 0).sum())}, 0 red orders "
          f"{int((ce['n_red_orders'] == 0).sum())}")

    print("\n=== check 3: flips ===")
    n_flip_rows = int(p["dir"].isin(FLIP_DIRS).sum())
    fl = g[g["is_flip"]]
    print(f"flip groups {len(fl):,}; fill rows with flip dir {n_flip_rows:,}")
    fr = p[p["dir"].isin(FLIP_DIRS)].groupby(["trader", "symbol", "ts"]).size()
    fk = fl.set_index(["trader", "symbol", "ts"]).index
    print(f"distinct (trader,symbol,ts) with a flip-dir row {len(fr):,}; of these are flip groups "
          f"{int(fr.index.isin(fk).sum()):,}; flip groups without a flip-dir row {int((~fk.isin(fr.index)).sum()):,}; "
          f"flip-dir rows per flip group max {int(fr.max()) if len(fr) else 0}")

    print("\n=== check 4: example addresses ===")
    e2 = prof[prof["enough"] & (prof["n_episodes"] >= 20)]
    picks = [("short-hold", e2.loc[e2["hold_med_h"].idxmin(), "trader"]),
             ("long-hold", e2.loc[e2["hold_med_h"].idxmax(), "trader"]),
             ("flipper", e2.loc[e2["flip_share"].idxmax(), "trader"])]
    pc = p.sort_values(["trader", "symbol", "ts", "tid"])
    for label, addr in picks:
        r = prof.set_index("trader").loc[addr]
        print(f"\n[{label}] {addr}: n_episodes {r['n_episodes']}, hold_med_h {r['hold_med_h']:.4f}, "
              f"flip_share {r['flip_share']:.3f}, n_flips {r['n_flips']}")
        el = ep[ep["trader"] == addr].sort_values(["symbol", "eid"])
        print(el[["symbol", "open_ts", "close_ts", "sign", "open_ntl", "left_cens", "right_cens",
                  "end_by_flip", "dur_h"]].head(6).to_string(index=False))
        cx = ce[ce["trader"] == addr]
        # the flipper's verified episode is one that both opens and ends at a flip
        if label == "flipper" and (cx["end_by_flip"] & cx["open_ts"].isin(cx["close_ts"])).any():
            cx = cx[cx["end_by_flip"] & cx["open_ts"].isin(cx["close_ts"])]
        x = cx.iloc[0]
        rows = pc[(pc["trader"] == addr) & (pc["symbol"] == x["symbol"])]
        prev = rows[rows["ts"] < x["open_ts"]].tail(2)
        inside = rows[(rows["ts"] >= x["open_ts"]) & (rows["ts"] <= x["close_ts"])]
        nxt_ = rows[rows["ts"] > x["close_ts"]].head(1)
        cols = ["ts", "dir", "side", "start_position", "size", "end_pos", "price", "oid"]
        print(f"  verify episode {x['symbol']} {x['open_ts']} -> {x['close_ts']} (dur {x['dur_h']:.4f} h, "
              f"{(x['close_ts'] - x['open_ts']).total_seconds():.3f} s); raw fills before / inside / after:")
        show = pd.concat([prev.assign(w="before"), inside.head(4).assign(w="inside"),
                          inside.iloc[max(4, len(inside) - 3):].assign(w="inside"), nxt_.assign(w="after")])
        print(show[["w"] + cols].to_string(index=False))
        o_rows = inside[inside["ts"] == x["open_ts"]]
        c_rows = inside[inside["ts"] == x["close_ts"]]
        mid = inside[(inside["ts"] > x["open_ts"]) & (inside["ts"] < x["close_ts"])]
        ok_open = ((o_rows["start_position"].abs() < 1e-9) | o_rows["dir"].isin(FLIP_DIRS)).any()
        ok_close = ((c_rows["end_pos"].abs() < 1e-9) | c_rows["dir"].isin(FLIP_DIRS)).any()
        ok_mid = not ((mid["end_pos"].abs() < 1e-9) | mid["dir"].isin(FLIP_DIRS)).any()
        print(f"  raw-fill verification: open row starts flat/flips {ok_open}; close row ends flat/flips "
              f"{ok_close}; no flat/flip strictly inside {ok_mid}; inside rows {len(inside)}")

    print("\n=== check 5: NaN share of headline columns ===")
    for c in ("hold_med_h", "lev_med", "av_synth_med", "add_loser_share"):
        print(f"{c}: all {prof[c].isna().mean():.3f} ({int(prof[c].isna().sum())}), "
              f"enough {prof.loc[prof['enough'], c].isna().mean():.3f}")
    print(f"\nwrote {OUT}/address_profile.csv ({len(prof)} x {prof.shape[1]}) and profile_quantiles.csv "
          f"({len(qrows)} rows); runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
