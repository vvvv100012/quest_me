"""Reproduce the data examples and external-data checks in final_report section 1.

Run from any directory after the package data and work/extdata/build are present:
    python reproduce/report_data.py
The external build manifest is produced from the frozen raw inputs by build_ext.py.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ArkStream笔试题_v2"
BUILD = ROOT / "work/extdata/build"

EXAMPLES = [
    ("1.2 划转", "0x84b5ffcb1d9abd86f4fd26a2c9e5a9b54c35c9be", "2025-09-26", "2025-09-27"),
    ("1.2 现货", "0x4648baa72d8cfe3f31882fe52bd9fd8e6d539373", "2025-08-27", "2025-08-28"),
    ("1.3 逐仓", "0x4648baa72d8cfe3f31882fe52bd9fd8e6d539373", "2025-08-18", "2025-08-22"),
    ("1.4 HIP-3", "0x87a726a806f48a435d8bf627678c85f26b46c1b6", "2025-12-24", "2025-12-28"),
]


def main():
    required = [PACKAGE / "笔试题数据包-traders.csv", PACKAGE / "笔试题数据包-fills",
                BUILD / "manifest.json", BUILD / "funding.parquet",
                BUILD / "symbol_map.csv", BUILD / "hidden_positions.parquet",
                ROOT / "work/v2dev/build/groups.parquet",
                ROOT / "work/v2dev/build/snap_value_pair.parquet"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("缺少数据，请先完成外部数据重建：\n" + "\n".join(missing))
    addresses = {x[1] for x in EXAMPLES}
    snapshots = pd.read_csv(required[0])
    snapshots = snapshots[snapshots.trader.isin(addresses)].copy()
    snapshots["date"] = pd.to_datetime(snapshots.timestamp).dt.date
    snapshots["account_value"] = snapshots.pnl + snapshots.net_deposit
    snapshots["isolated_margin_estimate"] = snapshots.account_value - snapshots.equity
    for label, address, first, last in EXAMPLES:
        rows = snapshots[(snapshots.trader == address) &
                         (snapshots.date >= pd.Timestamp(first).date()) &
                         (snapshots.date <= pd.Timestamp(last).date())]
        print("\n", label, address)
        print(rows[["timestamp", "equity", "net_deposit", "pnl", "account_value",
                    "isolated_margin_estimate"]].sort_values("timestamp").to_string(index=False))

    fills = pd.read_csv(required[1])
    fills = fills[fills.trader.isin(addresses)].copy()
    fills["timestamp"] = pd.to_datetime(fills.timestamp, utc=True, format="mixed")
    for label, address, first, last in EXAMPLES:
        start = pd.Timestamp(first, tz="UTC")
        stop = pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1)
        rows = fills[(fills.trader == address) & (fills.timestamp >= start) & (fills.timestamp < stop)]
        print("\n", label, "成交记录")
        print(rows[["timestamp", "symbol", "side", "dir", "price", "size", "fee",
                    "closed_pnl"]].sort_values("timestamp").to_string(index=False))

    manifest = json.loads(required[2].read_text())
    checks = manifest["checks"]
    print("\n1.5 外部数据构建结果（来自逐项检查的 manifest.json）")
    print("资金费记录数", manifest["funding_rows"], "请求地址数", manifest["addresses_requested"])
    print("逐仓示例资金费", checks["1"])
    print("全天持仓覆盖", checks["2"]["(ii)"])
    print("无成交日仓位核对", checks["5"])
    print("通过映射的成交金额占比", checks["3"]["accepted_notional_share"])
    print("币种映射状态", pd.read_csv(required[4])["status"].value_counts().to_dict())
    print("仅资金费可见的持仓", manifest["hidden_positions"])
    funding = pd.read_parquet(required[3])
    hidden = pd.read_parquet(required[5])
    print("读取资金费记录、隐藏持仓记录", len(funding), len(hidden))
    full_fills = pd.read_csv(required[1])
    full_fills["ts"] = pd.to_datetime(full_fills.timestamp, format="mixed", utc=True).dt.tz_localize(None)
    perp = full_fills[(full_fills.ts < pd.Timestamp("2026-01-05")) &
                      ~full_fills["dir"].isin(["Buy", "Sell", "Spot Dust Conversion"])].copy()
    symbols = pd.read_csv(required[4]).query("status == 'accepted'").package_symbol
    matches = []
    for symbol in symbols:
        bars = pd.read_parquet(BUILD / "klines_5m" / f"{symbol}.parquet",
                               columns=["open_time", "low", "high"]).set_index("open_time")
        trades = perp.loc[perp.symbol == symbol, ["ts", "price", "notional"]].copy()
        trades["bar"] = trades.ts.dt.floor("5min")
        matches.append(trades.join(bars, on="bar"))
    matched = pd.concat(matches, ignore_index=True)
    no_bar = matched.low.isna()
    observed = matched[~no_bar].copy()
    distance = np.where(observed.price > observed.high, observed.price / observed.high - 1,
                        np.where(observed.price < observed.low, 1 - observed.price / observed.low, 0))
    large = distance > 0.005
    weights = observed.notional
    print("\n1.5 成交价与同根五分钟最高最低价")
    print("有行情的成交数", len(observed), "缺行情成交数", int(no_bar.sum()),
          "缺行情成交金额占通过映射币种的比例", matched.loc[no_bar, "notional"].sum() / matched.notional.sum())
    print("区间内成交金额比例", weights[distance == 0].sum() / weights.sum())
    print("超出区间超过50个基点的成交金额比例", weights[large].sum() / weights.sum())
    print("其中2025-10-10占比", weights[large & (observed.ts.dt.date == pd.Timestamp("2025-10-10").date())].sum()
          / weights[large].sum())

    window = funding[funding.time >= pd.Timestamp("2025-07-27")]
    totals = window.groupby("trader").usdc.sum()
    fees = perp.groupby("trader").fee.sum()
    ratios = totals.abs() / fees.reindex(totals.index).abs().replace(0, np.nan)
    active_days = perp.assign(day=perp.ts.dt.floor("D")).groupby("trader").day.nunique()
    orders = perp.groupby(["trader", "symbol", "oid"]).size().groupby(level=0).size()
    enough_ids = set(active_days.index[(active_days >= 10) & (orders.reindex(active_days.index) >= 30)])
    print("\n1.2–1.3 样本足够的地址数", len(enough_ids))
    positions = pd.read_parquet(required[7])
    positions = positions[positions.trader.isin(enough_ids)].copy()
    positions["aexp"] = positions.exposure.abs().where(positions.exposure.abs() >= 1e-6, 0.0)
    positions["h3"] = np.where(positions.symbol.str.contains(":", regex=False), positions.aexp, 0.0)
    positions["cc"] = positions.value - positions.exposure
    snap_positions = positions.groupby(["trader", "ts"]).agg(
        gross=("aexp", "sum"), h3=("h3", "sum"), cc=("cc", "sum"), fee=("cfee", "sum")).reset_index()
    snap = pd.read_csv(required[0])
    snap["ts"] = pd.to_datetime(snap.timestamp, format="mixed", utc=True).dt.tz_localize(None)
    snap["iso_gap"] = snap.pnl + snap.net_deposit - snap.equity
    merged = snap_positions.merge(snap[["trader", "ts", "pnl", "net_deposit", "equity", "iso_gap"]],
                                  on=["trader", "ts"]).sort_values(["trader", "ts"])
    first_any = merged.groupby("trader").head(1).set_index("trader").ts
    clean = merged[merged.h3 == 0].groupby("trader").head(1).set_index("trader").ts
    anchor = first_any.copy()
    anchor.loc[clean.index] = clean
    grouped = merged.groupby("trader")
    for col in ["ts", "net_deposit", "equity", "cc", "fee"]:
        merged["d_" + col] = grouped[col].diff()
    merged["p_gross"] = grouped.gross.shift()
    merged["p_h3"] = grouped.h3.shift()
    valid = merged[(merged.ts > merged.trader.map(anchor)) &
                   (merged.d_ts.dt.total_seconds() / 86400 <= 1.5) &
                   (merged.h3 == 0) & (merged.p_h3 == 0)].copy()
    flat = valid[(valid.gross == 0) & (valid.p_gross == 0) &
                 (valid.d_cc.abs() <= 1e-6) & (valid.d_fee.abs() <= 1e-9) &
                 (valid.d_net_deposit.abs() > 1)]
    print("无仓位无成交时净存款变动超过1美元", len(flat),
          "其中权益变化与净存款变化比的绝对值不超过2%", int((flat.d_equity.abs() / flat.d_net_deposit.abs() <= .02).sum()))
    print("其中前后两张快照的权益数值完全相同", int(flat.d_equity.eq(0).sum()))
    print("无 HIP-3 持仓的已观察快照中逐仓差额大于1美元的地址数",
          valid.loc[valid.iso_gap > 1, "trader"].nunique())

    print("\n1.5 按地址汇总资金费与手续费")
    print("净付出地址", int((totals < 0).sum()), "净收到地址", int((totals > 0).sum()))
    print("净资金费绝对值超过手续费的比例", (ratios > 1).mean(),
          "样本足够地址的比例", (ratios[ratios.index.isin(enough_ids)] > 1).mean())

    groups = pd.read_parquet(required[6], columns=["trader", "symbol", "ts", "gstart"])
    groups = groups.sort_values(["trader", "symbol", "ts"])
    first = groups.groupby(["trader", "symbol"]).agg(first_ts=("ts", "first"),
                                                    first_gstart=("gstart", "first")).reset_index()
    daily = window.rename(columns={"coin": "symbol"}).copy()
    daily["day"] = daily.time.dt.floor("D")
    daily = daily.sort_values("time").groupby(["trader", "symbol", "day"]).agg(szi=("szi", "last")).reset_index()
    daily = daily.merge(first, on=["trader", "symbol"])
    pre = daily[(daily.day < daily.first_ts.dt.floor("D")) & (daily.day >= pd.Timestamp("2025-07-28"))].copy()
    tolerance = 1e-6 * np.maximum(pre.szi.abs(), pre.first_gstart.abs()) + 1e-9
    mismatch = pre[(pre.szi - pre.first_gstart).abs() > tolerance]
    print("首次成交前资金费仓位与首次成交时仓位不同", len(mismatch), "日，",
          mismatch.groupby(["trader", "symbol"]).ngroups, "地址币种对，", mismatch.trader.nunique(), "地址")
    print("\n构建脚本的采样价格偏差检查请见 manifest.json 的 checks['4']。")
    print("长期资金费粗估和未由以上检查生成的描述需参照原始研究记录。")


if __name__ == "__main__":
    main()
