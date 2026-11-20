"""Offline reconstruction of the address selection study."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OLD_ROOT = "/Users/xc/Documents/arkstream_quest"
COMMON = [
    "最终报告.md", "最终报告附录.md",
    "work/select/run_simulation.py", "reproduce/summary.py",
]
SNAPSHOTS = [
    "work/select/sel_panel2.parquet", "work/select/simulation_exposure.npz",
    "work/select/simulation_outcomes.npz", "work/select/simulation_labels.parquet",
    "work/select/simulation_volatility.parquet",
]
FULL = [
    "ArkStream笔试题_v2/笔试题数据包-fills",
    "ArkStream笔试题_v2/笔试题数据包-traders.csv",
    "work/extdata/raw", "work/extdata/code", "work/extdata/logs",
    "work/v2dev/code", "work/profile/profile.py", "work/profile/direction.py",
    "work/profile/direction_balance.py", "work/profile/direction_wf.py",
    "work/profile2/code/exposure.py", "work/style/code", "work/roll/code",
    "work/codex6/pilot.py", "work/codex6/sparse_pilot.py",
    "work/codex10/REVIEW.md", "reproduce/report_data.py", "work/select/sel_diag.py",
    "work/select/sel_diag2.py", "work/select/sel_diag3.py",
    "work/select/sel_diag4.py", "work/select/sel_diag5.py",
    "work/select/direction_wf60.py", "work/select/prepare_simulation.py",
    "work/select/audit_simulation.py",
]


def copy_inputs(names, target):
    for name in names:
        src = ROOT / name
        dst = target / name
        if not src.exists():
            raise FileNotFoundError(f"Required input missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(src, dst)
    for path in target.rglob("*.py"):
        content = path.read_text()
        if OLD_ROOT in content:
            path.write_text(content.replace(OLD_ROOT, target.as_posix()))


def run_script(root, rel):
    print(f"\n>>> {rel}", flush=True)
    path = root / rel
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARK_")}
    env.update(ARK_DATA_DIR=str(root / "ArkStream笔试题_v2"), PYTHONDONTWRITEBYTECODE="1")
    log = root / "logs" / (path.stem + ".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as output:
        result = subprocess.run([sys.executable, "-W", "ignore", str(path)], cwd=path.parent,
                                stdout=output, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        print(log.read_text()[-4000:], file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)
    print(f"日志 {log}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    parser.add_argument("--output", required=True, help="new directory for generated files")
    args = parser.parse_args()
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        raise SystemExit(f"Output directory already exists: {target}")
    target.mkdir(parents=True)
    copy_inputs(COMMON + (SNAPSHOTS if args.mode == "quick" else FULL), target)
    for rel in ("work/v2dev/build", "work/v2dev/tables", "work/roll/build",
                "work/roll/logs", "work/extdata/build", "work/profile/build_formation"):
        (target / rel).mkdir(parents=True, exist_ok=True)
    if args.mode == "full":
        steps = ["work/v2dev/code/02_replay.py", "work/extdata/code/build_ext.py",
                 "reproduce/report_data.py", "work/roll/code/ledger.py", "work/roll/code/excursion.py", "work/roll/code/returns.py",
                 "work/select/sel_diag.py", "work/select/sel_diag2.py",
                 "work/select/prepare_simulation.py", "work/select/run_simulation.py",
                 "work/select/audit_simulation.py", "work/select/direction_wf60.py",
                 "work/select/sel_diag3.py", "work/select/sel_diag4.py", "work/select/sel_diag5.py",
                 "work/profile/direction_wf.py", "reproduce/summary.py"]
    else:
        steps = ["work/select/run_simulation.py", "reproduce/summary.py"]
    for script in steps:
        run_script(target, script)
    import pandas as pd
    paths = pd.read_csv(target / "work/select/simulation_paths.csv")
    groups = pd.read_parquet(target / "work/select/simulation_groups.parquet")
    summary = paths.groupby("scenario")[["portfolio_return", "full_period_maximum_drawdown",
                                        "full_period_annualized_sharpe", "mean_funded_fraction",
                                        "manual_weeks"]].median()
    summary["group_sharpe_median"] = (groups.groupby(["scenario", "draw"])["annualized_sharpe_28d"]
                                      .median().groupby("scenario").median())
    print("\nScenario medians:")
    print(summary.to_string())
    print(f"\nResults: {target / 'work/select'}")


if __name__ == "__main__":
    main()
