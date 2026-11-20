# 回测复现

把代码包、外部原始数据包和题目数据包解压到同一目录。完整模式需要 Python 3.11，使用 `python -m pip install -r reproduce/requirements.txt` 安装依赖。通过这些输入离线重建外部数据、成交回放、滚动账本、选址和回测。

    python reproduce/run.py --mode full --output /path/to/new-output

快速模式仅从打包的中间数据重新执行九种方案，不需要另外两个数据包。

    python reproduce/run.py --mode quick --output /path/to/new-output

输出目录必须不存在。脚本会复制输入，并在输出目录中适配原研究脚本的绝对路径。各步骤失败会中止。输出 work/select/simulation_paths.csv 包含每条路径的组合结果，simulation_groups.parquet 包含每周分组，simulation_daily_first_path.csv 包含每种方案第一条路径逐日记录。终端给出九种方案摘要。主线方案为 4，参考值是 98 天收益 +23.04%，最大回撤 2.43%，年化夏普比率 5.69。

验证环境为 macOS、Python 3.11.15 和 requirements.txt 中的依赖版本。输出目录的路径可以包含空格。各步骤日志写在输出目录的 logs 中。

完整模式依次运行成交回放、外部数据构建、报告数据检查、滚动账本、选址、模拟及审计和方向诊断。各原始脚本的控制台输出可重定向保存。报告中全部具体数据检查的复现覆盖范围见 report_data.py 的输出。

输出 summary_portfolios.csv 和 summary_groups.csv 对应组合表现和分组表现，summary_funnel.csv 是筛选漏斗，summary_main_weekly.csv 列出主线每周读数，summary_matched_funding.csv 是资金匹配对照。分组地址名单保存在 simulation_groups.parquet。

### 最终报告第 1 节的数据来源

完整运行的 `logs/report_data.log` 列出报告所举四个地址示例的快照和成交，以及资金费、持仓、行情匹配等核对结果。`reproduce/report_data_reference.txt` 是随包的参考输出。第 1.2 节所述的 70 次是权益变化幅度不超过净存款变化幅度的 2%，其中有 60 次前后权益数值完全相同。第 1.1 节长期多仓资金费约为账户金额的 6% 是量级估算，回测使用每个地址的历史资金费流水。

第 1.5 节的 19.9% 和 40.4% 来自补充外部行情前的成交价格覆盖检查。随包的 [历史覆盖表](../work/codex6/horizon_sparse/coverage_overall.csv) 中，`population=eligible` 的 1 小时和 4 小时两行，`scored_coverage` 分别为 19.87% 和 40.41%。分母为合格地址纳入检查的开仓金额，分子还要求有波动估计和至少五个可对比时点。这项检查独立于九种选址方案。若需重建该历史表，在完整模式的**输出目录**中，用同一虚拟环境依次运行 `work/profile/profile.py`、`work/codex6/pilot.py` 和 `work/codex6/sparse_pilot.py`。
