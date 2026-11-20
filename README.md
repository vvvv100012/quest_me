# ArkStream 选址回测复现

[下载本次提交的数据和复现包](https://github.com/vvvv100012/arkstream_quest/releases/tag/backtest-v1)

[最终报告](final_report.md)给出数据核对、选址规则和主线结果。[每周选址规则与历史检验](selection.md)列出九种方案的完整对照。

| 文件 | 用途 |
|---|---|
| `code-and-reports.zip` | 两份报告、研究代码和冻结的中间输入。只下载这个包，就能重新模拟九种方案。 |
| `external-data.zip` | 完整重建使用的外部原始资金费和行情记录，保留抓取时间与行情文件校验信息。 |
| `original-data.zip` | 题方提供的成交记录和账户快照。已有相同版本题包的读者可以直接使用原文件。 |
| `SHA256SUMS` | 三个附件的 SHA-256 校验和，可用来检查下载文件是否完整。 |

## 快速复算

将代码包解压，在包含 `selection.md` 的目录中建立 Python 3.11 虚拟环境，安装复算依赖。安装依赖后，研究脚本可以离线运行。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r reproduce/requirements.txt
.venv/bin/python reproduce/run.py --mode quick --output ../arkstream-quick
```

快速模式从按历史时点准备好的地址特征、标签、敞口和每日盈亏开始，重新生成各周的名单、权重和收益。主线是方案 4。98 天累计收益应为 +23.04%，最大回撤为 2.43%，年化夏普比率为 5.69。

## 从原始数据重建

将三个附件解压至同一目录，使两份报告、`reproduce/`、`work/` 和 `ArkStream笔试题_v2/` 并列。例如，在保存附件的目录执行以下命令。

```sh
unzip code-and-reports.zip -d replay
unzip external-data.zip -d replay
unzip original-data.zip -d replay
cd replay
python3.11 -m venv .venv
.venv/bin/python -m pip install -r reproduce/requirements.txt
.venv/bin/python reproduce/run.py --mode full --output ../arkstream-full
```

每次运行使用一个新的输出目录。逐路径结果在输出目录的 `work/select/simulation_paths.csv`，每周名单与跟单比例在 `work/select/simulation_groups.parquet`。各阶段输入、运行顺序和产物见[复现说明](reproduce/README.md)。

## 读数核对

最终报告第 1.2 节的 313 次记录指空仓、无成交且净存款变化超过 1 美元的区间。其中 70 次权益变化幅度不超过净存款变化幅度的 2%，严格比较前后权益数值，完全相同的有 60 次。复核该句时以这两个口径为准。原始计算输出见[数据核对参考记录](reproduce/report_data_reference.txt)。
