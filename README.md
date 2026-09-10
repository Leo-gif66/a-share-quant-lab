# A-Share Quant Lab v0.1

这是一个从第一天就按“可持续训练/可持续扩展”设计的个人 A 股量化研究平台。

## 现在已有

- AKShare 数据 Provider 接口，可继续新增 Tushare/券商/自有数据
- Parquet 数据层 + DuckDB 查询入口
- Factor Registry：新增因子不改核心框架
- Model Registry：Rule / LightGBM 共用统一接口
- 可配置 Label：默认未来10日横截面超额收益
- 时间切分 train/valid/test
- RankIC 训练评估
- MLflow 实验记录
- 周频 Top-N 回测 + 大盘 MA 风控 + 交易成本
- CLI 命令行入口
- 不联网的 synthetic demo，用于先验证环境
- 可选 Docker MLflow Server

## 最快部署（Windows）

建议 Python 3.12 + uv。

```powershell
# 进入项目目录后
uv sync
uv run quant demo
```

看到 `Core pipeline works.` 就说明核心框架安装成功。

然后开始真实 A 股数据：

```powershell
uv run quant data-update
uv run quant features-build
uv run quant backtest
uv run quant signal
```

完整股票池时再提高 limit；第一轮不要直接全量下载。

## 切到 LightGBM

先构建训练数据，然后：

```powershell
uv run quant train --override configs/models/lightgbm.yaml
uv run quant backtest --override configs/models/lightgbm.yaml --model-path artifacts/lightgbm_latest.joblib
uv run quant signal --override configs/models/lightgbm.yaml --model-path artifacts/lightgbm_latest.joblib
```

每次训练会保存模型，并尝试写入本地 `mlruns/`。

## 自定义训练

不要改核心代码。复制一个 YAML：

```text
configs/strategies/my_v2.yaml
```

只覆盖你想变的部分，例如：

```yaml
label:
  horizon: 5

portfolio:
  top_n: 15
  max_single_weight: 0.08

model:
  name: lightgbm
  params:
    learning_rate: 0.02
    n_estimators: 700
```

然后：

```powershell
uv run quant train --override configs/strategies/my_v2.yaml
```

## 新增自己的因子

在 `src/quant/factors/` 新建模块，实现 `Factor.calculate()`，并通过 registry 注册。
之后把因子名加入 YAML 的 `features.enabled`。

## 目录

```text
configs/              策略、模型与训练配置
src/quant/data/       数据源 + 存储
src/quant/factors/    因子注册系统
src/quant/models/     模型注册系统
src/quant/features.py 特征与标签
src/quant/training.py 训练 + MLflow
src/quant/backtest.py 回测
artifacts/            已训练模型
data/                 本地 Parquet 数据
reports/              回测报告
mlruns/               本地实验历史
```

## 当前 v0.1 的已知限制

1. 默认股票池用“当前”沪深300+中证500成份股历史行情，存在 survivorship bias；不能把初版回测收益直接当作可实现收益。
2. 回测成交模型仍是日线简化版，下一版应补历史成份股、涨跌停、停牌、T+1、次日开盘成交等。
3. Rule 模型不是机器学习；它用于验证完整工程管线。LightGBM 才进入可训练模型阶段。
4. 当前不连接券商实盘。先完成研究/模拟盘验证。

## 推荐升级顺序

v0.2：历史股票池 + 严格交易规则 + 年度/滚动回测
v0.3：因子诊断（IC/RankIC/分层收益/相关性/衰减）
v0.4：Walk-forward LightGBM + 超参搜索
v0.5：行业/市值中性化 + 组合优化
v0.6：Paper Trading
v1.0：稳定后再考虑券商接口

## v0.1 中已经为后续定制预留的机制

- `quant doctor`：查看当前自动发现的因子和模型。
- 因子自动发现：把新的因子模块放入 `src/quant/factors/`，无需修改中央列表。
- 模型自动发现：把新的模型模块放入 `src/quant/models/` 并注册，核心训练/回测入口不变。
- 模型版本化：每次训练写入 `artifacts/<model>/<model_id>/`，不会覆盖旧模型；`artifacts/latest.json` 记录每类模型的最新版本。
- Purged time split：训练集/验证集边界按 label horizon 留出净空，减少未来收益标签跨分割边界造成的数据泄漏。
- `templates/factor_template.py`：个人因子模板。

第一次运行 `uv sync` 时会自动生成并维护 `uv.lock`。项目包内未预生成 lock 文件，是因为生成环境无法访问 PyPI；在你的联网电脑上执行一次 `bootstrap.ps1` 或 `uv sync` 即可锁定依赖。
