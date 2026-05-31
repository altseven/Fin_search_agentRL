
# 带搜索工具的股票预测 Agent RL 项目设计文档（veRL 实验版）

> 目标：构建一个带 function call 搜索工具的金融预测 agent，用 agent RL / RLVR 方法提升股票未来相对收益方向、概率校准和横截面排序能力。  
> 推荐基座模型：Qwen3-4B-Instruct 或同级别 4B instruct 模型。  
> 推荐 RL 框架：veRL，优先 GRPO，后续可扩展 PPO / multi-turn tool RL。  
> 核心思想：不是把所有新闻公告预先拼给模型做固定 RAG，而是构建一个 point-in-time 历史可回放搜索环境，让 agent 在有限搜索预算下主动调用工具、获取 observation、形成预测，再用未来真实收益自动计算 verifiable reward。

---

## 0. 一句话概括

本项目要把“股票涨跌预测”改造成一个可验证的 agentic RL 任务：

```text
给定股票 i、历史预测时点 t、预测窗口 h，
agent 只能看到 t 时点之前的信息，并可以通过 function call 主动搜索。
agent 输出未来 h 日 up / neutral / down 的概率。
训练系统用 t+h 后真实相对收益计算 reward。
RL 优化的是整条“搜索 → 证据整合 → 概率预测”轨迹。
```

最终希望证明：

```text
RL 前 agent reward < RL 后 agent reward
RL 后验证集 reward 提升
RL 后样本外预测指标提升
RL 后横截面 rank IC / 分组收益 / 回测指标提升
```

---

## 1. 项目边界与核心定义

### 1.1 这个项目不是直接预测股价点位

不建议把目标定义成：

```text
预测某只股票 5 天后的收盘价是多少元
```

原因：

1. 股价 level 不同股票不可比。
2. 股价受复权、分红、拆股影响。
3. 价格水平非平稳。
4. 真实投资更关心收益、方向、排序和可交易信号，而不是价格点位。

更合理的目标是：

```text
预测未来 T+5 / T+10 的相对收益方向、概率和横截面排序。
```

例如：

```text
future_relative_return = stock_return_{t,t+h} - industry_return_{t,t+h}
```

或者：

```text
future_relative_return = stock_return_{t,t+h} - market_index_return_{t,t+h}
```

### 1.2 什么叫“预测能力提升”

本项目中的“预测能力”建议定义为：

```text
在无未来信息泄漏的前提下，
agent 对未来 h 日相对收益方向、概率和横截面排序的样本外预测能力提升。
```

具体体现在：

| 能力层次 | 输出 | 指标 |
|---|---|---|
| 方向预测能力 | up / neutral / down | Accuracy、Balanced Accuracy、Macro-F1 |
| 概率预测能力 | p_up、p_neutral、p_down | Log Loss、Brier Score、ECE |
| 横截面排序能力 | alpha_score = p_up - p_down | IC、Rank IC、ICIR |
| 可交易能力 | top/bottom portfolio | Top-bottom return、Sharpe、Max Drawdown、Turnover |
| Agent 能力 | tool call trajectory | 平均工具调用次数、无效调用率、搜索成本、泄漏率 |

### 1.3 什么叫“训练前 reward”

“训练前 reward”不是一个特殊模型，而是指：

```text
在 agent RL 更新之前，用当前 agent 在一批 ticker-date-horizon 样本上 rollout，
再用同一个 verifier 计算出的平均 reward。
```

常见 baseline：

```text
Base-0: Qwen3-4B-Instruct zero-shot + function call prompt
Base-1: Qwen3-4B-Instruct + SFT 后 agent
Base-2: 固定搜索流程 RAG baseline
RL:     SFT agent + GRPO / PPO
```

推荐报告格式：

| 模型 | 搜索方式 | 是否 RL | Valid Reward | Macro-F1 | Brier↓ | Rank IC | Avg Tool Calls |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B zero-shot | agent function call | 否 | 0.12 | 0.31 | 0.68 | 0.006 | 3.8 |
| Qwen3-4B SFT | agent function call | 否 | 0.21 | 0.36 | 0.61 | 0.014 | 3.5 |
| Qwen3-4B SFT + GRPO | agent function call | 是 | 0.32 | 0.41 | 0.55 | 0.027 | 3.1 |
| 固定 RAG workflow | 固定查资料 | 否 | 0.24 | 0.38 | 0.60 | 0.017 | 4.0 |

---

## 2. 总体系统架构

### 2.1 整体流程

```text
原始金融数据
  ├── 行情 OHLCV
  ├── 复权因子
  ├── 行业指数
  ├── 公告
  ├── 新闻
  ├── 财报 / 基本面
  └── 宏观 / 行业数据

      ↓ point-in-time 处理

历史可回放搜索环境
  ├── price_factor_db
  ├── announcement_db
  ├── news_db
  ├── fundamental_db
  ├── industry_db
  └── peer_db

      ↓ 构造任务样本

ticker-date-horizon 样本
  ├── agent 可见字段：ticker、as_of、horizon、tools、search_budget
  └── verifier 隐藏字段：future_return、future_relative_return、true_label

      ↓ rollout

agent function-call trajectory
  ├── action: tool call
  ├── observation: tool result
  ├── action: tool call
  ├── observation: tool result
  └── final_answer: p_up / p_neutral / p_down

      ↓ verifier

reward
  ├── direction reward
  ├── probability reward
  ├── brier reward
  ├── pnl reward
  ├── search cost
  ├── format reward
  └── leakage penalty

      ↓ veRL

GRPO / PPO 更新 agent policy

      ↓ evaluation

验证集 / 测试集
  ├── reward
  ├── classification metrics
  ├── calibration metrics
  ├── Rank IC
  └── backtest metrics
```

### 2.2 关键原则

1. **数据可以提前建库，但检索动作不能提前替 agent 决定。**
2. **每条文档必须有 publication_time / effective_time，所有工具必须 as-of 过滤。**
3. **模型可见字段和 verifier 隐藏字段必须物理隔离。**
4. **训练 reward 上升不够，必须验证集 reward 和真实预测指标同步提升。**
5. **第一版不要过复杂，先用 T+5/T+10 相对收益方向作为主任务。**

---

## 3. 数据构造总览

本项目至少需要构造 5 类数据：

| 数据类型 | 用途 | 是否模型可见 |
|---|---|---|
| 任务样本数据 | 告诉 agent 要预测哪只股票、哪个日期、哪个周期 | 是 |
| 历史搜索环境 | 给 function call 返回 observation | 是，但只能通过工具动态访问 |
| 隐藏标签数据 | 给 verifier 计算 reward | 否 |
| Agent trajectory 数据 | 记录 rollout 过程 | 训练过程中生成 |
| Reward 结果数据 | 每条 trajectory 的打分 | 训练/评估使用，模型不直接看 |

---

## 4. 原始数据表设计

下面是推荐的最小数据表设计。第一版可以用 Parquet + DuckDB / SQLite 实现，后续再换成更复杂的检索服务。

### 4.1 交易日历表 `trading_calendar.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | string, YYYY-MM-DD | 交易日 |
| market | string | A-share / US / HK 等 |
| is_open | int | 是否开市 |
| next_trade_date | string | 下一个交易日 |
| prev_trade_date | string | 上一个交易日 |

示例：

```json
{
  "trade_date": "2024-06-28",
  "market": "A-share",
  "is_open": 1,
  "next_trade_date": "2024-07-01",
  "prev_trade_date": "2024-06-27"
}
```

### 4.2 股票池表 `universe.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| ticker | string | 股票代码 |
| company_name | string | 公司名 |
| market | string | 市场 |
| listing_date | string | 上市日期 |
| delisting_date | string/null | 退市日期 |
| industry_level1 | string | 一级行业 |
| industry_level2 | string | 二级行业 |
| is_st | int | 是否 ST，A 股可用 |
| is_tradeable | int | 当前是否可交易 |
| effective_start_date | string | 成分生效起始日期 |
| effective_end_date | string | 成分生效结束日期 |

注意：

- 股票池必须用历史成分，不要只用当前仍在指数里的股票。
- 退市股票不能直接删掉，否则会有 survivorship bias。
- ST、停牌、涨跌停样本要么过滤，要么单独标记。

### 4.3 日行情表 `stock_daily.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| ticker | string | 股票代码 |
| trade_date | string | 交易日 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| amount | float | 成交额 |
| adj_factor | float | 复权因子 |
| adj_close | float | 复权收盘价 |
| pct_change | float | 当日收益 |
| turnover | float | 换手率 |
| limit_up | int | 是否涨停 |
| limit_down | int | 是否跌停 |
| suspended | int | 是否停牌 |

示例：

```json
{
  "ticker": "688888.SH",
  "trade_date": "2024-06-28",
  "close": 37.21,
  "adj_close": 41.83,
  "pct_change": 0.018,
  "turnover": 0.047,
  "limit_up": 0,
  "limit_down": 0,
  "suspended": 0
}
```

### 4.4 行业行情表 `industry_daily.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| industry_id | string | 行业代码 |
| industry_name | string | 行业名 |
| trade_date | string | 交易日 |
| index_close | float | 行业指数收盘 |
| index_return | float | 行业当日收益 |
| valuation_pe_ttm | float | PE TTM |
| valuation_pb | float | PB |
| turnover | float | 行业成交活跃度 |

### 4.5 公告表 `announcements.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| doc_id | string | 文档唯一 ID |
| ticker | string | 股票代码 |
| company_name | string | 公司名 |
| title | string | 标题 |
| content | string | 正文 |
| summary | string | 可预先生成的摘要 |
| publish_time | string, ISO | 公告发布时间 |
| effective_time | string, ISO | 对交易可见的时间 |
| source | string | 来源 |
| doc_type | string | 年报/季报/合同/减持/诉讼等 |
| importance_score | float | 可选，规则或模型打分 |
| embedding | vector/null | 可选，用于向量检索 |

`effective_time` 很重要。例如 A 股公告晚上 20:00 发布，对当天收盘后的预测可见，但对当天盘中不可见。第一版可以简化为：

```text
如果 publish_time <= as_of，则可见。
```

更严谨的版本：

```text
如果 publish_time 在收盘后，则下一交易日开盘前才可交易。
```

### 4.6 新闻表 `news.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| news_id | string | 新闻 ID |
| tickers | list[string] | 关联股票 |
| title | string | 标题 |
| content | string | 正文 |
| summary | string | 摘要 |
| publish_time | string | 发布时间 |
| source | string | 媒体来源 |
| category | string | 公司/行业/宏观/政策 |
| sentiment_score | float/null | 预处理情绪分，可选 |
| relevance_score | float/null | 相关性分，可选 |
| embedding | vector/null | 可选 |

注意：

- 新闻要去重，尤其转载新闻。
- 不要让未来新闻进入历史搜索。
- 新闻情绪分如果由大模型生成，生成时也不能看未来收益。

### 4.7 基本面快照表 `fundamental_snapshot.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| ticker | string | 股票代码 |
| report_period | string | 报告期 |
| publish_time | string | 财报发布时间 |
| as_of_date | string | 可见日期 |
| revenue_ttm | float | TTM 收入 |
| net_profit_ttm | float | TTM 净利润 |
| revenue_yoy | float | 收入同比 |
| net_profit_yoy | float | 净利润同比 |
| gross_margin | float | 毛利率 |
| roe | float | ROE |
| debt_ratio | float | 资产负债率 |
| pe_ttm | float | PE TTM |
| pb | float | PB |

关键点：

```text
财报数据按 publish_time 可见，而不是按 report_period 可见。
```

例如 2024Q1 财报可能 2024-04-29 发布，不能在 2024-03-31 的预测样本中使用。

### 4.8 预计算因子表 `factor_snapshot.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| ticker | string | 股票代码 |
| as_of_date | string | 截止日期 |
| ret_1d | float | 过去 1 日收益 |
| ret_5d | float | 过去 5 日收益 |
| ret_20d | float | 过去 20 日收益 |
| vol_20d | float | 20 日波动率 |
| turnover_5d_avg | float | 5 日平均换手 |
| turnover_20d_avg | float | 20 日平均换手 |
| amount_20d_avg | float | 20 日平均成交额 |
| max_drawdown_20d | float | 20 日最大回撤 |
| rs_industry_20d | float | 相对行业 20 日收益 |
| rs_market_20d | float | 相对市场 20 日收益 |
| beta_60d | float | 60 日 beta |
| momentum_rank_in_industry | float | 行业内动量分位 |
| liquidity_rank | float | 流动性分位 |

这些因子必须只用 `as_of_date` 当天及之前的数据计算。

---

## 5. 任务样本构造

### 5.1 样本粒度

推荐样本粒度：

```text
一个样本 = 一只股票 + 一个预测时点 + 一个预测窗口
```

例如：

```text
ticker = 688888.SH
as_of = 2024-06-28 15:00:00
horizon = 5 trading days
```

### 5.2 预测时点选择

推荐第一版：

```text
每个交易日收盘后预测
as_of = trade_date 15:00:00
```

这样避免盘中数据和公告可见性问题。

### 5.3 预测窗口

推荐：

```text
horizon = 5 trading days
horizon = 10 trading days
```

不建议第一版做 T+1，因为噪声太大。

### 5.4 标签构造

计算股票未来收益：

```text
stock_return_{i,t,h} = adj_close_{i,t+h} / adj_close_{i,t} - 1
```

计算行业未来收益：

```text
industry_return_{ind(i),t,h} = industry_index_close_{ind,t+h} / industry_index_close_{ind,t} - 1
```

计算相对收益：

```text
future_relative_return_{i,t,h}
= stock_return_{i,t,h} - industry_return_{ind(i),t,h}
```

也可以使用市场相对收益：

```text
future_relative_return_{i,t,h}
= stock_return_{i,t,h} - market_return_{t,h}
```

### 5.5 三分类标签方案

推荐两种方案。

#### 方案 A：固定阈值

```text
if future_relative_return > +0.01:
    label = "up"
elif future_relative_return < -0.01:
    label = "down"
else:
    label = "neutral"
```

优点：简单。  
缺点：不同市场阶段类别分布不稳定。

#### 方案 B：按交易日横截面分位数

同一天所有股票按 future_relative_return 排序：

```text
top 30%       => up
middle 40%    => neutral
bottom 30%    => down
```

优点：类别均衡，更适合横截面选股。  
缺点：label 是相对概念，不是绝对涨跌。

第一版建议用方案 B。

### 5.6 任务样本表 `tasks.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| sample_id | string | 唯一 ID |
| ticker | string | 股票代码 |
| company_name | string | 公司名 |
| market | string | 市场 |
| industry_id | string | 行业 ID |
| industry_name | string | 行业名 |
| as_of | string | 预测时点 |
| trade_date | string | 预测基准交易日 |
| horizon | int | 预测窗口 |
| tools | list[string] | 可用工具 |
| max_tool_calls | int | 最大工具调用次数 |
| prompt | string | 给模型的任务 prompt |
| split | string | train / valid / test |

示例：

```json
{
  "sample_id": "688888.SH_2024-06-28_T5",
  "ticker": "688888.SH",
  "company_name": "海川机器人",
  "market": "A-share",
  "industry_id": "CI005001",
  "industry_name": "机器人设备",
  "as_of": "2024-06-28 15:00:00",
  "trade_date": "2024-06-28",
  "horizon": 5,
  "tools": [
    "get_price_factors",
    "search_announcements",
    "search_news",
    "get_industry_context"
  ],
  "max_tool_calls": 4,
  "split": "train"
}
```

### 5.7 隐藏标签表 `labels.parquet`

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| sample_id | string | 唯一 ID |
| ticker | string | 股票代码 |
| trade_date | string | 预测日期 |
| horizon | int | 预测窗口 |
| stock_return | float | 未来股票收益 |
| industry_return | float | 未来行业收益 |
| market_return | float | 未来市场收益 |
| future_relative_return | float | 未来相对收益 |
| future_volatility | float | 未来或历史波动率，用于归一化 |
| label | string | up / neutral / down |
| label_id | int | up=0, neutral=1, down=2 或自定义 |
| cross_section_rank | float | 当日横截面收益分位 |
| tradable_at_t_plus_1 | int | 是否可交易 |
| has_limit_issue | int | 是否涨跌停影响 |

示例：

```json
{
  "sample_id": "688888.SH_2024-06-28_T5",
  "ticker": "688888.SH",
  "trade_date": "2024-06-28",
  "horizon": 5,
  "stock_return": 0.041,
  "industry_return": 0.016,
  "market_return": 0.008,
  "future_relative_return": 0.025,
  "label": "up",
  "label_id": 0,
  "cross_section_rank": 0.86
}
```

重要：`labels.parquet` 绝对不能被拼进模型 prompt，只能给 reward/verifier 使用。

---

## 6. 数据切分

金融任务必须按时间切分，不要随机切分。

推荐：

```text
train: 2018-01-01 ~ 2022-12-31
valid: 2023-01-01 ~ 2023-12-31
test : 2024-01-01 ~ 2025-12-31
```

如果样本不够，可以做 rolling split：

```text
fold 1:
  train 2018-2021
  valid 2022
  test  2023

fold 2:
  train 2019-2022
  valid 2023
  test  2024
```

不要让同一个日期的未来收益同时出现在训练和测试里。

---

## 7. Point-in-time 搜索环境

### 7.1 核心原则

所有工具都必须满足：

```text
只返回 publish_time / effective_time <= as_of 的数据。
```

任何工具不得返回未来数据。

例如：

```python
def filter_as_of(df, as_of):
    return df[df["effective_time"] <= as_of]
```

### 7.2 检索后处理

每个搜索工具都需要：

1. `as_of` 时间过滤。
2. ticker / industry / keyword 过滤。
3. BM25 / embedding / 规则相关性排序。
4. 去重。
5. 截断到 top_k。
6. 返回结构化 observation。

Observation 不要返回太长，第一版每条文档摘要 100~200 中文字即可。

### 7.3 检索环境实现建议

第一版可用：

```text
Parquet + DuckDB / Polars
BM25: rank_bm25 / Elasticsearch / OpenSearch
Embedding: FAISS / Milvus / LanceDB
```

为了方便实验，第一版可以不用向量库，先用：

```text
ticker 精确匹配 + 时间过滤 + doc_type / keyword 简单过滤 + 时间倒序
```

后面再加 BM25 / embedding。

---

## 8. Function Call 工具设计

### 8.1 工具总览

第一版建议只做 5 个工具：

| 工具 | 作用 | 第一版必要性 |
|---|---|---|
| get_price_factors | 查量价和因子 | 必须 |
| search_announcements | 查公告 | 必须 |
| search_news | 查新闻 | 必须 |
| get_industry_context | 查行业上下文 | 必须 |
| finish_prediction | 最终预测 | 必须 |

第二版再扩展：

| 工具 | 作用 |
|---|---|
| get_fundamental_snapshot | 查基本面 |
| get_peer_context | 查同业比较 |
| search_macro_events | 查宏观/政策事件 |
| search_research_summary | 查研报摘要 |

### 8.2 统一工具返回格式

所有工具建议返回统一 JSON：

```json
{
  "tool_name": "search_news",
  "status": "ok",
  "as_of": "2024-06-28 15:00:00",
  "result_count": 3,
  "results": [
    {
      "id": "news_001",
      "time": "2024-06-25 09:20:00",
      "title": "xxx",
      "summary": "xxx",
      "source": "xxx",
      "relevance": 0.83
    }
  ],
  "warning": null
}
```

出错时：

```json
{
  "tool_name": "search_news",
  "status": "error",
  "error_type": "invalid_arguments",
  "message": "top_k must be between 1 and 10",
  "results": []
}
```

### 8.3 工具 1：`get_price_factors`

#### 函数签名

```python
def get_price_factors(
    ticker: str,
    as_of: str,
    lookback_days: int = 20
) -> dict:
    # Get point-in-time price and technical factor summary for a stock.
    # Args:
    #   ticker: Stock ticker, e.g. "688888.SH".
    #   as_of: Prediction timestamp. Only data up to this timestamp can be used.
    #   lookback_days: Lookback window in trading days.
    # Returns:
    #   A JSON dict containing recent returns, volatility, turnover, relative strength,
    #   liquidity, limit/suspension flags, and a short summary.
    ...
```

#### 入参约束

```text
ticker: 必须在 universe 中
as_of: 必须是合法时间
lookback_days: 建议限制在 [5, 120]
```

#### 返回字段

```json
{
  "tool_name": "get_price_factors",
  "status": "ok",
  "as_of": "2024-06-28 15:00:00",
  "ticker": "688888.SH",
  "lookback_days": 20,
  "factors": {
    "ret_1d": 0.018,
    "ret_5d": 0.032,
    "ret_20d": 0.087,
    "vol_20d": 0.036,
    "turnover_5d_avg": 0.052,
    "turnover_20d_avg": 0.038,
    "rs_industry_20d": 0.061,
    "rs_market_20d": 0.074,
    "momentum_rank_in_industry": 0.82,
    "liquidity_rank": 0.76,
    "limit_up_recent_5d": 0,
    "limit_down_recent_5d": 0,
    "suspended_recent_20d": 0
  },
  "summary": "近20日上涨8.7%，跑赢行业6.1%，成交活跃度上升，短期动量较强。"
}
```

#### 实现伪代码

```python
def get_price_factors(ticker: str, as_of: str, lookback_days: int = 20) -> dict:
    as_of_date = to_trade_date(as_of)
    factor_row = factor_snapshot.query(
        "ticker == @ticker and as_of_date == @as_of_date"
    )

    if factor_row.empty:
        return {
            "tool_name": "get_price_factors",
            "status": "error",
            "error_type": "not_found",
            "message": "No factor snapshot found.",
            "results": []
        }

    row = factor_row.iloc[0]
    return {
        "tool_name": "get_price_factors",
        "status": "ok",
        "as_of": as_of,
        "ticker": ticker,
        "lookback_days": lookback_days,
        "factors": {
            "ret_5d": row.ret_5d,
            "ret_20d": row.ret_20d,
            "vol_20d": row.vol_20d,
            "turnover_5d_avg": row.turnover_5d_avg,
            "turnover_20d_avg": row.turnover_20d_avg,
            "rs_industry_20d": row.rs_industry_20d,
            "rs_market_20d": row.rs_market_20d,
            "momentum_rank_in_industry": row.momentum_rank_in_industry,
            "liquidity_rank": row.liquidity_rank
        },
        "summary": build_price_summary(row)
    }
```

### 8.4 工具 2：`search_announcements`

#### 函数签名

```python
def search_announcements(
    ticker: str,
    as_of: str,
    start_time: str,
    end_time: str,
    top_k: int = 5,
    doc_types: list[str] | None = None
) -> dict:
    # Search company announcements visible before as_of.
    ...
```

#### 入参约束

```text
end_time <= as_of
top_k in [1, 10]
start_time 不建议早于 as_of - 365d
```

如果模型传了未来时间：

```text
end_time > as_of
```

工具必须自动截断到 `as_of`，并返回 warning，同时 reward 中可以对这种行为轻微扣分。

#### 返回示例

```json
{
  "tool_name": "search_announcements",
  "status": "ok",
  "as_of": "2024-06-28 15:00:00",
  "ticker": "688888.SH",
  "result_count": 2,
  "results": [
    {
      "doc_id": "ann_001",
      "publish_time": "2024-06-24 18:30:00",
      "title": "海川机器人关于签订重大销售合同的公告",
      "doc_type": "major_contract",
      "summary": "公司与某新能源车企签订3.8亿元工业机器人设备订单，预计未来12个月交付。",
      "importance_score": 0.91
    },
    {
      "doc_id": "ann_002",
      "publish_time": "2024-06-11 19:10:00",
      "title": "股东减持计划公告",
      "doc_type": "shareholder_reduction",
      "summary": "持股5.2%股东计划未来3个月内减持不超过公司总股本1%。",
      "importance_score": 0.74
    }
  ],
  "warning": null
}
```

### 8.5 工具 3：`search_news`

#### 函数签名

```python
def search_news(
    ticker: str,
    as_of: str,
    query: str,
    start_time: str,
    end_time: str,
    top_k: int = 5
) -> dict:
    # Search point-in-time news related to a stock.
    ...
```

#### query 设计

模型可以生成 query：

```text
海川机器人 订单 新能源车 工业机器人
海川机器人 股价 放量 机构
机器人设备 行业 政策 订单
```

工具实现时可以把 query 用于：

1. title/content BM25 检索。
2. embedding 检索。
3. 和 ticker 精确关联结合。
4. 结果重排序。

#### 返回示例

```json
{
  "tool_name": "search_news",
  "status": "ok",
  "as_of": "2024-06-28 15:00:00",
  "ticker": "688888.SH",
  "query": "海川机器人 订单 新能源车 工业机器人",
  "result_count": 2,
  "results": [
    {
      "news_id": "news_001",
      "publish_time": "2024-06-25 09:20:00",
      "title": "新能源车企加速自动化产线升级",
      "summary": "多家机器人设备厂商订单增长，市场关注工业机器人板块。",
      "source": "财经媒体A",
      "relevance": 0.86,
      "sentiment_score": 0.48
    },
    {
      "news_id": "news_002",
      "publish_time": "2024-06-27 13:15:00",
      "title": "海川机器人盘中放量上涨",
      "summary": "机构认为订单落地利好收入预期，但短期股价已反映部分乐观预期。",
      "source": "财经媒体B",
      "relevance": 0.79,
      "sentiment_score": 0.16
    }
  ]
}
```

### 8.6 工具 4：`get_industry_context`

#### 函数签名

```python
def get_industry_context(
    industry_id: str,
    as_of: str,
    lookback_days: int = 20
) -> dict:
    # Get point-in-time industry context.
    ...
```

#### 返回示例

```json
{
  "tool_name": "get_industry_context",
  "status": "ok",
  "as_of": "2024-06-28 15:00:00",
  "industry_id": "CI005001",
  "industry_name": "机器人设备",
  "metrics": {
    "industry_ret_5d": 0.012,
    "industry_ret_20d": 0.045,
    "valuation_pe_percentile_2y": 0.68,
    "turnover_percentile_1y": 0.72,
    "policy_sentiment": "positive"
  },
  "summary": "机器人设备行业近20日表现较强，政策和订单预期偏正面，但估值处于近两年偏高位置。"
}
```

### 8.7 工具 5：`get_fundamental_snapshot`（第二版）

#### 函数签名

```python
def get_fundamental_snapshot(
    ticker: str,
    as_of: str
) -> dict:
    # Get the latest fundamental snapshot visible before as_of.
    ...
```

返回：

```json
{
  "tool_name": "get_fundamental_snapshot",
  "status": "ok",
  "ticker": "688888.SH",
  "as_of": "2024-06-28 15:00:00",
  "latest_report_period": "2024Q1",
  "publish_time": "2024-04-29 20:00:00",
  "fundamentals": {
    "revenue_yoy": 0.28,
    "net_profit_yoy": 0.35,
    "gross_margin": 0.41,
    "roe": 0.13,
    "pe_ttm": 48.3,
    "pb": 6.2
  },
  "summary": "最新可见财报显示收入同比增长28%，净利润同比增长35%，盈利增速较高，但估值处于偏高水平。"
}
```

### 8.8 工具 6：`finish_prediction`

这个工具不是外部检索，而是 agent 的最终动作。也可以不实现为工具，而是要求模型输出 final JSON。

推荐 final schema：

```json
{
  "prediction": "up",
  "p_up": 0.62,
  "p_neutral": 0.25,
  "p_down": 0.13,
  "alpha_score": 0.49,
  "confidence": 0.62,
  "evidence_summary": [
    {
      "direction": "positive",
      "source_type": "announcement",
      "source_id": "ann_001",
      "summary": "公司签订重大销售合同，改善未来收入预期。"
    },
    {
      "direction": "negative",
      "source_type": "announcement",
      "source_id": "ann_002",
      "summary": "股东存在减持计划，对短期情绪有压力。"
    }
  ],
  "risk_factors": [
    "短期涨幅较大，部分利好可能已被价格反映。"
  ],
  "search_steps_used": 4
}
```

约束：

```text
p_up + p_neutral + p_down 必须接近 1
每个概率必须在 [0, 1]
prediction 必须等于最大概率对应类别
alpha_score = p_up - p_down
search_steps_used 不能超过 max_tool_calls
```

---

## 9. Prompt 设计

### 9.1 System Prompt

```text
你是一个金融搜索预测 agent。
你的任务是在历史预测时点 as_of 下，预测指定股票未来 horizon 个交易日的相对收益方向。
你只能使用 as_of 时点之前的信息。
你可以通过 function call 调用工具搜索公告、新闻、量价因子和行业信息。
你需要在有限搜索预算内决定查什么、是否继续查、何时停止。
最终必须输出合法 JSON，包含 p_up、p_neutral、p_down、prediction、alpha_score、evidence_summary。
不要输出未来信息，不要编造工具结果。
```

### 9.2 User Prompt 模板

```text
请预测以下股票未来 {horizon} 个交易日的相对收益方向。

股票代码：{ticker}
公司名称：{company_name}
行业：{industry_name}
预测时点：{as_of}
市场：{market}
最大工具调用次数：{max_tool_calls}

可用工具：
1. get_price_factors
2. search_announcements
3. search_news
4. get_industry_context

你需要通过 function call 主动获取信息。不要假设你已经知道相关新闻或公告。
最终输出 JSON：
{
  "prediction": "up|neutral|down",
  "p_up": float,
  "p_neutral": float,
  "p_down": float,
  "alpha_score": float,
  "confidence": float,
  "evidence_summary": [...],
  "risk_factors": [...],
  "search_steps_used": int
}
```

### 9.3 veRL 数据中的 prompt 字段

对于单条样本，prompt 可以序列化为 messages：

```json
[
  {
    "role": "system",
    "content": "你是一个金融搜索预测 agent..."
  },
  {
    "role": "user",
    "content": "请预测以下股票未来5个交易日的相对收益方向...\n股票代码：688888.SH\n..."
  }
]
```

或者直接 flatten 成字符串，取决于你使用的 tokenizer chat template。

---

## 10. Trajectory 数据结构

### 10.1 完整 trajectory JSONL

每条 rollout 可以记录为一行 JSON：

```json
{
  "trajectory_id": "688888.SH_2024-06-28_T5_rollout_0001",
  "sample_id": "688888.SH_2024-06-28_T5",
  "model_name": "Qwen3-4B-SFT",
  "policy_step": 0,
  "messages": [
    {
      "role": "system",
      "content": "你是一个金融搜索预测 agent..."
    },
    {
      "role": "user",
      "content": "请预测..."
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "search_announcements",
        "arguments": {
          "ticker": "688888.SH",
          "as_of": "2024-06-28 15:00:00",
          "start_time": "2024-05-28 00:00:00",
          "end_time": "2024-06-28 15:00:00",
          "top_k": 5
        }
      }
    },
    {
      "role": "tool",
      "name": "search_announcements",
      "content": {
        "status": "ok",
        "result_count": 2,
        "results": [
          {
            "doc_id": "ann_001",
            "publish_time": "2024-06-24 18:30:00",
            "title": "重大销售合同公告",
            "summary": "公司签订3.8亿元订单。"
          }
        ]
      }
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "get_price_factors",
        "arguments": {
          "ticker": "688888.SH",
          "as_of": "2024-06-28 15:00:00",
          "lookback_days": 20
        }
      }
    },
    {
      "role": "tool",
      "name": "get_price_factors",
      "content": {
        "status": "ok",
        "summary": "近20日上涨8.7%，跑赢行业6.1%。"
      }
    },
    {
      "role": "assistant",
      "content": "{\"prediction\":\"up\",\"p_up\":0.62,\"p_neutral\":0.25,\"p_down\":0.13,\"alpha_score\":0.49,\"confidence\":0.62,\"evidence_summary\":[...],\"search_steps_used\":2}"
    }
  ],
  "tool_calls": [
    {
      "tool_name": "search_announcements",
      "valid": true,
      "latency_ms": 23
    },
    {
      "tool_name": "get_price_factors",
      "valid": true,
      "latency_ms": 9
    }
  ],
  "final_answer": {
    "prediction": "up",
    "p_up": 0.62,
    "p_neutral": 0.25,
    "p_down": 0.13,
    "alpha_score": 0.49,
    "confidence": 0.62,
    "search_steps_used": 2
  },
  "reward": {
    "total": 0.583,
    "components": {
      "prob_reward": 0.62,
      "brier_reward": 0.888,
      "direction_reward": 1.0,
      "pnl_reward": 0.245,
      "format_reward": 0.05,
      "search_cost": -0.04
    }
  }
}
```

### 10.2 为什么要保存完整 trajectory

用于：

1. Debug agent 搜索行为。
2. 分析 reward hacking。
3. 做 SFT 数据蒸馏。
4. 做错误分析。
5. 对比 RL 前后搜索策略变化。

---

## 11. SFT 冷启动数据

### 11.1 为什么需要 SFT

原始 Qwen3-4B 可能会出现：

```text
function call JSON 不合法
忘记调用工具直接猜
工具参数乱填
概率字段缺失
输出中文散文而不是 JSON
```

所以建议先做 SFT，让模型学会：

1. 工具调用格式。
2. 基本搜索流程。
3. 读取 observation 后继续调用。
4. 最终输出结构化 JSON。

### 11.2 SFT 数据构造方式

可用三种方式：

#### 方式 A：规则轨迹

固定模板：

```text
get_price_factors
→ search_announcements
→ search_news
→ get_industry_context
→ final prediction
```

优点：简单稳定。  
缺点：搜索策略单一。

#### 方式 B：规则 + 随机扰动

例如：

```text
路径 1：price → announcement → news → industry
路径 2：announcement → price → news
路径 3：news → price → announcement
路径 4：price → industry → final
```

优点：给 RL 初始策略一些多样性。

#### 方式 C：强模型蒸馏

用更大模型生成轨迹，但必须保证：

```text
强模型只能看 as_of 前工具返回的结果，不能看未来 label。
```

不能让强模型根据未来收益倒推理由。

### 11.3 SFT 样本格式

```json
{
  "sample_id": "688888.SH_2024-06-28_T5",
  "messages": [
    {
      "role": "system",
      "content": "你是一个金融搜索预测 agent..."
    },
    {
      "role": "user",
      "content": "请预测..."
    },
    {
      "role": "assistant",
      "content": "{\"name\":\"get_price_factors\",\"arguments\":{\"ticker\":\"688888.SH\",\"as_of\":\"2024-06-28 15:00:00\",\"lookback_days\":20}}"
    },
    {
      "role": "tool",
      "content": "{\"summary\":\"近20日上涨8.7%，跑赢行业6.1%...\"}"
    },
    {
      "role": "assistant",
      "content": "{\"name\":\"search_announcements\",\"arguments\":{\"ticker\":\"688888.SH\",\"as_of\":\"2024-06-28 15:00:00\",\"start_time\":\"2024-05-28\",\"end_time\":\"2024-06-28\",\"top_k\":5}}"
    },
    {
      "role": "tool",
      "content": "{\"summary\":\"公司签订重大合同...\"}"
    },
    {
      "role": "assistant",
      "content": "{\"prediction\":\"up\",\"p_up\":0.58,\"p_neutral\":0.28,\"p_down\":0.14,\"alpha_score\":0.44,...}"
    }
  ]
}
```

### 11.4 SFT 注意事项

1. SFT 不追求最终预测最优，先追求格式和流程稳定。
2. 不要在 SFT target 中暴露未来收益。
3. 如果用真实标签辅助生成最终答案，需要小心 hindsight rationale，最好只用 label 约束概率方向，不生成过度确定的解释。
4. 可以对工具调用 JSON 和 final JSON 加强监督。

---

## 12. Agent RL 训练设计

### 12.1 State / Action / Observation / Reward

#### State

```text
当前任务：
  ticker
  company_name
  as_of
  horizon
  industry
  tools
  max_tool_calls

已经获得的信息：
  previous tool calls
  previous observations
  remaining tool budget
```

#### Action

```text
1. 调用工具：
   get_price_factors(...)
   search_announcements(...)
   search_news(...)
   get_industry_context(...)

2. 停止搜索并输出 final prediction。
```

#### Observation

```text
工具返回的结构化 JSON。
```

#### Reward

```text
由 final prediction + hidden label + trajectory metadata 计算。
```

### 12.2 GRPO 组内采样

对于同一个 sample，采样 G 条 rollout：

```text
sample = 688888.SH_2024-06-28_T5

rollout 1: announcement → price → industry → final
rollout 2: news → industry → final
rollout 3: price → news → announcement → final
rollout 4: price → final
```

每条都有 reward：

```text
R1 = 0.583
R2 = 0.235
R3 = 0.421
R4 = 0.310
```

GRPO 使用组内相对优势：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

高 reward 轨迹被强化，低 reward 轨迹被压制。

### 12.3 为什么推荐 GRPO

第一版推荐 GRPO 而不是 PPO：

1. 不需要单独 critic，工程复杂度低。
2. 适合 RLVR 这类 outcome reward。
3. 和 veRL 结合成熟。
4. 对小模型后训练比较友好。

---

## 13. Reward 函数设计

### 13.1 输入

reward function 需要拿到：

```python
response_text: str
extra_info: dict
trajectory_info: dict
```

其中 `extra_info` 包含 verifier 隐藏字段：

```json
{
  "sample_id": "688888.SH_2024-06-28_T5",
  "label": "up",
  "label_id": 0,
  "future_relative_return": 0.025,
  "future_volatility": 0.036,
  "cross_section_rank": 0.86,
  "max_tool_calls": 4
}
```

`trajectory_info` 包含：

```json
{
  "num_tool_calls": 4,
  "num_invalid_tool_calls": 0,
  "num_future_time_violations": 0,
  "tool_names": [
    "search_announcements",
    "get_price_factors",
    "search_news",
    "get_industry_context"
  ],
  "observation_token_count": 1200
}
```

### 13.2 Final Answer 解析

先解析模型最终 JSON：

```python
def parse_final_answer(response_text: str) -> dict | None:
    # Extract the final JSON answer from model response.
    # Return None if parsing fails.
    ...
```

需要校验：

```text
prediction in {"up", "neutral", "down"}
0 <= p_up <= 1
0 <= p_neutral <= 1
0 <= p_down <= 1
abs(p_up + p_neutral + p_down - 1) <= tolerance
alpha_score exists or can be computed
```

如果解析失败：

```text
total_reward = -1.0
```

### 13.3 Direction Reward

定义：

```python
pred_label = argmax([p_up, p_neutral, p_down])
true_label = extra_info["label"]

if pred_label == true_label:
    direction_reward = 1.0
else:
    direction_reward = -0.5
```

也可以更温和：

```python
direction_reward = 1.0 if correct else 0.0
```

第一版建议用 `1.0 / -0.5`，让方向错有明显惩罚，但不过度。

### 13.4 Probability Reward

真实类别概率：

```python
p_true = {
    "up": p_up,
    "neutral": p_neutral,
    "down": p_down
}[true_label]
```

可用：

```python
prob_reward = p_true
```

范围 `[0, 1]`。

也可以用 log reward：

```python
log_prob_reward = log(max(p_true, 1e-6))
```

但 log reward 是负数，数值范围更大，第一版可以先用 `p_true`，更稳定。

### 13.5 Brier Reward

三分类真实 one-hot：

```python
y = [1, 0, 0]  # true_label = up
p = [p_up, p_neutral, p_down]
brier_error = sum((p_k - y_k) ** 2 for k in classes)
```

三分类 Brier error 最大为 2，因此：

```python
brier_reward = 1 - brier_error / 2
```

范围 `[0, 1]`。

示例：

```text
p = [0.62, 0.25, 0.13]
y = [1, 0, 0]

brier_error = (0.62-1)^2 + 0.25^2 + 0.13^2 = 0.2238
brier_reward = 1 - 0.2238 / 2 = 0.8881
```

### 13.6 PnL Reward

将概率转成仓位：

```python
position = p_up - p_down
```

范围 `[-1, 1]`。

用未来相对收益计算：

```python
raw_pnl = position * future_relative_return
```

为了数值稳定，需要缩放和截断：

```python
pnl_reward = clip(raw_pnl / pnl_scale, -1.0, 1.0)
```

其中：

```text
pnl_scale 可以取训练集 abs(future_relative_return) 的 75% 分位数
比如 pnl_scale = 0.03
```

也可以按波动率归一化：

```python
risk_adjusted_return = future_relative_return / max(hist_volatility, 1e-4)
pnl_reward = clip(position * risk_adjusted_return / 3.0, -1.0, 1.0)
```

第一版建议：

```python
pnl_scale = 0.03
pnl_reward = clip(position * future_relative_return / 0.03, -1.0, 1.0)
```

如果：

```text
position = 0.49
future_relative_return = 0.025
```

则：

```text
raw_pnl = 0.01225
pnl_reward = 0.01225 / 0.03 = 0.408
```

### 13.7 Search Cost Reward

```python
search_cost = -lambda_call * num_tool_calls
```

推荐：

```python
lambda_call = 0.02
```

如果超出 max_tool_calls：

```python
search_cost -= 0.2 * (num_tool_calls - max_tool_calls)
```

无效工具调用：

```python
invalid_tool_penalty = -0.1 * num_invalid_tool_calls
```

未来时间违规：

```python
future_time_penalty = -0.3 * num_future_time_violations
```

### 13.8 Format Reward

```python
format_reward = 0.0

if valid_json:
    format_reward += 0.03

if probabilities_valid:
    format_reward += 0.03

if prediction_matches_argmax:
    format_reward += 0.02

if evidence_summary_non_empty:
    format_reward += 0.02
```

总计最高 `0.10`。

注意：format reward 不能太高，否则模型可能只学会输出漂亮格式。

### 13.9 Evidence Reward

第一版可以简单：

```python
evidence_reward = 0.0

if evidence_summary has at least 1 positive/negative/neutral evidence:
    evidence_reward += 0.03

if evidence_summary sources are from actual tool results:
    evidence_reward += 0.05

if source_id hallucinated:
    evidence_reward -= 0.1
```

如果暂时不好判断 source_id 是否真实，可以先不加 evidence_reward。

### 13.10 Rank Reward

Rank reward 很重要，但第一版不建议直接放进 per-sample RL reward，因为 Rank IC 需要同一天一组股票才能算。

推荐策略：

#### 第一版

```text
Rank IC 只做评估，不放入训练 reward。
```

#### 第二版

在每个 batch 中按 trade_date 分组，计算：

```python
alpha_score = p_up - p_down
rank_ic = spearman_corr(rank(alpha_score), rank(future_relative_return))
```

再把该 date group 的 rank_ic 分配给组内样本：

```python
rank_reward_i = rank_ic
```

问题：

1. 要保证同一 batch 里有足够多同一天股票。
2. veRL 默认 reward function 是否方便拿到 batch-level 信息要看实现。
3. group reward 方差可能较大。

因此第一版先不用 rank reward 训练，但必须在评估中报告 Rank IC。

### 13.11 总 Reward 推荐版本

#### MVP 版本

```python
total_reward = (
    0.25 * direction_reward
    + 0.25 * prob_reward
    + 0.25 * brier_reward
    + 0.20 * pnl_reward
    + format_reward
    + search_cost
    + invalid_tool_penalty
    + future_time_penalty
)
```

#### 稳定版

```python
total_reward = (
    0.20 * direction_reward
    + 0.25 * prob_reward
    + 0.25 * brier_reward
    + 0.20 * pnl_reward
    + 0.05 * evidence_reward
    + format_reward
    + search_cost
    + invalid_tool_penalty
    + future_time_penalty
)

total_reward = clip(total_reward, -1.0, 1.5)
```

### 13.12 Reward 伪代码

```python
import json
import numpy as np

CLASSES = ["up", "neutral", "down"]

def safe_clip(x, lo, hi):
    return max(lo, min(hi, x))

def parse_json_answer(text: str):
    # 实际实现需要更鲁棒：提取最后一个 JSON block
    try:
        start = text.rfind("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        return json.loads(text[start:end+1])
    except Exception:
        return None

def compute_reward(response_text: str, extra_info: dict, trajectory_info: dict) -> dict:
    ans = parse_json_answer(response_text)

    if ans is None:
        return {
            "score": -1.0,
            "components": {
                "parse_error": 1
            }
        }

    try:
        p_up = float(ans["p_up"])
        p_neutral = float(ans["p_neutral"])
        p_down = float(ans["p_down"])
        pred = ans.get("prediction", None)
    except Exception:
        return {
            "score": -1.0,
            "components": {
                "missing_fields": 1
            }
        }

    probs = np.array([p_up, p_neutral, p_down], dtype=float)

    valid_prob = (
        np.all(probs >= 0)
        and np.all(probs <= 1)
        and abs(probs.sum() - 1.0) <= 0.05
    )

    if not valid_prob:
        return {
            "score": -0.8,
            "components": {
                "invalid_probability": 1
            }
        }

    # 归一化，避免概率和轻微不等于 1
    probs = probs / probs.sum()

    true_label = extra_info["label"]
    true_idx = CLASSES.index(true_label)

    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]

    # 1. direction
    direction_reward = 1.0 if pred_label == true_label else -0.5

    # 2. probability
    prob_reward = float(probs[true_idx])

    # 3. brier
    y = np.zeros(3)
    y[true_idx] = 1.0
    brier_error = float(np.sum((probs - y) ** 2))
    brier_reward = 1.0 - brier_error / 2.0

    # 4. pnl
    future_rel = float(extra_info["future_relative_return"])
    position = float(probs[0] - probs[2])
    pnl_scale = float(extra_info.get("pnl_scale", 0.03))
    pnl_reward = safe_clip(position * future_rel / max(pnl_scale, 1e-6), -1.0, 1.0)

    # 5. format reward
    format_reward = 0.0
    format_reward += 0.03  # valid json
    format_reward += 0.03  # valid probabilities
    if pred == pred_label:
        format_reward += 0.02
    if isinstance(ans.get("evidence_summary", None), list) and len(ans["evidence_summary"]) > 0:
        format_reward += 0.02

    # 6. costs
    num_tool_calls = int(trajectory_info.get("num_tool_calls", 0))
    max_tool_calls = int(extra_info.get("max_tool_calls", 4))
    search_cost = -0.02 * num_tool_calls
    if num_tool_calls > max_tool_calls:
        search_cost -= 0.2 * (num_tool_calls - max_tool_calls)

    invalid_tool_penalty = -0.1 * int(trajectory_info.get("num_invalid_tool_calls", 0))
    future_time_penalty = -0.3 * int(trajectory_info.get("num_future_time_violations", 0))

    total = (
        0.20 * direction_reward
        + 0.25 * prob_reward
        + 0.25 * brier_reward
        + 0.20 * pnl_reward
        + format_reward
        + search_cost
        + invalid_tool_penalty
        + future_time_penalty
    )

    total = safe_clip(total, -1.0, 1.5)

    return {
        "score": total,
        "components": {
            "direction_reward": direction_reward,
            "prob_reward": prob_reward,
            "brier_reward": brier_reward,
            "pnl_reward": pnl_reward,
            "format_reward": format_reward,
            "search_cost": search_cost,
            "invalid_tool_penalty": invalid_tool_penalty,
            "future_time_penalty": future_time_penalty,
            "pred_label": pred_label,
            "true_label": true_label,
            "position": position,
            "future_relative_return": future_rel
        }
    }
```

---

## 14. veRL 实验落地方案

### 14.1 veRL 中的关键模块映射

本项目概念与 veRL 映射：

| 本项目概念 | veRL 对应 |
|---|---|
| Qwen3-4B agent | actor policy model |
| SFT model | initial policy / reference model |
| 同一问题多条 rollout | GRPO group samples |
| verifier reward | custom reward function |
| 工具调用过程 | multi-turn rollout / tool environment |
| 历史搜索环境 | custom tools / external tool server |
| labels hidden fields | dataset extra_info |

### 14.2 两种实现路线

#### 路线 A：veRL multi-turn tool rollout（推荐）

如果你当前 veRL 版本支持多轮工具调用，可以实现：

```text
model generates tool call
→ veRL rollout engine intercepts tool call
→ 调用 Python tool
→ append tool observation
→ model continues
→ final answer
→ reward_fn 计算 reward
```

优点：

1. 真正在线 agent RL。
2. RL 训练直接优化 tool-call trajectory。
3. 最符合项目定义。

缺点：

1. 需要处理工具注册、tool schema、消息格式。
2. Debug 成本较高。

#### 路线 B：外部 rollout runner + veRL 训练

如果多轮工具集成暂时不稳定，可以先做：

```text
外部 runner 用当前 policy rollout 轨迹
→ 保存完整 messages + final answer + reward
→ veRL 用这些 response/reward 做更新
```

优点：

1. 工程更可控。
2. 搜索环境容易 debug。

缺点：

1. 不是最标准的在线 RL。
2. 需要处理 rollout 和训练 policy 的同步问题。
3. 训练效率可能低。

第一阶段建议：

```text
先用路线 B 快速验证数据构造和 reward；
再切换路线 A 做真正 multi-turn agent RL。
```

### 14.3 veRL 数据集格式建议

构造 `train.parquet`：

| 字段 | 类型 | 说明 |
|---|---|---|
| data_source | string | stock_agent_rl |
| prompt | list[dict] or string | messages / prompt |
| ability | string | stock_prediction_agent |
| reward_model | dict | 可为空或包含 style |
| extra_info | dict | 隐藏标签和元数据 |
| sample_id | string | 样本 ID |

示例：

```json
{
  "data_source": "stock_agent_rl",
  "ability": "stock_prediction_agent",
  "prompt": [
    {
      "role": "system",
      "content": "你是一个金融搜索预测 agent..."
    },
    {
      "role": "user",
      "content": "请预测股票 688888.SH 在 2024-06-28 收盘后未来 5 个交易日的相对收益方向..."
    }
  ],
  "reward_model": {
    "style": "rule"
  },
  "extra_info": {
    "sample_id": "688888.SH_2024-06-28_T5",
    "ticker": "688888.SH",
    "as_of": "2024-06-28 15:00:00",
    "horizon": 5,
    "industry_id": "CI005001",
    "industry_name": "机器人设备",
    "max_tool_calls": 4,
    "label": "up",
    "label_id": 0,
    "future_relative_return": 0.025,
    "pnl_scale": 0.03,
    "split": "train"
  }
}
```

注意：

```text
extra_info 会被 reward function 使用，但不能出现在 prompt 里。
```

### 14.4 自定义 reward function 文件

推荐目录：

```text
src/rewards/stock_agent_reward.py
```

提供：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ...
```

实际函数签名要根据你使用的 veRL 版本适配。核心逻辑是：

1. 从 `solution_str` 中解析 final JSON。
2. 从 `extra_info` 中拿 label 和 future_relative_return。
3. 从 rollout metadata 中拿 tool call 次数、非法调用次数。
4. 返回 float reward 或带 components 的 dict。

### 14.5 多轮工具注册

推荐目录：

```text
src/tools/stock_tools.py
```

伪代码：

```python
from typing import Optional

def get_price_factors(ticker: str, as_of: str, lookback_days: int = 20) -> dict:
    """Get point-in-time price factor summary."""
    ...

def search_announcements(
    ticker: str,
    as_of: str,
    start_time: str,
    end_time: str,
    top_k: int = 5,
    doc_types: Optional[list[str]] = None,
) -> dict:
    """Search company announcements visible before as_of."""
    ...

def search_news(
    ticker: str,
    as_of: str,
    query: str,
    start_time: str,
    end_time: str,
    top_k: int = 5,
) -> dict:
    """Search point-in-time news related to ticker."""
    ...

def get_industry_context(
    industry_id: str,
    as_of: str,
    lookback_days: int = 20,
) -> dict:
    """Get point-in-time industry context."""
    ...
```

如果使用 veRL 的 function tool 机制，工具函数需要有清晰类型签名和 Google-style docstring，方便自动推断 schema。

### 14.6 GRPO 配置重点

推荐重点关注：

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001
  rollout:
    n: 4  # 每个 prompt 采样 G 条，第一版可用 4 或 8
    temperature: 0.7
    top_p: 0.95
    max_response_length: 2048

data:
  train_files: /path/to/train.parquet
  val_files: /path/to/valid.parquet
  max_prompt_length: 4096
  max_response_length: 2048
```

实际字段名需要以你本地 veRL 版本为准。

### 14.7 训练阶段建议

#### 阶段 0：无训练 baseline

```text
Qwen3-4B-Instruct + prompt + tools
```

记录：

```text
valid reward
format success rate
平均 tool calls
方向指标
rank IC
```

#### 阶段 1：SFT

目的：

```text
学会 tool call 格式和 final JSON。
```

数据：

```text
规则轨迹 + 少量强模型轨迹。
```

#### 阶段 2：GRPO MVP

工具：

```text
get_price_factors
search_announcements
search_news
get_industry_context
```

Reward：

```text
direction + probability + brier + pnl + cost + format
```

#### 阶段 3：加入更多工具

增加：

```text
fundamental
peer
macro
research summary
```

#### 阶段 4：reward ablation

对比：

```text
无 pnl reward
无 search cost
无 brier reward
只 direction reward
完整 reward
```

---

## 15. 推荐代码目录结构

```text
stock-agent-rl/
  README.md

  configs/
    sft_qwen3_4b.yaml
    grpo_stock_agent.yaml
    eval.yaml

  data/
    raw/
      stock_daily.parquet
      industry_daily.parquet
      announcements.parquet
      news.parquet
      fundamentals.parquet
      universe.parquet
      trading_calendar.parquet

    processed/
      factor_snapshot.parquet
      tasks.parquet
      labels.parquet
      train.parquet
      valid.parquet
      test.parquet

    rollouts/
      sft_rollouts.jsonl
      grpo_debug_rollouts.jsonl

  src/
    data/
      build_universe.py
      build_factors.py
      build_labels.py
      build_tasks.py
      build_verl_dataset.py
      point_in_time.py

    tools/
      stock_tools.py
      search_backend.py
      schemas.py

    rewards/
      stock_agent_reward.py
      reward_debug.py

    prompts/
      templates.py

    sft/
      build_sft_data.py

    eval/
      eval_predictions.py
      eval_rank_ic.py
      backtest.py
      analyze_tool_usage.py

    utils/
      json_utils.py
      trading_calendar.py
      logging.py

  scripts/
    00_build_data.sh
    01_build_sft_data.sh
    02_run_sft.sh
    03_run_grpo.sh
    04_eval.sh
```

---

## 16. 实验指标

### 16.1 Reward 指标

```text
train/mean_reward
train/reward_std
valid/mean_reward
valid/reward_by_month
valid/reward_by_industry
```

### 16.2 预测指标

```text
Accuracy
Balanced Accuracy
Macro-F1
Log Loss
Brier Score
ECE
```

### 16.3 排序指标

每天横截面：

```text
alpha_score = p_up - p_down
```

计算：

```text
IC = corr(alpha_score, future_relative_return)
Rank IC = spearman_corr(alpha_score, future_relative_return)
```

报告：

```text
mean IC
mean Rank IC
ICIR = mean(IC) / std(IC)
positive IC ratio
monthly IC
industry-neutral IC
```

### 16.4 回测指标

简单分组：

```text
每天按 alpha_score 排序
top 20% 做多
bottom 20% 做空或空仓
持有 h 天
```

报告：

```text
top group return
bottom group return
top-bottom return
annual return
Sharpe
max drawdown
turnover
transaction-cost-adjusted return
```

### 16.5 Agent 行为指标

```text
平均工具调用次数
每个工具调用占比
无效工具调用率
未来时间违规率
JSON 解析成功率
概率合法率
提前停止比例
平均 response token 数
平均 observation token 数
```

---

## 17. 消融实验设计

至少做：

| 实验 | 描述 |
|---|---|
| Zero-shot Agent | 原始 Qwen3-4B + tools |
| SFT Agent | 只做 SFT，不做 RL |
| Fixed RAG | 固定查 price + ann + news + industry |
| RL-Direction | reward 只用方向 |
| RL-Direction+Brier | 加概率校准 |
| RL-Direction+Brier+PnL | 加收益 reward |
| Full RL | 加搜索成本和格式约束 |
| No Ann Tool | 移除公告工具 |
| No News Tool | 移除新闻工具 |
| No Price Tool | 移除量价工具 |

核心结论需要回答：

```text
RL 是否比 SFT 强？
Agent 是否比固定 RAG 强？
哪些工具最有贡献？
reward 组件是否有效？
搜索成本是否下降或更合理？
```

---

## 18. 防泄漏清单

### 18.1 时间泄漏

检查：

```text
tool_result.publish_time <= as_of
fundamental.publish_time <= as_of
factor calculation end_date <= as_of trade_date
news.publish_time <= as_of
announcement.effective_time <= as_of
```

### 18.2 标签泄漏

禁止：

```text
prompt 中出现 future_return
prompt 中出现 label
工具返回未来涨跌
新闻摘要由看过未来收益的模型生成
teacher rationale 根据未来 label 编造原因
```

### 18.3 生存者偏差

需要保留：

```text
退市股票
历史指数成分
历史行业分类
停牌和 ST 标记
```

### 18.4 交易可行性

需要处理：

```text
涨停买不进
跌停卖不出
停牌不可交易
成交额过低
交易成本
```

第一版可以先在样本过滤中去掉不可交易样本，但要记录过滤逻辑。

---

## 19. 常见问题与建议

### 19.1 如果 reward 上升但 Rank IC 不升怎么办

可能原因：

1. reward 过度依赖 format / cost。
2. direction label 太噪。
3. pnl_scale 不合理。
4. 验证集和训练集分布差异。
5. agent 学会保守输出 neutral。
6. 分类 reward 与横截面目标不一致。

解决：

1. 降低 format reward 权重。
2. 用横截面分位 label。
3. 增加 alpha_score 评估。
4. 加强 PnL reward。
5. 调整 neutral 类比例。
6. 做 reward ablation。

### 19.2 如果模型总是少搜

原因：

```text
search_cost 太大，或者工具 observation 没有带来足够 reward。
```

解决：

```text
降低每次工具惩罚
提高信息相关 reward
改进工具返回质量
在 SFT 中加入多工具轨迹
```

### 19.3 如果模型总是乱搜

原因：

```text
search_cost 太小，或者 prompt 没有强调预算。
```

解决：

```text
提高 search_cost
增加 invalid tool penalty
限制 max_tool_calls
给工具参数更严格 schema
```

### 19.4 如果模型总是输出极端概率

原因：

```text
direction / pnl reward 过强，brier / calibration 不够。
```

解决：

```text
提高 brier reward 权重
加入 entropy floor 或 probability clipping
对过度自信错误加重惩罚
```

### 19.5 如果 GRPO 组内 reward 方差太小

表现：

```text
同一个 prompt 的 G 条 rollout reward 都差不多
advantage 接近 0
训练停滞
```

解决：

```text
提高 sampling temperature
增大 G
增加 trajectory 多样性
SFT 数据中加入多样路径
调整 reward，使不同搜索路径有区分度
```

---

## 20. 最小可行实验计划

### P0：离线数据和工具

目标：

```text
构造 tasks.parquet、labels.parquet
实现 4 个工具
能对任意 sample 做 point-in-time 查询
```

验收：

```text
随机抽 100 条样本，人工检查无未来泄漏。
工具返回稳定 JSON。
```

### P1：Zero-shot agent baseline

目标：

```text
Qwen3-4B-Instruct 能调用工具并输出 final JSON。
```

验收：

```text
JSON 成功率 > 70%
平均 reward 有可计算结果
```

### P2：SFT

目标：

```text
提高工具调用合法率和 final JSON 合法率。
```

验收：

```text
JSON 成功率 > 95%
工具调用合法率 > 95%
```

### P3：GRPO MVP

目标：

```text
用 veRL 对 SFT agent 做 GRPO。
```

Reward：

```text
direction + prob + brier + pnl + cost + format
```

验收：

```text
valid reward 高于 SFT baseline
Macro-F1 或 Brier 至少一个提升
Rank IC 不下降
```

### P4：样本外评估

目标：

```text
证明不是 reward hacking。
```

验收：

```text
test reward 提升
Rank IC 提升
Top-bottom return 提升
搜索成本可控
```

---

## 21. 第一版推荐参数

```yaml
task:
  horizon: 5
  label_method: cross_section_quantile
  up_quantile: 0.70
  down_quantile: 0.30
  max_tool_calls: 4

model:
  base: Qwen3-4B-Instruct
  sft_method: lora
  rl_method: grpo

rollout:
  num_generations: 4
  temperature: 0.7
  top_p: 0.95
  max_response_length: 2048

reward:
  direction_weight: 0.20
  prob_weight: 0.25
  brier_weight: 0.25
  pnl_weight: 0.20
  pnl_scale: 0.03
  tool_call_cost: 0.02
  invalid_tool_penalty: 0.10
  future_time_penalty: 0.30
  reward_clip_min: -1.0
  reward_clip_max: 1.5

eval:
  metrics:
    - mean_reward
    - accuracy
    - macro_f1
    - brier
    - rank_ic
    - top_bottom_return
    - avg_tool_calls
```

---

## 22. 给 mentor 汇报时的表达

可以这样讲：

> 这个项目的核心不是把新闻公告拼进 prompt 做固定 RAG，而是构造一个 RLVR-style 的金融搜索环境。每个样本是 ticker-date-horizon，agent 初始只知道股票、预测时点和可用工具；它必须通过 function call 主动搜索 as-of 前的公告、新闻、量价和行业信息，并在有限预算内输出未来 T+5/T+10 相对收益的概率预测。数据构造包括五部分：任务样本、point-in-time 搜索环境、隐藏未来收益标签、agent trajectory 和 reward 结果。Reward 由方向、概率校准、PnL、格式和搜索成本组成，用未来真实相对收益自动验证。训练上先 SFT 冷启动工具调用，再用 veRL 的 GRPO 优化整条搜索—分析—预测轨迹。评估不只看训练 reward，还看 valid/test reward、Macro-F1、Brier、Rank IC、分组收益和工具调用成本。

---

## 23. 实现顺序建议

不要一上来就写 veRL 训练代码。推荐顺序：

```text
1. build_labels.py
   先把未来相对收益和 up/neutral/down label 做对。

2. build_tools.py
   实现 point-in-time 工具，确保不会返回未来数据。

3. build_tasks.py
   构造 ticker-date-horizon prompt 样本。

4. run_agent_rollout.py
   不训练，先用 Qwen3-4B 跑 100 条样本，检查 trajectory。

5. stock_agent_reward.py
   写 reward_fn，对 rollout 结果打分。

6. eval_predictions.py
   写评估脚本，计算 reward、Brier、Rank IC。

7. build_sft_data.py
   构造 SFT 冷启动数据。

8. run_sft.sh
   训练工具调用格式。

9. run_grpo.sh
   接入 veRL 做 GRPO。

10. eval_all.sh
    对比 zero-shot / SFT / RL / fixed RAG。
```

---

## 24. 最后总结

本项目成败的关键不是“用不用 Qwen3-4B”或“用不用 GRPO”，而是：

```text
1. 是否构造了无泄漏的 point-in-time 搜索环境；
2. 是否把股票预测变成了可验证 reward 的 RLVR 任务；
3. 是否让 agent 动态 function call，而不是固定 RAG；
4. 是否用未来真实相对收益设计 reward；
5. 是否能在 valid/test 上证明 reward 和预测指标同步提升。
```

第一版最小闭环：

```text
Qwen3-4B-Instruct
+ 4 个工具：price / announcement / news / industry
+ T+5 横截面分位 label
+ SFT 冷启动
+ veRL GRPO
+ reward = direction + prob + brier + pnl + cost + format
+ evaluation = valid reward + Brier + Rank IC + top-bottom return
```

只要这个闭环跑通，后续再逐步增加工具、优化 reward、做多周期、多市场、多策略融合。

---

## 25. 参考资料

- veRL documentation: GRPO, PPO, multi-turn rollout, custom reward, function tool.
- Qwen3 model documentation and chat template documentation.
- 金融时间序列建模中的 point-in-time、survivorship bias、lookahead bias 处理原则。
- 常见量化评估指标：IC、Rank IC、ICIR、分组收益、Sharpe、最大回撤、换手率。
