# Fin_search_agentRL

MVP for a stock search-agent RLVR experiment with Tushare data and verl.

## Directory Layout

- `stock_agent_rl_mvp.py`: CLI/main entrypoint.
- `rl_common.py`: shared constants, local config, table cache helpers, and path helpers.
- `rl_config.py`: runtime config dataclass.
- `rl_data.py`: Tushare download, factor snapshots, task and label construction.
- `rl_tools.py`: verl function-call tools for price factors, market context, and announcements.
- `rl_reward.py`: standalone verl reward function file containing `compute_score`.
- `rl_dataset.py`: prompt building and verl parquet export.
- `rl_baseline.py`: local rule-agent baseline and metrics.
- `rl_verl.py`: model download helpers and generated verl command.
- `data/`: local downloaded market data cache. Ignored by git.
- `model/`: local model weights, for example `model/Qwen3-4B`. Ignored by git.
- `result/`: per-run outputs. Each run creates `HHMMSS_MMDD_result/`. Ignored by git except `.gitkeep`.
- `verl-main/`: local verl framework copy used by the generated training script.

## Quick Start

Clone 到 8 卡 A800 服务器后，建议放到你指定的目录：

```bash
mkdir -p /kunlun_data/temp_ag_rl
cd /kunlun_data/temp_ag_rl
git clone https://github.com/altseven/Fin_search_agentRL.git
cd /kunlun_data/temp_ag_rl/Fin_search_agentRL
```

然后直接执行统一入口：

```bash
bash run_stock_agent_rl.sh "your_token"
```

它会自动创建/复用 `stockverl` conda 环境、安装依赖、下载 `Qwen/Qwen3-4B`、检测 8 张 70GB+ GPU，然后启动全参数 GRPO 训练。默认 profile 是 8 卡 A800：`train_batch_size=64`、`rollout_n=4`、`rollout_tp=4`、`max_prompt_length=3072`、`max_response_length=1024`，LoRA 默认关闭。

统一入口会把所有 stdout/stderr 同步写到项目根目录的 `output.log`，每次运行都会覆盖旧日志。云服务器上任务结束后可以直接查看：

```bash
cd /kunlun_data/temp_ag_rl/Fin_search_agentRL
tail -n 200 output.log
```

如果已经装好环境和模型，只想直接跑训练：

```bash
bash run_stock_agent_rl.sh "your_token" --no-setup
```

如果不想每次在命令里传 token，可以用本地配置文件：

```bash
cp local_config.example.py local_config.py
```

Open `local_config.py` and paste your Tushare token. Then run:

```bash
bash run_stock_agent_rl.sh
```

If you only want to set up the environment without training:

```bash
bash setup_stockverl_env.sh --cn-mirror --download-model
```

CUDA 13.0 driver 下脚本默认安装 PyTorch `cu128` wheel，这是正常的；NVIDIA driver 向下兼容 CUDA 12.8 runtime。

Advanced direct Python entry:

```bash
python3 stock_agent_rl_mvp.py --mode all-train --tushare-token "your_token"
```

If you only want to build data and generate the training script:

```bash
python3 stock_agent_rl_mvp.py --mode all --tushare-token "your_token"
```

If training was interrupted and you want to run the latest generated training script:

```bash
python3 stock_agent_rl_mvp.py --mode train-latest
```

To download Qwen3-4B manually instead of auto-downloading:

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-4B --local_dir model/Qwen3-4B
```

Then disable auto-download:

```bash
python3 stock_agent_rl_mvp.py --mode all-train --no-auto-download-model --tushare-token "your_token"
```

For the custom Tushare endpoint, the script sets:

```python
pro._DataApi__http_url = "http://lianghua.nanyangqiankun.top"
```

Do not commit `data/`, `model/`, or `result/` contents.
