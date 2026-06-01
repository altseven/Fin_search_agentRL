# Fin_search_agentRL

Single-file MVP for a stock search-agent RLVR experiment with Tushare data and verl.

## Directory Layout

- `stock_agent_rl_mvp.py`: all-in-one data builder, tool definitions, reward function, rule baseline, and verl command generator.
- `data/`: local downloaded market data cache. Ignored by git.
- `model/`: local model weights, for example `model/Qwen3-4B`. Ignored by git.
- `result/`: per-run outputs. Each run creates `HHMMSS_MMDD_result/`. Ignored by git except `.gitkeep`.
- `verl-main/`: local verl framework copy used by the generated training script.

## Quick Start

On a new GPU server, first create the Python/verl environment:

```bash
bash setup_stockverl_env.sh
```

If normal PyPI is slow, use the China mirror for regular packages:

```bash
bash setup_stockverl_env.sh --cn-mirror
```

Optional model download during setup:

```bash
bash setup_stockverl_env.sh --cn-mirror --download-model
```

For a one-card A100 LoRA run, use the wrapper script:

```bash
bash run_a100_1gpu.sh "your_token"
```

Clone 后最省事的一条命令如下，不需要 `export`，不需要自己找最新 result 目录，模型也会自动下载到 `model/Qwen3-4B`：

```bash
python3 stock_agent_rl_mvp.py --mode all-train --tushare-token "your_token"
```

如果不想每次在命令里传 token，可以用本地配置文件：

```bash
cp local_config.example.py local_config.py
```

Open `local_config.py` and paste your Tushare token. Then run:

```bash
python3 stock_agent_rl_mvp.py --mode all-train
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
