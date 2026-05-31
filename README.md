# Fin_search_agentRL

Single-file MVP for a stock search-agent RLVR experiment with Tushare data and verl.

## Directory Layout

- `stock_agent_rl_mvp.py`: all-in-one data builder, tool definitions, reward function, rule baseline, and verl command generator.
- `data/`: local downloaded market data cache. Ignored by git.
- `model/`: local model weights, for example `model/Qwen3-4B`. Ignored by git.
- `result/`: per-run outputs. Each run creates `HHMMSS_MMDD_result/`. Ignored by git except `.gitkeep`.
- `verl-main/`: local verl framework copy used by the generated training script.

## Quick Start

```bash
cp local_config.example.py local_config.py
```

Open `local_config.py` and paste your Tushare token.

Download Qwen3-4B into `model/`:

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-4B --local_dir model/Qwen3-4B
```

Run the whole MVP and start verl training:

```bash
python3 stock_agent_rl_mvp.py --mode all-train
```

If you only want to build data and generate the training script:

```bash
python3 stock_agent_rl_mvp.py --mode all
```

If training was interrupted and you want to run the latest generated training script:

```bash
python3 stock_agent_rl_mvp.py --mode train-latest
```

You can also pass the Tushare token directly without editing `local_config.py`:

```bash
python3 stock_agent_rl_mvp.py --mode all-train --tushare-token "your_token"
```

For the custom Tushare endpoint, the script sets:

```python
pro._DataApi__http_url = "http://lianghua.nanyangqiankun.top"
```

Do not commit `data/`, `model/`, or `result/` contents.
