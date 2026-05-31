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
export TUSHARE_TOKEN="your_token"

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-4B --local-dir model/Qwen3-4B

python3 stock_agent_rl_mvp.py --mode all
LATEST_RUN=$(ls -td result/*_result | head -1)
bash "$LATEST_RUN/run_verl_stock_grpo.sh"
```

For the custom Tushare endpoint, the script sets:

```python
pro._DataApi__http_url = "http://lianghua.nanyangqiankun.top"
```

Do not commit `data/`, `model/`, or `result/` contents.
