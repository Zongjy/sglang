# PP+Spec 研究实验工具

围绕 Qwen3.6-27B + DFlash 在 TP/PP 两种并行下的 speculative decoding 对比实验。
脚本入库；`data/ logs/ traces/ results/` 为本地产物，已被 .gitignore 排除。

## 文件

| 文件 | 用途 |
|---|---|
| `bench_spectre.py` | SPECTRE 方式在线压测客户端：ShareGPT prompt 定长截断 + Poisson 开环到达 + max-concurrency FIFO 门控，1024 定长输出、greedy；输出 throughput / TTFT / TPOT / accept_length |
| `capture_profile.sh` | 一键抓 torch profiler trace：起 server（`TOPO=tp\|pp`、`BLOCK`、`CONC` 可调）→ 压稳态 decode → `/start_profile` 抓 N 步 → 自动跑 `slim_trace.py` 产出 lean 版 |
| `slim_trace.py` | trace 后处理：过滤 python 栈/cpu op/runtime 噪音（~5x 缩小）、缩短 kernel 名、按 baseTimeNanoseconds 合并对齐多 rank、按 run_batch 聚类合成 DRAFT/VERIFY/POST 阶段色块 |
| `merge_traces.py` | 多 rank 原始 trace 简单合并（对齐但不裁剪） |
| `trace_summary.py` | 高层摘要（不看图）：GPU busy%、NCCL 占比、阶段 span 均值、top kernels |

## 快速开始

```bash
# 数据集（一次性，~640MB）
mkdir -p data && curl -L -o data/sharegpt.json \
  https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 压测（server 已在跑时）
python3 bench_spectre.py --url http://127.0.0.1:30011 --label tp2_dflash

# 抓 trace（自动起停 server）
TOPO=pp BLOCK=8 CONC=4 bash capture_profile.sh

# 看图：把 traces/<label>/lean.json.gz 拖进 https://ui.perfetto.dev
# 高层摘要：
python3 trace_summary.py traces/<label>/lean.json.gz
```

## 已知约束

- hybrid GDN 模型 + DFlash 的 mamba per-step 中间态随 block size 线性膨胀，
  2x48G 上有效并发上限约为：b16→TP 8 / PP 4，b8→TP 14 / PP 8（显式
  `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 2.0` 缓解）
- 被 profile 的步有 torch profiler 开销，绝对时长偏大，看结构比例
