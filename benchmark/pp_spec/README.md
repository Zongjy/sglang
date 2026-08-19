# PP + Speculative Decoding 调优工具

这里提供 Qwen3.6-27B + DFlash 的 PP/TP profiling、离线分析和闭环切分调优。
`data/`、`logs/`、`traces/`、`results/` 和 `tuning_runs/` 都是本地产物。

## 自适应 PP 切分

完整调优只要求提供 PP size 和 TP size：

```bash
cd benchmark/pp_spec

# 当前单机双卡：每个 PP stage 使用一张卡
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1

# 未来单机八卡：4 个 PP stage，每个 stage 内 TP2
python adaptive_pp_tuner.py --pp-size 4 --tp-size 2
```

GPU 数量必须满足 `PP_SIZE * TP_SIZE`。未设置 `CUDA_VISIBLE_DEVICES` 时，工具会读取
`nvidia-smi topo -m`，优先把 TP rank 放在高速互联的一组卡上；显式设置后严格保留
用户给出的设备顺序：

```bash
CUDA_VISIBLE_DEVICES=0,1 python adaptive_pp_tuner.py --pp-size 2 --tp-size 1
```

执行流程如下：

1. 从本地 Hugging Face 配置自动读取层数和 layer type，优先使用本地 cache。
2. 按 SGLang 默认规则启动等分基线，自动施加稳态 decode load 并抓 50 step trace。
3. 对每个 PP stage 采用最慢的 TP lane，分离 target 层时间、draft/head/result/P2P
   等固定开销，并从启动日志估算每个 stage 的保守显存层数上界。
4. 动态规划求解最小化最慢 stage 的连续 layer partition，并生成 3 个邻近候选。
5. 固定 server random seed，在全新进程中分别短测基线和候选各 2 次；候选至少提升
   1% 才替换基线。启动失败或 OOM 的候选会被记录并跳过。

每次运行写入 `tuning_runs/<timestamp>_tpX_ppY/`：

| 文件 | 内容 |
|---|---|
| `result.json` | 硬件、参数、显存约束、trace 分析、所有候选实测和最终选择 |
| `best_config.json` | 最优 partition、GPU 映射和完整 server command |
| `best_config.env` | `CUDA_VISIBLE_DEVICES` 与 `SGLANG_PP_LAYER_PARTITION` |
| `launch_best.sh` | 可直接启动最优配置的脚本 |
| `baseline/traces/` | 各 rank 原始 trace 和合并后的 `lean.json.gz` |
| `baseline/analysis.txt` | 可读的 stage 测量与切分预测 |
| `baseline/`、`validation_baseline/`、`candidate_p*/` | profile、server 与 benchmark 日志、指标明细 |

调优结束后直接启动：

```bash
bash tuning_runs/<run>/launch_best.sh
```

### 当前默认参数

| 参数 | 默认值 |
|---|---|
| target / draft | `Qwen/Qwen3.6-27B` / `z-lab/Qwen3.6-27B-DFlash` |
| DFlash block size | 8 |
| static memory fraction | 0.82 |
| decode CUDA graph max batch | 32 |
| Mamba state | BF16，full-memory-ratio 2.0 |
| server random seed | 1；所有 partition 完全一致 |
| offered profile/validation concurrency | 32；可用 `--concurrency` 覆盖 |
| profile | 50 steps，256-token synthetic prompt，4096-token decode request |
| validation | 重复 2 次；每次 `max(16, 2 * concurrency)` 个请求，固定输出 256 token |
| memory reserve | 每个 rank 保留 2 GiB |

所有默认值都可通过 `python adaptive_pp_tuner.py --help` 覆盖。额外的 SGLang 参数放在
`--server-args` 后面，并且该选项必须位于命令最后。

`--concurrency` 固定客户端同时在途的 offered load。默认情况下，工具先读取均匀
基线实际解析出的 `max_running_requests`，然后取
`min(concurrency, baseline cap)` 作为共同 active request 数，重新启动基线和所有
候选做同负载验证。因此只提供 PP/TP size 时，切分收益不会混入 Mamba 容量差异。
也可以显式覆盖共同 active request 数：

```bash
# C32 持续供给，但每个候选都只允许 8 个请求进入 running batch
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1 \
  --concurrency 32 --fixed-active-requests 8
```

如果任一候选受 KV/Mamba 显存限制而达不到指定 active request 数，调优会明确失败，
不会悄悄用更小的 batch 继续比较。

若目标是线上最大吞吐，希望把不同切分带来的容量变化也计入收益，可以显式使用：

```bash
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1 \
  --concurrency 32 --allow-variable-active-requests
```

上述默认模式固定请求压力、server active request 数、prompt、输出长度和随机种子，但
保留真实 DFlash acceptance。若要做更严格的逐 step 计算对比，还可以在实验环境中设置
`SGLANG_SIMULATE_ACC_LEN=<value>` 固定 acceptance；这种结果只用于微架构归因，不应
替代真实 acceptance 下的最终吞吐验证。

### 辅助模式

```bash
# 只检查模型、GPU 数量、rank 映射和最终 server command，不启动服务
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1 --dry-run --offline

# 抓基线 trace 并给出预测，不重新加载候选做实测
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1 --skip-validation

# 对已有原始 trace 做纯离线分析；若 trace 来自非均匀切分需说明当前 partition
python adaptive_pp_tuner.py --pp-size 2 --tp-size 1 \
  --analyze-traces traces/pp2_dflash_b8_c8 --current-partition 32,32 --offline
```

## 单独使用分析器

`auto_pp_partition.py` 不启动 GPU 服务，可直接读取一个完整 trace 目录。混合 TPxPP
目录可以包含全部 rank，分析器会自动选择每个 PP stage 的最慢 TP lane。

```bash
python auto_pp_partition.py traces/pp2_dflash_b8_c8 --num-layers 64

python auto_pp_partition.py traces/pp2_dflash_b8_c8_p38-26 \
  --num-layers 64 --current-partition 38,26

python auto_pp_partition.py traces/pp2_dflash_b8_c8 --num-layers 64 \
  --max-layers-per-stage 40,34
```

分析器优先使用嵌套 `step[...]` 数量区分 target verify 和 DFlash draft；旧 trace
缺少嵌套 annotation 时才回退到相对时长。普通 CUDA graph trace 没有逐层 NVTX
边界时，会使用从非末 stage 校准的均匀单层成本，因此完整调优器始终用无 profiler
benchmark 对预测候选做最后确认。

## 手动抓 trace 与查看

`capture_profile.sh` 是低层抓取工具，支持任意单机 `TP_SIZE * PP_SIZE`，只管理自己
启动的进程组，不会 `pkill` 其他 SGLang server：

```bash
# 兼容旧用法：TOPO=pp 等价于 TP_SIZE=1 PP_SIZE=2
TOPO=pp BLOCK=8 CONC=32 bash capture_profile.sh

# 任意 hybrid 组合
TP_SIZE=2 PP_SIZE=4 CONC=32 bash capture_profile.sh

# 手动验证指定切分
TP_SIZE=1 PP_SIZE=2 PP_PARTITION=38,26 bash capture_profile.sh

# 固定 server active batch，便于不同切分做同负载 trace 对比
TP_SIZE=1 PP_SIZE=2 PP_PARTITION=39,25 CONC=32 \
  MAX_RUNNING_REQUESTS=8 bash capture_profile.sh
```

将 `lean.json.gz` 拖入 [Perfetto](https://ui.perfetto.dev) 查看时间线。新版 lean trace
会保留 PP/scheduler CPU 控制事件和阻塞 CUDA sync，并在 step 之间生成
`INTER_STEP_GAP`，其中包含该区间的 GPU busy/idle 比例、通信、结果处理和调度归因。
也可以运行：

```bash
python trace_summary.py traces/<label>/lean.json.gz
```

其他文件：`bench_spectre.py` 是在线压测客户端；`slim_trace.py` 负责裁剪和合并
Chrome trace；`merge_traces.py` 做无裁剪合并；`trace_summary.py` 输出 GPU busy、
NCCL 占比、阶段跨度和 top kernels。

## 注意事项

- 一次完整运行会启动 1 次 profile 基线、1 次干净的 validation 基线和最多 3 次
  候选，模型加载和 CUDA graph capture 是主要耗时；中断后已完成的阶段仍保留在
  `result.json` 和对应日志中。
- torch profiler 会抬高绝对时长，切分只使用相对 stage 结构，最终选择来自无
  profiler benchmark。
- hybrid GDN + DFlash 的有效并发可能受 Mamba state cache 限制。队列仍能维持
  饱和负载，实际 cap 会出现在 server log 中。
- 自动显存约束是基于基线启动日志的保守估计，候选启动实测仍是最终的 OOM 判定。
