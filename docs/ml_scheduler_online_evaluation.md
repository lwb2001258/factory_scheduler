# 学习型调度器 Webots 上线前评估

## 结论（2026-10-01）

本轮使用真实 Webots 物理控制循环、5 个训练和验证均未使用的新 seed，完成了
15 次配对实验。按预先设定的上线门槛：

- **GraphImitation 通过全部门槛**，配对平均吞吐提升 3.04%，部署清单选择
  `GraphImitation`；Hungarian 继续作为运行时安全回退。
- **LearnedHungarian 被拒绝**，配对平均吞吐下降 44.16%，平均完成时间增加
  27.47%，且最差调度延迟超过相对预算。

机器可读结论位于 `results/ml_scheduler/online_evaluation.json`，当前选择位于
`results/ml_scheduler/deployment_manifest.json`。`promoted/` 只表示模型通过离线
训练门禁，最终能否使用以 deployment manifest 为准。

GraphImitation 的吞吐 bootstrap 95% 区间仍跨过 0（-8.65%～+14.73%），5 个
seed 中 2 胜、1 平、2 负。因此“通过”表示满足当前预设的 2% 平均优势规则，
不是统计显著性结论。用于长期生产前应继续扩大独立 seed 和长时间运行样本；在此
之前必须保留 Hungarian 回退与在线监控。

## 实验设计

- 运行时：Webots R2023b，`runtime_mode=webots`，外部 TCP 控制器模式。
- 场景：C，8 个机器人，高任务到达率。
- 仿真时间：每次 300 秒。
- 训练/验证 seed：7001、7002、7003。
- 独立评估 seed：7201、7202、7203、7204、7205。
- 调度器：Hungarian、LearnedHungarian、GraphImitation，共 15 次实验。
- 配对原则：同一 seed、同一场景、同一机器人数量、同一 Webots 控制逻辑。
- 安全原则：候选模型只改变匹配评分；路径规划、避碰、充电、合法性校验和
  Hungarian 回退保持不变。

## Webots 配对结果

| 候选算法 | 吞吐变化 | 平均完成时间变化 | P95 完成时间变化 | 平均等待变化 | 安全/距离违规 | 最大调度 P95 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| LearnedHungarian | -44.16% | +27.47% | +12.76% | -43.78% | 0 / 0 | 104.61 ms | 拒绝 |
| GraphImitation | +3.04% | +2.22% | +0.52% | -2.56% | 0 / 0 | 74.11 ms | 通过，选择为候选 |

延迟门槛取 `max(20 ms, 基线同批次最大 P95 × 1.20)`。本批次 Hungarian 最大
P95 为 69.07 ms，允许上限为 82.89 ms。这个规则把 Webots 中双方共同承担的 A*
路径成本纳入基线，同时仍阻止候选引入超过 20% 的额外尾延迟。

GraphImitation 的逐 seed 吞吐（任务/分钟）为：

| Seed | Hungarian | GraphImitation | LearnedHungarian |
|---:|---:|---:|---:|
| 7201 | 2.60 | 2.20 | 1.00 |
| 7202 | 3.20 | 3.00 | 2.20 |
| 7203 | 2.80 | 3.40 | 1.80 |
| 7204 | 2.60 | 2.60 | 1.40 |
| 7205 | 2.60 | 3.00 | 1.40 |

## 同 seed 任务一致性

对五个 seed 的 supervisor 原始日志逐条比较后，三种调度器的每个 task id 的
生成时刻、取货点和送货点完全一致；每个 seed 的任务数量分别是 38、47、42、
33、40。任务优先级也由同一个 `TaskGenerator` 随机流生成，与调度器和机器人
运行状态无关，并有自动化测试覆盖完整任务流复现和 5%/20%/75% 分布边界。

因此，在机器人数量相同、场景相同、seed 相同且任务生成器调用节奏不变时，
相同 task id 的任务定义一致。不同调度算法只会改变分配、等待、完成顺序等运行
结果，不会改变任务本身。

## 运行速度说明

Webots 以 `--batch --mode=fast --no-rendering` 运行。短场景验证中，30 仿真秒约
需 27 秒墙钟时间（含约 3 秒启动）；300 秒压力场景的主要耗时不是渲染或模型
推理，而是 8 机器人拥堵时的 A* 与联合避碰重规划。三个调度器可以使用独立 TCP
端口并行运行，从而缩短整批等待时间，但不能增大 world 的 `basicTimeStep` 或跳过
控制器，否则训练/评估逻辑就不再与实际 Webots 一致。

## 复现命令

```powershell
$env:SMART_FACTORY_SIM_DURATION = "300"
$env:SMART_FACTORY_ROBOT_QUIET = "1"
$env:SMART_FACTORY_SUPERVISOR_QUIET = "1"

python scripts/run_experiments.py `
  --scenario C `
  --scheduler Hungarian LearnedHungarian GraphImitation `
  --seed-values 7201 7202 7203 7204 7205 `
  --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --controller-mode auto `
  --learned-cost-model results/ml_scheduler/promoted/learned_cost.npz `
  --graph-imitation-model results/ml_scheduler/promoted/graph_imitation.npz `
  --max-parallel 3 `
  --results-dir results/webots_evaluation_7201_7205

python scripts/evaluate_scheduler_results.py `
  --results-glob "results/webots_evaluation_7201_7205/**/experiment_*.json" `
  --scenario C --baseline Hungarian `
  --candidate LearnedHungarian GraphImitation `
  --required-runtime-mode webots `
  --deployment-manifest results/ml_scheduler/deployment_manifest.json `
  --output results/ml_scheduler/online_evaluation.json
```

## 历史结果边界

旧 seed 6001–6005 的结果来自 `runtime_mode=standalone`，只能用于快速算法回归，
不能作为 Webots 物理运行或生产晋级证据，已被生产评估器的 runtime 门禁排除。
