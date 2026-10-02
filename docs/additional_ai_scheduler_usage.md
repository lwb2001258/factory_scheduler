# 其它 AI 调度算法使用说明

## 已实现算法

- `GraphPPO`：共享二部图边策略，PPO/GAE 训练，Hungarian 匹配解码。
- `RainbowDQN`：Dueling、Double-DQN、PER、n-step、C51。
- `QRDQN`：固定分位数分布式价值，默认使用较低回报分位数做风险敏感决策。
- `CQL`：从固定 `MaskedCostGreedy` 轨迹数据训练的离散 Conservative Q-Learning。
- `LinUCB`：在 FCFS、NearestNeighbour、Greedy、Hungarian、Auction 之间选择的上下文 Bandit。

这些算法只位于任务分配层。路径规划、联合预留、避让、DWA、停车距离、路权和机器人控制代码没有改动。

## 训练与离线门禁

快速冒烟：

```powershell
python scripts/run_advanced_ai_workflow.py `
  --train-seeds 9200 9201 `
  --validation-seeds 109200 `
  --max-steps 6 `
  --offline-updates 3 `
  --output-dir results/advanced_ai_smoke `
  --fail-on-gate
```

正式研究训练使用默认的 32 个训练 seed 和 8 个不相交验证 seed：

```powershell
python scripts/run_advanced_ai_workflow.py `
  --output-dir results/advanced_ai_scheduler `
  --fail-on-gate
```

输出包括 `candidates/*.npz` 和 `workflow_report.json`。通过该 workflow 只表示 checkpoint、合法性、可复现性和 headless Webots 业务逻辑门禁通过，**不会自动生产晋级**。

## 与 Webots 的一致范围

训练环境现在按 16 ms 步长推进，并复用项目的 `MotionCoordinator`，执行与 Supervisor 相同的任务提交、取货、送货、低电量门禁、到站五秒换电、失败配对暂时屏蔽、路径预约和死锁监测语义。报告固定标记为 `headless_webots_logic` / `business_logic_only`。

它不包含 Webots 物理引擎、轮速动力学、距离传感器、机器人本地 DWA、无线通信失败和真实碰撞接触。因此 headless 训练适合预训练和回归，训练后的模型仍须在未见 seed 的 Webots 运行中微调或做配对验收。

项目中的 10 个 AI 调度器都可通过统一的 `step_scheduler` 接入该运行时：`LearnedHungarian`、`GraphImitation`、`PPO_RL`、`SARSA`、`DQN`、`GraphPPO`、`RainbowDQN`、`QRDQN`、`CQL`、`LinUCB`。它们共享执行环境，但学习方法并不相同：前两者使用监督/模仿数据，`CQL` 使用环境采集的离线数据，`LinUCB` 使用 bandit 反馈，其余算法使用在线运行时 transition/reward。`workflow_report.json` 的 `ai_training_capabilities` 会记录每种算法的实际训练入口。

## Standalone 配对实验

```powershell
python scripts/run_experiments.py --standalone `
  --scenario C --seed-values 110001 110002 `
  --scheduler Hungarian GraphPPO RainbowDQN QRDQN CQL LinUCB `
  --graph-ppo-model results/advanced_ai_scheduler/candidates/graph_ppo.npz `
  --rainbow-dqn-model results/advanced_ai_scheduler/candidates/rainbow_dqn.npz `
  --qrdqn-model results/advanced_ai_scheduler/candidates/qrdqn.npz `
  --cql-model results/advanced_ai_scheduler/candidates/cql.npz `
  --linucb-model results/advanced_ai_scheduler/candidates/linucb.npz `
  --results-dir results/advanced_ai_standalone
```

Standalone 只用于算法回归，不能形成上线结论。上线前必须换用未见 seed 在真实 Webots 控制循环中与 Hungarian 配对运行，再用 `evaluate_scheduler_results.py` 检查吞吐、平均/P95 完成时间、安全事件、距离违规、非法输出、fallback 和调度 P95 延迟。

任何模型缺失、损坏、版本/维度不兼容、产生 NaN/Inf、返回非法动作/匹配或推理超时，调度器工厂都会保留或切换到 Hungarian。

两套训练目标以及 3/5/8 机器人统一评估方案见
[`dual_training_evaluation_design.md`](dual_training_evaluation_design.md)。
