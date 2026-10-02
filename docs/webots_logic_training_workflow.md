# Webots 逻辑一致训练环境工作流

## 目标与真实性边界

在 `feature/ai-schedulers` worktree 中，把 AI 训练从“静态路径长度除以固定速度后跳到完成事件”升级为无 Webots 依赖的逐步运行时。该运行时必须与当前 `FactorySupervisor` 的可复用业务语义一致：任务提交后再执行、取货后规划送货、逐步运动、电池阈值、到站后五秒换电、失败配对暂时屏蔽、lifelong 路径预约、拥堵上下文和死锁监测。

“一致”不包括无法脱离 Webots 复现的物理引擎、轮速动力学、距离传感器、本地 DWA、无线通信失败和真实碰撞接触。纯代码运行时必须在元数据和文档中明确标记为 `headless_webots_logic`，不得把结果描述成 Webots 结果。

不可变边界：

- AI 只决定任务分配或选择既有调度器，不控制速度、路径、避让、让行、停车或恢复。
- 不修改 `factory_supervisor.py`、`config.py`、`robot_controller.py`、`grid_planner.py`、`joint_grid_planner.py`、`motion_coordinator.py`、`safety_coordination.py` 和 `collision_safety.py`。
- 训练环境调用现有 `MotionCoordinator`，但不得改变其路径规划和避让规则。
- 模型输出仍经过 action mask、`validate_assignment` 和现有部署安全回退。
- 纯代码训练/验证不能替代未见 seed 的真实 Webots 配对评测。

## 算法接入矩阵

| 算法 | 环境接入 | 学习方式 |
|---|---|---|
| SARSA、DQN、PPO_RL | 固定观察/机器人任务动作 | 在线 RL；PPO_RL 需保留其独立网络语义 |
| RainbowDQN、QRDQN、GraphPPO | 固定动作或图边动作 | 在线 RL |
| CQL | 环境采集带 mask 的固定数据集 | 离线 RL |
| LearnedHungarian、GraphImitation | `BaseScheduler.assign` 适配器 | 监督/模仿数据，不做伪 RL 更新 |
| LinUCB | `BaseScheduler.assign` 适配器 | 上下文 bandit 完成反馈 |

全部十个 AI 调度器都必须能够在同一个运行时执行和评估；“能够接入”不等于“使用相同训练公式”。

## 实施步骤

### 步骤 1：Webots 业务逻辑兼容运行时

- 新增独立 headless runtime，持有机器人状态、任务、waypoint、失败配对和 `MotionCoordinator`。
- 使用 `TIMESTEP / 1000` 推进；按当前 Supervisor 语义实现任务提交、取货、送货、低电量、充电站选择和五秒换电。
- 路径先走 `plan_grid_lifelong`，失败再走 `plan_path_for_robot`；动态失败不虚构成功。
- 输出完成、拒绝、重规划、死锁和模拟步数遥测。

完成条件：状态转换、电池、换电、不可达回滚、并发机器人和确定性均有定向测试。

### 步骤 2：训练环境接入与奖励

- `SchedulingEnvironment` 的训练模式改用 headless runtime；保留 `abstract` 作为兼容别名，但不再使用旧的直接完成模型。
- observation/action 维度和部署端 `webots` 快照接口保持不变。
- reward 使用真实 runtime 事件：提交、完成、拒绝、实际路程、等待和协调事件；不得奖励未来任务。
- 训练报告记录 dynamics 版本及真实性限制。

完成条件：现有高级 RL 短训练可复现，未来任务不可见，transition 的下一状态/mask 合法。

### 步骤 3：十个 AI 调度器统一接入

- 提供 `step_scheduler(BaseScheduler)`，将匹配型/元调度器的首个合法提交映射到统一运行时。
- 建立显式能力表，区分在线 RL、离线 RL、监督/模仿和 bandit。
- 为现有 SARSA、DQN、PPO_RL 和新增五种算法核对观察、动作或 scheduler 接口；不得以名称存在代替实际可执行。

完成条件：十个算法都有明确接入路径；动作型和 scheduler 型适配器均通过合法性、失败和终止测试。

### 步骤 4：Workflow、报告与回归

- 高级训练 workflow 默认使用 headless Webots-logic runtime，并在 JSON 报告中声明 fidelity 和限制。
- 增加训练冒烟、统一 scheduler 接入冒烟、严格 JSON 和安全文件哈希检查。
- 更新使用文档，不把 headless 结果表述成生产/Webots 结论。

完成条件：全量单元测试、训练 workflow 冒烟、CLI、确定性检查和安全文件逐字节哈希均通过。

## 每步三轮 Code Review

每一步完成后执行三轮审查；发现问题则修复并从该步骤第一轮重新开始：

1. Code Review A：状态机/算法、时间与单位、维度、确定性和异常路径。
2. Code Review B：低电量、不可达、重复任务、未来信息、NaN/Inf、安全边界和受保护文件哈希。
3. Code Review C：完整 diff、所有算法接口、训练/部署兼容、全量回归和文档真实性。

结果写入 `docs/webots_logic_training_review_log.md`。

## 两轮 Workflow Review

1. 编码前：审查范围、真实性边界、步骤依赖、算法接入分类和不可变安全文件。
2. 完工后：核对四步十二轮 code review、运行时证据、十算法接入、训练冒烟、全量回归、安全文件哈希和未完成的真实 Webots 门禁。
