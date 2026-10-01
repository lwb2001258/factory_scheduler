# 其它 AI 调度算法实施工作流

## 目标和不可变边界

本工作流在 `feature/ai-schedulers` worktree 中实现研究文档里适合当前“集中式、离散机器人—任务分配”架构的其它 AI 算法，同时保持下列边界不变：

- AI 只决定任务分配或选择既有调度器，不决定速度、路径、让行、停车、恢复或机器人控制。
- 所有候选分配继续经过现有成本矩阵、可行性检查、`validate_assignment(s)`、超时门禁和 Hungarian 回退。
- 不修改 `factory_supervisor.py`、`config.py`、`robot_controller.py`、`grid_planner.py`、`joint_grid_planner.py`、`motion_coordinator.py`、`safety_coordination.py`、`collision_safety.py` 及避让距离/优先权常量。
- 没有 checkpoint、checkpoint 不兼容、输出含 NaN/Inf、无合法动作、推理异常或超时时，拒绝 AI 输出并回退到 Hungarian。
- 新算法默认不替换生产基线；只有离线门禁、未见 seed 配对 Webots 评测和安全指标均通过后才可晋级。

## 实施范围

结合当前依赖仅有 NumPy/SciPy、动作空间为 161 个离散动作、Supervisor 集中决策的现实，实施以下算法：

1. **GraphPPO**：二部图边特征共享编码，PPO actor/critic 训练；部署时用 Hungarian 将边分数解码成合法的一对一匹配。
2. **RainbowDQN**：Dueling 网络、Double-DQN、优先经验回放、n-step 回报和 C51 分布式价值。
3. **QRDQN**：固定分位数分布式价值网络，用于尾部风险目标；这是调研中 IQN/QR-DQN 路线里依赖更轻、较容易严格验证的实现。
4. **CQL**：离散动作的离线 Conservative Q-Learning，只从固定数据集训练，不在 Webots 生产运行中探索。
5. **LinUCB**：在既有 FCFS、NearestNeighbour、Greedy、Hungarian、Auction 之间选择的上下文 Bandit 元调度器。

暂不实现 MAPPO、QMIX、CPO 和分层 RL。前三者需要分散执行或新的约束训练接口，分层 RL 会扩展到充电/路径协调；在当前项目中强行接入会跨越本任务“不得改变避让规则”的边界。SAC/MADDPG 与当前离散匹配动作不匹配，也不纳入本轮。

## 步骤和完成条件

### 步骤 1：统一模型契约与安全适配层

- 定义严格、带算法名/环境版本/维度的 `.npz` checkpoint 契约；加载时固定 `allow_pickle=False`。
- 提供 masked action、有限值、重复分配和推理预算检查。
- 让新增 RL 策略复用现有 `SchedulingEnvironment` 和 `RLSchedulerSafetyWrapper`。

完成条件：坏 checkpoint、错误算法、错误维度、NaN/Inf、空合法集均有测试，Hungarian 回退保持可用。

### 步骤 2：RainbowDQN、QRDQN 与 CQL

- 实现三个独立 agent，不能把普通 DQN 改名冒充。
- 实现固定 seed 可复现的抽象环境训练与 checkpoint 往返。
- CQL 数据采集与训练分离；数据必须包含 state、当前 action mask、action、reward、next state、next action mask、done、行为策略、seed 和环境版本。

完成条件：网络形状、mask、目标分布/分位数、PER、n-step、保守正则和保存加载都有定向测试；短训练冒烟测试输出有限损失。

### 步骤 3：GraphPPO + 匹配解码

- 复用 `learning_scheduler.py` 的版本化二部图边特征。
- 共享边编码器生成策略分数，图池化生成 value；实现 clipped PPO 和 GAE。
- 训练端按同一组边 logits 采样单个合法边并与抽象环境交互；部署端只把同一 logits 交给 Hungarian 批量一对一解码，再走共享合法性校验。

完成条件：机器人/任务顺序置换不改变语义；没有重复机器人/任务；checkpoint 错误时回退；短 PPO 训练可复现且输出有限。

### 步骤 4：LinUCB 元调度器

- 上下文只使用决策时刻可见的负载、电量、优先级、等待和拥堵信息。
- 每个 arm 调用未经修改的既有调度器；LinUCB 不重写 arm 的分配结果。
- 离线训练标签以同一快照上各 arm 的合法匹配总成本归一化负值为收益；非法/空匹配使用固定惩罚，不虚构任务完成反馈。
- 训练/更新记录按运行 seed 分组，避免同一仿真泄漏到训练和验证两侧。

完成条件：选择确定、矩阵可逆性稳定、坏模型回退、arm 异常隔离以及 checkpoint 往返都有测试。

### 步骤 5：训练 workflow、实验入口和文档

- 提供训练命令，产出候选 checkpoint、验证指标和机器可读报告。
- 实验入口接受每种算法独立模型路径，不复用错误 checkpoint。
- 报告记录 train/validation seed、算法配置、非法输出、fallback、推理 P95 和是否允许晋级。

完成条件：一条命令可执行小规模训练/验证；所有新增算法能被实验入口选择；全量测试通过。

## 每一步强制三轮 Code Review

每个步骤完成后必须依次执行三轮 review；发现问题则修复并从该步骤第一轮重新开始：

1. **Code Review A — 算法与实现**：核对公式、梯度/目标、维度、数值稳定性、确定性、checkpoint 兼容性和可维护性，并运行定向测试。
2. **Code Review B — 边界与安全**：核对 action mask、不可达边、空输入、重复 ID、低电量、异常/超时、NaN/Inf、模型缺失及 Hungarian 回退；确认未修改避让/防碰撞文件，并运行安全边界测试。
3. **Code Review C — 集成与回归**：检查该步骤完整 diff、CLI/工厂/指标兼容性、旧算法行为和全项目回归测试。

每轮结果记录在 `docs/additional_ai_scheduler_review_log.md`，包含审查范围、发现、修复、命令和结果。

## Workflow Review（必须两次）

### Workflow Review 1：编码前

审查范围、架构适配、算法真实性、数据边界、不可变安全边界、步骤依赖和每步完成条件。未通过不得开始步骤 1。

### Workflow Review 2：全部步骤完成后

逐项核对交付物、五个步骤的十五轮 code review 证据、未修改安全文件的 diff 证据、全量回归、训练冒烟、checkpoint 回退和复现实验命令。未通过不得标记完成。

## 最终验证门禁

- `git diff` 不包含不可变边界中列出的导航、避让、控制文件。
- 新旧调度器的非法/重复分配测试全部为零。
- 新算法 checkpoint 缺失、损坏或超时均回退 Hungarian。
- 训练与验证 seed 不重叠，所有报告 JSON 禁止 NaN。
- 单元测试、训练冒烟、standalone 配对冒烟全部通过。
- Webots 未见 seed 配对评测只作为上线门禁；无法在当前环境完成时明确保留为未满足，不把 standalone 结果描述为生产结论。
