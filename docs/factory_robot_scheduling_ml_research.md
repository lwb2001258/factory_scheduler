# 工厂多机器人调度的强化学习与机器学习算法选型调研

更新时间：2026-09-30

## 1. 结论摘要

本项目当前是一个集中式、动态多机器人任务分配系统：最多 8 台机器人、最多保留 20 个任务槽位，强化学习动作是“机器人—任务”组合加一个 no-op，共 `8 × 20 + 1 = 161` 个离散动作。任务按泊松过程到达，调度成本已经综合路径距离、任务优先级、等待时间和拥堵，并在强化学习模型失效时回退到 Hungarian 调度。

基于项目规模、现有代码结构和相关研究，建议优先级如下：

1. **图网络/注意力打分 + Hungarian 匹配（监督学习或模仿学习）**：最适合首先落地。机器学习负责预测长期效果更好的机器人—任务边权，Hungarian 负责生成合法的一对一分配。风险低、可解释性较好，并可保留现有安全校验。
2. **Graph-PPO 或 Graph-DQN + 匹配解码器**：最有希望在场景 C、高任务到达率、拥堵和电量状态相互耦合时超过当前规则算法。核心改进是用图结构替代固定长度平铺状态，而不只是更换 PPO/DQN 名称。
3. **Rainbow DQN 或 IQN**：对当前 Double-DQN 的渐进升级，代码改动小于重建图模型。Rainbow 更侧重样本效率和训练稳定性；IQN 可面向尾部等待时间、延期风险等风险敏感目标。
4. **上下文 Bandit 元调度器**：让模型根据当前负载在 FCFS、NearestNeighbour、Hungarian、Auction 等现有算法之间选择。适合先验证“什么状态下哪种算法更好”，实施成本明显低于完整 RL。
5. **离线 CQL**：只有在积累了大量 Webots 或真实工厂调度日志后再采用，可避免在线探索；不适合在数据覆盖不足时直接替代现有调度器。
6. **MAPPO / QMIX**：仅在未来改成分布式决策、通信受限或机器人异构时优先。当前 Supervisor 拥有全局状态，集中式图策略通常更简单。
7. **CPO/拉格朗日约束 PPO、分层 RL**：分别适合显式安全/SLA 约束和“任务分配—充电—路径协调”联合决策，属于中长期方案。

最重要的判断是：**如果目标仍是单个调度时刻上的加性成本最小化，Hungarian 已经对该成本矩阵给出精确匹配，学习算法没有天然优势。** 机器学习要获得稳定优势，必须利用常规单步算法没有建模的信息，例如未来任务到达、拥堵传播、电量消耗、充电排队、死锁风险和尾部等待时间。

## 2. 当前项目基线与约束

### 2.1 已有调度算法

当前实验入口包含 12 种任务调度算法：

- 规则/启发式：FCFS、NearestNeighbour、RoundRobin、Greedy、Random。
- 优化与元启发式：Hungarian、Auction、GA、SA。
- 强化学习：PPO_RL、SARSA、DQN。

代码位置：

- 调度器清单：[`scripts/run_experiments.py`](../scripts/run_experiments.py)
- 调度器工厂：[`controllers/factory_supervisor/schedulers.py`](../controllers/factory_supervisor/schedulers.py)
- RL 适配器与安全回退：[`controllers/factory_supervisor/rl_schedulers.py`](../controllers/factory_supervisor/rl_schedulers.py)

### 2.2 场景规模

| 场景 | 机器人数量 | 平均任务到达间隔 | 预期学习算法价值 |
|---|---:|---:|---|
| A | 3 | 30 秒 | 低负载，常规算法通常足够 |
| B | 5 | 15 秒 | 中等负载，适合验证混合学习方案 |
| C | 8 | 8 秒 | 高负载，最可能体现长期调度和拥堵建模优势 |

配置见 [`controllers/factory_supervisor/config.py`](../controllers/factory_supervisor/config.py)。

### 2.3 当前 RL 接口

当前统一环境的关键特征是：

- 最大机器人数量：8。
- 最大任务槽位：20。
- 动作空间：每个机器人—任务对加 no-op，共 161 个离散动作。
- 任务特征包含起终点、优先级、等待时间、可行机器人比例和预估成本。
- 奖励包含合法分配、任务优先级、完成任务、等待和估计距离等项。
- 推理结果仍需通过合法性检查；模型错误、超时或连续失败时使用 Hungarian。

环境见 [`controllers/factory_supervisor/rl_environment.py`](../controllers/factory_supervisor/rl_environment.py)。共享成本矩阵见 [`controllers/factory_supervisor/schedulers.py`](../controllers/factory_supervisor/schedulers.py)，当前成本近似为：

```text
cost = travel
       - priority_weight × (priority - 1)
       - waiting_weight × waiting_time
       + congestion_weight × congestion
```

这套架构很适合增加“学习型打分器”，但固定任务槽和平铺向量不利于跨任务数量、机器人数量和布局泛化。

## 3. 强化学习候选算法

### 3.1 Graph-PPO / Attention-PPO

**建议优先级：最高（作为主要研究方向）**

将状态表示成二部图：一侧是机器人节点，另一侧是任务节点，边表示机器人执行任务的路径成本、预计耗时、拥堵和可行性。GNN/GAT 编码后，PPO 对合法边或候选匹配进行决策。

适合本项目的原因：

- 机器人—任务天然构成二部图，正好对应现有成本矩阵。
- 参数可在不同任务数量之间共享，避免固定 20 个任务槽造成的位置敏感性。
- 可编码机器人之间的冲突、共享狭窄通道和充电站竞争。
- 可以保留现有 action mask、合法性检查和 Hungarian fallback。

研究依据：

- NeurIPS 2020 的图网络调度策略使用 GNN 表示作业车间状态并学习派工规则，报告了对更大未见实例的泛化能力：[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html)。
- 多机器人 GAT 调度研究将专家示范与图注意力结合，在不同机器人和任务规模上实现快速调度：[Learning Scheduling Policies for Multi-Robot Coordination With Graph Attention Networks](https://doi.org/10.1109/LRA.2020.3002198)。
- NeurIPS 2022 针对多机器人/机器任务分配提出可迁移的图 Q 函数和 auction-fitted Q-learning，问题形式与本项目的动态 pickup-and-delivery 很接近：[Learning NP-Hard Multi-Agent Assignment Planning using GNN](https://proceedings.neurips.cc/paper_files/paper/2022/hash/66ad22a4a1d2e6fe6f6f6581fadeedbc-Abstract-Conference.html)。
- CapAM 使用图/Capsule 注意力和策略梯度处理带截止时间、工作量与机器人能力约束的多机器人任务分配，说明图策略可以直接承载本项目未来可能加入的任务 SLA 和电量/载荷约束：[Learning Scalable Policies over Graphs for Multi-Robot Task Allocation](https://arxiv.org/abs/2205.03321)。
- 2025 年的异构多机器人任务分配研究进一步使用注意力式分散协作策略处理技能、任务和机器人日程之间的依赖；它更适合本项目未来出现异构机器人或协作任务时，而不是当前同构集中式版本：[Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning](https://doi.org/10.1109/LRA.2025.3534682)。

更可能超过常规算法的条件：

- 场景 C 或更高负载，当前分配会显著影响后续拥堵。
- 任务优先级、等待时间、电量和路径冲突需要长期权衡。
- 相似布局和工作负载被反复求解，可从历史分布中学习结构。

不会自然占优的条件：

- 每次只需最小化当前已知成本矩阵。
- 测试布局、任务类型或故障模式远离训练分布。
- 训练样本不足，奖励主要是稀疏的最终吞吐量。

### 3.2 Rainbow DQN

**建议优先级：高（当前 DQN 的低风险升级）**

Rainbow 将 Double DQN、Dueling 网络、优先经验回放、多步回报、分布式价值等多种 DQN 改进组合起来。原论文在离散动作基准上展示了更好的样本效率和最终性能：[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11796)。

适合本项目的原因：

- 当前已经有 Double-DQN、经验回放、固定离散动作和 action mask。
- 可以分阶段加入 Dueling head、Prioritized Replay 和 n-step return，不必一次重写环境。
- 比 PPO 更充分复用历史经验，适合 Webots 仿真采样成本较高的情况。

优势最可能出现在任务到达和完成奖励延迟明显、不同动作价值接近、普通 DQN 学习不稳定时。它仍受固定 161 动作和平铺状态的限制，因此长期建议与 GNN 编码结合，而不是只换训练技巧。

### 3.3 IQN / QR-DQN 等分布式强化学习

**建议优先级：中高（面向尾部风险）**

这里的“分布式”是指学习回报分布，不是多机器人分布式执行。IQN 学习完整回报分位数，可构造风险敏感策略：[Implicit Quantile Networks for Distributional Reinforcement Learning](https://proceedings.mlr.press/v80/dabney18a.html)。

适合以下目标：

- 不仅降低平均完成时间，还要降低 P95/P99 等尾部完成时间。
- 高优先级任务超时需要比平均吞吐更强的惩罚。
- 路径时长、阻塞和故障具有较大随机性。

如果项目只优化平均成本且仿真噪声很小，IQN 的收益可能不足以抵消复杂度。

### 3.4 MAPPO

**建议优先级：中；当前架构下不是首选**

MAPPO 采用集中训练、分散执行：训练时 critic 可以使用全局状态，各机器人 actor 只使用本地观测。相关研究表明，经过恰当实现的 PPO 在多个合作式多智能体基准中可与或超过多种 off-policy 方法：[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html)。

适用条件：

- Supervisor 不再能可靠获得全局状态。
- 机器人需要在通信延迟或带宽受限时独立出价、接单或避让。
- 机器人异构，每台机器人有不同能力和局部观测。

当前系统由 Supervisor 统一分配任务，直接使用 MAPPO 会引入多智能体非平稳性和信用分配问题，却未获得分散执行收益。因此应在架构真的去中心化后采用。

### 3.5 QMIX

**建议优先级：中低；仅适合离散分散决策**

QMIX 用单调 mixing network 将各智能体价值组合成全局价值，实现集中训练、分散执行：[Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://www.jmlr.org/papers/v21/20-081.html)。

它适合每个机器人独立选择“申请哪个任务/等待/充电”等离散动作，同时团队共享吞吐、等待或能耗奖励的情况。缺点是：

- 多个机器人可能选择同一任务，需要额外冲突消解或拍卖层。
- 单调价值分解限制了可表达的协作关系。
- 当前集中式一对一匹配由 Hungarian 处理得更直接。

### 3.6 CQL 离线强化学习

**建议优先级：中；取决于是否有高质量日志**

CQL 从固定历史数据训练保守 Q 函数，主要解决离线数据分布之外动作价值被高估的问题：[Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)。

适用条件：

- 已积累 Hungarian、Auction、人工策略和真实运行产生的大量轨迹。
- 不允许在真实工厂中通过随机探索收集数据。
- 日志包含足够多样的负载、故障和拥堵状态。

风险是日志只覆盖常规策略走过的状态；若高负载或异常样本很少，离线 RL 很难可靠学到这些状态下的改进策略。应继续保留 action mask、合法性检查和回退算法。

### 3.7 CPO / 拉格朗日约束 PPO

**建议优先级：中；当安全和 SLA 被显式建模时采用**

把碰撞、低电量接单、高优先级超时、死锁次数作为约束成本，而不是全部压进一个奖励标量。CPO 在约束 MDP 中优化收益，并给出接近期望约束满足的理论性质：[Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a)。

它适用于“平均效率可以优化，但碰撞率、任务失败率或优先级 3 超时率必须低于阈值”的场景。注意 CPO 的约束通常是期望意义上的，不能代替路径规划器、碰撞检查和硬安全规则。

### 3.8 分层强化学习

**建议优先级：中长期**

高层每隔较长时间决定任务分配、充电策略或区域责任；低层继续由 A*、CBS/RHCR、DWA 等确定路线和运动。Option-Critic 提供了同时学习时间扩展动作及其终止条件的方法：[The Option-Critic Architecture](https://ojs.aaai.org/index.php/AAAI/article/view/10916)。

适用条件：

- 需要联合优化任务分配、充电和冲突协调。
- 决策具有明显不同时间尺度。
- 单层策略奖励延迟严重、动作序列过长。

当前任务层和运动层边界清晰，过早合并可能使训练难度明显增加。建议先保持低层规划确定性，只学习高层策略。

### 3.9 SAC/MADDPG 为什么不是近期优先项

SAC 和 MADDPG 主要擅长连续控制。SAC 的优势包括 off-policy 样本复用、熵探索和较好的连续控制稳定性：[Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b)。但本项目当前是 161 个带 mask 的离散匹配动作，DQN、PPO 或离散 SAC 更自然。

只有将动作扩展为连续权重、速度上限、充电阈值、区域负载比例或调度参数时，SAC/MADDPG 才更值得采用。为使用算法而把离散匹配强行连续化通常得不偿失。

## 4. 非强化学习的机器学习方案

### 4.1 GNN/GAT 模仿学习调度器

**建议优先级：最高**

用 Hungarian、MILP、小规模穷举或高预算 GA/SA 作为专家，生成大量状态—最优/近优分配样本；GNN/GAT 学习机器人—任务边得分，在线时再用 Hungarian 将得分解码为合法匹配。

可采用两种部署方式：

1. **纯模仿**：网络直接预测专家偏好的边，推理非常快。
2. **学习边权 + 精确解码**：网络预测长期代价，Hungarian/MILP 保证一对一约束，更适合本项目。

优势条件：

- 可以离线生成大量高质量专家解。
- 在线决策预算严格，但离线计算预算充足。
- 任务数或机器人数量未来会扩大，精确求解开始变慢。

主要风险是行为克隆的分布偏移：模型犯错后会进入专家数据中少见的状态。DAgger 通过让当前策略运行、再让专家标注这些新状态并聚合数据来缓解该问题：[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)。

GNN 模仿强分支规则也已被用于加速组合优化，并能泛化到更大实例：[Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html)。

纯模仿通常不能系统性超过自己的专家；其主要价值是近似高预算专家、降低在线延迟和支持更大实例。若希望超过专家，需要 DAgger、在线反馈、RL 微调或将模型嵌入搜索算法。

### 4.2 学习路径耗时、拥堵和失败概率，再交给优化器

**建议优先级：最高，工程风险最低**

不让模型直接决定任务，而是预测更准确的输入：

- 机器人到取货点的真实行驶时间。
- 取货到送货的预计完成时间。
- 某条路线的拥堵、避让和重规划延迟。
- 电量消耗和中途充电风险。
- 机器人—任务组合失败或超时概率。

模型可以从线性/岭回归开始，再比较 Gradient Boosting、Random Forest、MLP 或 GNN。预测值替换当前 `_pair_cost` 中的静态 travel 部分，最终仍由 Hungarian、Auction 或 Greedy 分配。

这一方案在以下情况下可能明显优于当前常规算法：

- 欧氏距离或静态 A* 路径长度与真实完成时间偏差大。
- 狭窄通道、交叉口和充电站排队导致强状态依赖延迟。
- 有足够历史日志训练监督模型。

端到端路径耗时学习的研究显示，直接学习整条路径耗时可以减少逐段误差累积：[DeepTTE: Estimating Travel Time Based on Deep Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11877)。对本项目而言，重点不是照搬道路网络模型，而是采用“先预测真实代价、再优化”的分层思想。

### 4.3 学习辅助优化：ML + MILP/LNS/现有元启发式

**建议优先级：高（任务规模扩大后）**

机器学习不替代求解器，而是帮助求解器：

- 预测 MILP 分支变量、初始可行解或变量固定策略。
- 为 GA/SA 提供高质量初始种群或邻域。
- 学习 Large Neighborhood Search 的 destroy/repair 操作。
- 由模型筛选候选任务，再由精确算法求解小规模子问题。

NeurIPS 2021 的学习型 LNS 用 actor-critic 选择需要重优化的变量子集，再由 IP solver 修复，在多个 IP 基准上体现了同等时间预算下的优势，并能迁移到更大问题：[Learning Large Neighborhood Search Policy for Integer Programming](https://proceedings.neurips.cc/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html)。神经 LNS 在车辆路径问题中也展示了“学习启发式 + 传统搜索”优于纯学习或手写局部算子的潜力：[Neural Large Neighborhood Search for Routing Problems](https://doi.org/10.1016/j.artint.2022.103786)。

该方向适用于未来引入时间窗、任务依赖、异构能力、多任务批量规划后形成的复杂组合优化。目前每次候选规模最多 8×20，直接 Hungarian 很快，立即引入 MILP/LNS 的收益有限。

### 4.4 上下文 Bandit 元调度器

**建议优先级：高（快速实验方向）**

把现有调度器视为可选择的“臂”，上下文包括机器人数量、空闲比例、任务队列长度、优先级分布、平均电量和拥堵度。模型选择本轮使用 Hungarian、Greedy、Auction、NearestNeighbour 等哪一种，并根据本轮或短窗口收益更新。

可选算法：LinUCB、Thompson Sampling、NeuralUCB。上下文 Bandit 使用上下文选择动作并根据反馈在线适应，其计算和样本需求通常低于完整 RL；LinUCB 的经典工作给出了大规模日志上的离线评估方法：[A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)。

适合条件：

- 不同负载区间确实由不同现有算法占优。
- 主要反馈发生在决策后不久，不需要很长的信用传播。
- 希望在线适应工作负载，但不愿承担完整深度 RL 的复杂度。

它不擅长处理某次分配对很久以后拥堵和吞吐造成的影响；此时需要 RL 或显式滚动规划。

### 4.5 贝叶斯优化/监督式算法配置

**建议优先级：高（应先于增加复杂模型）**

当前项目存在大量可调参数：travel/priority/waiting/congestion 权重、GA/SA 时间预算、PPO 奖励权重、候选任务数和回退阈值。可以使用贝叶斯优化、SMAC 或 Optuna 在固定训练 seeds 上搜索，在独立验证 seeds 上选型。

贝叶斯优化适合单次仿真昂贵、参数数量中等且目标不可微的场景。相关工作展示了其在昂贵超参数调优中的有效性：[Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html)。

它不会产生新的在线调度策略，但可能以最低成本显著提升现有 Hungarian/Greedy/GA/SA 和 RL 基线，也是判断“学习策略是否真正超过调好后的常规算法”的必要步骤。

## 5. 什么时候机器学习更可能超过常规算法

### 5.1 有利条件

1. **长周期效应明显**：当前选择会改变未来拥堵、机器人位置、电量和任务等待。
2. **持续随机到达且负载较高**：静态优化每次只能看到当前快照。Decima 在持续随机作业到达和高负载下展示了 RL 调度相对手工启发式的收益，说明这类优势依赖工作负载结构，而不是 RL 名称本身：[Learning Scheduling Algorithms for Data Processing Clusters](https://web.mit.edu/decima/index.html)。
3. **真实代价难以手工建模**：路线长度无法准确反映避让、等待、死锁恢复和充电排队。
4. **同类问题被重复求解**：训练成本可由大量后续在线决策摊薄。
5. **状态规模或约束复杂度增长**：机器人异构、任务时间窗、任务依赖和动态故障使手写规则快速膨胀。
6. **允许大量仿真和域随机化**：训练覆盖任务率、位置、速度、电量、临时障碍和故障等变化。
7. **目标是多指标长期折中**：平均完成时间、尾部延迟、优先级 SLA、能耗、公平性和拥堵无法被单个短视成本充分表达。

### 5.2 常规算法更可能占优的条件

1. **规模小且目标明确**：当前 3–8 台机器人、一次最多 20 个任务，Hungarian 对单步线性指派已经很强。
2. **低负载**：场景 A 中任务稀疏，复杂策略能利用的耦合很少。
3. **硬约束多、必须给出可行性保证**：优化器和规则更容易审计；学习模型应只做打分或候选生成。
4. **分布频繁变化但无法再训练**：新布局、新机器人动力学或任务流程会造成分布外失效。
5. **缺乏高保真仿真和真实日志**：策略可能只学会抽象训练环境的偏差。
6. **在线决策样本很少**：深度 RL 的训练成本无法摊薄。

因此，不能以“是否超过 FCFS”判断学习算法有效，而应至少与调优后的 Hungarian、Greedy、Auction、GA/SA 以及预测成本增强版 Hungarian 比较。

## 6. 面向本项目的推荐方案

### 6.1 第一阶段：学习代价 + 精确匹配

推荐结构：

```text
机器人状态 ─┐
任务状态   ─┼─> 成本/耗时/风险预测模型 ─> 学习型成本矩阵 ─> Hungarian ─> 现有合法性与路径检查
拥堵与路径 ─┘
```

第一版可使用 Gradient Boosting/MLP；数据量足够后升级为机器人—任务二部 GNN。训练标签可用：

- 实际从分配到完成的时长。
- 空驶距离和总行驶距离。
- 重规划/等待/死锁恢复次数。
- 是否触发低电量、任务失败或 RL fallback。

优点是能直接复用当前成本矩阵和 Hungarian，实现失败时可立即退回静态成本。

### 6.2 第二阶段：GAT 模仿学习 + DAgger

1. 在小规模实例上让高预算 GA/SA、滚动时域搜索或 MILP 生成专家分配。
2. 训练 GAT 边打分器模仿专家。
3. 用 Hungarian 解码边得分。
4. 运行学习策略，收集其真正访问到的状态并让专家重新标注，执行 DAgger。
5. 先作为 shadow policy 记录建议，不直接控制；确认稳定后再启用。

### 6.3 第三阶段：Graph-PPO / Graph-Rainbow

在第二阶段图编码器和合法动作接口稳定后，再使用长期奖励微调：

- actor/Q 网络输出机器人—任务边分数。
- critic 使用全局池化图表示。
- action mask 继续屏蔽忙碌机器人、不可达任务和低电量组合。
- 奖励同时报告原始业务 KPI，避免只看 shaped reward。
- 训练中随机化任务率、地图阻塞、速度、电量和故障。
- 线上保留 50 ms 推理超时、Hungarian fallback 和现有路径安全层。

### 6.4 第四阶段：按需求分叉

- 有大量真实日志但不能探索：CQL。
- 要求降低 P95/P99 或高优任务违约：IQN/风险敏感 RL。
- 系统改成分散执行：MAPPO 或 QMIX。
- 联合任务、充电和运动层：分层 RL。
- 碰撞率、失败率必须满足显式阈值：约束 RL + 硬安全层。

## 7. 推荐实施优先级矩阵

| 方案 | 预期收益 | 实施成本 | 安全/可解释性 | 当前适配度 | 建议 |
|---|---|---|---|---|---|
| 学习耗时/风险 + Hungarian | 中高 | 低中 | 高 | 很高 | 第一优先 |
| GNN/GAT 模仿 + Hungarian | 高 | 中 | 较高 | 很高 | 第一优先 |
| 上下文 Bandit 选择现有调度器 | 中 | 低 | 高 | 高 | 快速验证 |
| 贝叶斯优化现有权重/参数 | 中 | 低 | 高 | 很高 | 立即进行 |
| Rainbow DQN | 中高 | 中 | 中 | 高 | 渐进升级 |
| Graph-PPO / Graph-DQN | 高 | 高 | 中 | 高 | 核心研究方向 |
| IQN/风险敏感 DQN | 中高 | 中高 | 中 | 中高 | 有尾部 SLA 时 |
| CQL | 中高 | 高 | 中 | 取决于日志 | 有真实数据后 |
| ML + LNS/MILP | 高 | 高 | 高 | 当前规模偏小 | 扩展约束后 |
| MAPPO | 高 | 高 | 中低 | 当前偏低 | 去中心化后 |
| QMIX | 中高 | 高 | 中低 | 当前偏低 | 分散离散动作时 |
| 分层 RL | 高 | 很高 | 低中 | 当前偏低 | 中长期 |

## 8. 公平验证方案

### 8.1 数据划分

- 训练、验证、测试使用互不重叠的 seeds。
- 同一测试 seed、同一场景和相同时间步下，所有算法共享相同任务流。
- 测试必须包含场景 A/B/C，以及训练范围之外的任务到达率和临时阻塞。
- 真实日志按时间切分，避免未来数据泄漏到训练集。

### 8.2 强基线

至少比较：

- FCFS、NearestNeighbour。
- 调优后的 Greedy、Hungarian、Auction。
- 相同在线时间预算的 GA、SA。
- 当前 PPO、DQN、SARSA。
- 静态成本 Hungarian 与学习成本 Hungarian。

### 8.3 指标

业务指标：

- 总吞吐量、平均完成时间、P95/P99 完成时间。
- 平均等待时间、最大等待时间、任务饥饿率。
- 优先级 3/2/1 分层完成时间与 SLA 违约率。
- 总距离、空驶距离、能耗和充电等待。

安全与鲁棒性指标：

- 碰撞、近碰撞、死锁、重规划和任务失败次数。
- 不合法动作率、模型超时率、fallback 比例。
- 新任务率、新布局、机器人故障和传感噪声下的性能退化。

计算指标：

- 平均、P95 和最大推理时间。
- 训练样本数、训练耗时、模型大小和内存。

### 8.4 判定规则

- 使用同 seed 配对比较，而不是只比较不同 seed 的均值。
- 至少报告均值、标准差和置信区间；建议使用配对 bootstrap 或 Wilcoxon signed-rank test。
- 预先规定主要指标，例如“场景 C 的 P95 完成时间”，避免训练后挑选有利指标。
- 学习算法只有在安全指标不退化、fallback 比例可接受且多个测试 seeds 上稳定改善时，才算超过常规算法。
- 单步 reward 高不代表业务性能好，最终结论必须基于完整仿真 KPI。

## 9. 最终建议

对于当前版本，推荐按以下顺序推进：

1. 先用贝叶斯优化调好现有成本权重和强基线。
2. 记录每个候选机器人—任务对的预测特征与真实执行结果。
3. 实现“监督学习成本模型 + Hungarian”，作为最小风险机器学习基线。
4. 实现二部 GAT 边打分器，用专家解进行模仿学习和 DAgger。
5. 复用该图编码器实现 Graph-Rainbow 或 Graph-PPO，重点验证场景 C 和分布外负载。
6. 只有在去中心化、日志驱动或硬约束需求出现后，再分别引入 MAPPO/QMIX、CQL 或 CPO。

从研究价值看，**“GNN/GAT 表示 + 学习长期边成本 + Hungarian/拍卖解码 + 安全回退”最符合本项目。** 它既利用机器学习对动态、非线性和长期效应的建模能力，也保留组合优化在约束满足、稳定性和可解释性上的优势。

## 10. 主要参考资料

以下优先列出原始论文或作者/会议官方页面：

1. Zhang et al., 2020. [Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html), NeurIPS.
2. Wang & Gombolay, 2020. [Learning Scheduling Policies for Multi-Robot Coordination With Graph Attention Networks](https://doi.org/10.1109/LRA.2020.3002198), IEEE RA-L.
3. Kang et al., 2022. [Learning NP-Hard Multi-Agent Assignment Planning using GNN](https://proceedings.neurips.cc/paper_files/paper/2022/hash/66ad22a4a1d2e6fe6f6f6581fadeedbc-Abstract-Conference.html), NeurIPS.
4. Kool et al., 2019. [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm), ICLR.
5. Yu et al., 2022. [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html), NeurIPS.
6. Rashid et al., 2020. [Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://www.jmlr.org/papers/v21/20-081.html), JMLR.
7. Hessel et al., 2018. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11796), AAAI.
8. Dabney et al., 2018. [Implicit Quantile Networks for Distributional Reinforcement Learning](https://proceedings.mlr.press/v80/dabney18a.html), ICML.
9. Kumar et al., 2020. [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html), NeurIPS.
10. Achiam et al., 2017. [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a), ICML.
11. Bacon et al., 2017. [The Option-Critic Architecture](https://ojs.aaai.org/index.php/AAAI/article/view/10916), AAAI.
12. Haarnoja et al., 2018. [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b), ICML.
13. Mao et al., 2019. [Learning Scheduling Algorithms for Data Processing Clusters](https://web.mit.edu/decima/index.html), ACM SIGCOMM.
14. Gasse et al., 2019. [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html), NeurIPS.
15. Ross et al., 2011. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS.
16. Wu et al., 2021. [Learning Large Neighborhood Search Policy for Integer Programming](https://proceedings.neurips.cc/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html), NeurIPS.
17. Hottung & Tierney, 2022. [Neural Large Neighborhood Search for Routing Problems](https://doi.org/10.1016/j.artint.2022.103786), Artificial Intelligence.
18. Li et al., 2010. [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146), WWW.
19. Snoek et al., 2012. [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html), NeurIPS.
20. Wang et al., 2018. [DeepTTE: Estimating Travel Time Based on Deep Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11877), AAAI.
21. Paul et al., 2022. [Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks](https://arxiv.org/abs/2205.03321).
22. Dai et al., 2025. [Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning](https://doi.org/10.1109/LRA.2025.3534682), IEEE RA-L.
