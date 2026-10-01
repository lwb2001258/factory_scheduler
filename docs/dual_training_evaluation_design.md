# 双训练目标与 3/5/8 机器人评估方案

## 1. 目标

为项目建立两套彼此独立、可复现的训练和评估方法：

1. **方法一：完成任务数优先（Count）**。只优化固定时间内完成的任务数，回答“谁做得更多”。
2. **方法二：优先级与安全综合效用（Utility）**。同时考虑任务优先级、从到达到完成的耗时、排队等待、行驶效率、负载均衡、调度可靠性和安全，回答“谁在安全前提下把重要任务做得更快、更稳”。

两套方法必须分别训练、分别保存 checkpoint、分别报告结果。不得先用综合目标训练，再把同一个模型称作“完成任务数模型”；否则无法判断性能来自哪种训练目标。

最终结论必须覆盖 3、5、8 台机器人。任一规模未达到门禁，不得用其他规模的收益抵消。

## 2. 与当前项目的对应关系

### 2.1 三种正式场景

| 场景 | 机器人 | 平均任务间隔 | 当前配置 |
|---|---:|---:|---|
| A | 3 | 30 秒 | `SCENARIOS["A"]` |
| B | 5 | 15 秒 | `SCENARIOS["B"]` |
| C | 8 | 8 秒 | `SCENARIOS["C"]` |

任务为泊松到达，优先级 1/2/3 的当前概率分别为 75%/20%/5%。三种场景必须使用相同的地图、任务端点分布、电量规则、路径规划和机器人避让规则；训练目标只能影响任务分配决策。

### 2.2 已有可用数据

项目目前已经记录：

- `total_tasks_completed`、`total_tasks_generated`、`throughput_per_minute`；
- 每个已完成任务的 `priority`、`waiting_time`、`completion_duration`、`execution_time`；
- `total_distance_all_robots`、`workload_balance_cv`、`tasks_per_robot`；
- `safety_event_count`、`pair_distance_violations`、`min_pair_distance`、`total_deadlocks`；
- `invalid_scheduler_outputs`、`fallback_scheduler_commits` 和调度 P95 延迟。

字段口径必须保持如下定义：

- `waiting_time = assignment_time - arrival_time`，表示分配前排队等待；
- `completion_duration = completion_time - arrival_time`，表示任务从到达到完成的总周转时间；
- `execution_time = completion_time - assignment_time`，表示分配后的执行时间。

用户所说的“任务完成时等待时间”在正式报告中使用 `completion_duration` 表示；同时单独报告 `waiting_time`，从而区分调度器造成的排队等待和机器人执行路径造成的时间。

### 2.3 当前数据缺口

实现本方案前需要补齐以下观测字段，但不改变控制或避让规则：

1. 将已有的 `MetricsCollector.task_arrivals` 写入结果 JSON。当前收集器记录了到达事件，但 `save_results()` 尚未输出它，导致无法准确按优先级统计未完成任务。
2. 输出实验结束时所有任务的 `task_id/status/priority/arrival_time/assignment_time/completion_time`，使未分配、执行中、失败任务也进入评分。
3. 当前 `pair_distance_violations` 是距离低于 0.50 m 的**时间采样数**，`safety_event_count` 也不是 Webots 刚体接触次数。应增加 `pair_distance_violation_episodes`：同一机器人对连续违规合并为一次，间隔超过 1 秒后再次发生才开始新 episode。若需要评价“真实碰撞”，还必须增加由 Webots 接触/碰撞检测产生的 `physical_collision_count`。在该字段落地前，只能称为“安全距离违规”，不能称为物理碰撞。
4. 当前 `factory_scenario()` 默认随机生成 2～8 台机器人。新训练 workflow 必须显式固定 `min_robots=max_robots` 为 3、5、8，不能用随机机器人数量代替场景矩阵。
5. 当前 headless 训练场景最多生成 12 个任务。正式训练应按 A/B/C 的任务间隔持续生成，或预生成完整定长任务流；否则 8 机器人高负载训练会偏离 Webots 场景 C。
6. `factory_scenario()` 使用 NumPy RNG 和随机自由起点，Webots `TaskGenerator` 使用 Python `random.Random` 且从场景初始停车位启动。因此“相同 seed”目前不代表 headless 与 Webots 逐任务相同。正式 workflow 应复用同一个任务清单生成器，或先生成版本化 task manifest 再由两种运行时读取；正式验证的初始机器人位置也必须固定为对应 Webots 场景位置。训练阶段可以额外随机化起点增强鲁棒性，但必须单独标记，不能混入配对验证。
7. 当前 `FactoryAStarCostOracle` 复用按 8 台机器人创建的静态 `MotionCoordinator`。实现 A/B/C 固定场景时，应按 3/5/8 分别建立或缓存 cost oracle/coordinator，核对场景 A/B 中停放的未参与机器人仍作为障碍物；不能让训练看到一张比 Webots 更空的地图。
8. LinUCB 的决策诊断当前包含 `selected_arm`，但提交后的任务 `learning_trace` 只保存算法名。应在任务真正提交时把 `selected_arm` 和 bandit context 版本固化进 trace，完成时据此发放延迟反馈；不能在完成时读取可能已经被后续决策覆盖的 `_last_arm`。

## 3. 两套方法共同的实验协议

### 3.1 模型与场景组织

默认训练一个可同时支持 3/5/8 台机器人的共享模型，继续使用最大 8 机器人观察空间和 action mask。训练 batch 对 A/B/C 按 **1:1:1 的 trajectory 数**采样。不同学习器按以下方式防止场景 C 支配更新：

- PPO/GraphPPO：在每个场景内部单独标准化 advantage，再把三个场景等量合并为 minibatch；
- DQN、RainbowDQN、QRDQN、CQL、SARSA：经验回放或在线更新按场景分层，每批 A/B/C transition 数量相同；
- LinUCB：每场景贡献相同数量的上下文样本，反馈按该场景预期任务量归一化；
- 监督/模仿算法：每场景样本或 episode 权重相同。

不能把三个场景的原始 episode 回报直接混合标准化。

推荐训练日程：

- 前 20% episode：3 机器人，学习基本分配；
- 中间 30% episode：3、5 机器人各半；
- 后 50% episode：3、5、8 机器人等概率混合；
- 最终模型选择：只看三个场景等权的验证结果，不看训练回报最高点。

如果后续确实需要按规模训练三个专用模型，必须作为单独实验报告，不能与共享模型结果混在一起。

### 3.2 时间范围

建议每个正式 episode 使用 600 仿真秒。原因是优先级 3 任务只有约 5%，场景 A 在过短 episode 中经常没有高优先级样本。

- 训练和 headless 验证：600 秒或达到等价的完整任务流终止条件；
- Webots 快速冒烟：300 秒，仅检查可运行性；
- Webots 正式评估：600 秒；
- 所有算法、同一场景和同一 seed 必须使用完全相同的时长。

训练终止条件以 `context.current_time` 达到预定仿真时长为准。`max_steps_per_episode` 只能作为防止死循环的高上限；如果在 600 秒前触发该上限，该 episode 标记为截断异常，不能与完整 episode 一起计分。不能让“决策次数较少”的算法获得更长或更短的有效仿真窗口。

可选的更严格模式是“600 秒到达窗口 + 最多 300 秒排空窗口”：600 秒后不再生成任务，只允许已有任务完成。若采用该模式，所有算法必须共享同一预生成任务清单，并同时报告 600 秒截面完成数和排空后的最终完成数。

### 3.3 Seed 分区

推荐最低配置：

| 数据集 | 每场景 seed 数 | 用途 |
|---|---:|---|
| 训练 | 100 | 参数更新、经验回放、专家轨迹 |
| 验证 | 20 | early stopping、权重选择、阈值校准 |
| 最终测试 | 10 | 冻结模型后的 Webots 配对评估 |

例如训练、验证、测试可分别使用 `21000–21099`、`31000–31019`、`41000–41009`，每个范围都在 A/B/C 中复用。复用 seed 是为了同场景内配对，不表示三个场景的任务流相同。

正式论文或生产决策建议把最终测试扩展到每场景 30 个 seed。训练、验证、测试 seed 严禁交叉；Webots 微调只能使用训练 seed，不能使用最终测试 seed。

### 3.4 分阶段训练

1. **Headless 预训练**：使用 `headless_webots_logic`，快速学习任务分配；保持 action mask、合法性验证和现有路径协调器。
2. **Headless 验证**：三个场景等权选择 checkpoint，不允许只选场景 C 最优模型。
3. **Webots 训练轨迹采集**：在训练 seed 上运行冻结策略，收集真实完成时间、安全距离违规、控制延迟和物理碰撞（字段实现后）。
4. **微调**：优先使用 Webots 轨迹做离线微调或小学习率更新，避免在线探索直接产生危险动作。动作仍须经过现有合法性和安全回退。
5. **冻结评估**：锁定模型、奖励版本、代码 commit 和配置，在最终测试 seed 上做配对实验；评估期间不得继续更新模型。

Headless 环境不包含 Webots 物理、距离传感器、本地 DWA、无线传输失败和刚体接触。因此 headless 阶段的碰撞项只能来自路径/距离代理；生产安全结论必须来自 Webots。

## 4. 方法一：完成任务数优先（Count）

### 4.1 训练目标

训练奖励保持最小且可解释：

```text
r_count(t) = 本次 transition 新完成的任务数
R_count    = episode 内完成任务总数
```

所有优先级在该方法中都记 1 分。合法 action mask、失败配对屏蔽、低电量规则和 Hungarian 安全回退继续生效，但不把距离、等待、优先级或死锁混入训练 reward。

Count 是有限时域的不折扣目标，默认 `gamma_count = 1.0`。若使用 `gamma < 1`，较早的完成会得到更大权重，模型实际优化的是“完成数 + 时效”，不能再标记为纯 Count 模型。

如果稀疏奖励导致训练困难，可以使用 value bootstrap、n-step return、优先经验回放或 curriculum；不得增加会改变最优策略排序的奖励项。若使用 potential-based shaping，必须单独记录公式并证明 episode 总排序仍由完成数决定。

### 4.2 适配不同算法

- PPO_RL、SARSA、DQN、GraphPPO、RainbowDQN、QRDQN：直接使用 `r_count`。
- CQL：Webots/headless 轨迹中的 reward 统一重算为 `r_count` 后建立独立离线数据集。
- LinUCB：利用任务 `learning_trace`，任务真正完成时给实际提交该任务的 arm `+1` 延迟反馈；不能把下一调度窗口的所有完成数粗略归给最后一次选中的 arm。
- LearnedHungarian、GraphImitation：本身不是 reward 驱动算法。可用“最大化后续完成数”的搜索/教师产生标签，并最终用 Count 验证集选择 checkpoint；不能声称它们直接执行了 RL 更新。

### 4.3 评估指标

固定 600 秒时，主指标是每个 `(场景, seed)` 的：

```text
C(s, k, a) = 算法 a 在场景 s、seed k 完成的任务数
```

吞吐量只是同一指标的单位换算：`throughput = C / 10`（任务/分钟）。不得同时把完成数和吞吐量当成两个独立加分项。

同 seed 相对 Hungarian 的配对变化：

```text
delta_count(s, k) = (C_candidate - C_Hungarian) / C_Hungarian
```

该相对量只在 `C_Hungarian > 0` 时有定义。600 秒正式实验若出现基线完成数为 0，应把该配对标记为运行/场景异常并调查，同时报告绝对差 `C_candidate - C_Hungarian`；不得用分母 1 伪装成百分比。

总结果采用场景等权宏平均：

```text
CountScore = mean(
    mean_k(delta_count(A, k)),
    mean_k(delta_count(B, k)),
    mean_k(delta_count(C, k))
)
```

同时必须报告 A/B/C 的原始均值、中位数、标准差、配对胜/平/负次数和按 seed 分层 bootstrap 95% 置信区间。

### 4.4 Count 模型通过条件

推荐门禁：

1. A、B、C 每个场景的平均完成数相对基线均不低于 -2%；
2. 三场景等权 `CountScore >= +2%` 才称为“优于基线”；
3. 若 `CountScore` 为正但 bootstrap 95% 区间包含 0，只能称为“当前样本下有提升趋势”；
4. `invalid_scheduler_outputs == 0` 且 `fallback_scheduler_commits == 0`，本地策略提交不得依赖 fallback 才取得优势；
5. 即使 Count reward 不包含安全项，部署评估仍必须通过第 6 节的安全硬门禁。

## 5. 方法二：优先级与安全综合效用（Utility）

### 5.1 优先级权重

使用非线性权重强调紧急任务：

| 优先级 | 含义 | 权重 `w(p)` |
|---:|---|---:|
| 1 | 普通 | 1 |
| 2 | 重要 | 2 |
| 3 | 紧急 | 4 |

权重和版本必须写入 checkpoint 元数据及报告。若未来改为 1/3/9，必须产生新的 reward/evaluation 版本，不得与旧结果直接合并。

### 5.2 建议训练 reward

以下为起始配置，最终系数只能用训练/验证 seed 校准：

下式中的 `H` 是该实验预先声明的评分终点：标准模式为 600 秒；若采用排空模式，则必须明确奖励按 600 秒截面还是排空终点计算，并在整批实验中保持一致。

Utility 已经显式处罚等待和执行时间，因此默认也采用有限时域 `gamma_utility = 1.0`，避免 event-driven transition 的不等时长再引入隐藏的时间折扣。若某个算法为稳定性必须折扣，应使用按实际 `delta_sim_time` 定义的统一时间折扣，并作为新的 reward 版本单独验证。

```text
任务 i 完成时：
  + 10.00 * w(priority_i)
  -  0.03 * w(priority_i) * min(waiting_time_i, 300)
  -  0.01 * w(priority_i) * min(execution_time_i, 300)

每次 transition：
  -  0.08 * 本 transition 所有机器人新增行驶距离（米）

episode 结束时每个未完成任务 i：
  - 10.00 * w(priority_i)
  -  0.03 * w(priority_i) * min(H - arrival_time_i, 300)

事件项：
  -100.00 * 新增物理碰撞事件
  - 20.00 * 新增安全距离违规 episode
  -  5.00 * 非法调度输出
  -  5.00 * 非物理恢复事件
  -  2.00 * 新增死锁恢复 episode
```

关键规则：

- 安全距离违规按连续 episode 计一次，不能按 16 ms 采样逐帧处罚，否则同一次接近事件会被重复数百次，并对 8 机器人场景形成不公平惩罚。
- `deadlock` 是效率/鲁棒性问题，不等同于碰撞；其权重应低于物理碰撞。
- Headless 没有真实碰撞时不得伪造 `physical_collision_count`。该项在 headless 中为“不可观测”，而不是自动证明为 0。
- 综合 reward 用于学习，最终排名使用下一节的固定评分和安全硬门禁，防止通过调整 reward 系数制造结论。

不同算法的 Utility 接法：

- PPO_RL、SARSA、DQN、GraphPPO、RainbowDQN、QRDQN：直接使用上述 transition/event reward；
- CQL：保留原始状态、动作、下一状态和事件，把数据集 reward 按同一版本公式离线重算；
- LinUCB：任务完成时根据 `learning_trace` 给实际提交 arm 返回该任务的优先级/时间效用；无法可靠归因的全局安全事件不强塞给某一次 arm 选择，而由最终安全门禁处理；
- LearnedHungarian、GraphImitation：由滚动 look-ahead 教师按 Utility 目标生成或重排标签，并用按优先级加权的监督损失训练；最终仍由 Utility 验证集选 checkpoint。

### 5.3 综合评估分数

先只用训练数据上的 Hungarian 结果冻结参考尺度：

- `tau(s,p)`：场景 s、优先级 p 的完成周期参考值；优先使用训练基线中位数；样本不足 30 个时跨训练场景合并同优先级样本，并设至少 1 秒的数值下限。
- `kappa(s,p)`：排队等待参考值，校准方式同上；等待中位数可能为 0，因此设至少 1 秒的数值下限。
- `d_ref(s)`：每单位已完成优先级权重的基线行驶距离中位数，必须有限且严格大于 0；若训练基线没有完成任务，则该场景校准失败。
- `lat_ref(s)`：基线调度 P95 延迟，设至少 1 ms 的数值下限。

这些值在验证和测试前写入版本化配置并冻结，禁止用最终测试结果重新归一化。

对每个场景和 seed，设所有已生成任务集合为 `G`，已完成集合为 `C`：

```text
PriorityCompletion =
  sum(i in C, w(p_i)) / sum(i in G, w(p_i))

Timeliness =
  sum(i in C, w(p_i) * exp(-completion_duration_i / tau(s,p_i)))
  / sum(i in G, w(p_i))

QueueResponsiveness =
  sum(i in C, w(p_i) * exp(-waiting_time_i / kappa(s,p_i)))
  / sum(i in G, w(p_i))

DistanceEfficiency =
  1 / (1 + distance_per_completed_weight / d_ref(s))

WorkloadBalance = 1 / (1 + workload_balance_cv)

SchedulerReliability =
  0, 若有非法输出或 fallback 提交；否则
  1 / (1 + candidate_latency_p95 / lat_ref(s))
```

其中：

```text
distance_per_completed_weight =
  total_distance_all_robots / sum(i in C, w(p_i))
```

实验开启时立即生成首个任务，因此正式运行的 `G` 应非空；若 `G` 为空则整次运行无效，不计算分数。未完成任务不进入分子，因此自然得到 0 的完成、时效和响应贡献；这避免只统计“完成得快的任务”产生幸存者偏差。若没有完成任务，则 `DistanceEfficiency=0`，而不是发生除零或把零行驶距离当成最高效率。所有参考尺度还应检查有限且严格大于 0。

基础综合分数范围为 0～100：

```text
UtilityScore = 100 * (
    0.45 * PriorityCompletion
  + 0.25 * Timeliness
  + 0.10 * QueueResponsiveness
  + 0.10 * DistanceEfficiency
  + 0.05 * WorkloadBalance
  + 0.05 * SchedulerReliability
)
```

优先级完成率和完成时效共占 70%，符合“重要任务优先且尽快完成”的主要目标。距离、均衡和推理延迟只占辅助权重，不能压过任务服务质量。

### 5.4 安全采用字典序门禁

碰撞不能靠多完成几个任务抵消，因此安全不作为 UtilityScore 中可交易的小权重，而是在排名前执行硬门禁：

1. `physical_collision_count == 0`；在该字段未实现前，模型只能获得“安全代理通过”，不能获得“无碰撞认证”。
2. `nonphysical_recoveries == 0`。
3. 候选的安全距离违规 episode 数不得高于同 seed Hungarian，正式生产目标为 0。
4. 正式默认 `min_pair_distance >= 0.50 m`，与当前违规阈值一致，并报告三个场景的最小值；若研究要改变阈值，必须在测试前登记新版本，不能看完结果再修改。
5. `invalid_scheduler_outputs == 0`、`fallback_scheduler_commits == 0`。
6. 调度 P95 延迟不得超过 `max(20 ms, 同批基线 P95 * 1.20)`。

仅在所有门禁通过的候选之间比较 UtilityScore。报告仍保留被拒模型的基础分数，便于诊断，但不得把它列为可部署模型。

### 5.5 Utility 模型通过条件

1. A、B、C 三个场景分别通过全部安全硬门禁；
2. 每个场景的优先级加权完成率相对 Hungarian 不劣于 -2%；
3. 每个场景 UtilityScore 不得比基线低 2 个分数点以上；
4. 三场景等权宏平均 UtilityScore 至少提高 2 个分数点，才称为“综合优于基线”；
5. 优先级 3 的完成率和 P95 `completion_duration` 必须单独报告，不能只报告总体均值；
6. 若某场景最终完成的优先级 3 任务少于 20 个，必须标记样本不足并增加测试 seed，不能把不稳定的 P95 用作通过依据；
7. 正式结论给出按场景分层、以 seed 为采样单位的配对 bootstrap 95% 区间。

## 6. 共同安全与有效性门禁

两套方法都必须遵循：

- AI 只输出任务分配，不控制路径、速度、避让、让行或停车；
- action mask 和 `validate_assignment(s)` 继续生效；
- 同一 `(场景, seed)` 的任务 ID、到达时刻、取货点、送货点和优先级必须逐项一致；
- 候选和 Hungarian 使用同一个代码 commit、Webots world、控制器、时长和环境变量；
- headless 结果只标记 `headless_webots_logic/business_logic_only`；最终安全结论要求 `runtime_mode=webots`；
- 训练、验证、最终测试报告必须记录模型 SHA-256、reward 版本、评估版本、代码 commit 和 seed 清单；
- 任何 NaN/Inf、维度错误、非法动作、非法匹配或 checkpoint 不兼容均失败关闭。

## 7. 3/5/8 机器人训练与评估矩阵

每套方法对每个候选算法至少生成以下矩阵：

| 阶段 | Count-A/B/C | Utility-A/B/C | 是否更新模型 |
|---|---|---|---|
| Headless 训练 | 各 100 seed | 各 100 seed | 是 |
| Headless 验证 | 各 20 seed | 各 20 seed | 否，仅选 checkpoint |
| Webots 微调轨迹 | 三场景训练 seed 子集 | 三场景训练 seed 子集 | 可离线更新 |
| Webots 冒烟 | 各 2 seed × 300 秒 | 各 2 seed × 300 秒 | 否 |
| Webots 正式测试 | 各 10 seed × 600 秒 | 各 10 seed × 600 秒 | 否 |

每种方法的每个候选在最低正式配置下需要 `3 场景 × 10 seed = 30` 次 Webots 测试；同一运行配置的 Hungarian 30 次基线可以被两套评估复用。对同一个算法同时评估 Count 和 Utility 两个 checkpoint 时，最低总量为 `30 Count + 30 Utility + 30 Hungarian = 90` 次。比较 N 个算法的两套 checkpoint 时，最低总量为 `60N + 30` 次。两套模型不能共享候选结果。

建议并行单位是独立 `(方法, 场景, 调度器, seed)`，但每个进程使用独立 Webots 端口和结果目录。结果文件缺失、重复或 seed 集不完整时，整组评估失败，不得只统计成功运行。

## 8. 统计与选型规则

### 8.1 配对而不是独立平均

所有候选都与 Hungarian 按相同场景、相同 seed 配对。先计算每个 seed 的差值，再求均值和 bootstrap 区间，不能把两组不相干 seed 的总体平均直接相减。

### 8.2 场景等权

场景 C 产生的任务更多，但最终总分采用 A/B/C 等权宏平均。报告同时提供按任务数加权的 micro average 作为辅助，不用它做唯一选型依据。

### 8.3 多算法比较

如果同时比较 10 个 AI 调度器，必须先应用统一门禁，再报告每个算法相对 Hungarian 的配对区间。需要宣称统计显著时，对多个候选的 p 值采用 Holm 校正；否则只使用“验证通过”“提升趋势”等描述。

### 8.4 推荐最终产物

```text
results/dual_objective/
  count/<algorithm>/checkpoint...
  utility/<algorithm>/checkpoint...
  calibration/reference_scales.json
  headless_validation.json
  webots_count_evaluation.json
  webots_utility_evaluation.json
  deployment_manifest.json
```

`deployment_manifest.json` 至少包含：方法、算法、模型路径与哈希、训练/验证/测试 seed、A/B/C 单场景门禁、宏平均结果、安全资格、代码 commit 和是否允许生产使用。

## 9. 建议的实现顺序

1. 补齐未完成任务、任务到达、物理碰撞或安全代理的结果字段，并给口径写测试。
2. 增加固定 A/B/C 的训练场景生成器，确保 3/5/8 机器人和 30/15/8 秒任务间隔准确。
3. 增加版本化 `CountRewardProfile` 与 `UtilityRewardProfile`，训练数据记录 reward profile。
4. 为在线 RL、CQL、LinUCB 和监督/模仿算法分别接入正确训练方式。
5. 扩展 workflow，生成两套 checkpoint 和完整的“2 种方法 × 3 个机器人场景”训练/验证矩阵。
6. 扩展评估器，计算 CountScore、UtilityScore、优先级分层指标、安全硬门禁和分层 bootstrap。
7. 先跑短 headless/300 秒 Webots 冒烟，再启动冻结的 600 秒正式测试。

## 10. 推荐结论格式

最终不要只说“算法 X 最好”，而应写成：

> 在 3/5/8 机器人、每场景 10 个未见 seed 的 600 秒 Webots 配对实验中，算法 X 的 Count 模型在三个场景均满足完成数非劣门禁，等权完成数提升为 …；算法 Y 的 Utility 模型通过全部安全硬门禁，等权综合分提升为 …，其中优先级 3 完成率为 …、P95 完成周期为 …。置信区间为 …。其余模型因 … 被拒绝。Headless 结果仅用于预训练，不作为物理安全结论。

这种格式能同时说明优化目标、机器人规模、统计证据、安全资格和仿真真实性边界。
