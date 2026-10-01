# 其它 AI 调度算法审查记录

## Workflow Review 1（编码前）

- 状态：通过（2026-10-01）。
- 范围：`additional_ai_scheduler_workflow.md`、现有 scheduler factory、RL 环境、学习型调度器、实验入口和测试基线。
- 发现并修正：checkpoint 格式原先未明确，已限定为 `allow_pickle=False` 的 NPZ；CQL 离线数据缺少当前/下一动作 mask 契约，已补充；GraphPPO 训练与部署解码关系原先不够明确，已规定共享 logits；LinUCB 收益口径原先未定义，已固定为同快照合法匹配成本代理并禁止伪造完成反馈。
- 架构结论：GraphPPO、RainbowDQN、QRDQN、离散 CQL 和 LinUCB 都只作用于集中式任务分配层；MAPPO、QMIX、CPO、分层 RL、SAC/MADDPG 本轮排除理由与研究文档一致。
- 安全结论：受保护的导航、避让、运动控制文件不进入实现范围；所有新模型继续使用现有可行性矩阵、合法性校验和 Hungarian 回退。
- 测试基线：`python -m unittest discover -s tests -v`，39 项通过。

受保护文件 SHA-256 基线：

```text
9B807D8760CE77A335419F2C56F535B04FB7FD4786E7C51ACBC965360ACAA97C  controllers/robot_controller/robot_controller.py
5E35449BAFADADC7A67AD1153BA16EF507C329BB7683A5FB422807595F3A1C7E  controllers/factory_supervisor/grid_planner.py
4F357344B3B9B6B6A03554B864BBA677D80F1D5136EF94221CD4302FFB9C8D5B  controllers/factory_supervisor/joint_grid_planner.py
5105E27F1ED422FB15E5C43C5709A06395A1C89C2CBBF63F9EED7D81BD7636FE  controllers/factory_supervisor/motion_coordinator.py
CC832979D16264465D1568F78A6DE2B01A13016ED4A286AE3513F87109933AA6  controllers/factory_supervisor/safety_coordination.py
9CCA0B0DEB8E8A23240AA848C64049873B8DAB4A64C75AB455FA05E5A1082385  controllers/factory_supervisor/collision_safety.py
```

## 每步 Code Review

### 步骤 1：统一模型契约与安全适配层

- **Code Review A — 通过。** 初检发现复杂数参数、JSON NaN 和极端维度异常未完全收口，修复后重新审查。`py_compile`、`git diff --check` 和 8 项定向测试通过；checkpoint 固定使用 NPZ/`allow_pickle=False`，参数要求实数且有限。
- **Code Review B — 通过。** 初检发现低电量、重复任务 ID 和推理超时缺少直接测试，补齐后重新从 A 审查。空 mask、NaN、缺失/损坏模型、低电量、重复 ID、超时及 Hungarian 回退均通过；六个受保护文件 SHA-256 与基线完全一致。
- **Code Review C — 通过。** `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v` 共 47 项通过；`git diff --check` 通过；旧 LearnedHungarian、GraphImitation、Webots 启动器测试无回归。

### 步骤 2：RainbowDQN、QRDQN 与 CQL

- **Code Review A — 通过。** 初检并修复 QR-DQN 分位数梯度归一化、Adam 恢复步数、配置 NaN/非整数门禁和 agent 维度检查；补充项目抽象环境固定 seed 训练。确认 Rainbow 包含 Dueling+C51+Double-DQN+PER+n-step，QRDQN 使用 quantile-Huber，CQL 使用 masked discrete log-sum-exp conservative penalty。17 项定向测试通过。
- **Code Review B — 通过。** 初检并修复离线数据集对浮点 action、NaN mask/done、非整数 seed 的宽松转换，以及采集器可能产生 masked no-op 的极端路径。模型/数据集固定 `allow_pickle=False`；当前/下一 mask、行为策略和 seed 均入契约；低电量、重复任务、非法 action、坏 checkpoint 和超时均 fail closed。受保护文件哈希与基线一致。
- **Code Review C — 通过。** 全量 `unittest` 56 项通过，`git diff --check` 通过；三个部署适配器均能加载真实环境维度 checkpoint 并只返回合法任务分配；旧调度器测试无回归。

### 步骤 3：GraphPPO + 匹配解码

- **Code Review A — 通过。** 首次项目环境运行发现未来到达任务被图构图器提前看见，修复为策略快照只暴露当前任务槽后重新审查。确认共享边编码、稳定 masked softmax、clipped PPO、GAE、entropy/value loss 和梯度方向；固定 seed 训练可逐参数复现，23 项定向测试通过。
- **Code Review B — 通过。** 策略快照任务集合改为不可变 tuple；错误/缺失 checkpoint、错误 action semantics、未来任务、busy 低编号机器人 action 映射均有直接测试。部署端仅将有限边概率交给 Hungarian，并由 `_result_from_matching` 做共享唯一性/可行性校验；推理异常 fail closed。受保护文件哈希与基线一致。
- **Code Review C — 通过。** 全量 `unittest` 62 项通过，`git diff --check` 通过；GraphPPO 返回无重复机器人/任务的合法匹配，旧调度器与 Webots 启动测试无回归。

### 步骤 4：LinUCB 元调度器

- **Code Review A — 通过。** 确认标准 `A += xxᵀ`、`b += reward*x` 和 UCB 置信项实现；上下文只含决策时可见负载/电量/优先级/等待/拥堵/可行成本。初检修复 arm 评分需分别拒绝重复机器人和任务、checkpoint 训练计数不得浮点截断；27 项定向测试通过。
- **Code Review B — 通过。** 未来任务从上下文和 arm 输入中排除；非法/空匹配使用固定缺配惩罚，arm 抛异常时 LinUCB fail closed 并由外层 Hungarian 回退；模型要求有限、对称、正定协方差和固定上下文版本。受保护文件哈希与基线一致。
- **Code Review C — 通过。** 全量 `unittest` 66 项通过，`git diff --check` 通过；固定 seed 训练逐数组可复现，scheduler 保留所选既有 arm 的合法分配，不改写分配内容。

### 步骤 5：训练 workflow、实验入口和文档

- **Code Review A — 通过。** workflow 实际生成 RainbowDQN、QRDQN、CQL、CQLDataset、GraphPPO、LinUCB 六个候选 artifact 并通过严格 JSON 研究门禁。初检修复静态验证提前包含未来任务、`.gitignore` 未白名单新增交付物、以及报告把外部哈希审查误写成自动结论的问题。29 项定向测试通过。
- **Code Review B — 通过。** 五种调度器均由统一 factory 严格加载；缺失/坏模型显式回退 Hungarian，`allow_safe_fallback=False` 时拒绝启动。扩展保护范围后，`factory_supervisor.py`、`config.py`、机器人控制器及六个导航/安全文件与原工作区当前代码逐字节 SHA-256 一致。
- **Code Review C — 通过。** 全量 `unittest` 68 项通过，`git diff --check` 通过；统一 `run_experiments.py` 入口以同 seed 成功运行 Hungarian、GraphPPO、RainbowDQN、QRDQN、CQL、LinUCB 六个 standalone 短仿真并生成完整结果矩阵。该冒烟只证明集成有效，不作为性能或上线结论。

## Workflow Review 2（完工后）

- 状态：通过（2026-10-01）。
- 完整性：五个步骤各有 Code Review A/B/C，共十五轮；发现的问题均在进入下一步前修复并重新审查。
- 算法真实性：GraphPPO 含共享图边编码、clipped PPO、GAE 和 Hungarian 解码；RainbowDQN 含 Dueling、Double-DQN、PER、n-step、C51；QRDQN 使用 quantile-Huber；CQL 使用固定离线数据和 conservative penalty；LinUCB 使用标准闭式线性估计与 UCB 置信项。未发现 TODO、占位 `pass`、`NotImplemented` 或新代码 pickle 加载。
- 工件与入口：高级 workflow 冒烟生成六个可加载 artifact 和严格 JSON 报告；两个 CLI `--help` 正常；统一实验入口以同 seed 跑通 Hungarian 与五种新算法的完整 standalone 结果矩阵。
- 回归：最终全量 `unittest` 68 项通过，`git diff --check` 通过。
- 工作区隔离：主工作区仍保持任务开始时的原有未提交状态；新增 worktree 为 `feature/ai-schedulers`，没有把实现写回主目录。
- 上线边界：workflow 只标记 `research_candidates_ready`，固定 `production_promotion=false`；未执行未见 seed 的真实 Webots 配对评测，因此没有给出生产优于基线的结论。

最终扩展安全文件 SHA-256（与原工作区当前代码逐字节一致）：

```text
5F3D8F9820E2D7815A35DF2F6CEBBCB50148838F00D297E26A6CF0BBB69E94CA  controllers/factory_supervisor/factory_supervisor.py
F1AB62019BA1A870640E532E58E35BE2DC4A72386B845211754D34D1A49F9A3D  controllers/factory_supervisor/config.py
9B807D8760CE77A335419F2C56F535B04FB7FD4786E7C51ACBC965360ACAA97C  controllers/robot_controller/robot_controller.py
5E35449BAFADADC7A67AD1153BA16EF507C329BB7683A5FB422807595F3A1C7E  controllers/factory_supervisor/grid_planner.py
4F357344B3B9B6B6A03554B864BBA677D80F1D5136EF94221CD4302FFB9C8D5B  controllers/factory_supervisor/joint_grid_planner.py
5105E27F1ED422FB15E5C43C5709A06395A1C89C2CBBF63F9EED7D81BD7636FE  controllers/factory_supervisor/motion_coordinator.py
CC832979D16264465D1568F78A6DE2B01A13016ED4A286AE3513F87109933AA6  controllers/factory_supervisor/safety_coordination.py
9CCA0B0DEB8E8A23240AA848C64049873B8DAB4A64C75AB455FA05E5A1082385  controllers/factory_supervisor/collision_safety.py
```
