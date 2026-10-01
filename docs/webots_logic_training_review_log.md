# Webots 逻辑一致训练环境审查记录

## Workflow Review 1（编码前）

- 状态：通过（2026-10-02）。
- 基线：`python -m unittest discover -s tests -v`，68 项通过。
- 架构结论：训练 runtime 复用当前配置、`MotionCoordinator` 和调度契约；不实例化假的 Webots `Supervisor`，不伪造传感器/物理结果。
- 语义结论：以当前 `factory_supervisor.py` 为业务真值，采用 16 ms 时钟、取货/送货两段规划、低电量门禁、五秒换电、失败配对和路径预约；旧 standalone 的连续充电不作为当前 Webots 真值。
- 接入结论：十个 AI 调度器均可共享运行时，但训练接口分为动作型 RL、离线 RL、监督/模仿和 contextual bandit 四类，禁止用统一 RL 更新器冒充全部算法。
- 安全结论：AI 不获得路径、速度或避让控制权；八个受保护文件不进入修改范围。

受保护文件 SHA-256 基线：

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

## 每步 Code Review

### 步骤 1：Webots 业务逻辑兼容运行时

- Code Review A — 通过。首次审查发现 waypoint 进入容差后仍被补记为瞬移到精确坐标，已修正为保持测量位置并仅确认到达；同时修复送货路径暂时失败后的重试目标和充电路径重试节流。重新审查后 6 项定向测试、`py_compile`、`git diff --check` 通过。
- Code Review B — 通过。首次审查发现新增测试和文档被白名单式 `.gitignore` 排除，补齐白名单后从 Review A 重启。最终低电量、五秒换电、动态不可达不提交、重复 ID、NaN、确定性及当前 Supervisor 的返充不耗电语义均通过；8 个受保护文件 SHA-256 与基线一致。
- Code Review C — 通过。全量 `unittest` 74 项通过；新 runtime 与旧调度器、学习模型、Webots launcher 测试无回归。运行时明确声明 `business_logic_only`，未把 headless 结果冒充 Webots 物理结果。

### 步骤 2：训练环境接入与奖励

- Code Review A — 通过。首次审查发现分别深复制机器人和任务会在恢复执行中快照时破坏 `current_task` 的规范对象身份，改为一次复制完整对象图并补充恢复测试后重审。最终训练统一使用 `headless`，`abstract` 仅为兼容别名；奖励来自 runtime 完成/失败/死锁事件和实际累计距离。9 项环境测试、29 项高级算法测试及编译通过。
- Code Review B — 通过。确认未来任务仍不可见、动态路径失败不提交、非法 action 失败关闭、Webots 快照模式不改变传入状态、严格 JSON telemetry 有限；旧 `_abstract_execution`/固定完成时间路径已从生产训练代码清除。8 个受保护文件哈希不变。
- Code Review C — 通过。全量 `unittest` 77 项通过，`git diff --check` 通过；RainbowDQN、QRDQN、CQL、GraphPPO 训练确定性以及旧模型/launcher 回归无异常。训练和验证报告标注 `headless_webots_logic`、dynamics 版本、`business_logic_only` 及未模拟项。

### 步骤 3：十个 AI 调度器统一接入

- Code Review A — 通过。首次审查发现旧 PPO 训练入口缺少非有限网络输出的显式拒绝和同权重同 seed 的更新确定性测试，补齐后从 A 轮重审。最终 `step_scheduler` 与 Supervisor 一样只提交完整合法匹配中的第一项；SARSA、DQN、PPO_RL 使用实际 runtime transition/reward，10 个算法均以真实 checkpoint 通过统一运行时执行测试。
- Code Review B — 通过。重复机器人/任务匹配在提交前被拒绝，未来任务仍不进入 action slots，模型异常采用失败关闭；能力注册表恰好覆盖 `LearnedHungarian`、`GraphImitation`、`PPO_RL`、`SARSA`、`DQN`、`GraphPPO`、`RainbowDQN`、`QRDQN`、`CQL`、`LinUCB`。8 个受保护文件 SHA-256 与基线一致。
- Code Review C — 通过。全量 `unittest` 82 项通过，`py_compile` 和 `git diff --check` 通过；10 个调度器的工件加载、统一执行、提交回调和 runtime telemetry 无回归。

### 步骤 4：Workflow、报告与回归

- Code Review A — 通过。训练 workflow 新增候选调度器的端到端 headless 执行门禁，不再只做 checkpoint 加载和静态快照合法性检查；报告稳定列出 10 个算法的学习方法、训练入口、是否消费 runtime reward 和 headless 执行能力。
- Code Review B — 通过。独立 CLI 冒烟通过 8 项门禁，严格 JSON 解析通过；GraphPPO、RainbowDQN、QRDQN、CQL、LinUCB 候选均在 `headless_webots_logic` 中产生实际提交且无拒绝输出。报告明确 `business_logic_only` 及未模拟物理项；8 个受保护文件哈希不变。
- Code Review C — 通过。全量 `unittest` 82 项通过，6 个改动 Python 入口 `py_compile` 通过，`git diff --check` 通过；CLI 帮助、文档、旧学习 workflow 和 Webots launcher 回归通过。

## Workflow Review 2（完工后）

- 状态：通过（2026-10-02）。
- 完整性：4 个实施步骤各完成 A/B/C 三轮审查，共 12 轮；发现问题的步骤均在修复后从 A 轮重新审查。
- 运行时证据：训练环境使用 16 ms 时钟、当前 `MotionCoordinator`、取货/送货状态机、实际逐步距离、电量门禁、五秒换电、失败配对、预约和死锁检查；旧的固定速度直接跳到完成事件实现已移除。
- 算法证据：10 个 AI 调度器均使用实际 checkpoint 通过统一 `step_scheduler` 执行测试；workflow 对 5 个新增候选执行端到端提交门禁，其余算法保留各自正确的监督、模仿或 RL 训练入口。
- 真实性：所有 headless 报告均标记 `headless_webots_logic` / `business_logic_only`，没有把未模拟的 Webots 物理、传感器、本地 DWA、无线传输或碰撞接触描述为已复现。
- 回归与安全：最终全量 83 项测试、workflow CLI 冒烟、严格 JSON、模块导入和 diff 检查通过；8 个受保护文件逐字节哈希与编码前一致。
- 最终刷新：整体 Review 1 修复 telemetry 边界后重新核对本 workflow，四步依赖、10 算法接入矩阵、报告真实性和上线前 Webots 门禁均不变，Workflow Review 2 最终通过。

## 完工后的整体 Code Review

- Overall Review 1 — 通过。首次审查发现显式 `py_compile` 污染了仓库中历史跟踪的 `.pyc`，精确恢复后重启；继续审查又发现尚未 `reset()` 的 headless 环境会将 telemetry 误标为 `webots_snapshot`，修复为 `headless_webots_logic`、`business_logic_only`、`initialized: false` 并补测试后再次从 Review 1 重启。最终完整 diff、状态机、接口和报告契约通过。
- Overall Review 2 — 通过。高级算法 29 项、既有学习调度 30 项通过；失败关闭、fallback、未来任务不可见、非法匹配、NaN/Inf、文件可跟踪性和 8 个受保护文件哈希通过。
- Overall Review 3 — 通过。最终全量 `unittest` 83 项通过；六个关键模块无字节码写入导入成功，10 算法能力契约和最终严格 JSON workflow 报告通过，`git diff --check` 通过，工作区仅包含本任务文件。
