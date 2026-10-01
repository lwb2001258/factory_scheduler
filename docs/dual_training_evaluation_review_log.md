# 双训练目标方案文档 Review 记录

文档：`docs/dual_training_evaluation_design.md`

按要求执行两轮详细文档 Review；若发现问题，修订后两轮均重新开始。

## Review 1：公式、实验设计与统计口径

- 状态：通过（2026-10-02）。
- 审查范围：两套目标是否真正独立；固定时域与折扣；3/5/8 场景均衡；优先级、等待、未完成任务和安全的公式；零分母/稀有样本；配对统计与宏平均；训练、验证、测试 seed 隔离。
- 首轮发现并修订：
  - 仅按 episode 1:1:1 采样仍可能让高负载场景 C 主导梯度，改为按学习器类型执行场景分层 minibatch/回放/样本权重。
  - Count 的基线完成数为 0 时不能用分母 1 伪装百分比，改为运行异常并单报绝对差。
  - Utility 参考等待时间可能为 0，距离/延迟及零完成任务存在边界，补充严格正值下限和失败定义。
  - Count 使用 `gamma=0.99` 会暗中优化时效，改为有限时域 `gamma=1.0`；Utility 也默认不折扣，避免不等时长 transition 引入隐藏偏好。
  - episode 必须按仿真时间结束，决策步数只能作为异常保护上限。
  - 补充各类算法的 Count/Utility 接入方式、Utility 分数点口径、优先级 3 最小样本量和完整 Webots 运行次数。
- 修订后重审：所有公式有定义，权重和为 1，未完成任务不会产生幸存者偏差，A/B/C 等权且任一场景不能被其他场景抵消；结构、代码块、空白和 `git diff --check` 通过。

## Review 2：项目字段、可落地性与真实性

- 状态：通过（2026-10-02）。
- 审查范围：逐项对照 `config.py`、`training_scenarios.py`、`task_generator.py`、`metrics_collector.py`、`learning_scheduler.py`、`bandit_scheduler.py` 和现有评估器；检查链接、Git 跟踪和受保护文件边界。
- 首轮发现并修订：
  - 相同 seed 不能自动保证 NumPy 训练生成器与 Webots `random.Random` 生成器产生相同任务，补充 canonical task manifest/统一生成器要求。
  - 当前 cost oracle 静态复用 8 机器人协调器，补充 A/B 中未参与机器人障碍物一致性要求。
  - LinUCB 虽在 diagnostics 中提供 `selected_arm`，任务 `learning_trace` 尚未保存 arm；补充提交时固化 arm/context 版本的实施要求。
  - 明确 `pair_distance_violations` 是 0.50 m 以下的采样数而非物理碰撞，新增连续违规 episode 和真实接触指标要求。
- 修订后重审：A/B/C 分别为 3/5/8 台机器人和 30/15/8 秒任务间隔；优先级概率、时间字段、已有指标和全部数据缺口均与源码一致。新增文档未被忽略，链接有效；Supervisor、机器人控制、路径规划和避让相关文件无差异。
