# GraphImitation / LearnedHungarian 专家系数学习设计与三轮审查

## 目标

在不学习可行性、路径合法性或碰撞安全边界的前提下，为 Count 与
Utility-v2 分别学习实时交通 Hungarian 专家的软成本系数；再用同一版专家
重新生成 GraphImitation 的匹配标签和 LearnedHungarian 的完整成本面标签，
依次完成 standalone 训练、Webots 微调和 seeds 61000--61005 的冻结评价。

## 拟采用的模型与搜索

- 专家仍是可解释线性成本模型，路径系数固定为 1，消除 Hungarian 对成本
  整体正比例缩放的不辨识性。
- Count 搜索拥堵、预计交通等待、历史等待、停车事件四个非负系数；优先级、
  等待年龄和 deadline 系数固定为 0。
- Utility-v2 在上述四项之外搜索优先级、等待年龄和 deadline pressure，
  共七个非负系数。
- 候选参数采用相对当前值的有界对数倍率，默认倍率 1，范围 0.1--10。
- 使用确定性的 separable CMA-ES 在 standalone 训练 seeds 上搜索；所有候选
  使用相同 manifests，fitness 为按 seed 聚类后的三场景均值减去稳定性惩罚。
- 最终评价 seeds 61000--61005 不参与搜索、训练、模型选择或 Webots 微调。

## Review 1：算法正确性

状态：通过，合入以下修正。

1. 专家系数优化必须先直接评价 TrafficHungarian，不能通过学生的 imitation
   loss 间接判断系数优劣；否则只会得到“更容易模仿”的教师，而不一定获得
   更高完成数或 Utility。
2. GraphImitation 的39维 Logistic 权重和 LearnedHungarian 的33维 Ridge
   系数是学生参数，不是专家公式的比例。checkpoint 必须额外绑定专家系数
   bundle 的版本和哈希。
3. Count 与 Utility-v2 必须独立搜索。Count 不引入 priority/wait/deadline
   奖励，避免目标语义被悄悄改成 Utility。
4. 先验证默认系数经过新参数化后逐元素复现旧专家成本，再允许任何搜索。

## Review 2：泄漏、安全与统计口径

状态：通过，合入以下修正。

1. 路径可达性、failed-pair mask、Hungarian 一对一约束和物理碰撞处理保持
   硬约束；CMA-ES 只能修改可行边之间的软成本排序。
2. 评价 seeds 严格隔离。系数搜索、student standalone、Webots 微调与最终
   评价分别使用互不重叠的注册分区，workflow 对越区 seed fail closed。
3. 所有候选在相同 seed/task manifest 上成对比较。动作改变后续状态是需要
   优化的真实轨迹效应，不尝试伪造同一状态的反事实结果。
4. 不直接用18个跨场景样本的标准差惩罚，因为场景规模差异会被误当成不稳；
   先对每个 seed 计算 A/B/C 均值，再对 seed 均值计算稳定性惩罚。
5. 系数保持非负且符号由公式决定，路径系数固定为 1；所有值、fitness 和
   checkpoint metadata 必须有限。

## Review 3：执行成本、恢复和审计

状态：通过，合入以下修正。

1. 第一版采用小规模、确定性、可恢复的 CMA-ES；每个 generation 原子落盘，
   已完成候选按系数哈希复用，避免中断后重跑。
2. standalone 学生训练和 Webots episode 都以输入模型哈希、manifest 哈希、
   专家系数哈希和目标身份作为恢复键；不允许覆盖不匹配的旧结果。
3. Webots 微调结束后由同一 workflow 立即进入冻结评价，不插入退化回滚；
   评价阶段关闭学习和 fallback。
4. 每个 workflow 小步骤必须有三份针对相同源码哈希的 review receipt；源码
   在 review 后变化会使门禁失效。
5. 最终同时报告优化 TrafficHungarian、GraphImitation、LearnedHungarian 的
   Count/Utility 双目标结果、逐场景逐 seed 数据、碰撞、fallback 和模型身份。

## 审查结论

三轮审查均通过。推荐实现为：版本化专家系数 bundle + standalone CMA-ES
搜索 + 两个学生的同教师重训 + Webots 微调 + 同一批6 seeds冻结评价。
