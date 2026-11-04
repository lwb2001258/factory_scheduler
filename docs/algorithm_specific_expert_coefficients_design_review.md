# 算法独立专家系数设计与两轮审查

## 目标

为 `GraphImitation`、`LearnedHungarian`、`TrafficHungarian` 分别训练
Count 与 Utility-v2 系数，共六个不可混用的 coefficient bundle。最终选择依据是
部署算法冻结后的业务指标，而不是共享教师本身的指标或学生的拟合损失。

安全可行性、实时路径冲突、路径规划、动作 mask、fallback 规则和碰撞边界保持
共享且不可学习。最终评价继续使用场景 A/B/C 与 seeds `61000--61005`。

## 初始方案

1. 使用当前共享系数结果作为六个搜索的 warm start。
2. 每个算法/目标运行独立的 separable CMA-ES。
3. `TrafficHungarian` 候选直接用冻结策略评价。
4. `GraphImitation` 与 `LearnedHungarian` 的每个候选重新生成监督、重新训练，
   再用冻结学生策略评价。
5. Count fitness 为配对 seed 聚类均值减稳定性惩罚；Utility-v2 额外设置
   相对本算法 warm-start 基线的吞吐量下限。
6. 每套系数、模型、manifest 和结果均绑定 SHA-256，允许按候选和 episode 恢复。

## 方案 Review 1：算法正确性

状态：通过，合入以下约束。

1. 外层 CMA-ES 必须评价最终部署策略。GraphImitation 的分类损失、
   LearnedHungarian 的 Ridge MSE 都不能代替 Count/Utility 业务指标。
2. 学生候选不得复用旧教师训练的模型；每个候选必须从固定 source checkpoint
   重新生成监督并训练，避免模型与 coefficient bundle 身份错配。
3. GraphImitation、LearnedHungarian、TrafficHungarian 使用独立 bundle；
   bundle 中仍同时保留 Count/Utility 两行，但只有对应 policy/objective 的行参加搜索。
4. `travel=1` 继续作为不可辨识尺度的固定锚点，所有可搜索系数保持非负。
5. 路径冲突和可行性规则不进入 CMA-ES 参数空间，防止以降低安全约束换取分数。

## 方案 Review 2：统计、泄漏与计算预算

状态：通过，合入以下修正。

1. coefficient manifests 只有五个 seeds，因此严格拆分为：
   - 候选内层训练：`11000--11002`；
   - 外层选择：`11003--11004`。
2. standalone 正式训练使用 `21000--21019`，Webots 微调使用 `41000--41009`，
   最终冻结评价使用 `61000--61005`；四个分区不得交叉。
3. 搜索从当前共享最优系数开始，仅探索每个可搜索系数的 `0.25--4.0` 倍，
   降低六套双层搜索的计算量与极端系数风险。
4. 每个算法/目标执行两代 CMA-ES；Count population 为 6，Utility-v2 为 7。
   基线候选始终进入最终选择，确保搜索失败时仍有合法参照，但不执行评价回退。
5. fitness 使用 seed 聚类后的主指标均值减 `0.25 × seed 标准差`。
   Utility-v2 候选若完成数低于本算法 warm-start 基线 2 个以上，按缺口扣分。
6. 只有搜索分区可用于选择系数；Webots 微调不做退化验证或 rollback，最终六个
   seeds 只运行一次冻结评价，不能用于重新选择系数、模型或轮次。

## 最终批准的优化定义

对算法 `a`、目标 `o`、系数 `theta`：

```text
model(a, o, theta) = inner_train(a, o, theta, seeds=11000..11002)
score = frozen_metric(model, seeds=11003..11004)
fitness = mean(seed_cluster_score) - 0.25 * pstdev(seed_cluster_score)
```

Utility-v2 额外应用：

```text
throughput_floor = warm_start_mean_completed_tasks - 2
fitness -= max(0, throughput_floor - candidate_mean_completed_tasks)
```

`TrafficHungarian` 没有内层可训练模型，`model` 为绑定候选 bundle 的冻结描述符。

## 预期与解释边界

- 六套系数可能提高每个模型的可学习性或部署性能，但不保证均值一定上升。
- 独立系数后，三个算法不再是“同一专家”的严格控制变量比较；最终业务指标仍可
  公平比较，教师/学生误差则只能在各自 policy 内解释。
- 两个 coefficient validation seeds 的统计能力有限，因此最终结论必须同时报告
  相同六个 Webots seeds 的配对差值、胜平负和 seed 聚类区间。
