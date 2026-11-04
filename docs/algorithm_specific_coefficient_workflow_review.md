# 算法独立系数 Workflow 与两轮审查

## Workflow

1. 冻结源码、设计文档、当前共享系数 warm start 和四个 seed 分区。
2. 分别搜索六个 `policy_id` 的 coefficient bundle。
3. 对搜索产物执行三轮 artifact/code review。
4. 使用选定 bundle 从固定 source checkpoint 完成四个学生策略的 standalone 训练。
5. 对 standalone 产物执行三轮 review。
6. 使用 Webots seeds `41000--41009` 连续微调四个学生策略。
7. 对微调产物执行三轮 review；同一进程立即进入冻结评价。
8. 评价四个学生策略和两个 TrafficHungarian 策略，共
   `6 policies × 3 scenarios × 6 seeds = 108` 次。
9. 生成相对共享系数基线的配对比较并执行三轮 review。

## Workflow Review 1：状态机、恢复与身份

状态：通过，要求如下。

- run root 必须是 `results` 的直接子目录，所有写入经过目录边界检查和原子替换。
- CMA-ES 每代、每个候选训练 episode、候选验证 episode、standalone episode、
  Webots episode 和模型 checkpoint 都具有独立完成标记及 SHA-256。
- 恢复时必须重新验证 manifest、输入 checkpoint、coefficient bundle、训练 contract
  和结果文件身份；任何不一致均 fail closed，不静默覆盖。
- 六个 bundle 不能由 objective 单独索引，必须由完整 `policy_id` 索引，避免
  GraphImitation 与 LearnedHungarian 再次意外共享同一套系数。
- final evaluation 的学生 checkpoint 只能来自最后一轮累计 Webots 微调；
  TrafficHungarian descriptor 只能来自其自身 policy bundle。

## Workflow Review 2：阶段门禁与连续执行

状态：通过，要求如下。

- 设计、系数核心、workflow 实现、静态/测试验证分别记录三种独立视角的 code review。
- coefficient search、standalone、Webots 微调和最终评价各自在进入下一阶段前执行
  三轮 artifact review，且三轮必须覆盖相同文件哈希。
- 不启用退化验证、rollback 或使用最终评价 seeds 的模型选择。
- `physical` 命令内部必须按“全部微调 → 三轮 review → 立即冻结评价 → 三轮 review
  → 配对比较”的顺序执行，中间不存在人工命令边界。
- 进程中断后重新运行同一 `physical` 命令，从最后一个有效 episode/checkpoint 恢复；
  已完成的纯策略结果不会重复运行。
- 汇总必须验证 108/108、配对 manifest、零 fallback、零无效输出和零调度超时，
  并与 `results/expert_coefficients_v1_20261030/evaluation_summary.json` 配对比较。

## 审批结论

两轮 workflow review 均通过。实现不得缩短最终六 seeds，不得把 coefficient
validation、Webots fine-tune 或 final evaluation 结果混入其他阶段的选择逻辑。

## 实现前冻结验证

- 算法独立 workflow、系数核心及兼容性专项：`86 passed, 17 subtests passed`。
- 当前 `tests/` 完整套件：`485 passed, 152 subtests passed`。
- 新增/修改 Python 模块均通过 `py_compile`，`git diff --check` 无空白错误。
- 必须显式使用 `pytest tests`；仓库 `results/` 中存在用于审计的冻结测试副本，
  直接无路径运行 pytest 会重复收集历史副本，不属于当前测试套件。
