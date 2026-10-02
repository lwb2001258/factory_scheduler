# 学习型机器人调度实现工作流

## 目标与边界

本工作流把研究建议中风险最低、最容易回退的路线落地为两级能力：

1. **LearnedHungarian**：用历史执行数据训练岭回归成本模型，预测每个“机器人—任务”组合的执行成本，再由 Hungarian 算法完成全局一一匹配。
2. **GraphImitation**：把候选组合构造成二部图边，加入相对行/列统计特征，以 Hungarian 专家标签训练边分类器，再由 Hungarian 保证最终匹配合法。

两种算法都保留原有任务生成、路径规划、充电、安全校验和调度回退逻辑。模型缺失、版本不兼容、输出非有限数或无合法边时，调度器拒绝模型结果并由现有 Hungarian 安全回退接管。

## 阶段和完成条件

### 小步 1：特征与模型基础设施

- 固定特征契约和模型版本。
- 实现确定性的机器人—任务特征提取。
- 实现可保存、可加载、带元数据校验的岭回归模型。
- 实现二部图相对特征和轻量模仿学习边模型。

完成条件：模型往返保存一致；错误版本、错误维度和非有限参数被拒绝；特征与输入顺序无关。

### 小步 2：在线数据闭环

- 只在路径和控制命令均成功后记录分配特征。
- 任务完成时把特征与执行时间一起写入现有指标结果。
- 低电量重排、命令发送失败和调度回退不产生错误标签。

完成条件：完成记录可直接转换为训练样本，同时旧结果文件仍可读取。

### Workflow Review 1

审查特征是否泄漏未来信息、数据是否只在成功提交后生成、模型文件是否具备严格兼容性校验、安全回退是否保持原有行为。

### 小步 3：学习成本 + Hungarian

- 注册 `LearnedHungarian` 调度器。
- 学习模型只替换成本估计，不绕过共享可行性矩阵和最终合法性校验。
- 支持命令行传入模型路径并纳入实验矩阵。

完成条件：合法模型可以调度；缺失/损坏模型自动回退；机器人和任务不会重复分配。

### 小步 4：图结构模仿学习 + 自动化工作流

- 注册 `GraphImitation` 调度器。
- 生成不同机器人/任务数量的工厂快照，以 Hungarian 作为专家。
- 训练、验证、保存候选模型，并执行确定性、合法性和质量门禁。
- 输出机器可读报告，只有通过门禁的候选模型才复制到 `promoted/`。

完成条件：一条命令可从数据生成运行到模型晋级；固定 seed 结果可复现；验证集不与训练集共享 seed。

### Workflow Review 2

审查训练/验证隔离、指标含义、晋级门禁、运行时预算、实验入口和文档可复现性。最后执行全量回归测试。

## 每个小步的三轮审查与测试

每个小步均按以下固定次序执行，任一轮失败都回到实现阶段修复后重新开始该小步的三轮检查：

1. **实现审查**：接口、类型、模型契约、确定性和可维护性；运行该步骤的定向单元测试。
2. **边界/安全审查**：空输入、不可达组合、坏 checkpoint、NaN/Inf、重复 ID、低电量和回退行为；运行边界测试。
3. **回归审查**：检查 diff 与原有算法兼容性；运行相关模块及全项目回归测试。

审查证据记录在自动化测试输出和工作流生成的 `workflow_report.json` 中。

## 数据与模型门禁

- 训练记录必须含特征版本、完整有限特征、调度算法、估计成本和实际执行时间。
- 线上数据按运行 seed/结果文件切分，不能随机拆散同一次仿真的记录后同时放入训练集和验证集。
- 岭回归模型必须在验证集上输出有限、非负成本；报告 MAE、RMSE 和相对基准的误差变化。
- 图模仿模型必须输出 `[0, 1]` 概率；报告边分类准确率、专家匹配重合率和匹配成本差。
- 任何合法性测试失败，模型不得晋级。
- 默认晋级门禁以“无非法输出 + 指标达到脚本参数阈值”为准，阈值可显式调整，但报告必须保留实际阈值。

## 运行方式

在项目根目录执行：

```powershell
python scripts/run_ml_scheduler_workflow.py --output-dir results/ml_scheduler
```

如果已有包含 `task_completions[*].learning_trace` 的实验结果，可追加：

```powershell
python scripts/run_ml_scheduler_workflow.py `
  --results-glob "results/**/*.json" `
  --output-dir results/ml_scheduler
```

之后可分别评估晋级模型：

```powershell
python scripts/run_experiments.py --standalone `
  --scheduler LearnedHungarian GraphImitation `
  --learned-cost-model results/ml_scheduler/promoted/learned_cost.npz `
  --graph-imitation-model results/ml_scheduler/promoted/graph_imitation.npz
```

## 强化学习升级路径

当前上线边界停在“学习打分 + 精确合法匹配”，这是研究文档建议的低风险第一阶段。图强化学习只有在以下条件全部满足后才进入候选：历史数据覆盖高负载、拥堵、低电量和故障场景；离线反事实评估稳定；仿真压力测试优于 GraphImitation；动作掩码、超时和 Hungarian 回退保持不变。届时可用 GraphImitation 权重做行为克隆预训练，再以 PPO/MAPPO 微调，避免从随机策略直接上线。

## 上线前配对评估

离线晋级只表示 checkpoint 合法且离线指标通过，不代表自动替换生产基线。使用训练、验证均未出现的 seed 运行同场景配对实验，再执行：

```powershell
python scripts/evaluate_scheduler_results.py `
  --results-glob "results/benchmark_unseen/*.json" `
  --scenario C --baseline Hungarian `
  --candidate LearnedHungarian GraphImitation `
  --required-runtime-mode webots `
  --deployment-manifest results/ml_scheduler/deployment_manifest.json `
  --output results/ml_scheduler/online_evaluation.json
```

## Webots 一致性与生产数据门禁（2026-10-01）

生产训练标签必须来自真实 Webots 控制循环。结果文件需满足
`experiment_info.runtime_mode == "webots"`；`standalone` 结果只用于快速回归，
不能推动模型进入 `promoted/`。默认 workflow 已执行该门禁；只有显式传入
`--allow-proxy-bootstrap` 时才允许代理数据通过，且该参数仅用于研究和冒烟测试。

在当前 Windows 受限环境中，Webots 可以运行，但 Webots 自身创建 Python 子进程
和本地 IPC 会被系统拦截。实验入口的默认 `--controller-mode auto` 会自动切换到
Webots 官方 external-controller TCP 模式。临时 world 只把 controller 字段改为
`<extern>`；物理场景、传感器、supervisor、机器人控制器、任务生成与调度代码均不变。

采集 Webots 训练数据：

```powershell
$env:SMART_FACTORY_SIM_DURATION = "300"
python scripts/run_experiments.py `
  --scenario C --scheduler Hungarian `
  --seed-values 7001 7002 7003 `
  --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --controller-mode auto `
  --max-parallel 3 `
  --results-dir results/webots_training
```

仅用上述 Webots 结果训练并执行生产门禁：

```powershell
python scripts/run_ml_scheduler_workflow.py `
  --results-glob "results/webots_training/*.json" `
  --required-runtime-mode webots `
  --output-dir results/ml_scheduler `
  --fail-on-gate
```

候选模型仍需用训练/验证均未出现的新 seed，在同一 Webots 场景中与 Hungarian
做配对评估。任何 runtime 不匹配、样本不足、质量退化、安全回归或非法输出都会
保留 Hungarian，并清理旧的 `promoted/` 文件，防止误把 standalone checkpoint
当成生产模型。

默认门禁要求 seed 覆盖一致、吞吐下降不超过 2%、平均和 P95 完成时间增加不超过 5%、安全事件与距离违规不增加、无非法输出或回退提交。调度 P95 延迟必须不超过 `max(20 ms, 同批次 Hungarian 最大 P95 × 1.20)`，避免把 Webots 中双方共同承担的 A* 路径计算误算成模型开销，同时限制候选的相对尾延迟回归。只有通过全部门禁并达到预设优势阈值的候选才会被建议上线；否则部署清单继续指定 Hungarian。

当前真实 Webots 执行记录、逐 seed 配对指标、任务流一致性验证和部署结论见
[`ml_scheduler_online_evaluation.md`](ml_scheduler_online_evaluation.md)。
