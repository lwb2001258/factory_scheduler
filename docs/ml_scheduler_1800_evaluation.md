# 1800 秒 Webots 调度训练与配对评估报告

## 结论（2026-10-01）

在场景 C、8 个机器人、真实 Webots 物理控制循环下，本轮完成了 10 个训练 seed
和 5 个独立评估 seed。评估包含 Hungarian、LearnedHungarian、GraphImitation、
GA、SA，共 25 次，每次均为 1800 仿真秒。评估期间任何时刻最多运行 3 个
Webots 实例。

最终部署清单选择 **LearnedHungarian**：相对 Hungarian 的配对平均吞吐提升
7.50%，平均完成时间下降 36.99%，平均等待下降 48.88%；吞吐变化的 bootstrap
95% 区间为 +2.59%～+11.45%。5 个 seed 中吞吐 4 胜 1 负、平均完成时间 5 胜。
候选的安全事件和距离违规均为 0，且没有非法输出或回退提交。

- **SA** 通过全部非劣门槛，但没有达到预设优势阈值，状态为
  `validated_candidate`，不替换 LearnedHungarian。
- **GA** 的吞吐平均提升 2.98%，但最大调度 P95 延迟 108.44 ms 超过
  95.53 ms 预算，因此被拒绝。
- **GraphImitation** 的吞吐下降 4.06%、平均完成时间增加 17.38%，且最大调度
  P95 延迟 142.80 ms 超预算，因此被拒绝。

机器可读报告位于 `results/ml_scheduler_1800/online_evaluation.json`，部署结论位于
`results/ml_scheduler_1800/deployment_manifest.json`。运行时仍必须保留 Hungarian
合法性检查和安全回退。

## 实验设计与数据边界

- Webots：R2023b，`--batch --mode=fast --no-rendering`，外部 TCP 控制器模式。
- 场景：C，8 个机器人，高任务到达率。
- 单次时长：1800 仿真秒；墙钟保护上限 7200 秒。
- 成功训练 seed：7001、7002、7003、7004、7005、7011、7012、7013、7014、
  7015。
- 独立评估 seed：7301、7302、7303、7304、7305。
- 评估并发：按算法分批运行；每批先 3 个 seed 并行、再 2 个 seed 并行，峰值
  始终为 3 个 Webots。不同算法不交叉运行，减少 CPU 竞争差异。
- 失败尝试 7006～7010 没有产生完整结果，其中 7004 的首次尝试也未完整落盘；
  它们均未进入训练。7004 由使用 7200 秒墙钟保护的完整重试替代。
- 所有生产数据均满足 `runtime_mode=webots` 和 `sim_duration≈1800`；standalone
  数据未进入训练或评估。

训练集共 10 个 Webots 文档、1211 条真实完成样本。按运行 seed 分组后，8 个
seed（971 条）用于训练，2 个 seed（240 条）用于验证，二者没有 seed 重叠。
7013 的 Hungarian 训练运行出现 1 次距离安全事件，其余 9 个训练 seed 为 0；
该事实保留在原始结果中，没有被隐藏或改写。

## 离线训练门禁

LearnedHungarian 使用真实 `execution_time` 作为 Ridge 目标；GraphImitation 使用
独立生成的工厂快照学习 Hungarian 专家匹配。离线工作流的 12 项门禁全部通过：

| 项目 | 结果 |
|---|---:|
| Ridge 验证样本 | 240 |
| Ridge MAE / RMSE | 31.88 s / 45.10 s |
| 原始路径成本基线 MAE | 95.23 s |
| Ridge 推理 P95 | 0.024 ms |
| Graph 专家匹配重合率 | 68.52% |
| Graph 平均目标差距 | 0.665% |
| Graph 决策 P95 | 0.174 ms |
| 非有限预测 / 非法匹配 | 0 / 0 |
| 确定性重训练 / checkpoint 往返加载 | 通过 / 通过 |

模型位于：

- `results/ml_scheduler_1800/promoted/learned_cost.npz`
- `results/ml_scheduler_1800/promoted/graph_imitation.npz`

## 1800 秒 Webots 配对结果

绝对指标为 5 个 seed 的均值；安全和距离违规为 5 个 seed 的总数；延迟取五次
运行中最大的调度 P95。

| 算法 | 吞吐（任务/分） | 平均完成时间（s） | 平均等待（s） | 安全/距离违规 | 最大调度 P95（ms） | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Hungarian | 3.820 | 388.01 | 284.60 | 2 / 2 | 79.61 | 基线 |
| LearnedHungarian | 4.107 | 244.17 | 145.14 | 0 / 0 | 90.42 | 优于基线，选择部署 |
| GraphImitation | 3.660 | 452.40 | 346.13 | 0 / 0 | 142.80 | 拒绝 |
| GA | 3.927 | 390.52 | 289.24 | 0 / 0 | 108.44 | 拒绝 |
| SA | 3.873 | 395.05 | 293.01 | 1 / 1 | 81.40 | 非劣候选，无优势 |

配对相对变化如下，正吞吐表示更好，负完成/等待时间表示更好：

| 候选 | 吞吐变化 | 平均完成变化 | P95 完成变化 | 平均等待变化 | 吞吐 bootstrap 95% 区间 |
|---|---:|---:|---:|---:|---:|
| LearnedHungarian | +7.50% | -36.99% | -42.51% | -48.88% | +2.59%～+11.45% |
| GraphImitation | -4.06% | +17.38% | +1.31% | +23.07% | -9.54%～+0.89% |
| GA | +2.98% | +0.65% | +0.27% | +1.68% | -2.05%～+9.58% |
| SA | +1.47% | +1.88% | +1.52% | +3.06% | -0.17%～+4.06% |

延迟门槛为 `max(20 ms, Hungarian 最大 P95 × 1.20)`，本轮允许上限为
95.53 ms。GA 只因延迟门槛失败；GraphImitation 同时失败于吞吐非劣、平均完成
时间非劣和延迟门槛。LearnedHungarian 与 SA 通过全部门槛，但只有
LearnedHungarian 达到优势阈值。

逐 seed 吞吐如下：

| Seed | Hungarian | LearnedHungarian | GraphImitation | GA | SA |
|---:|---:|---:|---:|---:|---:|
| 7301 | 3.833 | 3.800 | 3.867 | 3.967 | 3.867 |
| 7302 | 3.600 | 3.933 | 3.633 | 4.133 | 3.833 |
| 7303 | 3.900 | 4.100 | 3.400 | 3.700 | 3.867 |
| 7304 | 3.933 | 4.467 | 3.533 | 3.933 | 3.967 |
| 7305 | 3.833 | 4.233 | 3.867 | 3.900 | 3.833 |

## 同 seed 任务一致性

5 个评估 seed 在五种算法下的任务生成总数分别始终为 233、222、245、222、
247，25 个结果没有任何 seed 的数量不一致。任务生成器还有自动化测试验证：同一
seed 会产生完全相同的 task id、到达时间、取货点、送货点和优先级；5% 高优先级
3、20% 中优先级 2、75% 低优先级 1 的随机边界也有独立测试。因此在机器人数量、
场景、seed 和生成器调用节奏相同时，调度算法不会改变任务定义，只会改变任务的
分配和完成过程。

本轮为减少 I/O 使用 quiet 模式，结果 JSON 只保存完整完成记录而不是全部未完成
任务的定义。因此“完整 task 属性一致”由确定性生成器测试保证，本轮结果文件直接
交叉验证的是每个 seed 的生成总数和运行配置；不把未保存的字段宣称为本轮日志的
直接观测。

## PyTorch 与 GPU

设备为 NVIDIA GeForce RTX 4060 8 GB，驱动 566.24，驱动报告 CUDA 12.7；
Python 为 3.12.6 64 位。Masked Dueling Expected SARSA 使用 CUDA 12.6 构建
`torch==2.14.1+cu126`。安装命令为：

```powershell
$env:PIP_NO_INDEX = $null
python -m pip install -r requirements-torch-cu126.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

2026-10-28 已安装到 Webots 实际使用的 `C:\Python312\python.exe`，CUDA、cuDNN、
前向和反向传播均验证通过。Standalone 神经 SARSA 使用 GPU；多 Webots 实例中的
推理和物理微调使用 CPU，避免进程争用 GPU。

## 复现命令

训练数据采集建议稳定使用 3 实例：

```powershell
$env:SMART_FACTORY_SIM_DURATION = "1800"
$env:SMART_FACTORY_WEBOTS_WALL_TIMEOUT_SECONDS = "7200"
$env:SMART_FACTORY_ROBOT_QUIET = "1"
$env:SMART_FACTORY_SUPERVISOR_QUIET = "1"

python scripts/run_experiments.py `
  --scenario C --scheduler Hungarian `
  --seed-values 7001 7002 7003 7004 7005 7011 7012 7013 7014 7015 `
  --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --controller-mode auto --max-parallel 3 `
  --results-dir results/webots_training_1800

python scripts/run_ml_scheduler_workflow.py `
  --results-glob "results/webots_training_1800/experiment_*.json" `
  --required-runtime-mode webots `
  --output-dir results/ml_scheduler_1800 `
  --fail-on-gate
```

评估按算法顺序运行，`--max-parallel 3` 不变。PowerShell 的 `foreach` 会等待
当前 Python 进程退出后再运行下一算法，因此 Hungarian、GA、SA 不会互相重叠：

```powershell
foreach ($scheduler in @("Hungarian", "GA", "SA")) {
  python scripts/run_experiments.py `
    --scenario C --scheduler $scheduler `
    --seed-values 7301 7302 7303 7304 7305 `
    --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
    --controller-mode auto --max-parallel 3 `
    --results-dir results/webots_evaluation_1800_7301_7305
}

python scripts/run_experiments.py `
  --scenario C --scheduler LearnedHungarian `
  --seed-values 7301 7302 7303 7304 7305 `
  --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --controller-mode auto --max-parallel 3 `
  --learned-cost-model results/ml_scheduler_1800/promoted/learned_cost.npz `
  --results-dir results/webots_evaluation_1800_7301_7305

python scripts/run_experiments.py `
  --scenario C --scheduler GraphImitation `
  --seed-values 7301 7302 7303 7304 7305 `
  --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --controller-mode auto --max-parallel 3 `
  --graph-imitation-model results/ml_scheduler_1800/promoted/graph_imitation.npz `
  --results-dir results/webots_evaluation_1800_7301_7305

python scripts/evaluate_scheduler_results.py `
  --results-glob "results/webots_evaluation_1800_7301_7305/**/experiment_*.json" `
  --scenario C --baseline Hungarian `
  --candidate LearnedHungarian GraphImitation GA SA `
  --required-runtime-mode webots `
  --deployment-manifest results/ml_scheduler_1800/deployment_manifest.json `
  --output results/ml_scheduler_1800/online_evaluation.json
```

## 审查与测试证据

Workflow Review 1 检查了特征无未来信息、完成后才产生训练标签、模型元数据与
安全回退边界；Workflow Review 2 检查了 seed 分组隔离、离线门禁、五算法配对
覆盖、延迟预算和部署决策。各小步均执行了三轮检查：定向单元测试、边界/契约与
语法审查、全项目回归。最终证据包括：

- 启动器定向测试 9 项通过；模型/工作流定向测试 8 项通过；在线评估器定向测试
  9 项通过。
- 25 个评估 JSON：5 算法 × 5 seed，0 重复键、0 非 Webots/非 1800 秒结果、
  0 非法调度器输出、0 回退提交、0 任务总数错配 seed。
- 全项目 171 项自动化测试通过；Windows 沙箱不可写的系统临时目录通过仅用于
  测试的项目内临时目录适配器处理，没有修改产品逻辑。
