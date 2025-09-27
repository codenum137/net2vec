# RouteNet TF2 - 增强版

这是一个基于TensorFlow 2.x实现的增强版RouteNet，在原版RouteNet的基础上，我们引入了三项关键创新，旨在提升模型的性能、泛化能力和物理一致性。

## 🚀 三大核心创新

### 1. **Kolmogorov-Arnold Networks (KAN) 架构**
我们引入了KAN作为替代传统MLP（多层感知机）的选项。

- **工作原理**: KAN的激活函数位于网络的边上，而不是节点上。这些激活函数是可学习的、基于样条的函数，使得网络能够以更高的精度和更少的参数学习复杂的非线性关系。
- **优势**:
  - **更高精度**: 在相同参数量下，KAN通常比MLP具有更高的精度。
  - **可解释性**: KAN的结构使其更容易理解输入变量如何影响输出。
  - **参数效率**: 对于某些复杂函数，KAN可以用更少的参数达到与MLP相同的性能。
- **如何使用**: 在训练脚本中添加 `--use_kan` 参数即可切换到KAN架构。

### 2. **物理约束 (Physics-Informed Learning)**
为了让模型学习到符合网络物理规律的知识，我们引入了基于梯度的物理约束。

#### (1) **软约束 (Soft Constraint)**
- **工作原理**: 将物理约束作为一个正则化项添加到损失函数中。模型在训练过程中会“鼓励”去满足物理规律，但不是强制性的。
  ```
  Total Loss = Original Loss + λ * Physics Loss
  ```
- **优点**:
  - **灵活性**: 允许模型在数据与物理规律有冲突时找到一个平衡点。
  - **稳定性**: 训练过程更稳定，不容易发散。
- **适用场景**: 当数据存在噪声或不完全符合理想物理模型时。

#### (2) **硬约束 (Hard Constraint)**
- **工作原理**: 在每个训练样本上强制模型满足物理约束。这是一种更严格的约束方式，确保模型输出在任何情况下都符合物理规律。
- **优点**:
  - **强一致性**: 保证模型在任何输入下都符合物理直觉。
  - **更好的泛化**: 在分布外（OOD）数据上通常表现更好。
- **适用场景**: 当物理规律非常明确且数据质量高时。

- **如何使用**:
  - `--physics_loss`: 启用物理约束。
  - `--hard_constraint`: 切换到硬约束（默认软约束）。
  - `--lambda_physics`: 调节物理约束的强度 (λ)。

### 3. **课程学习 (Curriculum Learning)**
为了解决物理约束可能导致的训练初期不稳定问题，我们引入了课程学习策略。

- **工作原理**: 训练从易到难。在训练初期，物理约束的权重(λ)为0，让模型先学习数据本身。随着训练的进行，λ的值逐渐从0线性增加到预设的最大值，从而逐步引入物理约束。
- **阶段**:
  1. **热身期 (Warmup)**: λ = 0，模型只学习数据。
  2. **增长期 (Ramp-up)**: λ从0线性增加到最大值。
  3. **稳定期 (Hold)**: λ保持在最大值，进行微调。
- **优势**:
  - **训练稳定**: 避免了在模型还未充分学习时引入强约束导致的训练崩溃。
  - **性能更优**: 通常能达到比固定λ更好的性能。
- **如何使用**:
  - `--curriculum`: 启用课程学习。
  - `--warmup_epochs`: 设置热身期轮数。
  - `--ramp_epochs`: 设置增长期轮数。

## 🛠️ 使用方法

### 训练
统一使用 `train_models.py` 脚本进行训练，通过配置文件 `experiment_config.yaml` 管理所有模型。

**示例: 训练一个带硬约束和课程学习的KAN模型**
```yaml
# experiment_config.yaml
kan_hard_cl_0.5:
  model_type: "kan"
  physics_type: "hard"
  lambda_physics: 0.5
  use_kan: true
  curriculum_learning: true
  warmup_epochs: 5
  ramp_epochs: 10
```
```bash
python train_models.py --models kan_hard_cl_0.5
```

### 评估
使用 `run_experiments.py` 进行评估，它会自动加载模型并进行性能对比。

```bash
python run_experiments.py --models kan_hard_cl_0.5 mlp_none
```

### 结果分析
使用 `collect_performance_results.py` 汇总所有实验结果，并生成详细的Markdown性能分析报告。

```bash
python collect_performance_results.py
```

## 📊 实验管理

- **模型库**: 所有训练好的模型保存在 `fixed_model/` 目录下。
- **实验配置**: `experiment_config.yaml` 集中管理26种不同的模型配置。
- **早停机制**: 所有训练默认启用早停，防止过拟合，节约训练时间。

通过这三大创新，我们显著提升了RouteNet模型的性能和可靠性，使其不仅能“预测”，更能“理解”网络中的物理规律。