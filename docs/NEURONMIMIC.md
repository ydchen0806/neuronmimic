# NeuronMimic：把具身轨迹模仿接到连接组动力学

## 结论先行

可以借用 SuperMimic 类具身工作的思想，但应借用它的**状态估计、轨迹拟合、残差动力学和闭环验证**，不能直接把策略网络、身体运动损失或行为标签当作神经功能证据。当前仓库没有发现名为 `SuperMimic` 的独立源码、论文或 checkpoint；可核实的具身系统是 `embodied_v16`：FlyWire v783 全脑有向 LIF、FlyGym/MuJoCo 身体、虚拟触角和手写的 DNa02→CPG 控制接口。因此本项目把新层命名为 **NeuronMimic**，表示“带机制约束的神经轨迹校准器”，而不是已经完成的 neural twin。

## 当前动力学的可复用基础和缺口

`embodied_v16/build/lif_circuit.py` 已经提供了可复现的全脑事件后端：0.1 ms 步长、1.8 ms 固定延迟、2.2 ms 不应期、全 138,640 节点和 15,091,983 个有向连接对。它与 Brian2 的实现级对照达到事件时间格一致、最大电压误差约 (7.6\times10^{-13}) mV。这是数值一致性结果。

它仍有四个动力学层面的自由度没有被真实细胞数据约束：每突触 0.275 mV 是统一有效权重；递质符号是注释推断而非测得电导；膜/突触时间常数主要是全局参数；ORN 输入是合成 150 Hz 场。具身闭环的 5 ms 感觉交换和 DNa02 增益 0.02 也是接口假设，尚未连接 VNC、运动神经元、肌肉和实测感觉响应。已有结果显示完整网络与局部 KC 干预的洗脱期差异不超过匹配随机掩码，行为闭环的偏转还会被控制增益放大。因此下一步应校准神经动力学本身，再把身体当作独立的外部验证层。

## SuperMimic 思想如何映射到神经模型

| 具身方法中的思想 | NeuronMimic 的对应实现 | 证据边界 |
|---|---|---|
| 从示范轨迹恢复隐状态 | 用电压、脉冲计数和模块读出拟合 (v_i(t),s_i(t)) | 只有有观测的细胞/读出可校准；隐状态不是实测 |
| 机制模型加残差 | 冻结有向连接和 LIF 方程，只学习有界的每细胞 tau、阈值、增益和低秩残差电流 | 残差是校准项，不等于树突机制 |
| 轨迹损失 | 电压 MSE、脉冲 surrogate loss、模块率和时延损失联合优化 | 必须在未参与拟合的刺激/扰动上评估 |
| 闭环观察 | 将感觉输入、KC/MBON/DN 读出和身体状态作为分层 observation operator | 身体读出不能反向定义 KC 功能 |
| 反事实和干预 | 留出刺激、边删除、突触权重扰动、模块静默 | 模型反事实仍不是活体因果 |
| 不确定性 | 多 seed、参数后验/ensemble、记录输入和输出误差 | 技术重复不能称动物重复 |

应避免的迁移是把端到端行为 imitation policy 直接训练成“神经功能”；这会绕过胞体、分枝和突触位置，导致行为拟合很好但无法定位到生物实验对象。

## NeuronMimic 的动力学形式

机制骨架保留为有向、带符号的有效耦合：

\[
\tau_i^m \dot v_i = -(v_i-v_\mathrm{rest}) + g_i(I_i^\mathrm{ext}+s_i+r_i),\qquad
\tau_i^s\dot s_i=-s_i+\sum_{j\to i}w_{ji}\,\phi(v_j).
\]

其中 (w_{ji}) 直接来自输入的有向边，不能在适配时对称化。(r_i) 是受幅度约束的低秩残差：

\[
r(t)=\alpha\,\tanh(B\,\tanh(Ux(t))),\quad \alpha\text{ 在配置中冻结}.
\]

训练阶段用 sigmoid surrogate 近似阈值事件，导出科学事件时仍使用现有经过 Brian2 对照的硬阈值后端。损失写成：

\[
L=L_v+\lambda_sL_\mathrm{spike}+\lambda_rL_\mathrm{rate}+
\lambda_\theta\|\theta-\theta_0\|^2+\lambda_r'\|r\|^2.
\]

最后两项让拟合优先解释观测，而不是用自由残差取代连接组。`simulation/src/bio_fly/neuronmimic/` 已实现该最小层，支持 directed edge、每细胞参数边界、低秩残差和训练/留出分开评估。

## 当前实现和实跑结果

已加入：

- `bio_fly.neuronmimic.NeuronMimic`：可微 surrogate-LIF、保持有向边、可选 rank-(r) 残差；
- `bio_fly.neuronmimic.calibrate`：只在训练轨迹拟合，带参数先验和梯度裁剪；
- `simulation/scripts/run_neuronmimic_fixture.py`：三节点有向链 fixture；
- `simulation/configs/neuronmimic_fixture.yaml`：冻结 fixture 协议；
- `simulation/tests/test_neuronmimic.py`：有向性、边界、训练损失和 seed 复放测试。

fixture 结果仅为 `FIXTURE_TEST_ONLY`：训练电压 MSE 从 0.16293 降到 0.000125，留出驱动 MSE 从 0.0001266 降到 0.0000292。它证明接口能优化和复放，不证明 FlyWire 生理真实性，也不应进入论文主图。

## 与现有 KC 结构图谱的接法

1. **结构冻结**：20 个 finest units、stable8、parent/residual、真实 branch/contact registry 只作为输入和分层 readout，不用动态结果改写成员。
2. **细胞层**：用真实 root、branch、synapse 位置建立子图；输入输出端点按 `fine_unit_candidate_endpoints.parquet` 查询。
3. **模块层**：报告每个单元的 activated-cell fraction、单细胞概率分布、延迟和 MBON readout；不能用“模块任一细胞放电”比较不同规模单元。
4. **验证层**：训练刺激与评价刺激分开；动态特征若参与候选扩展，必须启用新的 protocol 版本，不能回写 v9/v10 冻结候选。
5. **实验层**：优先选择多尺度稳定、parent purity 高、分枝接点可定位且存在上游/MBON route 的少量单元；输出 root/branch/synapse 清单给遗传或成像实验。

## 建议的验证阶梯

| 阶段 | 数据 | 预注册终点 | 当前状态 |
|---|---|---|---|
| 0 | 三节点/小图 fixture | 有向性、seed、损失下降 | 已通过 |
| 1 | Brian2/Arbor 同配置 | 事件时间、膜电压、延迟 | 现有 LIF 已有数值核对；NeuronMimic 需做同配置核对 |
| 2 | 真实 γ-KC/WT 电生理 | 留一记录 RMSE、脉冲时延和基线 | 已有校准审计，但多房室没有跨记录优势；不能升级成生物验证 |
| 3 | 未参与拟合的感觉输入 | response probability、latency、module fraction | 需要真实输入源和可运行后端 |
| 4 | 结构扰动 | 边/分枝/单元静默后的留出响应 | 可做模型反事实，尚非活体因果 |
| 5 | 具身闭环 | 仅作接口一致性与控制敏感性 | 现有闭环可运行，但增益和感觉场未生理校准 |
| 6 | 独立动物/遗传干预 | 单元身份、响应和行为因果 | 未完成，达到 Nature 主刊还需要这一层 |

## 研究创新点和不可跳过的限制

创新点应写成“**结构守恒的机制模型与轨迹校准接口**”：它把全脑连接、分枝/接点、可解释的低秩残差和留出反事实放在同一 provenance 链上。不要写成“用 SuperMimic 证明了 KC 功能”或“完成神经孪生”。

下一轮真正有价值的实证是：用已核实的 3–5 个单源输入，固定背景和 no-stim controls，先在少量真实子图上做 smoke/pilot，再扩展到每源 100 trials；同时用未参与校准的电生理或成像轨迹做留出评价。没有真实输入映射、观测数据或可复现 SuperMimic 源码时，报告 `BLOCKED_MISSING_DATA`，不得用 fixture 代替。

## 可追溯来源

- `embodied_v16/科学结果与模型边界.md`、`embodied_v16/STATE.md`；
- `embodied_v16/build/lif_circuit.py`、`embodied_v16/build/neural_assay.py`、`embodied_v16/build/run_body.py`；
- Shiu et al., *Nature* (2024), “A Drosophila computational brain model reveals sensorimotor processing”；
- NeuroMechFly v2, *Nature Methods* (2024)；
- Pandarinath et al., LFADS, *Nature Methods* (2018)，用于“潜在轨迹拟合”方法比较；
- Chen et al., Neural ODE, *NeurIPS* (2018)，用于连续时间残差建模比较；
- Hasani et al., Liquid Time-Constant Networks, *AAAI* (2021)，用于可变时间常数的比较；
- `nature_methods_connectome/neural_twin_upgrade_20260916/reports/NEURAL_TWIN_DECISION.md`，当前状态同步 pilot 未通过，不能把新接口升级成 neural twin。
