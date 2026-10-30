# P0：指标修正与结构证据审计

2026-10-30。按用户提供的P0文字执行；本机未找到`/root/codex_kc_nature.sh`，未假装运行该脚本。复用现有模型和Teacher，修正评估及搜索契约；未重写模拟器、未改变候选、未下载数据、未自动推送。所有动力学新数值均为FIXTURE_TEST_ONLY。

## 已完成

- 旧源码保存到`before/`，历史结果原位保留；`invalidated_results.csv`逐文件列出失效指标、原因及哈希。
- 电压/电导分开；逐时刻对齐；一步预测用所有留出Teacher状态；无响应保留NA并计入FP/FN；非有限预测使评估失败，不静默忽略；负电导比例单独记录。
- 梯度比较同一事件、幅度、时刻和读出；有限差分用两档epsilon，并测试未来输入的因果零梯度。
- 搜索统一为25ms末端房室电压到目标的平方误差；刺激16ms、K=2、每点幅度1。松弛幅度在[0,1]且和为K，最终精确Top-K；三方法各25次Teacher候选评估，梯度每个起点40步。每个候选均由Teacher重评；几何启发式仅依树距离，不能把它写成旧版Teacher单点启发式的同一实验。
- 保留原6训练/3留出刺激、seed20261030、120epochs、Adam0.002；新增不学习的PhysicsOnly，记录训练与推理时间、模型权重和完整源码哈希。
- 23项回归测试通过。两个独立进程复放，非计时模型指标、梯度和搜索输出一致，产物哈希校验通过。

## 修正后的模型结果

|模型|全部留出状态的一步电压RMSE|对齐多步电压RMSE|有效时延比较/测试刺激|
|---|---:|---:|---:|
|MLP|0.068539|5.823275|0/3|
|Graph|0.213696|33.859930|0/3|
|PhysicsResidual|0.000164|0.014663|3/3|
|PhysicsOnly|0.00000244|0.00000815|3/3|

电压单位mV。MLP/Graph均漏检3个参考graded-response，不能以NA时延说明表现良好。Graph负电导比例为1，MLP约0.417；数值有限不等于物理状态合法。PhysicsResidual与PhysicsOnly均未出现负电导。

原PhysicsResidual多步4.58数值受错误广播和通道混算影响，旧梯度cosine约0.044的干预变量也不一致，均不再引用为模型失败证据。修正后该单例梯度相对L2误差约0.00706；但只有一个非零方向及一个未来事件控制，cosine=1并不证明普遍梯度准确。

在相同25次Teacher候选评估下，最佳目标电压误差：几何2.313808、随机0.134327、梯度Top-K0.128074mV。梯度仅略优于随机的最佳值，不作优越性声明；重复提案均计费，唯一提案数分别25/19/14。每次完整基准生成9条Teacher轨迹、36次梯度参考求解、1次搜索目标求解和75次候选求解，共121次Teacher调用；复放不增加独立科学样本数。

**解释：当前已知方程的小电缆上，不学习的物理参考最好。PhysicsResidual比纯数据Student准确，但没有证明比物理求解器更有必要、更快或更可泛化。** Teacher与PhysicsOnly采用同一分辨率与方程，该比较首先是正确性对照。不得把它写成真实KC或高分辨率代理成功。

## 结构与数据审计

173项检查通过，包括52候选冻结得分、coverage、成员保留率复算、stable8/965不变、20单元2125唯一成员、源哈希和ID往返。

- 训练KC图234,964有向对，测试58,798对，直接重叠0；二者相加等于原293,762对。
- 仅从训练图重建spectral32，与存档最大差1.57e-11。
- 15,119条测试边的反向边在训练图中。它不是同方向边直接泄漏，但不是无向pair隔离或独立细胞检验；新协议需处理这类依赖。
- 旧ap2协议明确说明结果已被查看，当前不能将该测试集重称为盲测。
- 20单元仍为图发现，胞体/形态不在本协议发现输入；尺度条件稳定不等于旧clean61跨尺度稳定。
- 已生成CM/SU成员交叉表。CM019与SU019分别132/104成员，不能混名；CM016与SU021为147/98，CM022与SU027为114/84。
- MaleCNS本地输入边表944,638有向聚合对、2,610,758突触；不是全CNS总量。41,105端点body ID中31,355未出现在当前有效注释ID集合。保留unknown，不能归零或硬映射类型。
- FlyWire和MaleCNS字符串ID经Parquet及实际Node.js JSON往返一致。

## P1门槛

允许继续做回顾性方案设计、完整parent特征清单与跨脑QC；暂不生成“独立确认了离散/混合组织”的新结论。必须先明确新评价集的历史使用情况、完整发现/拟合/筛选隔离，并修复跨脑注释覆盖。BANC已出现在本地既有分析，不应说本轮首次引入；其版本、原始数据和论文出处仍须分别核验。

用户给出的2025/2026文献描述尚未完成本轮书目核对，标UNVERIFIED，不用于证明创新性或设计依赖。GitHub已发布版本仍有旧指标；本轮按要求不自动推送，后续发布必须带失效说明及新结果版本。

## 复现

```bash
cd /unify/ydchen/unidit/bio_fly
PYTHONPATH=simulation/src pytest -q simulation/tests/test_physics_guided_scientific_metrics.py simulation/tests/test_physics_guided_protocol.py simulation/tests/test_physics_guided_students.py simulation/tests/test_physics_guided_teacher.py simulation/tests/test_neuronmimic.py simulation/tests/test_male_cns_adapter.py simulation/tests/test_stable8_dynamic_cohesion.py
PYTHONPATH=simulation/src python simulation/scripts/audit_nature_kc_p0.py
PYTHONPATH=simulation/src python simulation/scripts/run_physics_guided_baseline.py --output-dir artifacts/nature_kc_20261030/p0_audit/fixture_v2_reproduction_new
```

基准拒绝覆盖已完成输出；复放须指定新目录。状态入口`P0_STATUS.json`；图件`figures/p0_corrected_baseline.pdf`；新结果`fixture_v2/results/`；完整输入哈希`source_manifest.json`和`fixture_v2/run_manifest.json`。
