# KC 逐团结构复用包：FlyWire v783 + MaleCNS v1.0

代码沿用 [NeuronMimic](https://github.com/ydchen0806/neuronmimic)，数据位于 [同一 HF 数据集](https://huggingface.co/datasets/cyd0806/neuronmimic-data/tree/main/structure_handoff_v1)。这是完整 KC 范围的真实解剖结构包，便于逐团、逐细胞分析；不把候选结构升级为已验证功能模块。

| 内容 | FlyWire v783 | MaleCNS v1.0 |
| --- | ---: | ---: |
| KC 名册及完整覆盖骨架 | 5,177 | 4,064 |
| 骨架节点 | 7,024,185 | 4,184,720 |
| 涉及 KC 的物理突触对（至少一端为 KC） | 1,881,678 | 4,027,967 |
| KC 输入突触 | 970,760 | 2,177,431 |
| KC 输出突触 | 1,290,256 | 3,004,381 |
| KC→KC 突触（输入输出两表重叠部分） | 379,338 | 1,153,845 |
| 有向 body/root 连接对 | 888,641 | 1,828,440 |
| 组查询视图 | 38 | 29 |

输入数+输出数−KC内部数=去重物理突触对数。突触数不是唯一神经元连接对数；多输出突触位点可以连接多个后突触。MaleCNS body ID 包含未注释重建片段，不应把连接端点数全称为完整神经元数。

## 师弟先看什么

1. `registry/group_members.parquet`：组名→真实细胞 ID，保留 namespace、状态和证据类别。
2. `registry/cells.parquet`：所有 KC 的注释与胞体位置；未知值未填造。
3. `data/synapse_pairs.parquet`：组内、传入、传出的逐突触前后 ID、**双方坐标**、原始检测质量与脑区字段。
4. `registry/geometry_index.parquet`：每个 KC 对应的骨架 archive/member、SHA256。`geometry/*.tar.gz` 无损保存所有原始几何内容，查询代码自动解出所选细胞。
5. `COORDINATES.json`、`SCHEMAS.json`、`PROVENANCE.json`：坐标单位、全部 Parquet 字段、来源和处理范围。

FlyWire 另含全部2,261,016个KC端点的分支映射、映射误差、路径距离、1,252,322个拓扑分支、伙伴指纹、parent/residual角色、全脑胞体背景和原有脑区/脑表面GLB显示网格。MaleCNS 另含990,665个相关突触前位点的完整递质预测、所有源注释、伙伴注释和递质表、官方聚合权重核对、4,064份原始SWC下载凭据与拓扑QC。

**单位不能混用：** FlyWire 突触表的原始XYZ已经是nm，乘0.001变μm；它的JSON骨架和端点表已经是μm。MaleCNS原始突触XYZ及SWC坐标是8nm体素，乘0.008变μm。FlyWire原始胞体注释的体素变换另为[4,4,40]nm。两套数据没有做共同坐标配准。

## 一次安装，逐团导出

在已有 NeuronMimic checkout 中更新代码，不需要重建仿真系统：

```bash
git pull
pip install -e '.[structure]'
# 本轮已上传并固定的 HF 数据版本。
hf download cyd0806/neuronmimic-data --repo-type dataset \
  --revision b4143c83be3df42a95983d2be34af4354a25a84c \
  --include 'structure_handoff_v1/*' --local-dir data

python scripts/query_kc_structure.py --bundle data/structure_handoff_v1 \
  --dataset flywire_v783 --list-groups
python scripts/query_kc_structure.py --bundle data/structure_handoff_v1 \
  --dataset flywire_v783 --group CM019 --output outputs/CM019_structure
python scripts/query_kc_structure.py --bundle data/structure_handoff_v1 \
  --dataset flywire_v783 --group SU019 --output outputs/SU019_structure
python scripts/query_kc_structure.py --bundle data/structure_handoff_v1 \
  --dataset malecns_v1_0 --group 'TYPE:KCab-p:L' --output outputs/Male_KCabp_left
python scripts/query_kc_structure.py --bundle data/structure_handoff_v1 \
  --dataset malecns_v1_0 --neuron 11862 --output outputs/Male_11862
```

导出含成员表、所有相关突触、方向保持的聚合连接、每个成员的原始骨架及查询计数；FlyWire额外导出所有成员的分支与接点映射，MaleCNS导出实际相关的突触前位点递质预测。现有导出目录拒绝覆盖。`--group ALL_KC` 导出全体，`--no-geometry` 可仅导出表格。ID作为字符串读取，尤其不要让JS把FlyWire root ID转为Number。

实跑例子：CM019132个成员→56,983条去重相关突触；MaleCNS左KCab-p64个成员→66,815条。验证收据在`validation/`。

## 候选版本与完整性边界

- FlyWire38个查询视图=8个冻结stable候选+9个既有共识视图+20个SU条件性结构候选+1个探索性18细胞集合，彼此可以重叠。**不是38个已确认功能单元。** 全部KC及其原始历史状态均保留；未被命名视图覆盖的细胞也有完整几何和接点。旧CM/SU编号不能互相替代。
- MaleCNS29个视图仅为官方已注释KC类型×脑侧；不是新聚类，也没有假定与FlyWire候选一一同源。v0.9历史候选或16份旧SWC没有混入本v1.0包。
- 这里“完整”指上述KC名册范围内，现有发布源中的全部骨架和全部双向相关突触。**不包含全脑EM体素/分割体积、全部神经元表面mesh、非KC伙伴的骨架，或未公开的字段。** FlyWire显示mesh是脑/脑区边界近似；MaleCNS SWC原始头标为NeuTu coarse中心线，不能称为未降采样的完整细胞表面。
- MaleCNS有339,798个不同body端点，其中332,405个没有当前源细胞注释；这些ID及连接全部保留为unknown。这个范围比先前只看一部分输入的31,355缺失端点更广，两者分母不同。没有用其缺注释制造神经元类型。
- MaleCNS接点到具体骨架分支的投影**尚未计算**，没有伪造映射距离。现在可用真实SWC和双方突触坐标继续做这项分析。递质后验和检测置信度分别保存，不把递质或边统一当作兴奋性。
- 不包含真实生理记录、已校准通道参数或已证明功能；这是`VERIFIED_REAL_DATA`来源核对的解剖包，组解释保留原来的`DESCRIPTIVE_ONLY`/探索性状态。

## 校验与来源

所有9,241份骨架逐文件哈希验证；MaleCNS全部骨架父子引用、无环森林拓扑已检查；全部MaleCNS相关突触数与独立官方权重表逐对一致。全部FlyWire端点方向及nm→μm转换核对通过。MaleCNS相关接点的递质位点匹配缺失数为0。

MaleCNS两份固定ID的SWC坐标×8与官方precomputed骨架在float32精度下逐点一致。直接用float64要求绝对相等产生0.0125nm差异，失败诊断保留，没有把舍入误差隐藏掉。公开软件31项测试通过，真实CM019、SU体系独立名册、18细胞与MaleCNS查询均按版本保留。

原始FlyWire图谱的历史审计按原目录保存，因此旧`audit/package_hashes.json`还会引用未打包为逐文件布局的路径；**当前发布校验请使用HF根目录`SHA256SUMS.json`及几何索引。** 原始图谱生成代码保留在GitHub `research/flywire_atlas_source_snapshot/`，新打包、下载、查询与验证代码在`scripts/`和`src/neuronmimic/structure_bundle.py`。

授权与引用见`sources/ATTRIBUTION.md`。原始文件哈希、SWC公开URL和generation、所有质量检查及缺失字段均随包保存。
