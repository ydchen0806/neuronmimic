"""Derive P1 reports and data-quality checks from actual run artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/p1_replay'

def main():
    reg=pd.read_parquet(OUT/'registry/all5177_identity.parquet')
    pairs=pd.read_parquet(OUT/'data/directed_pair_split.parquet')
    models=pd.read_csv(OUT/'MODEL_COMPARISON.csv').set_index('model')
    ab=pd.read_csv(OUT/'analysis/discovery_ablation.csv').set_index('arm')
    parent=pd.read_csv(OUT/'analysis/complete_parent_splits.csv')
    child=pd.read_parquet(OUT/'registry/parent_child_members.parquet')
    checks=[]
    def check(name,ok,actual):checks.append({'check':name,'passed':bool(ok),'actual':str(actual)})
    check('all_KC_unique_strings',len(reg)==5177 and reg.root_id.nunique()==5177 and reg.root_id.map(lambda x:isinstance(x,str)).all(),len(reg))
    check('stable8_union',reg.stable_backbone.sum()==965,reg.stable_backbone.sum())
    for side,n in [('left',311),('right',310)]:
        got=parent.query('parent_type == "KCab-s" and side == @side').parent_n.unique()
        check('full_KCab-s_'+side,len(got)==1 and got[0]==n,got)
    check('synapse_conservation',pairs.synapses.sum()==379338,pairs.synapses.sum())
    sets={s:set(zip(d.pre_id,d.post_id)) for s,d in pairs.groupby('split')}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        overlap=sets[a]&sets[b];reverse=sets[a]&{(v,u) for u,v in sets[b]}
        check(a+'_'+b+'_direct_and_reverse_isolation',not overlap and not reverse,[len(overlap),len(reverse)])
    check('node_JSON_string_roundtrip',json.loads(json.dumps(reg.root_id.tolist()))==reg.root_id.tolist(),len(reg))
    su=set(reg.loc[reg.unit_id.eq('SU019'),'root_id']);cm=set(reg.loc[reg.candidate_id.eq('CM019')&reg.stable_backbone,'root_id'])
    check('SU019_CM019_not_aliased',len(su&cm)==104 and len(cm)==132,[len(su),len(cm),len(su&cm)])
    pd.DataFrame(checks).to_csv(OUT/'audit/structural_checks.csv',index=False)
    assert all(c['passed'] for c in checks)
    # Complete parent membership, including minority candidate members outside dominant parent.
    summaries=[];members=[]
    for cmid in ['CM012','CM014','CM016','CM019','CM020','CM022','CM031','CM033']:
        cand=reg.candidate_id.eq(cmid)&reg.stable_backbone.astype(bool)
        typ,side=reg.loc[cand].groupby(['frozen_detailed_subtype','official_hemisphere']).size().idxmax()
        par=reg.frozen_detailed_subtype.eq(typ)&reg.official_hemisphere.eq(side)
        summaries.append({'candidate':cmid,'parent_fine_type':typ,'side':side,'candidate_total':int(cand.sum()),'candidate_within_parent':int((cand&par).sum()),'parent_total':int(par.sum()),'residual':int((par&~cand).sum()),'minority_outside_parent':int((cand&~par).sum())})
        for _,r in reg.loc[par|cand].iterrows():
            i=r.name;members.append({'root_id':r.root_id,'candidate':cmid,'parent_type':typ,'side':side,'member_state':'selected_parent' if cand[i] and par[i] else 'minority_outside_parent' if cand[i] else 'parent_residual'})
    pd.DataFrame(summaries).to_csv(OUT/'analysis/parent_candidate_counts.csv',index=False)
    pd.DataFrame(members).to_parquet(OUT/'registry/parent_candidate_residual_atlas.parquet',index=False)
    # Describe where the 16 legacy CM019 residual cells land, without null-based claims.
    residual=set(reg.loc[reg.frozen_detailed_subtype.eq("KCa'b'-ap1")&reg.official_hemisphere.eq('left')&~reg.root_id.isin(cm),'root_id'])
    rr=[]
    for (arm,k),g in child.loc[child.parent_type.eq("KCa'b'-ap1")&child.side.eq('left')].groupby(['arm','k']):
        for c,cg in g.groupby('child'):
            nres=int(cg.root_id.isin(residual).sum())
            rr.append({'arm':arm,'k':k,'child':c,'child_n':len(cg),'CM019_residual_n':nres,'residual_recall':nres/len(residual),'residual_fraction_of_child':nres/len(cg),'evidence':'DESCRIPTIVE_NO_MATCHED_NULL'})
    pd.DataFrame(rr).to_csv(OUT/'analysis/CM019_residual_child_overlap.csv',index=False)
    sub=child.loc[child.parent_type.eq("KCa'b'-ap1")&child.side.eq('left')&child.arm.eq('graph_soma_tree')]
    groups={k:[set(g.root_id) for _,g in sub.loc[sub.k.eq(k)].groupby('child') if residual.issubset(set(g.root_id))] for k in [3,4]}
    assert len(groups[3])==len(groups[4])==1 and groups[3][0]==groups[4][0]
    small=reg.loc[reg.root_id.isin(groups[3][0])].copy()
    small['P1_candidate_id']='P1_ap1L_18_EXPLORATORY';small['status']='NOT_FROZEN_STABLE'
    small.to_parquet(OUT/'registry/ap1L_18_exploratory_members.parquet',index=False)
    # Fixed-model vertex-weighted bootstrap: accounts for shared endpoints, not animals.
    pred=pd.read_parquet(OUT/'data/test_predictions.parquet'); ids=sorted(set(pred.pre_id)|set(pred.post_id)); idx={r:i for i,r in enumerate(ids)}
    a=pred.pre_id.map(idx).to_numpy();b=pred.post_id.map(idx).to_numpy();y=pred.edge_present.to_numpy()
    losses={name:-(y*np.log(np.clip(pred[name],1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-pred[name],1e-12,1-1e-12))) for name in models.index}
    contrasts=[('M1','M2_descriptor'),('M2_descriptor','M3_descriptor_discrete'),('M2_descriptor','M4_descriptor_continuous'),('M3_descriptor_discrete','M4_descriptor_continuous')]
    diffs=np.column_stack([losses[x]-losses[z] for x,z in contrasts]);rng=np.random.default_rng(20261104);boot=[]
    for _ in range(1000):
        w=rng.poisson(1,len(ids));dw=w[a]*w[b];boot.append(dw@diffs/dw.sum())
    ci=np.quantile(boot,[.025,.975],axis=0)
    pd.DataFrame([{'base':x,'augmented':z,'delta_log_loss':float(diffs[:,j].mean()),'interval_low':ci[0,j],'interval_high':ci[1,j],'method':'POSTHOC_FIXED_MODEL_VERTEX_POISSON_BOOTSTRAP','n_resamples':1000,'seed':20261104,'limitation':'within-brain endpoint dependence only; no refit, spatial block, tuning or animal uncertainty'} for j,(x,z) in enumerate(contrasts)]).to_csv(OUT/'analysis/vertex_bootstrap_exploratory.csv',index=False)
    report=f'''# P1-v2.2：身份、双向隔离与真实空间融合试验

本轮已完成 P1A–P1C 的可执行试验，以及 P1D 的描述量级模型比较。原 stable8 仍为 8 个、965 个成员；新增稳定功能单元数为 0。全部结果来自实际运行，未用 fixture 替代结构数据。

## 当前结论

1. 胞体与树几何已真正进入发现目标。全体 5,177 KC 的 5,177×5,177 欧氏距离矩阵按 μm 生成；树特征从真实骨架计算，未读取接点标签。六个固定消融方案、每方案五个初始化及三次 20% 无序对删边测试均已运行。
2. 空间更紧凑不等于更稳定。纯图的平均簇胞体半径为 {ab.loc['graph','mean_cluster_soma_radius_um']:.2f} μm；大胞体权重降至 {ab.loc['graph_soma4','mean_cluster_soma_radius_um']:.2f} μm，但删边 ARI 为 {ab.loc['graph_soma4','pair_dropout_ARI_mean']:.3f}，未超过纯图的 {ab.loc['graph','pair_dropout_ARI_mean']:.3f}。图+胞体+树的删边 ARI 为 {ab.loc['graph_soma_tree','pair_dropout_ARI_mean']:.3f}，而平均半径仍为 {ab.loc['graph_soma_tree','mean_cluster_soma_radius_um']:.2f} μm。这里的 ARI 是整体分区重现性，不是旧 frozen S/coverage，也不代表个别新簇通过稳定门槛。
3. 几何描述之后，离散标签还有预测增量，但连续训练图表征更强。测试 log loss：M0（已知粗/细类型+侧别）{models.loc['M0','test_log_loss']:.5f}；M1（另加胞体/距离/训练图强度）{models.loc['M1','test_log_loss']:.5f}；M2_descriptor（另加树描述量）{models.loc['M2_descriptor','test_log_loss']:.5f}；M3（另加离散标签）{models.loc['M3_descriptor_discrete','test_log_loss']:.5f}；M4（另加连续图嵌入）{models.loc['M4_descriptor_continuous','test_log_loss']:.5f}。这不等价于“连续几何胜出”：M4 使用的是连续图表征；M2 还没有精细区域树接近性和分枝接点，M4 特征维数也更高。

4. CM019 residual 有具体追踪线索：融合胞体与树几何后，K=3 与 K=4 得到完全相同的18细胞子簇，含全部16个原 residual 与2个原CM019成员；K=2时它们进入20细胞子簇。成员见 registry/ap1L_18_exploratory_members.parquet。这是融合特征空间中的分离，不能直接推断其胞体构成单一连续空间团；尚需扰动、空间匹配零模型、输入输出留出与跨脑检验。

## 数据与协议

FlyWire v783，293,762 条有向神经元对、379,338 个聚合突触计数。无序 pair 哈希决定训练/验证/测试：205,483 / 44,335 / 43,944 条有向边；三个集合之间同向和反向直接交集均为 0。所有图嵌入与度/强度仅来自训练图。接点派生特征在本轮完全排除；这不是已经完成接点级训练测试映射。

预测样本为 60,000 正例和 60,000 非边；训练/验证/测试样本 83,858 / 18,168 / 17,974。各模型同为两个超参设置×100轮，验证集选择，测试集只评估最终设置。这是 case-control 采样，预测值不是自然患率下的连接概率，更不是突触传递概率。原全图与历史测试数据曾参与开发，本次所有切分均标记回顾性，不能称为盲测。

旧 CM、SU、新 P1 分区独立命名。SU019 与 CM019 为 104/132 重叠。P1 全局 K=30 仅为固定分辨率消融，并非新增 30 个稳定单元。局部试验覆盖完整细类型 parent：左右 KCab-s 311/310、ap1 148/133、KCab-p 67/61；K=2/3/4 是探索尺度，全部子簇成员与残余可追溯。CM019 的 16 个 residual 在各子簇中的去向见 CM019_residual_child_overlap.csv；没有匹配零模型和独立验证前，不称为一致补集。

## 修正与失败项

首轮 p1_v2 使用粗粒度 official_KC_subtype，漏掉 KCab-s parent。该试跑原样保留；当前 p1_v2_fine_types 使用 frozen_detailed_subtype 定义完整 parent，M0 同时保留粗/细标签。右 ap1 的粗标签计数132、细标签133不同，未静默覆盖。

MaleCNS 的完整原始 id 列有211,577个唯一ID，body列187,016个；已存在 body 的行两者一致。但换用完整 id 后仍有31,355个边端点未匹配，不能靠填充body解决。跨版本/来源/范围需继续查原始连接表生产链。本轮不宣称跨脑复现。

本地旧跨连接组比较的 provenance 指向名含 invalid_seed_schedule 的历史benchmark，不能升级为独立复现。BANC 与 FlyWire 的样本来源关系亦须核实，数据集名称不同不能自动算不同动物。

用户提供 Pleijzier 2026 预印本已登记为竞争基线，当前代理访问 Crossref/bioRxiv 失败，题目、日期与具体主张仍待原文核查；不把转述当已核实文献结论。

## 剩余裁决工作

P1D 仍需区域树/分枝接点的训练隔离、模型容量与调参预算敏感性、空间/细胞阻断、类型内困难负例、匹配零模型、PN→KC 与 KC→MBON 的预留终点。附加的1000次顶点权重bootstrap仅为固定模型、同脑内的探索性区间，未覆盖模型重拟合、空间相关和动物变异。

P1E 须先解决 MaleCNS 来源映射、BANC 生物独立性与其他数据版本，再比较可迁移组织规律及实验可定位性。目前不应指定“已验证的2–3个功能微回路”。不继续优化 NeuronMimic 架构。

## 复现与产物

仓库命令：`PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1_v2.py`（已完成目录拒绝覆盖）。公开紧凑数据复现命令见 GitHub 的 docs/P1_REPRODUCE.md。发布包含双向切分、身份与父子登记、树描述量、冻结协议、比较表、图及原始输入哈希；完整骨架须向原数据源获取。

关键文件：MODEL_COMPARISON.csv、P1_STATUS.json、analysis/discovery_ablation.csv、analysis/complete_parent_splits.csv、registry/parent_child_members.parquet、audit/structural_checks.csv、audit/source_manifest.json。
'''
    (OUT/'P1_REPORT.md').write_text(report)
    (OUT/'EXECUTION_LOG.md').write_text('''# Execution log

- P1-v2.1: original coarse-type pilot completed; preserved in ../p1_v2. Identified omitted KCab-s fine parent and weaker M0.
- P1-v2.2: protocol amended before rerun; P1A/B/C and descriptor-level D pilot completed. See audit/run.log and source_manifest.json for actual timings, seed and source hashes.
- Command: PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1_v2.py
- Validation: PYTHONPATH=simulation/src pytest -q simulation/tests/test_p1_structure.py
- Posthoc report: python simulation/scripts/finalize_kc_p1_v2.py
- MaleCNS mapping / independent-connectome confirmation: BLOCKED. Exact missing endpoints are in registry/malecns_unmatched_endpoints.csv.
- Literature primary-source retrieval: BLOCKED_NETWORK for Crossref/bioRxiv; HF proxy transport is working.
- NeuronMimic architecture optimization: not part of this structural experiment.
''')
    print('structural checks',len(checks),'passed; report written')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    OUT=parser.parse_args().output_dir.resolve()
    main()
