"""Summarize executed P1D-v3 evidence; no candidate promotion or biology claims."""
from pathlib import Path
import json,hashlib,sys,subprocess,importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nature_kc_20261030/p1d_v3'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def bootstrap_contrast(pred,base,augmented,evaluation=None):
    d=pred if evaluation is None else pred[pred.evaluation.eq(evaluation)]
    if 'seed' in d:
        d=d.copy();prob=np.clip(d.prediction.to_numpy(),1e-12,1-1e-12);y=d.observed_edge.to_numpy()
        d['loss']=-(y*np.log(prob)+(1-y)*np.log1p(-prob))
        v=d.groupby(['pre_id','post_id','model']).loss.mean().unstack('model')
        delta=(v[base]-v[augmented]).to_numpy()
    else:
        v=d.pivot(index=['pre_id','post_id','observed_edge'],columns='model',values='prediction')
        y=v.index.get_level_values('observed_edge').to_numpy()
        loss={c:-(y*np.log(np.clip(v[c],1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-v[c],1e-12,1-1e-12))) for c in [base,augmented]}
        delta=(loss[base]-loss[augmented]).to_numpy()
    ids=sorted(set(v.index.get_level_values('pre_id'))|set(v.index.get_level_values('post_id')));lookup={r:i for i,r in enumerate(ids)}
    a=np.array([lookup[r] for r in v.index.get_level_values('pre_id')]);b=np.array([lookup[r] for r in v.index.get_level_values('post_id')]);rng=np.random.default_rng(2026113000);draws=[]
    for _ in range(1000):
        w=rng.poisson(1,len(ids));weights=w[a]*w[b]
        if weights.sum():draws.append(np.average(delta,weights=weights))
    lo,hi=np.quantile(draws,[.025,.975])
    return {'evaluation':evaluation or 'nested_shared_mapping_seed_average','base':base,'augmented':augmented,'delta_log_loss':delta.mean(),'interval_low':lo,'interval_high':hi,'n_test_cells':len(ids),'n_test_pairs':len(v),'method':'POSTHOC_FIXED_MODEL_VERTEX_POISSON_BOOTSTRAP','n_resamples':len(draws),'seed':2026113000,'limitation':'within-brain endpoint dependence; excludes retraining/tuning/animal uncertainty'}

def main():
    gate=json.loads((OUT/'audit/GATE_STATUS.json').read_text());geo=json.loads((OUT/'audit/GEOMETRY_STATUS.json').read_text())
    male=json.loads((OUT/'audit/CROSSBRAIN_SOURCE_QC.json').read_text());route=json.loads((OUT/'audit/ROUTING_STATUS.json').read_text())
    model=pd.read_csv(OUT/'analysis/MODEL_COMPARISON.csv');nested=pd.read_csv(OUT/'analysis/nested_shared_geometry_comparison.csv')
    perturb=pd.read_csv(OUT/'analysis/CM019_18_perturbations.csv');density=pd.read_csv(OUT/'analysis/density_matched_dropout.csv')
    pred=pd.read_parquet(OUT/'data/model_test_predictions.parquet');npred=pd.read_parquet(OUT/'data/nested_test_predictions.parquet')
    cis=[bootstrap_contrast(pred,'M1_descriptor','M2_regional_arbor','pair_grouped'),bootstrap_contrast(pred,'M1_descriptor_no_graph_degree','M2_regional_arbor_no_graph_degree','cell_disjoint_geometry'),bootstrap_contrast(npred,'M2_fixed_geometry','M3_plus_frozen18'),bootstrap_contrast(npred,'M2_fixed_geometry','M4_plus_graph_PC3')]
    pd.DataFrame(cis).to_csv(OUT/'analysis/paired_model_uncertainty.csv',index=False)
    drop=perturb[perturb.condition.str.startswith('pair_drop')].groupby('K').agg(mean_jaccard=('jaccard','mean'),min_jaccard=('jaccard','min'),mean_retention=('retention','mean'))
    drop.to_csv(OUT/'analysis/dropout_summary.csv')
    nwide=nested.pivot(index='seed',columns='model',values='log_loss');gain=nwide.M2_fixed_geometry-nwide.M3_plus_frozen18
    sg=model.set_index(['evaluation','model']);g1=sg.loc[('pair_grouped','M1_descriptor'),'log_loss'];g2=sg.loc[('pair_grouped','M2_regional_arbor'),'log_loss'];b1=sg.loc[('cell_disjoint_geometry','M1_descriptor_no_graph_degree'),'log_loss'];b2=sg.loc[('cell_disjoint_geometry','M2_regional_arbor_no_graph_degree'),'log_loss']
    old=ROOT/'artifacts/nature_kc_20261030/p1_v2_fine_types';manifest=json.loads((old/'audit/source_manifest.json').read_text());oldhash={s['path']:s['sha256'] for s in manifest['sources']}
    sources=pd.read_csv(OUT/'audit/regional_geometry_sources.csv',dtype={'root_id':str});checks=[]
    for _,r in sources.iterrows():
        path=f'artifacts/wu_20261030_all5177_structural_atlas/geometry/{r.root_id}.json';checks.append({'root_id':r.root_id,'matches_P1_frozen_source':oldhash[path]==r.geometry_sha256})
    assert all(c['matches_P1_frozen_source'] for c in checks);pd.DataFrame(checks).to_csv(OUT/'validation/geometry_source_identity_checks.csv',index=False)
    # Assertions refer only to scientific gates already computed, not desired outcomes.
    status={'protocol':'P1D-v3.0-focused-ap1-retrospective','P1D_0':'PASS_COUNT_SPLIT_QC; ROSETTE_ANNOTATION_BLOCKED','P1D_1':'DONE_FOCUSED_148_CELL_GEOMETRY_AND_NODE_DISJOINT_COMPARISON','P1D_2':'PARTIAL_RANDOM_DROPOUT_ROBUST_WEAK_EDGE_SENSITIVE_MATCHING_FAILED','P1D_3':'PARTIAL_OBSERVED_ROUTES_ONLY_NO_VALID_MATCHED_SIGNIFICANCE','P1E':'MALE_SOURCE_ID_GAP_EXPLAINED; BANC_QC_PREPARED; EXTERNAL_MECHANISM_NOT_VALIDATED','hypothesis_verdict':'INSUFFICIENT_FOR_GLOBAL_CONTINUOUS_VS_HYBRID_VERDICT','positive_result':'Regional skeleton opportunity improves within-ap1 pair-heldout prediction; node-disjoint point estimate improves but interval crosses zero','new_validated_functional_units':0,'stable8_unchanged':True,'scope':'one FlyWire left ap1 parent, same brain, retrospective','next_step':'Replicate regional geometry effect across parents and an independent registered specimen; assess whether frozen residual label adds robust out-of-brain routing information'}
    (OUT/'P1D_STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    report=f'''# P1D-v3：区域神经树几何是当前最强的计算解释

**实跑结论：区域树接近性提高了同一已知细类型内部的连接预测；18细胞集合对常规随机删边稳定，但尚不能证明超出几何的独立功能路由。当前新增已验证功能单元为0。**

## 区域树几何提供了什么

在完整左侧 ap1 parent 的148个细胞中，使用全部21,756个非自连有向细胞对，包含全部非边，没有重新抽取容易负例。重新从真实骨架、闭合脑区mesh计算 FULL/CA/PED/ML/VL 中长度加权的点到线段接近性（高斯尺度1/2/5μm），没有读入接点标签。1μm积分与0.5μm参考通过核查，精确距离与暴力求解误差为0；148个骨架哈希与P1冻结输入一致。脑区display mesh是解剖近似，接近性不等于实际突触接触。

成对留出：描述量模型 log loss {g1:.6f} → 区域树模型 {g2:.6f}，增益 {g1-g2:.6f}；AUROC 0.5757→0.7138。细胞完全分离：{b1:.6f} → {b2:.6f}，增益 {b1-b2:.6f}；AUROC 0.6045→0.6779。细胞分离评价仅650个测试有向对、26个测试细胞，排除了图度数、图嵌入和历史发现标签。这里的blocked指细胞身份分离，不是空间block或另一动物。

成对留出的几何增益同脑顶点重采样95%区间为[0.03268,0.07115]；细胞分离增益区间为[-0.01580,0.07235]，跨零。因此细胞分离目前只有同方向点估计，尚未得到精确的泛化证据。这些均为固定模型的事后区间。

M0–M4均为128随机特征+L2 logistic，共129个名义系数，4个验证集超参设置；相同名义系数不保证有效容量完全相同。本轮M1已包含旧树描述量，因此和P1-v2.2全KC case-control数值不是可直接比较的重复实验。

## 不能将初步M3负增益写成标签完全无效

主比较中，加入标签或图嵌入会改变随机特征映射。为排除这种方法效应，另行冻结并执行了事后嵌套对照：相同128维几何映射保持不变，分别附加零向量、18细胞标签3维、训练图PC3，均132名义系数。五个随机特征种子中，标签增益均为正，均值 **{gain.mean():.6f}**，范围 **{gain.min():.6f}–{gain.max():.6f}**。

标签增益的顶点重采样95%区间为[-0.00254,0.00716]，跨零，不能确认非零增量。它是固定模型、同脑内的探索性区间，没有计入重拟合、调参或动物变异。PC3是限制维数的对照，不能据其表现判断完整图嵌入不含额外信息。初步非嵌套结果及后续对照全部保留。

## 18细胞集合：重复识别了历史对象，尚非新功能单元

其成员为历史CM030的16个细胞加CM019的2个细胞。20次随机删去20%无序神经元对后，K=3平均Jaccard **{drop.loc[3,'mean_jaccard']:.3f}**，平均保留率 **{drop.loc[3,'mean_retention']:.3f}**；检测质量下四分位剔除后的Jaccard为0.947（cleft）和0.857（connection score）。

原KC图79.8%的有向对仅有1个突触。去掉单突触边，K=3 Jaccard为0.409；只保留≥3或≥5突触边时为0.250。后续同删边数量随机对照中，≥2对照的平均Jaccard约0.664（5次范围0.405–0.842）；≥3/≥5的随机对照也大幅失稳。这说明强度筛选同时带来严重稀疏化，不能把失稳全部归于检测伪影。5次数量对照未匹配度分布/突触质量，不作为正式检测偏差检验。

1000组同细类型/同侧的位置、训练图强度和树PCA匹配对照中，**0组**满足预设最大均值差≤0.25个parent标准差；最小总距离匹配的最大均值差仍为0.540。因此PN与MBON两项匹配显著性均保留NA，未生成有效p值。不能把未匹配条件下的路由差异当作几何之外的独立作用，也不能以匹配失败证明没有作用。

真实输入/输出表包含65种ALPN类型、25种MBON类型，已导出72条PN→候选KC→MBON的实际有向结构路径（附root ID与逐边突触数）。它们仅是路由清单，没有仿真、真实活动或行为验证；整个工程早前接触过这些外部路由，亦不称全新盲测。

## 数据审计改变了哪些判断

FlyWire的293,762个有向对、379,338个突触与原表逐项一致，无重复原始突触ID。已有cleft/connection_score可供分层；**rosette标签缺失**，弱边不是rosette代理标签。旧P1负例只有约22%同细类型，正例约81%，所以本轮改用完整细类型内所有非边。

MaleCNS原始v1.0三个文件的发布哈希全部匹配；历史944,638行输入连接子表与151,856,684行原始权重表中相同目标范围逐项一致。31,355个未匹配端点**全部存在于同版原始递质表**：这是注释覆盖缺口，不是已证实的版本错误。不得为这些ID补造类型；类型同源和区域几何迁移仍需处理。先前P0/P1关于“来源/版本待核验”的结果保留，由本轮证据补充解释。

BANC与FlyWire来自不同个体及其右转偏好，按用户已核实的原文信息登记；不能把同源类型映射误认为同一生物样本。本地BANC v888/v3名册有4552个KC，其中4532个side为unknown，240个current root_id与banc_888_id不同；经核对，root_888与banc_888_id在全部4552个KC中一致；本轮按该冻结快照ID提取153,033个KC有向对，保留240个current root_id更新映射，不覆盖快照身份。区域几何尚未统一，未执行跨脑机制预测。不能使用历史invalid_seed_schedule结果充当这轮验证。

## 裁决与下一步

局部计算证据支持“区域树几何具有预测价值”；**对全局连续几何、混合组织或特化微回路的裁决仍是证据不足**。18细胞标签可能有较小残余信息，尚无有效匹配路由或独立个体功能证据。当前不增加簇数、不优化NeuronMimic、不提升历史CM030状态。

下一步优先把同一套区域机会模型迁移到其他完整parent及BANC，保持类型内困难负例、输入来源和阈值冻结；解决独立样本的区域注册和类型映射。只有标签残余效应能够迁移、且有可操作的输入输出预测，才进入针对该候选的生理/干预设计。

## 复现

命令见 report/EXECUTION_LOG.md。入口 simulation/scripts/run_kc_p1d_v3.py；几何运行使用已有 .venv_fine_structure，其他阶段使用系统Python。所有新产物位于 artifacts/nature_kc_20261030/p1d_v3；旧结果未作为本轮完成声明。图注、配置、成员、预测值、负结果与来源哈希均保留。
'''
    (OUT/'P1D_REPORT.md').write_text(report)
    (OUT/'report/NEXT_MEETING_REPORT.md').write_text(f'''# 下一次组会：先讲区域树几何，再讲18细胞的边界

1. **问题修正**：胞体聚得更紧不等于组织更可信。本轮直接建模神经树在各MB区域中的真实接近性，不继续扩大簇数。
2. **正向结果**：完整左ap1共148细胞、21,756有向对。区域树模型的log loss在pair留出下{g1:.4f}→{g2:.4f}，细胞分离下{b1:.4f}→{b2:.4f}。这是当前最明确的新增结果。
3. **标签残余仍需裁决**：保持几何映射不变后，18细胞标签平均增益{gain.mean():.5f}，95%同脑探索区间跨零；细胞分离的几何增益区间亦跨零。该集合含16个历史CM030成员；常规删边稳定，却对大量弱边剔除敏感。
4. **路由结论暂不升级**：1000组匹配对照均未平衡，因此没有有效PN/MBON匹配显著性。72条真实路径可供检查，但不是功能验证。
5. **数据基础推进**：MaleCNS的31,355缺注释ID已在同版原始表中核实；来源冲突疑问得到澄清，类型缺失仍保留。BANC准备区域几何外推。

当前结论：区域树几何有预测价值；尚无新验证功能单元。下一轮做跨parent/跨脑的同协议检验，不继续按目标数量分簇。
''')
    log='''# Executed stages and failures

All commands run from repository root. No downloaded Codex wrapper was present; these are the actually implemented local commands.

```bash
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py gate
PYTHONPATH=simulation/src .venv_fine_structure/bin/python simulation/scripts/run_kc_p1d_v3.py geometry
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py models
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py cm019
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py routing
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py crossbrain
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py banc_qc
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py density_control
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_v3.py nested_models
PYTHONPATH=simulation/src pytest -q simulation/tests/test_p1d_v3.py
python simulation/scripts/finalize_kc_p1d_v3.py
```

- gate DONE: exact raw-count closure, negative-sampling audit; rosette motif annotation unavailable.
- geometry DONE: initial system-Python invocation failed because trimesh was absent. Reused existing .venv_fine_structure, no simulator/package rewrite; numerical gate passed before pair/model computation.
- models DONE for complete left ap1 only; 4-setting validation budget; cell-disjoint evaluation omits graph-derived predictors.
- cm019 DONE screen: 27 conditions ×3 resolutions, 20 fresh random-pair perturbations, all unfavorable outcomes retained.
- routing PARTIAL: 1000 fixed matching draws, zero balanced groups; no matched significance. Observed route examples are not simulated propagation.
- crossbrain PARTIAL: original MaleCNS release IDs and subset counts verified; no new biological homology. BANC metadata/edge QC only.
- density/nested controls are POSTHOC and frozen in separate addenda before their execution; primary results untouched.
- Primary seed20261031, inherited pair split seed20261030; perturb seeds2026111000–19; nested random-feature seeds20261031–35; detailed control seeds in addenda.
- Resource consumption/times are in audit/*STATUS.json and audit/geometry_with_existing_env.log. CPU runs; no new GPU training or dynamics.
'''
    (OUT/'report/EXECUTION_LOG.md').write_text(log)
    literature=[{'work':'Pleijzier et al. Structured Connectivity Across Multiple Drosophila Mushroom Bodies','source_url':'https://www.biorxiv.org/content/10.64898/2026.09.06.749654v1','verification_status':'PRIMARY_SOURCE_VERIFIED_BY_USER; AGENT_RETRIEVAL_BLOCKED_PROXY_ACCESS_DENIED','novelty_boundary':'Conserved PN-KC-MBON bias alone is not a new claim'}, {'work':'BANC Nature 2026','source_url':'exact paper URL not supplied in latest message','verification_status':'SPECIMEN_INDEPENDENCE_AND_BEHAVIOR_REPORTED_VERIFIED_BY_USER','novelty_boundary':'Independent animal; extreme turning bias and cross-dataset QC must be modeled'}, {'work':'Li et al. eLife 2020 / structured sampling 2022 / Thornton-Kolbe 2025','source_url':'See user supplied literature assessment; exact claim-level bibliography pending','verification_status':'USER_SUPPLIED_COMPETITIVE_BASELINE_NOT_REVERIFIED_THIS_RUN','novelty_boundary':'Known types and coarse spatial opportunity cannot be claimed novel'}]
    pd.DataFrame(literature).to_csv(OUT/'audit/literature_competition_ledger.csv',index=False)
    executed=[ROOT/'simulation/scripts/run_kc_p1d_v3.py',Path(__file__),ROOT/'simulation/src/bio_fly/p1_structure.py',ROOT/'simulation/src/bio_fly/kc_weighted_structure.py',ROOT/'simulation/src/bio_fly/wu_20261019/arbor_opportunity.py',ROOT/'simulation/src/bio_fly/wu_20261019/calyx_geometry.py',ROOT/'simulation/scripts/analyze_ap2_arbor.py']
    provenance={'dataset_version':'FlyWire v783 / MaleCNS v1.0 / BANC v888-v3 QC','protocol_version':status['protocol'],'run_id':'p1d_v3_20261030','seed':20261031,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty_source_hashes':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in executed],'default_python':sys.version,'default_packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','pandas']},'geometry_environment':'existing .venv_fine_structure: numpy1.26.4, trimesh5.2.0, sklearn1.8.0, pandas2.3.3','stage_input_hashes':['audit/gate_sources.json','audit/regional_geometry_sources.csv','audit/mesh_audit.json','audit/malecns_release_hashes.json','audit/BANC_QC.json'],'protocol_files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for p in sorted((OUT/'audit').glob('*addendum.json'))]+[{'path':'audit/protocol.json','sha256':sha(OUT/'audit/protocol.json')} ]}
    (OUT/'audit/final_provenance_manifest.json').write_text(json.dumps(provenance,indent=2))
    figures(model,nested,perturb,density)
    print(pd.DataFrame(cis).to_string(index=False));print(json.dumps(status,ensure_ascii=False))

def figures(model,nested,perturb,density):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for group,color in [('pair_grouped','#147d92'),('cell_disjoint_geometry','#b65e28')]:
        d=model[model.evaluation.eq(group)].iloc[[1,2]];axes[0].plot([0,1],d.log_loss,marker='o',label=group,color=color)
    axes[0].set(xticks=[0,1],xticklabels=['Descriptors','Regional arbor'],ylabel='Held-out log loss',title='Real regional geometry');axes[0].legend(fontsize=7)
    w=nested.pivot(index='seed',columns='model',values='log_loss');delta=w.M2_fixed_geometry-w.M3_plus_frozen18
    axes[1].scatter(range(5),delta,color='#147d92');axes[1].axhline(0,color='gray',lw=.6);axes[1].set(xlabel='Random-feature seed index',ylabel='Log loss improvement',title='Frozen18: shared geometry control')
    ci=pd.read_csv(OUT/'analysis/paired_model_uncertainty.csv').query("augmented == 'M3_plus_frozen18'").iloc[0]
    axes[1].axhspan(ci.interval_low,ci.interval_high,color='#147d92',alpha=.12,label='Vertex bootstrap interval');axes[1].legend(fontsize=7)
    d=perturb[perturb.K.eq(3)].set_index('condition');vals=[d.loc['count','jaccard'],d[d.index.str.startswith('pair_drop')].jaccard.mean(),d.loc['count_ge2','jaccard'],d.loc['count_ge3','jaccard']]
    axes[2].bar(range(4),vals,color=['#147d92','#147d92','#b65e28','#b65e28']);axes[2].set(xticks=range(4),xticklabels=['Original','Drop20%','Count>=2','Count>=3'],ylabel='Best-match Jaccard',ylim=(0,1.05),title='18-cell boundary sensitivity')
    fig.text(.01,.01,'FlyWire v783; left ap1 n=148; P1D-v3; retrospective structure only. Sources: MODEL_COMPARISON / nested_shared_geometry_comparison / CM019_18_perturbations.csv',fontsize=6)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/'figures'/f'p1d_v3_evidence.{ext}',dpi=170)
    plt.close(fig)
    pd.DataFrame([{'figure':'p1d_v3_evidence.pdf/svg/png','source_file':'analysis/MODEL_COMPARISON.csv;analysis/nested_shared_geometry_comparison.csv;analysis/CM019_18_perturbations.csv','dataset_version':'FlyWire v783','protocol':'P1D-v3.0-focused-ap1-retrospective plus declared posthoc addenda','n':148,'denominator':'complete left ap1;18 frozen candidate members','metric_definition':'heldout log loss; paired score difference; best-match Jaccard under perturbation','simulation_vs_biology_label':'COMPUTED_STRUCTURAL_PREDICTION_NOT_PHYSIOLOGY','run_id':'p1d_v3_20261030'}]).to_csv(OUT/'figure_source_ledger.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
