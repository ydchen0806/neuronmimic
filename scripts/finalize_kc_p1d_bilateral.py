"""Summarize same-brain bilateral geometry stress tests and uncertainty."""
from pathlib import Path
import argparse,json,hashlib,subprocess,platform,importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nature_kc_20261101/p1d_v3_bilateral'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def contrasts(pred):
    pred=pred.copy();p=np.clip(pred.probability.to_numpy(),1e-12,1-1e-12);y=pred.observed_edge.to_numpy()
    pred['loss']=-(y*np.log(p)+(1-y)*np.log1p(-p));rows=[];perfold=[]
    for (source,target,evaluation),g in pred.groupby(['source_side','target_side','evaluation']):
        # Seeds are numerical replicates: average loss, never count as animals or cells.
        table=g.groupby(['pre_id','post_id','fold','model']).loss.mean().unstack('model')
        diff=(table.M1-table.M2).to_numpy()
        ids=sorted(set(table.index.get_level_values('pre_id'))|set(table.index.get_level_values('post_id')));lookup={v:i for i,v in enumerate(ids)}
        a=np.array([lookup[v] for v in table.index.get_level_values('pre_id')]);b=np.array([lookup[v] for v in table.index.get_level_values('post_id')]);rng=np.random.default_rng(2026110109);vals=[]
        for _ in range(1000):
            v=rng.poisson(1,len(ids));w=v[a]*v[b]
            if w.sum():vals.append(np.average(diff,weights=w))
        low,high=np.quantile(vals,[.025,.975])
        rows.append({'source_side':source,'target_side':target,'evaluation':evaluation,'M1_log_loss_seed_mean':float(table.M1.mean()),'M2_log_loss_seed_mean':float(table.M2.mean()),'improvement_M1_minus_M2':float(diff.mean()),'interval_low':low,'interval_high':high,'n_target_cells':len(ids),'n_test_pairs':len(table),'n_resamples':len(vals),'method':'FIXED_MODEL_VERTEX_POISSON_SEED_MEAN_LOSS','interpretation':'same brain; interval excludes model/tuning/animal uncertainty'})
        for fold,part in table.groupby(level='fold'):
            d=part.M1-part.M2
            perfold.append({'source_side':source,'target_side':target,'evaluation':evaluation,'fold':fold,'n_pairs':len(part),'M1_log_loss':part.M1.mean(),'M2_log_loss':part.M2.mean(),'improvement':d.mean()})
    return pd.DataFrame(rows),pd.DataFrame(perfold)

def main():
    cfg=json.loads((OUT/'audit/protocol.json').read_text());modelstatus=json.loads((OUT/'audit/MODELS_STATUS.json').read_text())
    pred=pd.read_parquet(OUT/'data/test_predictions.parquet');scores=pd.read_csv(OUT/'analysis/model_comparison.csv')
    result,folds=contrasts(pred);result.to_csv(OUT/'analysis/paired_geometry_contrasts.csv',index=False);folds.to_csv(OUT/'analysis/fold_contrasts.csv',index=False)
    seedg=scores.pivot(index=['source_side','target_side','evaluation','fold','seed'],columns='model',values='log_loss')
    seedg['M1_minus_M2']=seedg.M1-seedg.M2;seedg.to_csv(OUT/'analysis/seed_sensitivity.csv')
    sep=pd.read_csv(OUT/'analysis/spatial_separation_audit.csv')
    lines=[]
    for _,r in result.iterrows():
        title={'pair_grouped':'神经元对留出','spatial_blocked':'胞体空间分块','cross_side':'跨侧直接迁移'}[r.evaluation]
        lines.append(f'| {r.source_side}→{r.target_side}，{title} | {r.M1_log_loss_seed_mean:.5f} | {r.M2_log_loss_seed_mean:.5f} | {r.improvement_M1_minus_M2:+.5f} | [{r.interval_low:+.5f}, {r.interval_high:+.5f}] | {r.n_test_pairs} |')
    spatial=folds[folds.evaluation.eq('spatial_blocked')];pos=int(spatial.improvement.gt(0).sum())
    transfer=result[result.evaluation.eq('cross_side')]
    transfer_improved=int(transfer.improvement_M1_minus_M2.gt(0).sum())
    status={'protocol_version':cfg['protocol_version'],'run_id':'p1d_v3_bilateral_20261101','status':'DONE_SAME_BRAIN_GEOMETRY_STRESS_TEST','left_parent':148,'right_parent':133,'new_discovery_or_label_selection':False,'new_validated_functional_units':0,'spatial_folds_improved':pos,'spatial_folds_total':len(spatial),'cross_side_directions_improved':transfer_improved,'cross_side_directions_total':2,'independent_specimen_validation':'NOT_COMPLETED','functional_or_causal_validation':'NOT_COMPLETED','primary_caveat':'Bilateral patterns are same-brain evidence; soma blocks may have nearby boundaries, and rigid reflection is not exact anatomy registration.'}
    (OUT/'STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    report=f'''# P1D-v3.1：区域树几何的左右复核与空间外推

本轮用冻结协议完成左右ap1的同脑检验：左148、右133个细胞；没有重新发现候选，也没有修改18细胞集合或stable8。右侧区域树接近性重新计算，左侧复用经哈希核对的原始几何结果。候选标签、KC图嵌入、图度数和接点统计均未进入预测。

| 评价 | M1描述量 log loss | M2区域树 log loss | 改善（正值更好） | 95%探索区间 | 测试对数 |
| --- | ---: | ---: | ---: | --- | ---: |
{chr(10).join(lines)}

以上为三个随机特征种子的平均逐对损失，区间为1000次固定模型顶点Poisson重采样，不把种子当独立动物。该区间未计入重拟合、调参和动物差异，跨零时不主张已确认的正负效应。

## 空间外推检验

每侧按胞体主轴坐标形成5个连续空间块，轮流留出1块测试、相邻1块验证，其余3块训练；仅使用各集合内部有向对，训练/验证/测试无共享细胞。10个空间折中，{pos}个M2点估计优于M1，逐折负结果保留在analysis/fold_contrasts.csv。块间未设置排除缓冲，最小训练—测试胞体距离为{sep.min_train_test_soma_distance_um.min():.3f}μm，因此该设计是胞体区域外推，不等于完全去除树形态空间相关，也不是跨脑验证。

## 跨侧迁移

仅在来源侧训练对上拟合参数和标准化，来源侧验证对选择C/gamma；直接预测目标侧全部非自连细胞对。右侧X坐标与树质心X使用全体KC侧别中位数定义的单一镜像变换，区域树距离本身镜像不变；没有目标侧重拟合、非刚性配准或逐细胞匹配。两个方向中{transfer_improved}个M2点估计优于M1。若迁移退化，只能说明该特征/模型在当前配准下外推不足，不能把退化归因为生物偏侧化。

## 模型与证据范围

M0为类型内截距；M1含胞体坐标、两两距离、真实骨架的细胞级描述量；M2增加FULL/CA/PED/ML/VL的长度加权骨架接近性、缺失标记与区域树长度。每个模型128随机特征+L2 logistic，共129名义参数，C=.1/1×gamma=.5/2，共4个验证集设置；种子20261101–03。各模型输入维度不同，相同名义参数不保证相同有效容量；本轮目的在于固定方法的重现与外推，不作因果消融声明。

所有类型内非边均保留，无case-control抽样；没有将单突触边剔除后留下的缺失边改写为真实不存在连接。FULL/CA/PED/ML/VL几何源自已冻结真实骨架和闭合display mesh；mesh是区域近似，潜在接近性不等于真实突触接触。右侧133×132=17,556对与左侧148×147=21,756对全部保留。

本次新增结果回答“几何预测能否在右侧和空间块中重现”，不回答18个细胞是否构成独立功能微回路。前轮18细胞匹配对照失败、标签增益区间跨零和弱边敏感性结论仍保留。新增验证功能单元仍为0；BANC/MaleCNS独立样本区域注册、可靠类型映射和实测活动验证尚待完成。

## 复现与来源

完整来源模式：
```bash
PYTHONPATH=simulation/src .venv_fine_structure/bin/python simulation/scripts/run_kc_p1d_bilateral.py geometry
PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1d_bilateral.py models
PYTHONPATH=simulation/src pytest -q simulation/tests/test_p1d_bilateral.py
python simulation/scripts/finalize_kc_p1d_bilateral.py
```

已完成输出拒绝覆盖。公开紧凑包支持模型重算：`python scripts/run_kc_p1d_bilateral.py models --bundle-root DATA/p1d_v3_bilateral --output-dir outputs/bilateral_replay`；该模式使用已审计几何设计矩阵，不重做原始骨架/mesh提取。data/保存全部测试预测与精确划分，models/包含可直接数值回放的权重及输入标准化参数。旧P1D-v3与本轮协议均独立保留。
'''
    (OUT/'REPORT.md').write_text(report)
    (OUT/'report/EXECUTION_LOG.md').write_text(f'''# Executed same-brain geometry stress test

- Protocol frozen before right-side geometry/model outcomes; see audit/protocol.json.
- Geometry: existing fine_structure environment; reuse certified segment solver and ROI mesh code, no simulator rewrite.
- Models: {modelstatus['n_fitted_models']} fitted final models, four validation settings each, seeds20261101–03; actual elapsed {modelstatus['elapsed_s']:.2f}s.
- Pairs: all nonself pairs within left148/right133 ap1, no negative sampling.
- Spatial folds: no shared neuron among train/validation/test; cross-block edges omitted rather than relabeled absent.
- Cross-side: source-only scaling/tuning; complete target-side evaluation. Same-brain only.
- Reports retain per-seed, per-fold, and per-direction losses; no candidate promotion.
- Validation commands and source-mode reproduction are in REPORT.md. Finalization uses seed2026110109 for1000 vertex-weight bootstrap replicates.
''')
    imported=json.loads((OUT/'audit/imported_source_provenance.json').read_text())
    inputs=json.loads((OUT/'audit/bundle_inputs.json').read_text())
    provenance={'scope':'COMPACT_MODEL_REPLAY_ONLY; original geometry/source audits imported, not rerun',
        'original_run_id':imported['run_id'],'run_id':'replay_'+OUT.name,
        'dataset_version':imported['dataset_version'],'protocol_version':cfg['protocol_version'],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hashes':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [ROOT/'scripts/run_kc_p1d_bilateral.py',Path(__file__)]],
        'input_hashes':inputs['inputs'],'original_provenance_sha256':sha(OUT/'audit/imported_source_provenance.json'),
        'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn']},
        'seed_schedule':cfg['seeds'],'bootstrap_seed':2026110109}
    (OUT/'audit/provenance.json').write_text(json.dumps(provenance,indent=2))
    status.update(status='DONE_COMPACT_MODEL_REPLAY',run_id=provenance['run_id'],raw_source_audit_rerun=False)
    (OUT/'STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    note='> COMPACT MODEL REPLAY: geometry extraction, source-count audits and original frozen-member checks were imported, not rerun. The source-mode statements below describe the original experiment. Refit and evaluation results above use the published design matrices.\n\n'
    (OUT/'REPORT.md').write_text(note+report)
    plot(result,folds)
    print(result.to_string(index=False));print(json.dumps(status))

def plot(result,folds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));colors={'left':'#187d92','right':'#b65e28'}
    for i,r in result.reset_index(drop=True).iterrows():
        axes[0].errorbar(r.improvement_M1_minus_M2,i,xerr=[[max(0,r.improvement_M1_minus_M2-r.interval_low)],[max(0,r.interval_high-r.improvement_M1_minus_M2)]],fmt='o',color=colors[r.source_side],capsize=3)
    labels=[f'{r.source_side[0].upper()}→{r.target_side[0].upper()} {r.evaluation}' for _,r in result.iterrows()]
    axes[0].set(yticks=range(len(result)),yticklabels=labels,xlabel='Log-loss improvement (M1 − M2)',title='Geometry gain and endpoint-weighted intervals');axes[0].axvline(0,color='gray',lw=.8)
    for side in ['left','right']:
        d=folds[(folds.evaluation=='spatial_blocked')&(folds.source_side==side)]
        axes[1].plot(d.fold,d.improvement,'o-',label=side,color=colors[side])
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(xlabel='Held-out soma block',ylabel='Log-loss improvement',title='Every spatial fold retained');axes[1].legend()
    fig.text(.01,.01,'FlyWire v783, one brain; left ap1 148 / right ap1 133. P1D-v3.1. Fixed-model within-brain intervals; not animal replication or functional validation.',fontsize=7)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/bilateral_spatial_geometry.{ext}',dpi=170)
    plt.close(fig)
    pd.DataFrame([{'figure':'bilateral_spatial_geometry.pdf/svg/png','source_file':'analysis/paired_geometry_contrasts.csv;analysis/fold_contrasts.csv','dataset_version':'FlyWire v783','protocol':'P1D-v3.1-bilateral-spatial-ap1','n_denominator':'left148/right133 KC; all declared nonself pairs; fold-specific n in source CSV','metric_definition':'paired seed-average test log-loss difference M1-M2;1000 vertex-Poisson draws','simulation_vs_biology_label':'COMPUTED_STRUCTURAL_PREDICTION_NOT_FUNCTIONAL_VALIDATION','run_id':'p1d_v3_bilateral_20261101'}]).to_csv(OUT/'figure_source_ledger.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();OUT=args.output_dir.resolve()
    with threadpool_limits(limits=4):main()
