"""Run the frozen retrospective P1A-C and bounded M0-M4 descriptor pilot.

Run from repository root: PYTHONPATH=simulation/src python simulation/scripts/run_kc_p1_v2.py
"""
from __future__ import annotations
import argparse,hashlib,json,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,roc_auc_score,log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from neuronmimic.p1_structure import pair_split,pair_uniform,tree_features,fuse_blocks
from neuronmimic.structural_embedding import _embedding

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nature_kc_20261030/p1_v2_fine_types'
ATLAS=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
GRAPH=ROOT/'nature_methods_connectome/next_stage_structure_first_20260912/artifacts/flywire_all_kc_partner_pilot_20260912'
BUNDLE=None

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def main():
    start=time.time()
    if (OUT/'P1_STATUS.json').exists():raise RuntimeError('completed run exists; use a separately versioned protocol/output')
    for n in ['registry','data','analysis','figures','audit']:(OUT/n).mkdir(exist_ok=True,parents=True)
    cfg=json.loads((OUT/'audit/protocol.json').read_text()); seed=cfg['seed']; sources=[]
    def source(p):
        p=Path(p);sources.append({'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)});return p
    if BUNDLE:
        reg=pd.read_parquet(source(BUNDLE/'registry/all5177_identity.parquet'))
        reg.root_id=reg.root_id.astype(str);ids=reg.root_id.to_numpy(dtype=str)
    else:
        ids=np.array(json.loads(source(GRAPH/'node_order.json').read_text()),dtype=str)
        reg=pd.read_parquet(source(ROOT/'artifacts/wu_20261019/registry/kc_all.parquet'))
        reg.root_id=reg.root_id.astype(str);reg=reg.set_index('root_id').loc[ids].reset_index()
        su=pd.read_parquet(source(ROOT/'artifacts/wu_20261030_fine_units_integrated_atlas/registry/fine_unit_members.parquet'))
        su.root_id=su.root_id.astype(str)
        reg=reg.merge(su[['root_id','unit_id']],on='root_id',how='left',validate='one_to_one')
    assert len(reg)==len(set(ids))==5177
    expected={'CM012':163,'CM014':152,'CM016':147,'CM019':132,'CM020':129,'CM022':114,'CM031':67,'CM033':61}
    counts=reg.loc[reg.stable_backbone.astype(bool),'candidate_id'].value_counts().to_dict()
    assert counts==expected
    reg['legacy_CM_namespace']='v9_frozen';reg['SU_namespace']='20261030_scale_conditional';reg['P1_role']='full_registry_no_promotion'
    reg.to_parquet(OUT/'registry/all5177_identity.parquet',index=False)
    cross=reg.groupby(['candidate_id','unit_id'],dropna=False).size().reset_index(name='n_shared')
    cross.to_csv(OUT/'registry/CM_SU_crosswalk.csv',index=False)
    print('P1A: 5177 identities / frozen stable8 verified',flush=True)
    # Audit raw annotation alternatives, never silently repair historical normalized tables.
    if BUNDLE:
        qc=json.loads(source(BUNDLE/'audit/malecns_id_QC.json').read_text())
        qc['replay_note']='Imported audit receipt; raw MaleCNS source audit NOT rerun by compact bundle replay.'
    else:
        a=pd.read_parquet(source(ROOT/'audit/meeting_revision_20261027/reproduced_structure/MaleCNS_v1_0_source_annotations.parquet'))
        me=pd.read_parquet(source(ROOT/'audit/meeting_revision_20261027/reproduced_structure/MaleCNS_v1_0_annotated_input_edges.parquet'))
        ends=set(me.pre.astype(str))|set(me.post.astype(str)); body=set(a.body.dropna().astype('int64').astype(str)); aid=set(a.id.astype(str))
        qc={'annotation_rows':len(a),'body_valid_unique':len(body),'id_valid_unique':len(aid),'edge_endpoints':len(ends),
        'body_unmatched':len(ends-body),'id_unmatched':len(ends-aid),'recovered_by_id':len((ends-body)&aid),
        'body_vs_id_disagreement':int((a.loc[a.body.notna(),'id'].astype(str)!=a.loc[a.body.notna(),'body'].astype('int64').astype(str)).sum()),
            'status':'BLOCKED_SOURCE_VERSION_OR_SCOPE_UNRESOLVED','note':'Full source id column does not resolve missing edge endpoints; no cross-version identity inference.'}
        pd.DataFrame({'unmatched_endpoint_id':sorted(ends-aid)}).to_csv(OUT/'registry/malecns_unmatched_endpoints.csv',index=False)
    dump(OUT/'audit/malecns_id_QC.json',qc)
    # Split all observed pairs BEFORE embeddings, features or clustering.
    if BUNDLE:
        pp=pd.read_parquet(source(BUNDLE/'data/directed_pair_split.parquet'))
        lookup={v:i for i,v in enumerate(ids)}
        full=sparse.csr_matrix((pp.synapses.to_numpy(float),(pp.pre_id.map(lookup),pp.post_id.map(lookup))),shape=(len(ids),len(ids)))
    else:full=sparse.load_npz(source(GRAPH/'data/kc_graph.npz')).astype(float).tocsr()
    co=full.tocoo()
    split=pair_split(ids,co.row,co.col,seed)
    train=sparse.csr_matrix((co.data[split==0],(co.row[split==0],co.col[split==0])),shape=full.shape)
    sparse.save_npz(OUT/'data/pair_grouped_train.npz',train)
    pairs=pd.DataFrame({'pre_id':ids[co.row],'post_id':ids[co.col],'synapses':co.data.astype(int),'split':np.array(['train','validation','test'])[split]})
    pairs.to_parquet(OUT/'data/directed_pair_split.parquet',index=False)
    tr=set(zip(co.row[split==0],co.col[split==0]));he=set(zip(co.row[split!=0],co.col[split!=0]))
    assert not tr&he and not tr&{(j,i) for i,j in he}
    splitqc={'n_directed_pairs':full.nnz,'synapses_total':float(full.sum()),'train_pairs':int((split==0).sum()),'validation_pairs':int((split==1).sum()),'test_pairs':int((split==2).sum()),'heldout_direct_overlap':0,'heldout_reverse_overlap':0,'contact_features_used':False,'exposure':'RETROSPECTIVE_NOT_BLIND'}
    dump(OUT/'audit/pair_split_QC.json',splitqc)
    emb=_embedding(train,32,seed);np.save(OUT/'data/train_only_embedding.npy',emb)
    print('P1B: unordered pair split / training-only embedding complete',flush=True)
    # Tree-only geometry, with a hash for every original morphology.
    if BUNDLE:
        geo=pd.read_parquet(source(BUNDLE/'data/contact_free_tree_features.parquet'))
        geo=geo.set_index('root_id').loc[ids].reset_index()
    else:
        features=[]
        for i,rid in enumerate(ids):
            p=source(ATLAS/'geometry'/f'{rid}.json');d=json.loads(p.read_text())
            assert str(d['id'])==rid and d['dataset']=='FlyWire v783'
            features.append({'root_id':rid,**tree_features(d)})
            if (i+1)%1000==0:print('geometry',i+1,flush=True)
        geo=pd.DataFrame(features)
    geo.to_parquet(OUT/'data/contact_free_tree_features.parquet',index=False)
    soma=reg[['soma_x_um','soma_y_um','soma_z_um']].to_numpy(float)
    assert np.isfinite(soma).all(),'missing soma prevents full distance matrix'
    matrix=np.lib.format.open_memmap(OUT/'data/soma_distance_um.npy',mode='w+',dtype='float32',shape=(len(ids),len(ids)))
    for i in range(0,len(ids),256):matrix[i:i+256]=cdist(soma[i:i+256],soma)
    matrix.flush()
    soma_scale=float(np.sqrt(np.mean(np.sum((soma-soma.mean(0))**2,axis=1))))
    S=(soma-soma.mean(0))/soma_scale
    T=StandardScaler().fit_transform(geo.drop(columns='root_id'))
    dump(OUT/'data/feature_schema.json',{'soma_unit':'um','soma_scale_um':soma_scale,'tree_columns':geo.columns[1:].tolist(),'tree_source':'skeleton geometry only; no contacts, partner labels, old CM/SU labels','fusion':'concatenate E/sqrt(dE), sqrt(ws)*(soma-mean)/scale, sqrt(wt)*z(tree)/sqrt(dT)','embedding_source':'pair_grouped_train.npz','normalization':'transductive unlabeled full registry; not inductive new-cell evaluation'})
    arms=cfg['discovery_arms']; assignments=reg[['root_id','candidate_id','unit_id','official_KC_subtype','official_hemisphere']].copy()
    rows=[]; labelsets={}
    def fit(x,k,rs):return KMeans(n_clusters=k,n_init=cfg['n_init'],random_state=rs).fit_predict(x)
    for name,(ws,wt) in arms.items():
        x=fuse_blocks(emb,S,T,ws,wt);labs=[fit(x,30,rs) for rs in cfg['init_seeds']]
        labelsets[name]=labs[0];assignments[name]=labs[0]
        compact=np.mean([np.linalg.norm(soma[labs[0]==c]-soma[labs[0]==c].mean(0),axis=1).mean() for c in np.unique(labs[0])])
        rows.append({'arm':name,'soma_weight':ws,'tree_weight':wt,'mean_cluster_soma_radius_um':compact,'seed_ARI_mean':np.mean([adjusted_rand_score(labs[0],l) for l in labs[1:]]),'n_clusters':30,'n_cells':len(ids),'status':'EXPLORATORY_PARTITION_NOT_STABLE_CANDIDATES'})
    dropout={name:[] for name in arms}
    tc=train.tocoo()
    for rs in cfg['pair_dropout_seeds']:
        keep=pair_uniform(ids,tc.row,tc.col,rs)>=.2
        pert=sparse.csr_matrix((tc.data[keep],(tc.row[keep],tc.col[keep])),shape=train.shape)
        pe=_embedding(pert,32,seed)
        for name,(ws,wt) in arms.items():
            lab=fit(fuse_blocks(pe,S,T,ws,wt),30,seed)
            dropout[name].append(adjusted_rand_score(labelsets[name],lab))
    for row in rows:row['pair_dropout_ARI_mean']=float(np.mean(dropout[row['arm']]));row['pair_dropout_ARI_values']=json.dumps(dropout[row['arm']])
    pd.DataFrame(rows).to_csv(OUT/'analysis/discovery_ablation.csv',index=False)
    assignments.to_parquet(OUT/'registry/P1_exploratory_assignments.parquet',index=False)
    parent_rows=[];parent_members=[]
    # Complete official parents, including all residuals, never candidate-only reclustering.
    for (typ,side),rr in reg.groupby(['frozen_detailed_subtype','official_hemisphere']):
        if typ not in cfg['parent_scope']:continue
        idx=rr.index.to_numpy()
        for name in ['graph','graph_soma_tree']:
            ws,wt=arms[name];x=fuse_blocks(emb,S,T,ws,wt)[idx]
            for k in cfg['parent_k']:
                labs=[fit(x,k,rs) for rs in cfg['init_seeds']]
                parent_rows.append({'parent_type':typ,'side':side,'arm':name,'k':k,'parent_n':len(idx),'stable8_selected_n':int(rr.stable_backbone.sum()),'residual_n':int((~rr.stable_backbone.astype(bool)).sum()),'seed_ARI_mean':np.mean([adjusted_rand_score(labs[0],l) for l in labs[1:]]),'child_sizes':json.dumps(np.bincount(labs[0]).tolist()),'status':'DESCRIPTIVE_ONLY'})
                parent_members.extend({'root_id':ids[j],'parent_type':typ,'side':side,'arm':name,'k':k,'child':int(labs[0][ii])} for ii,j in enumerate(idx))
    pd.DataFrame(parent_rows).to_csv(OUT/'analysis/complete_parent_splits.csv',index=False)
    pd.DataFrame(parent_members).to_parquet(OUT/'registry/parent_child_members.parquet',index=False)
    print('P1C: six geometry ablations / full-parent child registries complete',flush=True)
    comparison(full,train,ids,reg,geo,soma,emb,labelsets['graph_soma_tree'],seed,source)
    figures(reg,soma,labelsets,rows)
    dump(OUT/'audit/source_manifest.json',{'sources':sources,'runner_sha256':sha(Path(__file__)),'primitives_sha256':sha(ROOT/'src/neuronmimic/p1_structure.py'),'protocol_sha256':sha(OUT/'audit/protocol.json'),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seed':seed,'dataset_version':'FlyWire v783 / local MaleCNS-v1.0 named source','run_id':cfg['protocol_version'],'wall_clock_s':time.time()-start})
    status={'P1A':'DONE_WITH_MALECNS_BLOCKER','P1B':'DONE_RETROSPECTIVE_PAIR_ISOLATION','P1C':'DONE_EXPLORATORY_GEOMETRY_DISCOVERY','P1D':'PARTIAL_DESCRIPTOR_PILOT_NOT_FULL_BRANCH_CONTACT_MODEL','P1E':'BLOCKED_INDEPENDENCE_AND_ID_QC','stable8_unchanged':True,'new_stable_candidates':0,'scientific_conclusion':'No discrete/continuous verdict yet; full independent competition remains open','wall_clock_s':time.time()-start}
    dump(OUT/'P1_STATUS.json',status)
    print(json.dumps(status),flush=True)

def comparison(full,train,ids,reg,geo,soma,emb,labels,seed,source):
    rng=np.random.default_rng(seed);co=full.tocoo();ix=rng.choice(len(co.data),min(60000,len(co.data)),replace=False)
    a=list(co.row[ix]);b=list(co.col[ix]);positive=set(zip(co.row,co.col));neg=set()
    while len(neg)<60000:
        q=rng.integers(0,len(ids),size=(10000,2))
        for i,j in q:
            if i!=j and (i,j) not in positive:neg.add((int(i),int(j)))
            if len(neg)==60000:break
    na,nb=np.array(sorted(neg)).T;y=np.r_[np.ones(len(a)),np.zeros(len(na))].astype(int)
    a=np.r_[a,na].astype(int);b=np.r_[b,nb].astype(int);sp=pair_split(ids,a,b,seed)
    pd.DataFrame({'pre_id':ids[a],'post_id':ids[b],'edge_present':y,'split':np.array(['train','validation','test'])[sp]}).to_parquet(OUT/'data/prediction_pairs.parquet',index=False)
    codes=pd.Categorical(reg.official_KC_subtype.fillna('unknown')).codes
    fine=pd.Categorical(reg.frozen_detailed_subtype.fillna('unknown')).codes
    sides=pd.Categorical(reg.official_hemisphere.fillna('unknown')).codes
    base=np.column_stack([codes[a],codes[b],fine[a],fine[b],sides[a],sides[b]])
    deg=np.column_stack([np.asarray(train.sum(1)).ravel(),np.asarray(train.sum(0)).ravel()])
    spatial=np.column_stack([soma[a],soma[b],np.abs(soma[a]-soma[b]),np.linalg.norm(soma[a]-soma[b],axis=1),np.log1p(deg[a]),np.log1p(deg[b])])
    t=geo.drop(columns='root_id').to_numpy(float)
    tree=np.column_stack([t[a],t[b],np.abs(t[a]-t[b])])
    m1=np.column_stack([base,spatial]);m2=np.column_stack([m1,tree])
    matrices={'M0':base,'M1':m1,'M2_descriptor':m2,'M3_descriptor_discrete':np.column_stack([m2,labels[a],labels[b],labels[a]==labels[b]]),'M4_descriptor_continuous':np.column_stack([m2,emb[a],emb[b],np.abs(emb[a]-emb[b])])}
    records=[];preds=pd.DataFrame({'pre_id':ids[a[sp==2]],'post_id':ids[b[sp==2]],'edge_present':y[sp==2]})
    for name,x in matrices.items():
        cat=np.zeros(x.shape[1],dtype=bool);cat[:6]=True
        if name.startswith('M3'):cat[-3:-1]=True
        fits=[]
        for leaves in [15,31]:
            tm=time.time();model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=leaves,learning_rate=.08,l2_regularization=1,categorical_features=cat,early_stopping=False,random_state=seed)
            model.fit(x[sp==0],y[sp==0]);vl=log_loss(y[sp==1],model.predict_proba(x[sp==1])[:,1]);fits.append((vl,leaves,model,time.time()-tm))
        vl,leaves,model,_=min(fits,key=lambda z:z[0]);p=model.predict_proba(x[sp==2])[:,1];preds[name]=p
        records.append({'model':name,'test_log_loss':log_loss(y[sp==2],p),'test_AUROC':roc_auc_score(y[sp==2],p),'validation_log_loss':vl,'selected_max_leaves':leaves,'n_features':x.shape[1],'n_train':int((sp==0).sum()),'n_validation':int((sp==1).sum()),'n_test':int((sp==2).sum()),'training_budget':'2 settings x100 iterations','training_wall_clock_s':sum(z[3] for z in fits),'evidence':'RETROSPECTIVE_DESCRIPTOR_PILOT','uncertainty':'NOT_ESTIMATED_NO_SIGNIFICANCE_CLAIM','sampling':'case-control 60000 positive/60000 negative; probabilities not natural prevalence calibrated'})
        print(name,records[-1]['test_log_loss'],flush=True)
    pd.DataFrame(records).to_csv(OUT/'MODEL_COMPARISON.csv',index=False);preds.to_parquet(OUT/'data/test_predictions.parquet',index=False)

def figures(reg,soma,labels,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=reg.candidate_id.eq('CM019') & reg.stable_backbone.astype(bool)
    parent=reg.frozen_detailed_subtype.eq("KCa'b'-ap1") & reg.official_hemisphere.eq('left')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(soma[:,i],soma[:,j],s=1,c='#dddddd',rasterized=True,label='all KC')
        ax.scatter(soma[parent&~selected,i],soma[parent&~selected,j],s=16,facecolors='none',edgecolors='#d95f02',label='parent residual')
        ax.scatter(soma[selected,i],soma[selected,j],s=5,c='#147d92',label='CM019')
        ax.set(xlabel='XYZ'[i]+' (um)',ylabel='XYZ'[j]+' (um)');ax.set_aspect('equal')
    axes[-1].legend(fontsize=7);fig.suptitle(f'CM019: selected {selected.sum()}, left ap1 parent {parent.sum()}, residual {(parent&~selected).sum()}')
    fig.text(.01,.01,'FlyWire v783; all5177_identity.parquet; P1-v2; see protocol.json; DESCRIPTIVE_ONLY; soma positions, no dynamic validation',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.93))
    for ext in ['pdf','svg']:fig.savefig(OUT/'figures'/f'CM019_full_parent_somata.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,(name,lab) in zip(axes.ravel(),labels.items()):
        ax.scatter(soma[:,0],soma[:,1],s=2,c=lab,cmap='turbo',rasterized=True);ax.set_title(name);ax.set_aspect('equal');ax.set(xlabel='X (um)',ylabel='Y (um)')
    fig.suptitle('Actual spatial fusion: same K=30, six frozen feature arms')
    fig.text(.01,.01,'FlyWire v783; P1_exploratory_assignments.parquet; n=5177 each; P1-v2; see protocol.json; exploratory partitions, not new stable units',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ['pdf','svg']:fig.savefig(OUT/'figures'/f'spatial_fusion_ablation.{ext}')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root',type=Path,help='HF p1_v2_fine_types directory; reuses audited geometry and identity tables, not raw source audits')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_dir.resolve();BUNDLE=args.bundle_root.resolve() if args.bundle_root else None
    if BUNDLE and not (OUT/'audit/protocol.json').exists():
        (OUT/'audit').mkdir(parents=True,exist_ok=True)
        (OUT/'audit/protocol.json').write_bytes((BUNDLE/'audit/protocol.json').read_bytes())
    with threadpool_limits(limits=4):main()
