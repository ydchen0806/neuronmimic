"""Staged, additive P1D-v3 analysis; raw source audits precede inference."""
from pathlib import Path
import argparse,json,hashlib,time,subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits
from bio_fly.p1_structure import pair_split,pair_uniform,fuse_blocks
from bio_fly.kc_weighted_structure import _embedding

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nature_kc_20261030/p1d_v3'
P1=ROOT/'artifacts/nature_kc_20261030/p1_v2_fine_types'
ATLAS=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
RAW=ROOT/'nature_methods_connectome/fine_structure_mechanism/runs/20260929_flywire_atlas_completion_v2/results/all_KC_incident_synapses.parquet'
SEED=20261031
REGIONS=['FULL','CA','PED','ML','VL']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def setup():
    for d in ['audit','data','registry','analysis','figures','report','cache/cells','cache/targets','validation']:(OUT/d).mkdir(parents=True,exist_ok=True)
    reg=pd.read_parquet(P1/'registry/all5177_identity.parquet')
    assert reg.root_id.map(lambda x:isinstance(x,str)).all()
    roots=reg.root_id.to_numpy(str)
    parent=reg.loc[reg.frozen_detailed_subtype.eq("KCa'b'-ap1")&reg.official_hemisphere.eq('left')].copy()
    assert len(parent)==148
    candidate=pd.read_parquet(P1/'registry/ap1L_18_exploratory_members.parquet')
    assert sha(P1/'registry/ap1L_18_exploratory_members.parquet')==json.loads((OUT/'audit/protocol.json').read_text())['candidate_sha256']
    assert len(candidate)==18 and set(candidate.root_id).issubset(set(parent.root_id))
    return reg,roots,parent,candidate

def gate():
    reg,ids,parent,candidate=setup();tm=time.time()
    if (OUT/'audit/GATE_STATUS.json').exists():raise RuntimeError('Gate completed; preserve existing outputs')
    raw=pd.read_parquet(RAW); raw.pre_pt_root_id=raw.pre_pt_root_id.astype(str);raw.post_pt_root_id=raw.post_pt_root_id.astype(str)
    k=raw.loc[raw.pre_pt_root_id.isin(ids)&raw.post_pt_root_id.isin(ids)].copy()
    counts=k.groupby(['pre_pt_root_id','post_pt_root_id']).size().reset_index(name='raw_count')
    p=pd.read_parquet(P1/'data/directed_pair_split.parquet')
    joined=p.merge(counts,left_on=['pre_id','post_id'],right_on=['pre_pt_root_id','post_pt_root_id'],how='outer',indicator=True)
    bad=joined.loc[joined._merge.ne('both')|joined.synapses.ne(joined.raw_count)]
    bad.to_csv(OUT/'audit/graph_raw_count_discrepancies.csv',index=False)
    passed=len(bad)==0 and raw.id.nunique()==len(raw)
    lookup={v:i for i,v in enumerate(ids)};r=k.pre_pt_root_id.map(lookup).to_numpy();c=k.post_pt_root_id.map(lookup).to_numpy()
    k['split']=pair_split(ids,r,c,20261030)
    k=k.rename(columns={'pre_pt_root_id':'pre_id','post_pt_root_id':'post_id'})
    k[['id','pre_id','post_id','cleft_score','connection_score','neuropil','split']].to_parquet(OUT/'data/KC_KC_contact_QC.parquet',index=False)
    hist=p.groupby('synapses').agg(directed_pairs=('pre_id','size')).reset_index();hist['synapses_total']=hist.synapses*hist.directed_pairs
    hist.to_csv(OUT/'analysis/edge_count_histogram.csv',index=False)
    p['pre_fine']=p.pre_id.map(reg.set_index('root_id').frozen_detailed_subtype);p['post_fine']=p.post_id.map(reg.set_index('root_id').frozen_detailed_subtype)
    p['same_fine']=p.pre_fine.eq(p.post_fine)
    sample=pd.read_parquet(P1/'data/prediction_pairs.parquet');sam=sample.assign(same_fine=sample.pre_id.map(reg.set_index('root_id').frozen_detailed_subtype).eq(sample.post_id.map(reg.set_index('root_id').frozen_detailed_subtype)))
    sam.groupby(['edge_present','split']).agg(n=('pre_id','size'),same_fine_fraction=('same_fine','mean')).to_csv(OUT/'analysis/old_negative_sampling_QC.csv')
    qs={col:float(k.loc[k.split.eq(0),col].quantile(.25)) for col in ['cleft_score','connection_score']}
    k.groupby('neuropil',dropna=False).agg(synapse_pairs=('id','size'),median_cleft=('cleft_score','median'),median_connection_score=('connection_score','median')).to_csv(OUT/'analysis/contact_ROI_quality.csv')
    candidate[['root_id','candidate_id','unit_id','frozen_detailed_subtype','official_hemisphere']].to_parquet(OUT/'registry/frozen18_with_history.parquet',index=False)
    parent.to_parquet(OUT/'registry/complete_left_ap1_parent.parquet',index=False)
    n_single=int(p.synapses.eq(1).sum())
    status={'passed':passed,'scope':'FlyWire source/count/split gate; rosette identification remains separately blocked','n_KC_synapse_pairs':len(k),'n_directed_pairs':len(p),'single_synapse_pairs':n_single,'single_synapse_pair_fraction':n_single/len(p),'mean_synapses_per_pair':float(p.synapses.mean()),'raw_graph_conflicts':len(bad),'duplicate_raw_synapse_ids':len(raw)-raw.id.nunique(),'quality_training_quartiles':qs,'rosette_status':'BLOCKED_MISSING_CONTACT_MOTIF_ANNOTATION','contact_features_for_discovery':False,'evaluation_exposure':'RETROSPECTIVE_NOT_BLIND','frozen18_historical_candidate_counts':candidate.candidate_id.value_counts().to_dict(),'elapsed_s':time.time()-tm}
    dump(OUT/'audit/GATE_STATUS.json',status)
    dump(OUT/'audit/gate_sources.json',{'dataset_version':'FlyWire v783','protocol':'P1D-v3.0-focused-ap1-retrospective','seed':SEED,'run_id':'p1d_v3_gate','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':[{ 'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [RAW,P1/'data/directed_pair_split.parquet',P1/'registry/all5177_identity.parquet',P1/'registry/ap1L_18_exploratory_members.parquet',Path(__file__),OUT/'audit/protocol.json']]})
    print(json.dumps(status),flush=True)
    assert passed,'Source/count gate failed'

def crossbrain():
    setup();tm=time.time()
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.compute as pc
    raw=ROOT/'simulation/data/external/male_cns_v1_0'
    manifest=json.loads((raw/'download_manifest.json').read_text());checks=[]
    for item in manifest:
        p=raw/item['file'];checks.append({'path':str(p.relative_to(ROOT)),'expected_sha256':item['sha256'],'observed_sha256':sha(p),'url':item['url']})
    assert all(x['expected_sha256']==x['observed_sha256'] for x in checks)
    ann=pd.read_feather(raw/'body-annotations-male-cns-v1.0-minconf-0.5.feather');nt=pd.read_feather(raw/'body-neurotransmitters-male-cns-v1.0.feather',columns=['body'])
    subset=pd.read_parquet(ROOT/'audit/meeting_revision_20261027/reproduced_structure/MaleCNS_v1_0_annotated_input_edges.parquet')
    ends=set(subset.pre.astype('int64'))|set(subset.post.astype('int64'));annotated=set(ann.bodyId.astype('int64'));ntids=set(nt.body.astype('int64'));missing=ends-annotated
    # Stream original full weights, selecting the exact target scope of the historical subset.
    targets=pa.array(sorted(set(subset.post.astype('int64'))),type=pa.int64())
    reader=ipc.open_file(pa.memory_map(str(raw/'connectome-weights-male-cns-v1.0-minconf-0.5.feather')));parts=[];totalrows=0
    for i in range(reader.num_record_batches):
        b=reader.get_batch(i);totalrows+=b.num_rows;mask=pc.is_in(b.column('body_post'),value_set=targets)
        if pc.any(mask).as_py():parts.append(pa.Table.from_batches([b]).filter(mask))
    selected=pa.concat_tables(parts).to_pandas().rename(columns={'body_pre':'pre','body_post':'post','weight':'raw_weight'})
    compare=subset[['pre','post','synapses']].merge(selected,on=['pre','post'],how='outer',indicator=True)
    bad=compare.loc[compare._merge.ne('both')|compare.synapses.ne(compare.raw_weight)]
    bad.to_csv(OUT/'audit/malecns_raw_subset_discrepancies.csv',index=False)
    pd.DataFrame({'body_id':[str(x) for x in sorted(missing)],'in_original_v1_nt_table':[x in ntids for x in sorted(missing)],'annotation_status':'not_in_released_body_annotation_table'}).to_parquet(OUT/'registry/malecns_missing_annotation_identity.parquet',index=False)
    status={'dataset_version':'MaleCNS v1.0 minconf0.5','release_hashes_match':True,'original_weight_rows':totalrows,'historical_input_subset_rows':len(subset),'exact_original_subset_conflicts':len(bad),'endpoint_ids':len(ends),'unannotated_endpoints':len(missing),'unannotated_but_in_original_NT_table':len(missing&ntids),'unannotated_and_absent_NT':len(missing-ntids),'interpretation':'Genuine released body IDs lacking curated body annotations if exact original weights and NT presence pass; do not invent type or homology. Missing annotation is not by itself proof of version mismatch.','BANC_independence':'User supplied primary-literature verification: distinct specimen from FlyWire, extreme right-turn bias. Record as user-verified until bibliographic source captured; no numerical replication in this stage.','external_mechanism_replication':'NOT_RUN','elapsed_s':time.time()-tm}
    dump(OUT/'audit/CROSSBRAIN_SOURCE_QC.json',status);dump(OUT/'audit/malecns_release_hashes.json',checks);print(json.dumps(status),flush=True)


def banc_qc():
    setup();tm=time.time();base=ROOT/'simulation/data/external/banc_v888'
    mp=base/'banc_888_meta.feather';ep=base/'banc_888_edgelist_simple_v3.feather'
    meta=pd.read_feather(mp);edges=pd.read_feather(ep,columns=['pre','post','count'])
    for c in ['banc_888_id','root_id']:meta[c]=meta[c].astype(str)
    edges.pre=edges.pre.astype(str);edges.post=edges.post.astype(str)
    kc=meta.loc[meta.cell_class.eq('kenyon_cell')].copy();assert kc.banc_888_id.is_unique
    kc.to_parquet(OUT/'registry/BANC_v888_KC_QC.parquet',index=False)
    ids=set(kc.banc_888_id);kk=edges.loc[edges.pre.isin(ids)&edges.post.isin(ids)]
    kk.to_parquet(OUT/'data/BANC_v888_v3_KC_directed_pairs.parquet',index=False)
    cross=kc.groupby(['cell_type','fafb_cell_type','hemibrain_cell_type','side'],dropna=False).size().reset_index(name='n_cells')
    cross.to_csv(OUT/'analysis/BANC_annotation_crosswalk_QC.csv',index=False)
    status={'dataset':'BANC v888 metadata + simple edgelist v3','n_annotation_rows':len(meta),'n_KC':len(kc),'KC_side_counts':kc.side.fillna('unknown').value_counts().to_dict(),'n_directed_KC_pairs':len(kk),'KC_synapse_count':int(kk['count'].sum()),'KC_single_edge_fraction':float(kk['count'].eq(1).mean()),'n_duplicate_directed_pairs':int(kk.duplicated(['pre','post']).sum()),'root_id_vs_banc888_id_KC_disagreements':int(kc.root_id.ne(kc.banc_888_id).sum()),'root_888_vs_banc888_id_KC_disagreements':int(kc.root_888.astype(str).ne(kc.banc_888_id).sum()),'canonical_join_key':'banc_888_id; root_888 agreement audited separately; current root_id preserved as versioned alias','specimen_independence':'USER_PRIMARY_SOURCE_VERIFIED_DISTINCT_FROM_FLYWIRE','sampling_caveat':'User reports extreme right-turn bias (~97th percentile); not an average population specimen','homology_mapping':'type crosswalk only; individual KC/18-cell unit homology NOT established','external_geometric_prediction':'BLOCKED_MISSING_HARMONIZED_REGIONAL_GEOMETRY','elapsed_s':time.time()-tm,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [mp,ep]]}
    dump(OUT/'audit/BANC_QC.json',status);print(json.dumps(status),flush=True)

def require_gate():assert json.loads((OUT/'audit/GATE_STATUS.json').read_text())['passed']


def geometry():
    require_gate();reg,ids,parent,candidate=setup();tm=time.time()
    import analyze_ap2_arbor as arbor
    # Reuse the existing certified segment-distance and closed-mesh implementation.
    arbor.OUT=OUT;arbor.ATLAS=ATLAS
    mm=arbor.meshes();roots=parent.root_id.to_numpy(str);cells={};rows=[];sources=[]
    for n,rid in enumerate(roots):
        path=ATLAS/'geometry'/f'{rid}.json';dest=OUT/'cache/cells'/f'{rid}.npz'
        if dest.exists():
            c=dict(np.load(dest));assert str(c['source_sha256'])==sha(path)
        else:
            c=arbor.cell_geometry(rid,'left',mm,step=1.)
            np.savez_compressed(dest,**c,source_sha256=np.array(sha(path)))
        cells[rid]=c
        rows.append({'root_id':rid,**{f'{roi}_cable_um':float(c['weight'][c['region_mask'][:,i]].sum()) for i,roi in enumerate(REGIONS)}})
        sources.append({'root_id':rid,'geometry_sha256':sha(path),'cache_sha256':sha(dest)})
        if (n+1)%30==0:print('geometry cells',n+1,'elapsed',round(time.time()-tm),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'data/regional_cable_lengths.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'audit/regional_geometry_sources.csv',index=False)
    # Quadrature and exact-distance checks on a fixed identity-selected panel.
    panel=sorted(roots)[:4];fine={r:arbor.cell_geometry(r,'left',mm,step=.5) for r in panel};numerics=[];exact=[]
    for target in panel:
        ti=arbor.indices(cells[target]);fi=arbor.indices(fine[target])
        for source in panel:
            if source==target:continue
            k1,_=arbor.directional(cells[source],ti);k2,_=arbor.directional(fine[source],fi)
            numerics.append({'source':source,'target':target,'max_kernel_diff':float(np.nanmax(np.abs(k1-k2))),'missingness_agrees':bool(np.array_equal(np.isnan(k1),np.isnan(k2)))})
            points=cells[source]['midpoint'][:20]
            brute=arbor.projection_distances(points[:,None,:],ti[0].a[None,:,:],ti[0].b[None,:,:]).min(1)
            exact.append(float(np.max(np.abs(brute-ti[0].distances(points)))))
    passed=max(x['max_kernel_diff'] for x in numerics)<=.02 and all(x['missingness_agrees'] for x in numerics) and max(exact)<1e-9
    dump(OUT/'validation/regional_geometry_numerics.json',{'passed':passed,'panel':panel,'pairs':numerics,'max_exact_distance_error_um':max(exact),'source_step_um':1,'reference_step_um':.5})
    assert passed,'Refine geometry under explicit protocol amendment, do not interpret failed numerical approximation'
    cols=[f'{roi}_gaussian_{s}um' for roi in REGIONS for s in [1,2,5]];parts=[]
    for n,target in enumerate(roots):
        path=OUT/'cache/targets'/f'{target}.parquet'
        if path.exists():d=pd.read_parquet(path)
        else:
            ti=arbor.indices(cells[target]);rr=[]
            for source in roots:
                if source==target:continue
                kernels,dist=arbor.directional(cells[source],ti)
                rr.append({'pre_id':source,'post_id':target,**dict(zip(cols,kernels)),**{f'{roi}_mean_distance_um':v for roi,v in zip(REGIONS,dist)}})
            d=pd.DataFrame(rr);d.to_parquet(path,index=False)
        assert len(d)==147 and d.post_id.eq(target).all();parts.append(d)
        if (n+1)%10==0:print('regional target',n+1,'/148 elapsed',round(time.time()-tm),flush=True)
    pd.concat(parts,ignore_index=True).to_parquet(OUT/'data/regional_pair_opportunities.parquet',index=False)
    dump(OUT/'audit/GEOMETRY_STATUS.json',{'status':'DONE','n_cells':148,'n_directed_pairs':148*147,'contact_records_used':0,'regions':REGIONS,'numerical_checks_passed':True,'elapsed_s':time.time()-tm,'scope':'complete left ap1 parent only, not all5177'})


def shared():
    reg,ids,parent,candidate=setup();ix=pd.Index(ids).get_indexer(parent.root_id)
    train=sparse.load_npz(P1/'data/pair_grouped_train.npz').astype(float).tocsr()
    xyz=reg[['soma_x_um','soma_y_um','soma_z_um']].to_numpy(float)
    scale=np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2,axis=1)));S=(xyz-xyz.mean(0))/scale
    geo=pd.read_parquet(P1/'data/contact_free_tree_features.parquet').set_index('root_id').loc[ids]
    T=StandardScaler().fit_transform(geo)
    return reg,ids,parent,candidate,ix,train,xyz,S,T

def best_overlap(labels,member_indices):
    rows=[];members=set(member_indices)
    for lab in np.unique(labels):
        group=set(np.flatnonzero(labels==lab));n=len(members&group)
        rows.append((n/len(members|group),n/len(members),n/len(group),len(group),group))
    return max(rows,key=lambda r:r[0])

def cm019():
    require_gate();reg,ids,parent,candidate,ix,train,xyz,S,T=shared();tm=time.time()
    target=np.flatnonzero(parent.root_id.isin(candidate.root_id));tc=train.tocoo();records=[];membership=[]
    variants={'count':train,'binary':(train>0).astype(float)}
    for threshold in [2,3,5]:
        mask=tc.data>=threshold;variants[f'count_ge{threshold}']=sparse.csr_matrix((tc.data[mask],(tc.row[mask],tc.col[mask])),shape=train.shape)
    k=pd.read_parquet(OUT/'data/KC_KC_contact_QC.parquet');k=k.loc[k.split.eq(0)];quant=json.loads((OUT/'audit/GATE_STATUS.json').read_text())['quality_training_quartiles'];lookup={v:i for i,v in enumerate(ids)}
    for col,q in quant.items():
        kk=k.loc[k[col]>=q].groupby(['pre_id','post_id']).size().reset_index(name='weight')
        variants['quality_'+col]=sparse.csr_matrix((kk.weight.to_numpy(float),(kk.pre_id.map(lookup),kk.post_id.map(lookup))),shape=train.shape)
    for j in range(20):
        keep=pair_uniform(ids,tc.row,tc.col,2026111000+j)>=.2
        variants[f'pair_drop20_{j:02d}']=sparse.csr_matrix((tc.data[keep],(tc.row[keep],tc.col[keep])),shape=train.shape)
    for name,g in variants.items():
        e=_embedding(g,32,20261030);x=fuse_blocks(e,S,T,1,1)[ix]
        for kval in [2,3,4]:
            labels=KMeans(n_clusters=kval,n_init=5,random_state=20261030).fit_predict(x)
            jac,recall,purity,n,selected=best_overlap(labels,target)
            records.append({'condition':name,'K':kval,'jaccard':jac,'retention':recall,'purity':purity,'matched_n':n,'graph_directed_pairs':g.nnz,'graph_weight':float(g.sum()),'state':'EXPLORATORY_NO_PROMOTION'})
            membership.extend({'condition':name,'K':kval,'root_id':parent.iloc[i].root_id,'in_frozen18':i in set(target)} for i in sorted(selected))
        print('cm019',name,'elapsed',round(time.time()-tm),flush=True)
    pd.DataFrame(records).to_csv(OUT/'analysis/CM019_18_perturbations.csv',index=False)
    pd.DataFrame(membership).to_parquet(OUT/'registry/CM019_18_matched_children.parquet',index=False)
    dump(OUT/'audit/CM019_STATUS.json',{'status':'DONE_PERTURBATION_SCREEN','n_frozen':18,'n_conditions':len(variants),'n_partitions':len(records),'new_stable_units':0,'elapsed_s':time.time()-tm})

def routing():
    require_gate();reg,ids,parent,candidate,ix,train,xyz,S,T=shared();tm=time.time()
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    target=np.flatnonzero(parent.root_id.isin(candidate.root_id));controls=np.flatnonzero(~parent.root_id.isin(candidate.root_id));n=len(parent)
    strength=np.column_stack([np.asarray(train.sum(1)).ravel(),np.asarray(train.sum(0)).ravel()])
    PCs=PCA(n_components=8,random_state=SEED).fit_transform(T[ix])
    X=StandardScaler().fit_transform(np.c_[xyz[ix],np.log1p(strength[ix]),PCs])
    distances=np.linalg.norm(X[target,None,:]-X[controls][None,:,:],axis=2)
    nearest=np.argsort(distances,axis=1)[:,:20];rng=np.random.default_rng(SEED);null=[];balance=[]
    for rep in range(1000):
        chosen=[]
        for i in rng.permutation(len(target)):
            pool=[controls[j] for j in nearest[i] if controls[j] not in chosen]
            if not pool:raise RuntimeError('Nearest matching pool exhausted')
            chosen.append(int(rng.choice(pool)))
        null.append(chosen);balance.append(np.max(np.abs(X[target].mean(0)-X[chosen].mean(0))))
    null=np.asarray(null);balance=np.asarray(balance);balanced=balance<=.25
    ai,bi=linear_sum_assignment(distances);best=controls[bi]
    pd.DataFrame({'replicate':np.repeat(np.arange(1000),18),'root_id':parent.root_id.to_numpy()[null.ravel()]}).to_parquet(OUT/'registry/matched_null_groups.parquet',index=False)
    pd.DataFrame({'replicate':np.arange(1000),'max_parent_SD_mean_difference':balance,'passes_0_25_parent_SD':balanced}).to_csv(OUT/'analysis/matched_null_balance.csv',index=False)
    pd.DataFrame({'target_root_id':parent.iloc[target[ai]].root_id.to_numpy(),'control_root_id':parent.iloc[best].root_id.to_numpy(),'covariate_distance':distances[ai,bi]}).to_csv(OUT/'registry/optimal_matched_controls.csv',index=False)
    # Record geometry with the SAME frozen control groups, never rematch by response.
    geom=[]
    for label,Z in [('soma_um',xyz[ix]),('standardized_tree',T[ix])]:
        observed=float(np.mean(pdist(Z[target])));vals=np.array([np.mean(pdist(Z[g])) for g in null])
        geom.append({'feature':label,'observed_mean_pair_distance':observed,'all_null_median':np.median(vals),'balanced_groups':int(balanced.sum()),'balanced_null_median':np.median(vals[balanced]) if balanced.any() else None})
    pd.DataFrame(geom).to_csv(OUT/'analysis/matched_geometry_dispersion.csv',index=False)
    fp=pd.read_parquet(ATLAS/'analysis/partner_fingerprints.parquet');fp=fp[fp.root_id.isin(parent.root_id)].copy()
    fp.to_parquet(OUT/'data/parent_partner_fingerprints.parquet',index=False)
    profiles={};stats=[];all_null=[]
    coverage=fp.assign(annotated=fp.partner_cell_class.ne('')).groupby(['root_id','direction']).apply(lambda g:pd.Series({'all_observed_synapse_pairs':g.synapse_pairs.sum(),'annotated_class_synapse_pairs':g.loc[g.annotated,'synapse_pairs'].sum()}),include_groups=False).reset_index()
    coverage.to_csv(OUT/'analysis/partner_annotation_coverage.csv',index=False)
    for endpoint,direction,cls in [('PN_input','in','ALPN'),('MBON_output','out','MBON')]:
        d=fp.loc[fp.direction.eq(direction)&fp.partner_cell_class.eq(cls)].copy();d['partner_cell_type']=d.partner_cell_type.replace('','UNKNOWN_TYPE')
        pivot=d.pivot_table(index='root_id',columns='partner_cell_type',values='synapse_pairs',aggfunc='sum',fill_value=0).reindex(parent.root_id,fill_value=0)
        A=pivot.to_numpy(float);mass=A.sum(1);H=np.sqrt(A/np.maximum(mass[:,None],1));profiles[endpoint]=(A,H,pivot.columns)
        def contrast(group):
            other=np.setdiff1d(np.arange(n),group);return float(np.sum((H[group].mean(0)-H[other].mean(0))**2))
        observed=contrast(target);vals=np.array([contrast(g) for g in null]);eligible=vals[balanced]
        pvalue=(1+np.sum(eligible>=observed))/(1+len(eligible)) if len(eligible)>=100 else None
        stats.append({'endpoint':endpoint,'n_candidate':18,'n_parent':n,'n_partner_types':len(pivot.columns),'candidate_zero_observed_mass':int((mass[target]==0).sum()),'observed_squared_Hellinger_centroid_distance':observed,'n_balanced_null':len(eligible),'matched_p':pvalue,'state':'MATCHING_FAILED_NO_INFERENCE' if pvalue is None else 'RETROSPECTIVE_MATCHED_STRUCTURAL_CONTRAST','not_functional_validation':True})
        all_null.extend({'endpoint':endpoint,'replicate':i,'contrast':v,'balanced':bool(balanced[i])} for i,v in enumerate(vals))
        diff=(A[target].sum(0)/max(A[target].sum(),1)-A[controls].sum(0)/max(A[controls].sum(),1))
        pd.DataFrame({'partner_type':pivot.columns,'candidate_synapses':A[target].sum(0).astype(int),'parent_other_synapses':A[controls].sum(0).astype(int),'fraction_difference':diff}).to_csv(OUT/f'analysis/{endpoint}_type_contrasts.csv',index=False)
    # Holm adjustment for the two prespecified endpoints only; undefined stays missing.
    valid=[i for i,r in enumerate(stats) if r['matched_p'] is not None]
    prev=0
    for rank,i in enumerate(sorted(valid,key=lambda i:stats[i]['matched_p'])):
        prev=max(prev,min(1,(2-rank)*stats[i]['matched_p']));stats[i]['holm_p']=prev
    pd.DataFrame(stats).to_csv(OUT/'analysis/routing_matched_null_results.csv',index=False);pd.DataFrame(all_null).to_parquet(OUT/'analysis/routing_null_distributions.parquet',index=False)
    # Concrete raw routes; observed synapse counts, no inferred propagation probabilities.
    routes=[]
    for rid in candidate.root_id:
        q=fp[fp.root_id.eq(rid)];a=q[q.direction.eq('in')&q.partner_cell_class.eq('ALPN')].sort_values('synapse_pairs',ascending=False).head(2);b=q[q.direction.eq('out')&q.partner_cell_class.eq('MBON')].sort_values('synapse_pairs',ascending=False).head(2)
        for _,pre in a.iterrows():
            for _,post in b.iterrows():routes.append({'pre_PN_id':pre.partner_id,'PN_type':pre.partner_cell_type,'KC_id':rid,'post_MBON_id':post.partner_id,'MBON_type':post.partner_cell_type,'PN_to_KC_synapses':pre.synapse_pairs,'KC_to_MBON_synapses':post.synapse_pairs,'evidence':'OBSERVED_DIRECTED_STRUCTURE_ONLY'})
    pd.DataFrame(routes).to_csv(OUT/'analysis/observed_PN_KC_MBON_routes.csv',index=False)
    dump(OUT/'audit/ROUTING_STATUS.json',{'status':'DONE_WITH_BALANCE_LIMITS','balanced_null_groups':int(balanced.sum()),'required':100,'median_max_parent_SD_difference':float(np.median(balance)),'optimal_matching_max_parent_SD_difference':float(np.max(np.abs(X[target].mean(0)-X[best].mean(0)))),'match_definition':'mean difference divided by whole-parent SD; not pooled two-group SMD','same_fine_type_side':True,'historical_external_route_exposure':'RETROSPECTIVE_NOT_NEW_BLIND','n_route_examples':len(routes),'elapsed_s':time.time()-tm})


def models(nested_only=False):
    require_gate();assert json.loads((OUT/'audit/GEOMETRY_STATUS.json').read_text())['status']=='DONE'
    reg,ids,parent,candidate,ix,train,xyz,S,T=shared();tm=time.time();n=len(parent);roots=parent.root_id.to_numpy(str)
    i,j=np.where(~np.eye(n,dtype=bool));a,b=ix[i],ix[j]
    allpairs=pd.read_parquet(P1/'data/directed_pair_split.parquet');lookup={r:i for i,r in enumerate(ids)}
    full=sparse.csr_matrix((allpairs.synapses.to_numpy(float),(allpairs.pre_id.map(lookup),allpairs.post_id.map(lookup))),shape=train.shape)
    y=(full[a,b].A1>0).astype(int);sp=pair_split(ids,a,b,20261030)
    deg=np.c_[np.asarray(train.sum(1)).ravel(),np.asarray(train.sum(0)).ravel()];ds=deg[a].copy();dt=deg[b].copy();w=train[a,b].A1;wr=train[b,a].A1
    ds[:,0]-=w;ds[:,1]-=wr;dt[:,1]-=w;dt[:,0]-=wr
    assert min(ds.min(),dt.min())>=0
    geom=pd.read_parquet(P1/'data/contact_free_tree_features.parquet').set_index('root_id').loc[ids].to_numpy(float)
    continuous=np.c_[xyz[a],xyz[b],np.abs(xyz[a]-xyz[b]),np.linalg.norm(xyz[a]-xyz[b],axis=1),geom[a],geom[b],np.abs(geom[a]-geom[b])]
    regional=pd.read_parquet(OUT/'data/regional_pair_opportunities.parquet').set_index(['pre_id','post_id']);cols=[f'{r}_gaussian_{s}um' for r in REGIONS for s in [1,2,5]]
    forward=regional.loc[pd.MultiIndex.from_arrays([roots[i],roots[j]]),cols].to_numpy();reverse=regional.loc[pd.MultiIndex.from_arrays([roots[j],roots[i]]),cols].to_numpy()
    lengths=pd.read_csv(OUT/'data/regional_cable_lengths.csv',dtype={'root_id':str}).set_index('root_id').loc[roots].to_numpy()
    region=np.c_[np.nan_to_num(forward),np.nan_to_num(reverse),np.isnan(forward).astype(float),np.isnan(reverse).astype(float),np.log1p(lengths[i]),np.log1p(lengths[j])]
    m0=np.ones((len(i),1));m1=np.c_[continuous,np.log1p(ds),np.log1p(dt)];m2=np.c_[m1,region]
    member=np.isin(roots,candidate.root_id).astype(float);labels=np.c_[member[i],member[j],member[i]*member[j]]
    emb=np.load(P1/'data/train_only_embedding.npy')
    designs={'M0_known_type':m0,'M1_descriptor':m1,'M2_regional_arbor':m2,'M3_frozen18':np.c_[m2,labels],'M4_graph_embedding':np.c_[m2,emb[a],emb[b],np.abs(emb[a]-emb[b])]}
    if nested_only:
        graph=np.c_[emb[a],emb[b],np.abs(emb[a]-emb[b])]
        return nested_comparison(m2,labels,graph,y,sp,roots[i],roots[j])
    data=pd.DataFrame({'pre_id':roots[i],'post_id':roots[j],'observed_edge':y,'pair_split':sp})
    # A real node-disjoint secondary endpoint; graph-derived labels/embedding are excluded.
    nodefold=np.array([int.from_bytes(hashlib.blake2b((r+'|20261031').encode(),digest_size=8).digest(),'little')%5 for r in roots])
    group=np.where(nodefold<3,0,np.where(nodefold==3,1,2));block=np.where(group[i]==group[j],group[i],-1);data['cell_disjoint_split']=block
    data.to_parquet(OUT/'data/complete_parent_prediction_pairs.parquet',index=False)
    pd.DataFrame({'root_id':roots,'cell_split':group}).to_csv(OUT/'registry/cell_disjoint_membership.csv',index=False)
    all_results=[];testpred=[];tuning=[]
    for evaluation,split,dd in [('pair_grouped',sp,designs),('cell_disjoint_geometry',block,{'M0_known_type':m0,'M1_descriptor_no_graph_degree':continuous,'M2_regional_arbor_no_graph_degree':np.c_[continuous,region]})]:
        tr=split==0;va=split==1;te=split==2
        assert set(y[tr])==set(y[va])==set(y[te])=={0,1}
        for name,x in dd.items():
            scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x)/np.sqrt(x.shape[1]);fits=[]
            for gamma in [.5,2.]:
                rff=RBFSampler(n_components=128,gamma=gamma,random_state=SEED);rff.fit(z[tr]);f=rff.transform(z)
                for C in [.1,1.]:
                    model=LogisticRegression(C=C,max_iter=1000,solver='lbfgs',random_state=SEED);model.fit(f[tr],y[tr]);val=log_loss(y[va],model.predict_proba(f[va])[:,1])
                    tuning.append({'evaluation':evaluation,'model':name,'gamma':gamma,'C':C,'validation_log_loss':val,'n_iter':int(model.n_iter_[0])})
                    fits.append((val,gamma,C,model,rff))
            val,gamma,C,model,rff=min(fits,key=lambda x:x[0]);pred=model.predict_proba(rff.transform(z[te]))[:,1]
            all_results.append({'evaluation':evaluation,'model':name,'log_loss':log_loss(y[te],pred),'AUROC':roc_auc_score(y[te],pred),'n_train':int(tr.sum()),'n_val':int(va.sum()),'n_test':int(te.sum()),'test_edge_prevalence':float(y[te].mean()),'n_input_features':x.shape[1],'n_trainable_coefficients':129,'gamma':gamma,'C':C,'state':'RETROSPECTIVE_FIXED_PARENT','capacity_note':'equal nominal coefficients and search budget; effective capacity not guaranteed identical'})
            for row,pv in zip(data.loc[te,['pre_id','post_id','observed_edge']].to_dict('records'),pred):testpred.append({**row,'evaluation':evaluation,'model':name,'prediction':float(pv)})
            print(evaluation,name,all_results[-1]['log_loss'],flush=True)
    pd.DataFrame(all_results).to_csv(OUT/'analysis/MODEL_COMPARISON.csv',index=False);pd.DataFrame(tuning).to_csv(OUT/'analysis/model_tuning.csv',index=False);pd.DataFrame(testpred).to_parquet(OUT/'data/model_test_predictions.parquet',index=False)
    dump(OUT/'audit/MODELS_STATUS.json',{'status':'DONE_FOCUSED_RETROSPECTIVE','n_pairs':len(y),'n_models':len(all_results),'nominal_parameters_each':129,'tuning_settings_each':4,'cell_holdout_excludes_all_graph_derived_predictors':True,'cell_holdout_M3_M4':'NOT_EVALUATED_FROZEN_LABELS_AND_EMBEDDING_NOT_INDUCTIVE','elapsed_s':time.time()-tm})


def nested_comparison(base,labels,graph,y,split,pre,post):
    """Posthoc mechanism check: keep geometric features identical across add-ons."""
    tr=split==0;va=split==1;te=split==2;sc=StandardScaler().fit(base[tr]);z=sc.transform(base)/np.sqrt(base.shape[1])
    ls=StandardScaler().fit(labels[tr]);lab=ls.transform(labels)
    gs=StandardScaler().fit(graph[tr]);gz=gs.transform(graph)
    pca=PCA(n_components=3,random_state=SEED).fit(gz[tr]);pc=pca.transform(gz);pc=StandardScaler().fit(pc[tr]).transform(pc)
    additions={'M2_fixed_geometry':np.zeros_like(lab),'M3_plus_frozen18':lab,'M4_plus_graph_PC3':pc}
    records=[];predrows=[]
    for seed in range(SEED,SEED+5):
        fits={name:[] for name in additions}
        for gamma in [.5,2.]:
            rf=RBFSampler(n_components=128,gamma=gamma,random_state=seed).fit(z[tr]);features=rf.transform(z)
            for name,extra in additions.items():
                x=np.c_[features,extra]
                for C in [.1,1.]:
                    model=LogisticRegression(C=C,max_iter=1000,random_state=SEED).fit(x[tr],y[tr]);val=log_loss(y[va],model.predict_proba(x[va])[:,1]);fits[name].append((val,gamma,C,model.predict_proba(x[te])[:,1]))
        for name,ff in fits.items():
            val,gamma,C,p=min(ff,key=lambda r:r[0]);records.append({'seed':seed,'model':name,'log_loss':log_loss(y[te],p),'AUROC':roc_auc_score(y[te],p),'gamma':gamma,'C':C,'nominal_coefficients':132,'addon_dimensions':3,'status':'POSTHOC_SHARED_GEOMETRY_MAPPING_CONTROL'})
            predrows.extend({'seed':seed,'model':name,'pre_id':a,'post_id':b,'observed_edge':int(t),'prediction':float(v)} for a,b,t,v in zip(pre[te],post[te],y[te],p))
    pd.DataFrame(records).to_csv(OUT/'analysis/nested_shared_geometry_comparison.csv',index=False)
    pd.DataFrame(predrows).to_parquet(OUT/'data/nested_test_predictions.parquet',index=False)
    dump(OUT/'audit/nested_control_status.json',{'status':'DONE_POSTHOC_CONTROL','seeds':list(range(SEED,SEED+5)),'geometry_mapping':'identical128 random features for each seed/gamma across the three arms','nominal_coefficients':132,'effective_capacity_note':'M2 addon is zero; M3 and M4 each have3 nonzero addon columns. Equal nominal dimension does not imply identical effective capacity.','graph_projection_train_only':True,'graph_PCA_variance_fraction':float(pca.explained_variance_ratio_.sum()),'not_full_graph_representation':True})

def nested_models():return models(nested_only=True)


def density_control():
    """Posthoc diagnostic: same number of removed edges, unrelated to counts."""
    require_gate();reg,ids,parent,candidate,ix,train,xyz,S,T=shared();tc=train.tocoo()
    target=np.flatnonzero(parent.root_id.isin(candidate.root_id));rows=[]
    keys=np.minimum(tc.row,tc.col)*len(ids)+np.maximum(tc.row,tc.col)
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    for threshold in [2,3,5]:
        desired=int((tc.data>=threshold).sum())
        for repeat in range(5):
            rng=np.random.default_rng(2026112000+100*threshold+repeat);order=rng.permutation(len(unique));cumulative=np.cumsum(counts[order]);take=order[cumulative<=desired]
            chosen=np.zeros(len(unique),bool);chosen[take]=True;kept=int(counts[take].sum())
            # Fill one remaining single-direction pair when possible without splitting reciprocal pairs.
            if kept<desired:
                singles=np.flatnonzero((~chosen)&(counts==1))
                if len(singles):chosen[singles[0]]=True
            mask=chosen[inverse];g=sparse.csr_matrix((tc.data[mask],(tc.row[mask],tc.col[mask])),shape=train.shape)
            assert abs(g.nnz-desired)<=1
            e=_embedding(g,32,20261030);x=fuse_blocks(e,S,T,1,1)[ix]
            for kval in [2,3,4]:
                lab=KMeans(n_clusters=kval,n_init=5,random_state=20261030).fit_predict(x);jac,ret,purity,size,_=best_overlap(lab,target)
                rows.append({'count_threshold_comparator':threshold,'replicate':repeat,'K':kval,'directed_pairs':g.nnz,'target_directed_pairs':desired,'jaccard':jac,'retention':ret,'purity':purity,'matched_n':size,'status':'POSTHOC_EDGE_COUNT_MATCHED_RANDOM_CONTROL'})
    pd.DataFrame(rows).to_csv(OUT/'analysis/density_matched_dropout.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['gate','geometry','cm019','routing','crossbrain','banc_qc','models','density_control','nested_models'])
    args=parser.parse_args()
    with threadpool_limits(limits=4):globals()[args.stage]()
