"""Reused regional geometry, bilateral and soma-blocked ap1 stress tests."""
from pathlib import Path
import argparse,hashlib,json,time,subprocess,shutil
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits
from neuronmimic.p1_structure import pair_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nature_kc_20261101/p1d_v3_bilateral'
OLD=ROOT/'artifacts/nature_kc_20261030/p1d_v3'
P1=ROOT/'artifacts/nature_kc_20261030/p1_v2_fine_types'
ATLAS=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
REGIONS=['FULL','CA','PED','ML','VL']
BUNDLE=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):
    Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def setup():
    for sub in ['audit','data','registry','analysis','figures','report','models','validation','cache/cells','cache/targets']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    assert json.loads((OLD/'audit/GATE_STATUS.json').read_text())['passed']
    reg=pd.read_parquet(P1/'registry/all5177_identity.parquet');pop=reg[reg.frozen_detailed_subtype.eq("KCa'b'-ap1")].copy()
    assert pop.groupby('official_hemisphere').size().to_dict()=={'left':148,'right':133}
    assert reg.root_id.map(lambda x:isinstance(x,str)).all()
    return reg,pop,json.loads((OUT/'audit/protocol.json').read_text())

def spatial_blocks(xyz,n_blocks=5):
    """Position-only, ordered contiguous blocks. No outcome or graph features."""
    pc=PCA(n_components=1,svd_solver='full').fit(xyz)
    axis=pc.components_[0].copy()
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    score=(xyz-xyz.mean(0))@axis
    order=np.argsort(score,kind='stable');block=np.empty(len(xyz),int)
    for i,ix in enumerate(np.array_split(order,n_blocks)):block[ix]=i
    return block,score,axis

def spatial_pair_fold(block,i,j,fold):
    state=np.where(block==fold,2,np.where(block==(fold+1)%5,1,0))
    return np.where(state[i]==state[j],state[i],-1)

def geometry():
    reg,pop,cfg=setup();tm=time.time()
    if (OUT/'audit/GEOMETRY_STATUS.json').exists():raise RuntimeError('Completed geometry output; no overwrite')
    import analyze_ap2_arbor as arbor
    arbor.OUT=OUT;arbor.ATLAS=ATLAS;mm=arbor.meshes()
    rows=[];sources=[];cells={};right=pop[pop.official_hemisphere.eq('right')].root_id.to_numpy(str)
    for n,rid in enumerate(right):
        path=ATLAS/'geometry'/f'{rid}.json';dest=OUT/'cache/cells'/f'{rid}.npz'
        if dest.exists():
            c=dict(np.load(dest));assert str(c['source_sha256'])==sha(path)
        else:
            c=arbor.cell_geometry(rid,'right',mm,step=1.)
            np.savez_compressed(dest,**c,source_sha256=np.array(sha(path)))
        cells[rid]=c
        rows.append({'root_id':rid,'side':'right',**{f'{roi}_cable_um':float(c['weight'][c['region_mask'][:,i]].sum()) for i,roi in enumerate(REGIONS)}})
        sources.append({'root_id':rid,'side':'right','geometry_path':str(path.relative_to(ROOT)),'sha256':sha(path),'cache_sha256':sha(dest)})
        if (n+1)%30==0:print('right cells',n+1,round(time.time()-tm),flush=True)
    panel=sorted(right)[:4];fine={r:arbor.cell_geometry(r,'right',mm,step=.5) for r in panel};num=[]
    for target in panel:
        ti=arbor.indices(cells[target]);fi=arbor.indices(fine[target])
        for source in panel:
            if source==target:continue
            k1,_=arbor.directional(cells[source],ti);k2,_=arbor.directional(fine[source],fi)
            pts=cells[source]['midpoint'][:20]
            brute=arbor.projection_distances(pts[:,None,:],ti[0].a[None,:,:],ti[0].b[None,:,:]).min(1)
            num.append({'source':source,'target':target,'max_kernel_diff':float(np.nanmax(np.abs(k1-k2))),'missingness_equal':bool(np.array_equal(np.isnan(k1),np.isnan(k2))),'exact_distance_error':float(np.max(np.abs(brute-ti[0].distances(pts))))})
    passed=all(x['max_kernel_diff']<=.02 and x['missingness_equal'] and x['exact_distance_error']<1e-9 for x in num)
    dump(OUT/'validation/right_geometry_numerics.json',{'passed':passed,'pairs':num,'n_reference_cells':4});assert passed
    cols=[f'{roi}_gaussian_{s}um' for roi in REGIONS for s in [1,2,5]];parts=[]
    for n,target in enumerate(right):
        path=OUT/'cache/targets'/f'{target}.parquet'
        if path.exists():d=pd.read_parquet(path)
        else:
            ti=arbor.indices(cells[target]);rr=[]
            for source in right:
                if source==target:continue
                k,dist=arbor.directional(cells[source],ti)
                rr.append({'pre_id':source,'post_id':target,**dict(zip(cols,k)),**{f'{roi}_mean_distance_um':v for roi,v in zip(REGIONS,dist)}})
            d=pd.DataFrame(rr);d.to_parquet(path,index=False)
        assert len(d)==132 and d.post_id.eq(target).all();parts.append(d)
        if (n+1)%10==0:print('right targets',n+1,'/133',round(time.time()-tm),flush=True)
    left_sources=pd.read_csv(OLD/'audit/regional_geometry_sources.csv',dtype={'root_id':str})
    for _,r in left_sources.iterrows():
        path=ATLAS/'geometry'/f'{r.root_id}.json';assert sha(path)==r.geometry_sha256
        sources.append({'root_id':r.root_id,'side':'left','geometry_path':str(path.relative_to(ROOT)),'sha256':r.geometry_sha256,'cache_sha256':r.cache_sha256})
    left_len=pd.read_csv(OLD/'data/regional_cable_lengths.csv',dtype={'root_id':str});left_len['side']='left'
    pd.concat([left_len,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/'data/regional_cable_lengths.csv',index=False)
    dl=pd.read_parquet(OLD/'data/regional_pair_opportunities.parquet');dl['side']='left';dr=pd.concat(parts,ignore_index=True);dr['side']='right'
    pd.concat([dl,dr],ignore_index=True).to_parquet(OUT/'data/regional_pair_opportunities.parquet',index=False)
    pd.DataFrame(sources).to_csv(OUT/'audit/geometry_sources.csv',index=False);pop.to_parquet(OUT/'registry/bilateral_complete_ap1.parquet',index=False)
    dump(OUT/'audit/GEOMETRY_STATUS.json',{'status':'DONE','left_n':148,'right_n':133,'left_pairs':len(dl),'right_pairs':len(dr),'right_recomputed':True,'left_reused_and_hash_checked':True,'contacts_used':0,'elapsed_s':time.time()-tm})

def prepare_designs():
    reg,pop,cfg=setup();assert (OUT/'audit/GEOMETRY_STATUS.json').exists()
    ids=reg.root_id.to_numpy(str);lookup=pd.Index(ids)
    edges=pd.read_parquet(P1/'data/directed_pair_split.parquet');full=sparse.csr_matrix((edges.synapses.to_numpy(),(lookup.get_indexer(edges.pre_id),lookup.get_indexer(edges.post_id))),shape=(len(ids),len(ids)))
    geom=pd.read_parquet(P1/'data/contact_free_tree_features.parquet').set_index('root_id')
    regional=pd.read_parquet(OUT/'data/regional_pair_opportunities.parquet').set_index(['pre_id','post_id'])
    lengths=pd.read_csv(OUT/'data/regional_cable_lengths.csv',dtype={'root_id':str}).set_index('root_id')
    cols=[f'{roi}_gaussian_{s}um' for roi in REGIONS for s in [1,2,5]]
    reflect=float(reg.groupby('official_hemisphere').soma_x_um.median().loc[['left','right']].sum())
    results={};blockrows=[];blockaudit=[]
    for side,frame in pop.groupby('official_hemisphere'):
        roots=frame.root_id.to_numpy(str);n=len(roots);ix=lookup.get_indexer(roots);i,j=np.where(~np.eye(n,dtype=bool));a,b=ix[i],ix[j]
        xyz=frame[['soma_x_um','soma_y_um','soma_z_um']].to_numpy(float);gg=geom.loc[roots].copy()
        blocks,scores,axis=spatial_blocks(xyz)
        blockrows.extend({'root_id':r,'side':side,'block':int(bl),'PCA1_score_um':float(v)} for r,bl,v in zip(roots,blocks,scores))
        from scipy.spatial.distance import cdist
        for fold in range(5):
            test=blocks==fold;val=blocks==(fold+1)%5;tr=~(test|val)
            blockaudit.append({'side':side,'fold':fold,'n_train_cells':int(tr.sum()),'n_val_cells':int(val.sum()),'n_test_cells':int(test.sum()),'min_train_test_soma_distance_um':float(cdist(xyz[tr],xyz[test]).min()),'axis_xyz':json.dumps(axis.tolist()),'label':'SOMA_BLOCKED_NO_EXCLUSION_GAP'})
        if side=='right':
            xyz[:,0]=reflect-xyz[:,0];gg['centroid_x_um']=reflect-gg.centroid_x_um
        tree=gg.to_numpy(float)
        basic=np.c_[xyz[i],xyz[j],np.abs(xyz[i]-xyz[j]),np.linalg.norm(xyz[i]-xyz[j],axis=1),tree[i],tree[j],np.abs(tree[i]-tree[j])]
        f=regional.loc[pd.MultiIndex.from_arrays([roots[i],roots[j]]),cols].to_numpy();r=regional.loc[pd.MultiIndex.from_arrays([roots[j],roots[i]]),cols].to_numpy()
        ll=lengths.loc[roots,[f'{roi}_cable_um' for roi in REGIONS]].to_numpy()
        roi=np.c_[np.nan_to_num(f),np.nan_to_num(r),np.isnan(f).astype(float),np.isnan(r).astype(float),np.log1p(ll[i]),np.log1p(ll[j])]
        pair=pd.DataFrame({'pre_id':roots[i],'post_id':roots[j],'observed_edge':(full[a,b].A1>0).astype(int),'side':side,'pair_split':pair_split(ids,a,b,20261030)})
        for fold in range(5):pair[f'spatial_fold_{fold}']=spatial_pair_fold(blocks,i,j,fold)
        pair.to_parquet(OUT/f'data/{side}_pairs.parquet',index=False)
        np.savez_compressed(OUT/f'data/{side}_designs.npz',M0=np.ones((len(pair),1)),M1=basic,M2=np.c_[basic,roi])
        results[side]=(pair,{'M0':np.ones((len(pair),1)),'M1':basic,'M2':np.c_[basic,roi]})
    pd.DataFrame(blockrows).to_csv(OUT/'registry/spatial_blocks.csv',index=False);pd.DataFrame(blockaudit).to_csv(OUT/'analysis/spatial_separation_audit.csv',index=False)
    dump(OUT/'data/feature_schema.json',{'M1':'soma pre/post,absdelta,Euclidean distance; tree descriptors pre/post,absdelta','M1_tree_columns':list(geom.columns),'M2':'M1 plus directional regional kernels,missing flags,loglength','region_columns':cols,'right_to_left_mirror_x_sum_um':reflect,'degrees_labels_contact_features':False,'spatial_block_axis_sign':'largest absolute loading positive','normalization':'each fitted model training pairs only','registration':'single reflection from whole-KC side medians, not optimal cell/arbor registration'})
    return results,cfg

def fit_model(x,y,train,val,seed):
    scale=StandardScaler().fit(x[train]);z=scale.transform(x)/np.sqrt(x.shape[1]);fits=[];tuning=[]
    assert set(y[train])==set(y[val])=={0,1}
    for gamma in [.5,2.]:
        rff=RBFSampler(n_components=128,gamma=gamma,random_state=seed).fit(z[train]);features=rff.transform(z)
        for C in [.1,1.]:
            model=LogisticRegression(C=C,max_iter=1000,random_state=seed).fit(features[train],y[train]);loss=log_loss(y[val],model.predict_proba(features[val])[:,1])
            tuning.append({'gamma':gamma,'C':C,'validation_loss':loss,'n_iter':int(model.n_iter_[0])})
            fits.append((loss,gamma,C,model,rff))
    loss,gamma,C,model,rff=min(fits,key=lambda t:t[0])
    state={'mean':scale.mean_,'scale':scale.scale_,'n_features':np.array(x.shape[1]),'rff_weights':rff.random_weights_,'rff_offset':rff.random_offset_,'coef':model.coef_.ravel(),'intercept':model.intercept_,'gamma':np.array(gamma),'C':np.array(C)}
    return state,tuning

def predict(state,x):
    z=(x-state['mean'])/state['scale']/np.sqrt(state['n_features'])
    features=np.sqrt(2/state['rff_weights'].shape[1])*np.cos(z@state['rff_weights']+state['rff_offset'])
    return expit(features@state['coef']+state['intercept'][0])

def models():
    if (OUT/'audit/MODELS_STATUS.json').exists():raise RuntimeError('Completed model run; no overwrite')
    tm=time.time()
    if BUNDLE is None:data,cfg=prepare_designs()
    else:
        cfg=json.loads((BUNDLE/'audit/protocol.json').read_text());data={}
        for side in ['left','right']:
            pair=pd.read_parquet(BUNDLE/f'data/{side}_pairs.parquet');designs=dict(np.load(BUNDLE/f'data/{side}_designs.npz'))
            assert len(pair)==({'left':148,'right':133}[side])*({'left':148,'right':133}[side]-1)
            assert pair.pre_id.map(lambda x:isinstance(x,str)).all() and not pair.duplicated(['pre_id','post_id']).any()
            assert all(len(x)==len(pair) and np.isfinite(x).all() for x in designs.values())
            data[side]=pair,designs
        for sub in ['audit','data','registry','analysis','models','validation','report','figures']:(OUT/sub).mkdir(parents=True,exist_ok=True)
        (OUT/'audit/protocol.json').write_bytes((BUNDLE/'audit/protocol.json').read_bytes())
        for name in ['data/left_pairs.parquet','data/right_pairs.parquet','data/left_designs.npz','data/right_designs.npz','data/feature_schema.json','registry/spatial_blocks.csv','registry/bilateral_complete_ap1.parquet','analysis/spatial_separation_audit.csv','audit/geometry_sources.csv','audit/mesh_audit.json','audit/GEOMETRY_STATUS.json']:
            shutil.copy2(BUNDLE/name,OUT/name)
        shutil.copy2(BUNDLE/'audit/provenance.json',OUT/'audit/imported_source_provenance.json')
        dump(OUT/'audit/bundle_inputs.json',{'scope':'Prediction replay only; original geometry extraction and source audit not rerun','inputs':[{'path':str(p),'sha256':sha(p)} for p in [BUNDLE/'audit/protocol.json',*[BUNDLE/f'data/{side}_{name}' for side in ['left','right'] for name in ['pairs.parquet','designs.npz']]]]})
    rows=[];tunes=[];preds=[];inventory=[]
    for source,(pair,designs) in data.items():
        y=pair.observed_edge.to_numpy();other='right' if source=='left' else 'left'
        for fold in [-1,0,1,2,3,4]:
            split=pair.pair_split.to_numpy() if fold==-1 else pair[f'spatial_fold_{fold}'].to_numpy()
            train=split==0;val=split==1;test=split==2
            evaluation='pair_grouped' if fold==-1 else 'spatial_blocked'
            for seed in cfg['seeds']:
                for name,x in designs.items():
                    state,tuning=fit_model(x,y,train,val,seed)
                    stem=f'{source}_{evaluation}_fold{fold}_seed{seed}_{name}'
                    path=OUT/'models'/f'{stem}.npz';np.savez_compressed(path,**state)
                    inventory.append({'model_id':stem,'path':str(path.relative_to(OUT)),'sha256':sha(path),'source_side':source,'evaluation':evaluation,'fold':fold,'seed':seed,'model':name})
                    tunes.extend({'model_id':stem,**t} for t in tuning)
                    evaluations=[(evaluation,source,pair.loc[test],x[test])]
                    if fold==-1:
                        op,od=data[other];evaluations.append(('cross_side',other,op,od[name]))
                    for ev,target,testpair,tx in evaluations:
                        yy=testpair.observed_edge.to_numpy();prob=predict(state,tx)
                        rows.append({'source_side':source,'target_side':target,'evaluation':ev,'fold':fold,'seed':seed,'model':name,'n_train':int(train.sum()),'n_validation':int(val.sum()),'n_test':len(yy),'test_prevalence':float(yy.mean()),'log_loss':log_loss(yy,prob),'AUROC':roc_auc_score(yy,prob),'nominal_parameters':129,'gamma':float(state['gamma']),'C':float(state['C']),'status':'RETROSPECTIVE_SAME_BRAIN_NO_FUNCTIONAL_VALIDATION'})
                        pp=testpair[['pre_id','post_id','observed_edge']].copy();pp['probability']=prob
                        for k,v in {'model_id':stem,'evaluation':ev,'source_side':source,'target_side':target,'fold':fold,'seed':seed,'model':name}.items():pp[k]=v
                        preds.append(pp)
            print('completed',source,evaluation,fold,'elapsed',round(time.time()-tm),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'analysis/model_comparison.csv',index=False);pd.DataFrame(tunes).to_csv(OUT/'analysis/tuning.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'models/manifest.csv',index=False);pd.concat(preds,ignore_index=True).to_parquet(OUT/'data/test_predictions.parquet',index=False)
    dump(OUT/'audit/MODELS_STATUS.json',{'status':'DONE','n_fitted_models':len(inventory),'tuning_each':4,'n_evaluation_rows':len(rows),'seeds':cfg['seeds'],'scope':'same FlyWire brain, left/right complete fine-ap1','elapsed_s':time.time()-tm})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['geometry','models']);parser.add_argument('--bundle-root',type=Path);parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.output_dir.resolve();BUNDLE=args.bundle_root.resolve() if args.bundle_root else None
    if BUNDLE is None:parser.error('Public replay requires --bundle-root; raw geometry requires the original BioFly source snapshot and raw inputs')
    if BUNDLE is not None and args.stage!='models':parser.error('compact bundle supports models only; real geometry requires source repository')
    with threadpool_limits(limits=4):globals()[args.stage]()
