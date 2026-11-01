"""Source-only cable geometry and retrospective blocked prediction of PN/MBON profiles.
No new grouping, no dynamics and no claim of complete potential-contact ground truth.
"""
from pathlib import Path
import argparse,json,struct,time,importlib.metadata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from . import audit as a
from .hierarchy_expansion import load_data,GRAPH,utc,write_once
from .spatial_units import MESHROOT,COORDS
from .biology_grounding import OUT as BIO
OUT=a.ROOT/'artifacts/wu_20261027_calyx_geometry'
VERSION='wu_20261027_calyx_geometry_v1'
MANIFEST=a.BASE/'results_v9/skeleton_source_manifest.csv'
FOCUS=['G30_007_S_D001','G30_009_S_D001','G30_011_S_D001','G30_016_S_D002']

def freeze():
    for sub in ['audit','registry','reference','analysis','figures','report','validation','reference/cells']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    p=dict(protocol_version=VERSION,created_utc=utc(),scope='All5177 geometry; complete right/left official KCab parents799/844 for nested prediction. Four previously identified balance-passing leaves retained as fixed focus; every other KCab leaf also reported. No new labels or promotions.',
    interpretation='Retrospective same-connectome stress test after prior outcomes were seen; this is not fresh independent biological validation. Pure skeleton features do not contain PN/MBON identity. Decimated official display mesh is an approximate region, not an experimentally precise biological boundary. Actual contacts are observed outcomes, not all possible appositions.',
    geometry='Original skeleton nm/1000 ->um. Each edge split into ceil(length/2um) equal segments; weighted midpoint samples, weights sum to original cable length. Side-specific CA GLB likewise nm/1000. Use union of its CLOSED face-connected components. Right display mesh has one nonmanifold edge with4 incident faces, resolving into existing646-face and4-face closed components; retain both, do not add/delete faces. Fail closed if any component open. Within-CA cable occupancy at fixed global10um cubic grid; normalization by CA cable length; sqrt probabilities. Zero CA cable stays missing/flagged, not no function. Export5/10/20um spatial occupancy summaries as anatomical sensitivity, primary modeling10um only.',
    matching_audit='Keep previous spatial/degree matched IDs fixed, no rematching; compare source-only CA occupancy and CA cable burden to those references for all feasible23 units. Continuous differences only; no new balance/p-value or stability threshold. Missing CA cells explicit.',
    models='B0=XYZ cubic splines5knots + log original KCtrain in/out strength + log full cable length/branch/end counts + existing MBp onehot. B1=B0+train-fold16PCs of sqrt CA cable occupancy + CA loglength,weighted centroid/spread,missing indicator. B2=B1+frozen41 leaf membership indicators. No MBON/ALPN degree predictors. Original labels are transductive, labels unseen in training flagged rather than termed biological failure.',
    families='Linear ridge and nonlinear Nystroem RBF64 components on standardized continuous features; gamma[.1,1,10]/continuous_feature_count. Add lineage and frozen-label indicators outside nonlinear map. Ridge alpha[.1,1,10]. All transforms/imputation/scaling/PCA/Nystroem fitted in training fold only. Inner3fold selects hyperparameters jointly on ALPN+MBON Hellinger loss; both unit-normalized profile blocks contribute equally. Both model families and all outer splits retained.',
    cv='Outer random5fold and5contiguous rank blocks separately forX,Y,Z within each full parent. Inner shuffled3fold. Seeds outer2026102704,inner2026102703,PCA/Nystroem2026102702. Zero-contact profile invalid, not zero activity; model population requires both endpoint profiles valid and reports excluded cells. Test folds never enter transforms/tuning. Static input graph and endpoints previously inspected: SUPPORTING only.',
    outputs='Out-of-fold predicted profiles, fold loss and train-mean reference loss, R2relative to train-mean OOF reference, geometry incrementB0-B1 and leaf incrementB1-B2, all per-cell losses, full fold ranges, focus leaf members present/unseen training count. Intervals not animal CI; no new p/q. Label increment with missing training support is INCONCLUSIVE, not no subtype.',
    resource='Start10 real skeleton smoke then full5177 if extrapolated<1200CPU seconds and<2GiB; per-cell checkpoint. Initial one outer randomfold all3blocks x2families pilot then complete8parent/split jobs if extrapolated<2400CPU seconds; checkpoint eachfold. No new simulation.',
    limitations='Does not measure membrane-to-membrane potential contacts, PN full arbors or molecular affinity; no inference of all calyx cable as dendrite/input. No dynamic fitting. No new connectome or animals.')
    pp=OUT/'audit/protocol.json'
    if pp.exists():p['created_utc']=json.loads(pp.read_text())['created_utc']
    write_once(pp,p)
    files=[Path(__file__),pp,MANIFEST,a.OUT/'registry/kc_all.parquet',GRAPH/'data/kc_graph_train.npz',GRAPH/'node_order.json',BIO/'registry/all5177_lineage_atlas.parquet',BIO/'registry/ALPN_partner_contacts.parquet',BIO/'registry/MBON_partner_contacts.parquet',BIO/'reference/frozen_audit_units.json',BIO/'reference/spatial_matched_pairs.parquet']+[MESHROOT/f'MB_CA_{s}.glb' for s in ['L','R']]
    manifest=dict(sources=[dict(path=a.rel(f),sha256=a.sha(f)) for f in files],code_commit=a.commit(),dataset='FlyWire v783; annotation hash-specific',packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','trimesh','rtree']})
    correction=OUT/'audit/implementation_correction.json'
    if correction.exists():
        c=json.loads(correction.read_text());assert a.sha(__file__)==c['executed_sha256'];manifest['sources'][0]['sha256']=c['initial_sha256']
    write_once(OUT/'audit/input_manifest.json',manifest)

def samples(v,e,step=2.):
    seg=v[e];delta=seg[:,1]-seg[:,0];length=np.linalg.norm(delta,axis=1);good=length>0;seg=seg[good];delta=delta[good];length=length[good];counts=np.ceil(length/step).astype(int);row=np.repeat(np.arange(len(counts)),counts);local=np.arange(counts.sum())-np.repeat(np.r_[0,np.cumsum(counts)[:-1]],counts);t=(local+.5)/counts[row];return seg[row,0]+delta[row]*t[:,None],length[row]/counts[row]

def meshes():
    import trimesh
    out={};audit=[]
    for side,label in [('left','L'),('right','R')]:
        m=trimesh.load(MESHROOT/f'MB_CA_{label}.glb').to_mesh();m.apply_scale(.001);parts=m.split(only_watertight=False);assert all(x.is_watertight for x in parts),'BLOCKED_MISSING_VALID_CLOSED_CA_SURFACE';assert sum(len(x.faces) for x in parts)==len(m.faces)
        out[side]=(m,parts);audit.append(dict(side=side,raw_watertight=bool(m.is_watertight),n_raw_faces=len(m.faces),component_faces=[len(x.faces) for x in parts],component_volume_um3=[float(x.volume) for x in parts],new_faces=0,deleted_faces=0,units='um',backend=type(parts[0].ray).__module__))
    (OUT/'audit/closed_component_mesh_audit.json').write_text(json.dumps(audit,indent=2));return out

def contains_union(parts,points):
    inside=np.zeros(len(points),bool)
    for part in parts:
        ix=np.flatnonzero(((points>=part.bounds[0])&(points<=part.bounds[1])).all(1)&~inside)
        for start in range(0,len(ix),2000):q=ix[start:start+2000];inside[q]=part.contains(points[q])
    return inside

def geometry(limit=None):
    freeze();ids,at,_,_=load_data();mm=meshes();sm=pd.read_csv(MANIFEST,dtype={'root_id':str}).set_index('root_id');start=time.perf_counter();records=[];done=0
    for root in ids[:limit]:
        dest=OUT/'reference/cells'/f'{root}.npz'
        if dest.exists():continue
        row=sm.loc[root];path=Path(row.path);assert a.sha(path)==row.sha256
        blob=path.read_bytes();nv,ne=struct.unpack_from('<II',blob);v=np.frombuffer(blob,'<f4',nv*3,8).reshape(-1,3).astype(float)/1000;e=np.frombuffer(blob,'<u4',ne*2,8+12*nv).reshape(-1,2);p,w=samples(v,e);m,parts=mm[at.loc[root,'official_hemisphere']];inside=contains_union(parts,p);q=p[inside];ww=w[inside];s=ww.sum();degree=np.bincount(e.ravel(),minlength=nv);centroid=np.average(q,axis=0,weights=ww) if s>0 else np.full(3,np.nan);spread=np.sqrt(np.average((q-centroid)**2,axis=0,weights=ww)) if s>0 else np.full(3,np.nan)
        keys=np.floor(q/10).astype(np.int16);ukey,inv=np.unique(keys,axis=0,return_inverse=True);cable=np.bincount(inv,weights=ww,minlength=len(ukey));summary=np.r_[len(v),len(e),w.sum(),s,s/w.sum() if w.sum() else np.nan,int((degree>2).sum()),int((degree==1).sum()),centroid,spread,len(q),len(p)]
        np.savez_compressed(dest,voxel_ijk=ukey,cable_um=cable,summary=summary,skeleton_sha256=np.array(row.sha256),root_id=np.array(root),ca_points_um=q.astype(np.float32),sample_weights_um=ww.astype(np.float32));done+=1
        if done%250==0:print('geometry',done,'seconds',round(time.perf_counter()-start),flush=True)
    elapsed=time.perf_counter()-start
    if limit is not None:
        (OUT/'validation/geometry_smoke.json').write_text(json.dumps(dict(n=done,elapsed_seconds=elapsed,extrapolated_full_seconds=elapsed/max(done,1)*5177),indent=2));print('geometry smoke',done,elapsed,flush=True);return
    allkeys=[];cols=['n_nodes','n_edges','full_cable_um','ca_cable_um','ca_fraction','branch_nodes','terminal_nodes','ca_x_um','ca_y_um','ca_z_um','ca_sd_x_um','ca_sd_y_um','ca_sd_z_um','n_ca_samples','n_full_samples'];sensitivity=[]
    for root in ids:
        z=np.load(OUT/'reference/cells'/f'{root}.npz');records.append(dict(root_id=root,**dict(zip(cols,z['summary'])),geometry_status='CA_CABLE_PRESENT' if z['summary'][3]>0 else 'NO_CABLE_IN_DISPLAY_CA_NOT_NO_FUNCTION',skeleton_sha256=str(z['skeleton_sha256'])));allkeys.extend(map(tuple,z['voxel_ijk']))
        for grid in [5,10,20]:
            key,inv=np.unique(np.floor(z['ca_points_um']/grid).astype(int),axis=0,return_inverse=True);val=np.bincount(inv,weights=z['sample_weights_um'],minlength=len(key));p=val/val.sum() if val.sum()>0 else val;sensitivity.append(dict(root_id=root,grid_um=grid,n_occupied=len(key),effective_voxels=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))) if len(p) else np.nan))
    keys=sorted(set(allkeys));index={v:i for i,v in enumerate(keys)};rr=[];cc=[];vv=[]
    for i,root in enumerate(ids):
        z=np.load(OUT/'reference/cells'/f'{root}.npz');rr.extend([i]*len(z['voxel_ijk']));cc.extend(index[tuple(v)] for v in z['voxel_ijk']);vv.extend(z['cable_um'])
    mat=sparse.csr_matrix((vv,(rr,cc)),shape=(len(ids),len(keys)));sparse.save_npz(OUT/'registry/calyx_cable_10um.npz',mat);pd.DataFrame(keys,columns=['i','j','k']).to_csv(OUT/'registry/voxel_index.csv',index=False);pd.DataFrame(records).to_parquet(OUT/'registry/calyx_geometry.parquet',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'analysis/voxel_resolution_sensitivity.csv',index=False)
    (OUT/'validation/geometry_runtime.json').write_text(json.dumps(dict(n=5177,elapsed_seconds=elapsed,resume_included=True),indent=2));print('geometry complete',len(keys),'voxels',elapsed,flush=True)

def project(y,ends,mean):
    out=np.zeros_like(y)
    for lo,hi in zip([0]+ends[:-1],ends):
        v=np.maximum(y[:,lo:hi],0);den=np.linalg.norm(v,axis=1);bad=den<=1e-12
        if bad.any():v[bad]=mean[lo:hi];den=np.linalg.norm(v,axis=1)
        out[:,lo:hi]=v/np.maximum(den[:,None],1e-12)
    return out

def losses(y,p,ends):return np.column_stack([.5*np.sum((y[:,lo:hi]-p[:,lo:hi])**2,axis=1) for lo,hi in zip([0]+ends[:-1],ends)])

def transform_fit(base,hist,cal,lin,labels,train,test,block,family,gamma):
    # Entire fit is on supplied training indices. Hist vocabulary has fixed physical meaning.
    def impute_scale(x):
        tr=x[train].copy();te=x[test].copy();med=np.nanmedian(tr,axis=0);med=np.nan_to_num(med);tr=np.where(np.isfinite(tr),tr,med);te=np.where(np.isfinite(te),te,med);return tr,te
    bt,be=impute_scale(base);spl=SplineTransformer(n_knots=5,degree=3,include_bias=False);xt=np.c_[spl.fit_transform(bt[:,:3]),bt[:,3:]];xe=np.c_[spl.transform(be[:,:3]),be[:,3:]]
    if block!='B0':
        keep=np.asarray(hist[train].sum(0)).ravel()>0;ht=hist[train][:,keep].toarray();he=hist[test][:,keep].toarray();n=min(16,len(train)-1,ht.shape[1]);pc=PCA(n_components=n,svd_solver='randomized',random_state=2026102702);ct,ce=impute_scale(cal);xt=np.c_[xt,ct,pc.fit_transform(ht)];xe=np.c_[xe,ce,pc.transform(he)]
    scaler=StandardScaler();xt=scaler.fit_transform(xt);xe=scaler.transform(xe)
    if family=='RBF':
        model=Nystroem(kernel='rbf',gamma=gamma/xt.shape[1],n_components=min(64,len(train)),random_state=2026102702,n_jobs=1);xt=model.fit_transform(xt);xe=model.transform(xe)
    xt=np.c_[xt,lin[train]];xe=np.c_[xe,lin[test]]
    if block=='B2':xt=np.c_[xt,labels[train]];xe=np.c_[xe,labels[test]]
    scale=StandardScaler();return scale.fit_transform(xt),scale.transform(xe)

def normalized_cable_hist(h):
    den=np.asarray(h.sum(1)).ravel();inv=np.divide(1.,den,out=np.zeros_like(den,dtype=float),where=den>0);h=(sparse.diags(inv)@h).tocsr();h.data=np.sqrt(h.data);return h

def prepare_data():
    ids,at,_,_=load_data();geo=pd.read_parquet(OUT/'registry/calyx_geometry.parquet').set_index('root_id').loc[ids];g=sparse.load_npz(GRAPH/'data/kc_graph_train.npz');base=np.c_[at[COORDS].to_numpy(),np.log1p(np.asarray(g.sum(0)).ravel()),np.log1p(np.asarray(g.sum(1)).ravel()),np.log1p(geo[['full_cable_um','branch_nodes','terminal_nodes']].to_numpy())];cal=np.c_[np.log1p(geo.ca_cable_um),geo[['ca_x_um','ca_y_um','ca_z_um','ca_sd_x_um','ca_sd_y_um','ca_sd_z_um']],geo.ca_cable_um.eq(0).astype(float)];h=sparse.load_npz(OUT/'registry/calyx_cable_10um.npz');den=np.asarray(h.sum(1)).ravel();h=normalized_cable_hist(h);lin=pd.read_parquet(BIO/'registry/all5177_lineage_atlas.parquet').set_index('root_id').loc[ids].lineage_annotation.to_numpy();lin=(lin[:,None]==np.array(['MBp1','MBp2','MBp3','MBp4'])[None,:]).astype(float)
    allunits=json.loads((BIO/'reference/frozen_audit_units.json').read_text())['units'];units=[u for u in allunits if u['unit_id'].startswith('G30_')];lab=np.array([np.isin(ids,u['member_ids']) for u in units],dtype=float).T;blocks=[];valid=np.ones(len(ids),bool);ends=[]
    for name in ['ALPN','MBON']:
        q=pd.read_parquet(BIO/f'registry/{name}_partner_contacts.parquet').loc[ids].to_numpy(float);s=q.sum(1);valid&=s>0;blocks.append(np.sqrt(q/np.maximum(s[:,None],1)));ends.append(sum(x.shape[1] for x in blocks))
    y=np.c_[*blocks];pd.DataFrame({'root_id':ids,'type':at.official_KC_subtype.to_numpy(),'side':at.official_hemisphere.to_numpy(),'both_profiles_valid':valid,'ca_cable_present':den>0}).to_csv(OUT/'registry/model_eligibility.csv',index=False)
    return ids,at,base,h.tocsr(),cal,lin,lab,y,valid,ends,units

def match_audit():
    ids,at,b,h,c,l,lab,y,valid,ends,units=prepare_data();look=pd.Index(ids);pairs=pd.read_parquet(BIO/'reference/spatial_matched_pairs.parquet');rows=[]
    for name,q in pairs.groupby('unit_id'):
        ix=look.get_indexer(q.target_root_id);jx=look.get_indexer(q.reference_root_id);vt=c[ix,0]>0;vr=c[jx,0]>0;mx=np.asarray(h[ix[vt]].mean(0)).ravel() if vt.any() else None;mr=np.asarray(h[jx[vr]].mean(0)).ravel() if vr.any() else None
        rows.append(dict(unit_id=name,n=len(q),candidate_CA_present=int(vt.sum()),reference_CA_present=int(vr.sum()),mean_profile_L2_difference=float(np.linalg.norm(mx-mr)) if mx is not None and mr is not None else np.nan,candidate_mean_log_CA_cable=float(c[ix,0].mean()),reference_mean_log_CA_cable=float(c[jx,0].mean()),geometry_scope='fixed prior soma-degree matching; no rematch or threshold; missing excluded from mean geometry only'))
    pd.DataFrame(rows).to_csv(OUT/'analysis/fixed_match_calyx_balance.csv',index=False)

def fit(pilot=False):
    freeze();ids,at,base,h,cal,lin,lab,y,valid,ends,units=prepare_data();records=[];t0=time.perf_counter();index=[]
    with threadpool_limits(limits=2):
      for side in ['right','left']:
        parent=np.flatnonzero(at.official_KC_subtype.eq('KCab')&at.official_hemisphere.eq(side));eligible=parent[valid[parent]];assert len(parent)==({'right':799,'left':844}[side]);splits={'random':list(KFold(5,shuffle=True,random_state=2026102704).split(eligible))}
        for ax in range(3):
            order=np.argsort(base[eligible,ax],kind='stable');splits['spatial_'+'XYZ'[ax]]=[(np.setdiff1d(np.arange(len(eligible)),te),te) for te in np.array_split(order,5)]
        for scheme,folds in splits.items():
          for fold,(tr,te) in enumerate(folds):
            if pilot and (side!='right' or scheme!='random' or fold!=0):continue
            train=eligible[tr];test=eligible[te];index.extend(dict(parent_side=side,cv=scheme,fold=fold,root_id=ids[i],role='test') for i in test);foldname=f'{side}_{scheme}_{fold}';path=OUT/'reference'/f'pred_{foldname}.npz'
            if path.exists():continue
            pred={};pars=[];mean=y[train].mean(0);ref=project(np.repeat(mean[None,:],len(test),axis=0),ends,mean)
            for block in ['B0','B1','B2']:
              for family in ['LINEAR','RBF']:
                gammas=[0.] if family=='LINEAR' else [.1,1.,10.];scores=[]
                for gamma in gammas:
                    innerloss=np.zeros(3)
                    for itr,ite in KFold(3,shuffle=True,random_state=2026102703).split(train):
                        ti=train[itr];vi=train[ite];xt,xv=transform_fit(base,h,cal,lin,lab,ti,vi,block,family,gamma)
                        for ai,alpha in enumerate([.1,1.,10.]):
                            model=Ridge(alpha=alpha).fit(xt,y[ti]);pp=project(model.predict(xv),ends,y[ti].mean(0));innerloss[ai]+=losses(y[vi],pp,ends).mean()/3
                    scores.extend((v,gamma,alpha) for v,alpha in zip(innerloss,[.1,1.,10.]))
                score,gamma,alpha=min(scores);xt,xv=transform_fit(base,h,cal,lin,lab,train,test,block,family,gamma);model=Ridge(alpha=alpha).fit(xt,y[train]);pred[block+'_'+family]=project(model.predict(xv),ends,mean).astype(np.float32);pars.append(dict(block=block,family=family,alpha=alpha,gamma=gamma,inner_loss=score))
            np.savez_compressed(path,test_indices=test,train_indices=train,y=y[test].astype(np.float32),reference=ref.astype(np.float32),**pred)
            (path.with_suffix('.json')).write_text(json.dumps(dict(parent_side=side,cv=scheme,fold=fold,parameters=pars,elapsed_total_seconds=time.perf_counter()-t0,seed=2026102704,protocol=VERSION,code_commit=a.commit(),sha256=a.sha(path)),indent=2));print(foldname,'done',round(time.perf_counter()-t0),flush=True)
    (OUT/'validation'/('model_pilot.json' if pilot else 'model_runtime.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-t0,pilot=pilot,new_simulations=0),indent=2))
    if not pilot:pd.DataFrame(index).to_parquet(OUT/'reference/cv_test_members.parquet',index=False);summarize()

def summarize():
    ids,at,base,h,cal,lin,lab,y,valid,ends,units=prepare_data();foldrows=[];cellrows=[];unitrows=[]
    for path in sorted((OUT/'reference').glob('pred_*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());z=np.load(path);test=z['test_indices'];train=z['train_indices'];ref=losses(z['y'],z['reference'],ends)
        for family in ['LINEAR','RBF']:
            ll={block:losses(z['y'],z[block+'_'+family],ends) for block in ['B0','B1','B2']}
            for j,end in enumerate(['ALPN','MBON']):
                for block in ll:foldrows.append(dict(side=meta['parent_side'],cv=meta['cv'],fold=meta['fold'],family=family,endpoint=end,block=block,n_test=len(test),loss=float(ll[block][:,j].mean()),reference_loss=float(ref[:,j].mean()),R2_train_mean_reference=float(1-ll[block][:,j].sum()/ref[:,j].sum())))
                for k,i in enumerate(test):cellrows.append(dict(root_id=ids[i],side=meta['parent_side'],cv=meta['cv'],fold=meta['fold'],family=family,endpoint=end,loss_B0=ll['B0'][k,j],loss_B1=ll['B1'][k,j],loss_B2=ll['B2'][k,j],reference_loss=ref[k,j]))
                for col,u in enumerate(units):
                    sel=lab[test,col]>0
                    if not sel.any():continue
                    ntrain=int(lab[train,col].sum());unitrows.append(dict(unit_id=u['unit_id'],focus=u['unit_id'] in FOCUS,side=meta['parent_side'],cv=meta['cv'],fold=meta['fold'],family=family,endpoint=end,n_test=int(sel.sum()),n_train_label=ntrain,mean_geometry_loss_gain=float((ll['B0'][sel,j]-ll['B1'][sel,j]).mean()),mean_label_loss_gain=float((ll['B1'][sel,j]-ll['B2'][sel,j]).mean()),status='LABEL_UNSEEN_IN_TRAINING_INCONCLUSIVE' if ntrain==0 else 'RETROSPECTIVE_SUPPORTING'))
    pd.DataFrame(foldrows).to_csv(OUT/'analysis/fold_results.csv',index=False);cells=pd.DataFrame(cellrows);cells.to_parquet(OUT/'analysis/oof_cell_losses.parquet',index=False);pd.DataFrame(unitrows).to_csv(OUT/'analysis/leaf_fold_increment.csv',index=False)
    rows=[]
    for key,q in cells.groupby(['side','cv','family','endpoint']):
        den=q.reference_loss.sum();rows.append(dict(zip(['side','cv','family','endpoint'],key))|dict(n_cells=len(q),R2_B0=1-q.loss_B0.sum()/den,R2_B1=1-q.loss_B1.sum()/den,R2_B2=1-q.loss_B2.sum()/den,geometry_delta_R2=(q.loss_B0-q.loss_B1).sum()/den,label_delta_R2=(q.loss_B1-q.loss_B2).sum()/den,geometry_mean_loss_gain=(q.loss_B0-q.loss_B1).mean(),label_mean_loss_gain=(q.loss_B1-q.loss_B2).mean()))
    pd.DataFrame(rows).to_csv(OUT/'analysis/model_comparison.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','smoke','geometry','match_audit','pilot','fit','summarize']);cmd=p.parse_args().command
    if cmd=='smoke':geometry(10)
    elif cmd=='pilot':fit(True)
    else:globals()[cmd]()
