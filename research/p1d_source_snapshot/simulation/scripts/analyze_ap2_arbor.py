"""Frozen complete-parent ap2 geometry: real skeletons, no contact input."""
from pathlib import Path
import argparse, hashlib, json, subprocess, time, struct
import numpy as np
import pandas as pd
from bio_fly.wu_20261019.arbor_opportunity import subdivide, SegmentIndex, cable_kernels, projection_distances
from bio_fly.wu_20261019.calyx_geometry import contains_union

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wu_20261029_ap2_arbor_opportunity'
ATLAS=ROOT/'artifacts/wu_20261029_structure_entries'
MESH=ROOT/'nature_methods_connectome/two_week_research_presentation_20260917/visual_v2/assets/meshes/neuropils'
REGIONS=['FULL','CA','PED','ML','VL']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def population():
    m=pd.read_parquet(ATLAS/'registry/all5177_cells.parquet')
    m=m[m.official_KC_subtype.eq('KCapbp-ap2')].sort_index()
    assert m.groupby('official_hemisphere').size().to_dict()=={'left':138,'right':160}
    assert all(isinstance(x,str) for x in m.index)
    return m


def meshes():
    import trimesh
    result={};rows=[]
    for side,s in [('left','L'),('right','R')]:
        for region in REGIONS[1:]:
            path=MESH/f'MB_{region}_{s}.glb'
            m=trimesh.load(path).to_mesh();m.apply_scale(.001)
            parts=m.split(only_watertight=False)
            assert len(parts) and all(p.is_watertight for p in parts)
            assert sum(len(p.faces) for p in parts)==len(m.faces)
            result[side,region]=parts
            rows.append(dict(side=side,region=region,path=str(path.relative_to(ROOT)),sha256=sha(path),
                raw_watertight=bool(m.is_watertight),component_faces=[len(p.faces) for p in parts],
                component_volume_um3=[float(p.volume) for p in parts],faces_added=0,faces_removed=0,scale_to_um=.001))
    (OUT/'audit/mesh_audit.json').write_text(json.dumps(rows,indent=2))
    return result


def cell_geometry(root,side,mm,step=1.):
    path=ATLAS/'geometry'/f'{root}.json';g=json.loads(path.read_text())
    assert g['id']==root
    c=subdivide(g['vertices'],g['edges'],g['branch_per_edge'],step)
    c['region_mask']=np.column_stack([np.ones(len(c['weight']),bool)]+
        [contains_union(mm[side,r],c['midpoint']) for r in REGIONS[1:]])
    return c


def indices(c):
    return [SegmentIndex(c['a'][c['region_mask'][:,r]],c['b'][c['region_mask'][:,r]]) for r in range(5)]


def directional(source, targets):
    kernels=[];distances=[]
    for r,t in enumerate(targets):
        keep=source['region_mask'][:,r]
        k,d=cable_kernels(source['midpoint'][keep],source['weight'][keep],t)
        kernels.extend(k);distances.append(d)
    return np.asarray(kernels),np.asarray(distances)


def prepare(smoke=False):
    start=time.time()
    for d in ['audit','data','analysis','figures','report','validation','models','cache/cells','cache/targets']:(OUT/d).mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/'audit/protocol.json').read_text())
    m=population();mm=meshes();sources=pd.read_csv(ATLAS/'audit/skeleton_sources.csv',dtype={'root_id':str}).set_index('root_id')
    selected=m.groupby('official_hemisphere',group_keys=False).head(2) if smoke else m
    provenance=[];rows=[]
    for root,row in selected.iterrows():
        src=sources.loc[root];path=ATLAS/src.geometry_path;raw=ROOT/src.raw_path
        # Historical field hashes the original viewer JSON, not its reformatted atlas copy.
        viewer=raw.parents[2]/'viewer/data'/f'{root}_skeleton.json'
        assert sha(viewer)==src.geometry_source_sha256 and sha(raw)==src.raw_sha256
        gg=json.loads(path.read_text());assert gg==json.loads(viewer.read_text())
        blob=raw.read_bytes();nv,ne=struct.unpack_from('<II',blob)
        vv=np.frombuffer(blob,'<f4',nv*3,8).reshape(-1,3).astype(float)/1000
        ee=np.frombuffer(blob,'<u4',ne*2,8+nv*12).reshape(-1,2)
        assert np.max(np.abs(vv-np.asarray(gg['vertices'])))<=1e-6
        assert np.array_equal(ee,np.asarray(gg['edges']))
        dest=OUT/'cache/cells'/f'{root}.npz'
        if dest.exists():
            c=dict(np.load(dest));assert str(c['source_sha256'])==sha(path)
        else:
            c=cell_geometry(root,row.official_hemisphere,mm)
            np.savez_compressed(dest,**c,root_id=np.array(root),source_sha256=np.array(sha(path)))
        cable=float(c['weight'].sum())
        gv=np.asarray(gg['vertices']);ge=np.asarray(gg['edges'])
        json_length=np.linalg.norm(gv[ge[:,1]]-gv[ge[:,0]],axis=1).sum()
        raw_length=np.linalg.norm(vv[ee[:,1]]-vv[ee[:,0]],axis=1).sum()
        assert np.isclose(cable,json_length,rtol=1e-12,atol=1e-10)
        assert np.isclose(raw_length,row.cable_length_um,rtol=1e-10,atol=1e-8)
        rounding_bound=2*ne*np.linalg.norm(vv-gv,axis=1).max()
        assert abs(cable-raw_length)<=rounding_bound+1e-9
        count=c['region_mask'][:,1:].sum(axis=1)
        rows.append(dict(root_id=root,side=row.official_hemisphere,subsegments=len(c['weight']),full_cable_um=cable,raw_cable_um=raw_length,json_rounding_delta_um=cable-raw_length,json_rounding_bound_um=rounding_bound,
            overlapping_ROI_cable_um=float(c['weight'][count>1].sum()),uncovered_cable_um=float(c['weight'][count==0].sum()),
            **{f'{r}_cable_um':float(c['weight'][c['region_mask'][:,k]].sum()) for k,r in enumerate(REGIONS)}))
        provenance.append(dict(root_id=root,geometry_path=str(path.relative_to(ROOT)),geometry_sha256=sha(path),original_viewer_path=str(viewer.relative_to(ROOT)),original_viewer_sha256=sha(viewer),raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),cache_sha256=sha(dest)))
    prefix='smoke_' if smoke else ''
    pd.DataFrame(rows).to_csv(OUT/f'data/{prefix}cell_geometry.csv',index=False)
    pd.DataFrame(provenance).to_csv(OUT/f'audit/{prefix}skeleton_sources.csv',index=False)
    if smoke:
        ss=[]
        for side,g in selected.groupby('official_hemisphere'):
            roots=g.index.tolist();a=dict(np.load(OUT/'cache/cells'/f'{roots[0]}.npz'));b=dict(np.load(OUT/'cache/cells'/f'{roots[1]}.npz'))
            k,d=directional(a,indices(b));ss.append(dict(side=side,source=roots[0],target=roots[1],kernels=k.tolist(),mean_distance_um=d.tolist()))
        (OUT/'validation/geometry_smoke.json').write_text(json.dumps(dict(n_cells=4,seconds=time.time()-start,pairs=ss),indent=2))
    else:
        manifest=dict(protocol_version=protocol['protocol_version'],dataset_version='FlyWire v783',run_id=protocol['protocol_version'],seed=protocol['seed'],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            inputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [OUT/'audit/protocol.json',ATLAS/'registry/all5177_cells.parquet',ATLAS/'audit/skeleton_sources.csv']],
            executed_code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__),ROOT/'simulation/src/bio_fly/wu_20261019/arbor_opportunity.py',ROOT/'simulation/src/bio_fly/wu_20261019/calyx_geometry.py']],
            geometry_manifest='skeleton_sources.csv',meshes='mesh_audit.json',contact_data_used_for_geometry=False,n_cells=len(m),seconds=time.time()-start)
        (OUT/'audit/geometry_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(prefix+'geometry',len(selected),'seconds',time.time()-start,flush=True)


def numerics():
    start=time.time();m=population();mm=meshes();rows=[];checks=[]
    for side,g in m.groupby('official_hemisphere'):
        roots=g.index[:4].tolist()
        coarse={r:dict(np.load(OUT/'cache/cells'/f'{r}.npz')) for r in roots}
        fine={r:cell_geometry(r,side,mm,step=.5) for r in roots}
        for target in roots:
            t1=indices(coarse[target]);t2=indices(fine[target])
            for source in roots:
                if source==target:continue
                k1,d1=directional(coarse[source],t1);k2,d2=directional(fine[source],t2)
                for ri,region in enumerate(REGIONS):
                    a=coarse[source];p=a['midpoint'][a['region_mask'][:,ri]][:20]
                    target_index=t1[ri]
                    if len(p) and target_index.n:
                        brute=projection_distances(p[:,None,:],target_index.a[None,:,:],target_index.b[None,:,:]).min(axis=1)
                        err=float(np.max(np.abs(brute-target_index.distances(p))))
                        checks.append(dict(side=side,source_id=source,target_id=target,region=region,n_samples=len(p),max_distance_error_um=err))
                    for si,scale in enumerate([1,2,5]):
                        ix=ri*3+si
                        rows.append(dict(side=side,source_id=source,target_id=target,region=region,scale_um=scale,kernel_1um=k1[ix],kernel_half_um=k2[ix],absolute_difference=abs(k1[ix]-k2[ix]),missingness_agrees=bool(np.isnan(k1[ix])==np.isnan(k2[ix]))))
    df=pd.DataFrame(rows);bf=pd.DataFrame(checks)
    df.to_csv(OUT/'validation/quadrature_convergence.csv',index=False);bf.to_csv(OUT/'validation/exact_distance_bruteforce.csv',index=False)
    status=dict(n_cells=8,n_directed_pairs=24,n_kernel_comparisons=len(df),max_kernel_difference=float(df.absolute_difference.max()),max_exact_distance_error_um=float(bf.max_distance_error_um.max()),missingness_agrees=bool(df.missingness_agrees.all()),seconds=time.time()-start)
    status['passed']=status['max_kernel_difference']<=.02 and status['max_exact_distance_error_um']<1e-9 and status['missingness_agrees']
    (OUT/'validation/numerical_checks.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    assert status['passed'],'Numerical audit failed; refine explicitly before interpretation'


def pairs():
    assert json.loads((OUT/'validation/numerical_checks.json').read_text())['passed']
    start=time.time();m=population();cols=[f'{r}_gaussian_{s}um' for r in REGIONS for s in [1,2,5]]
    records=[]
    for side,g in m.groupby('official_hemisphere'):
        roots=g.index.tolist();cells={r:dict(np.load(OUT/'cache/cells'/f'{r}.npz')) for r in roots}
        for ti,target in enumerate(roots):
            dest=OUT/'cache/targets'/f'{target}.parquet'
            if dest.exists():df=pd.read_parquet(dest)
            else:
                tt=indices(cells[target]);rows=[]
                for source in roots:
                    if source==target:continue
                    k,d=directional(cells[source],tt)
                    rows.append(dict(side=side,source_id=source,target_id=target,**dict(zip(cols,k)),**{f'{r}_mean_distance_um':v for r,v in zip(REGIONS,d)}))
                df=pd.DataFrame(rows);df.to_parquet(dest,index=False)
            assert set(df.source_id)==set(roots)-{target} and df.target_id.eq(target).all()
            records.append(df)
            if (ti+1)%20==0:print(side,'targets',ti+1,'/',len(roots),'seconds',round(time.time()-start),flush=True)
    df=pd.concat(records,ignore_index=True)
    assert len(df)==160*159+138*137 and not df.duplicated(['source_id','target_id']).any()
    df.to_parquet(OUT/'data/directional_arbor_features.parquet',index=False)
    (OUT/'validation/pair_geometry_checks.json').write_text(json.dumps(dict(n_directed_pairs=len(df),seconds=time.time()-start,contact_records_used=0,geometry_sha256=sha(OUT/'data/directional_arbor_features.parquet')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['smoke','prepare','numerics','pairs']);cmd=parser.parse_args().command
    if cmd=='smoke':prepare(True)
    elif cmd=='prepare':prepare()
    else:globals()[cmd]()
