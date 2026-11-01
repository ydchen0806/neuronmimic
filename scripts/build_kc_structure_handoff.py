"""Add full KC anatomy to the existing release; preserve original evidence states."""
from pathlib import Path
import argparse,hashlib,json,shutil,time,tarfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def strings(t,cols):
    for c in cols:
        i=t.schema.get_field_index(c)
        if i>=0:t=t.set_column(i,c,pc.cast(t.column(c),pa.string()))
    return t.replace_schema_metadata(None)
def reader(p):return ipc.open_file(pa.memory_map(str(p)))
def point_key(x,y,z):
    x,y,z=[np.asarray(v,dtype=np.uint64) for v in [x,y,z]]
    assert max(x.max(initial=0),y.max(initial=0),z.max(initial=0))<2**21
    return (z<<42)|(y<<21)|x

def tables(root,out):
    start=time.time();f=out/'flywire_v783';m=out/'malecns_v1_0';atlas=root/'artifacts/wu_20261030_all5177_structural_atlas'
    for sub in ['registry','data','analysis','audit','report','figures','validation']:
        shutil.copytree(atlas/sub,f/sub,dirs_exist_ok=True)
    # All original physical contact fields are retained under explicit ID renames.
    original=pq.read_schema(root/'nature_methods_connectome/fine_structure_mechanism/runs/20260929_flywire_atlas_completion_v2/results/all_KC_incident_synapses.parquet')
    renamed={'id':'synapse_id','pre_pt_root_id':'pre_id','post_pt_root_id':'post_id'}
    assert set(renamed.get(k,k) for k in original.names)==set(pq.read_schema(f/'data/synapse_pairs.parquet').names)
    reg=pd.read_parquet(f/'registry/all5177_cells.parquet').reset_index();assert 'root_id' in reg and reg.root_id.map(lambda v:isinstance(v,str)).all();reg.to_parquet(f/'registry/cells.parquet',index=False)
    roles=pd.read_parquet(f/'registry/entry_roles.parquet');members=pd.read_parquet(f/'registry/entry_members.parquet')
    groups=pd.read_csv(f/'registry/entries.csv',dtype=str);base=members.rename(columns={'entry_id':'group_id'})[['group_id','root_id','entry_kind','evidence_status']];base['namespace']='atlas_20261030'
    su=root/'artifacts/wu_20261030_fine_unit_entries/registry'
    for n in ['fine_unit_entries.csv','fine_unit_entry_roles.csv']:shutil.copy2(su/n,f/'registry'/n)
    suroles=pd.read_csv(su/'fine_unit_entry_roles.csv',dtype=str);surows=suroles[suroles.role.eq('candidate')].rename(columns={'unit_id':'group_id'})[['group_id','root_id']];surows['entry_kind']='SCALE_CONDITIONAL_STRUCTURAL_CANDIDATE';surows['evidence_status']='NOT_INDEPENDENTLY_FUNCTIONALLY_VALIDATED';surows['namespace']='fine_units_20261030'
    frozen=root/'artifacts/nature_kc_20261030/p1_v2_fine_types/registry/ap1L_18_exploratory_members.parquet';shutil.copy2(frozen,f/'registry/frozen_ap1L18.parquet');e=pd.read_parquet(frozen)[['root_id']].copy();e['group_id']='P1_AP1L18';e['entry_kind']='EXPLORATORY_HISTORICAL_OVERLAP';e['evidence_status']='NOT_PROMOTED';e['namespace']='p1_v2_fine_types'
    gm=pd.concat([base,surows,e],ignore_index=True);assert set(gm.root_id)<=set(reg.root_id);assert not gm.duplicated(['namespace','group_id','root_id']).any();gm.to_parquet(f/'registry/group_members.parquet',index=False)
    for sub in ['data','registry','audit','geometry']:(m/sub).mkdir(parents=True,exist_ok=True)
    raw=root/'simulation/data/external/male_cns_v1_0';ap=raw/'body-annotations-male-cns-v1.0-minconf-0.5.feather';ann=pd.read_feather(ap);kc=ann[ann.type.fillna('').str.startswith('KC')].copy();assert kc.bodyId.is_unique
    ids=pa.array(kc.bodyId.to_numpy(),type=pa.int64());kt=pa.Table.from_pandas(kc,preserve_index=False);kt=strings(kt,['bodyId']);pq.write_table(kt,m/'registry/cells.parquet',compression='zstd')
    full=strings(pa.Table.from_pandas(ann,preserve_index=False),['bodyId']);pq.write_table(full,m/'registry/all_source_annotations.parquet',compression='zstd')
    kg=pd.DataFrame({'root_id':kc.bodyId.astype(str),'group_id':'TYPE:'+kc.type+':'+kc.somaSide.fillna('unknown'),'namespace':'official_type_side_v1_0','entry_kind':'KNOWN_TYPE_SIDE_GROUP','evidence_status':'ANNOTATION_ONLY_NOT_NEW_FUNCTIONAL_UNIT'});kg.to_parquet(m/'registry/group_members.parquet',index=False)
    path=raw/'syn-partners-male-cns-v1.0-minconf-0.5.feather';r=reader(path);parts=[];offset=0
    for i in range(r.num_record_batches):
        b=r.get_batch(i);mask=pc.or_(pc.is_in(b.column('body_pre'),value_set=ids),pc.is_in(b.column('body_post'),value_set=ids));ix=pc.indices_nonzero(mask)
        if len(ix):
            t=pa.Table.from_batches([b]).take(ix).replace_schema_metadata(None);t=t.append_column('source_row',pa.array(ix.to_numpy()+offset,type=pa.int64()));parts.append(t)
        offset+=b.num_rows
        if i%1500==0:print('partners batches',i,'/',r.num_record_batches,flush=True)
    contacts=pa.concat_tables(parts);del parts
    rawrows=contacts['source_row'].to_numpy();contacts=contacts.append_column('synapse_id',pa.array(['male_v1_synpartners_row:'+str(x) for x in rawrows]));contacts=strings(contacts,['body_pre','body_post']);pq.write_table(contacts,m/'data/synapse_pairs.parquet',compression='zstd')
    c=contacts.to_pandas();directed=c.groupby(['body_pre','body_post']).size().reset_index(name='synapse_pairs');directed.to_parquet(m/'data/directed_pairs.parquet',index=False)
    keys=np.unique(point_key(c.x_pre.to_numpy(),c.y_pre.to_numpy(),c.z_pre.to_numpy()));keyset=pa.array(keys,type=pa.uint64());ntpath=raw/'tbar-neurotransmitters-male-cns-v1.0.feather';r=reader(ntpath);ntparts=[];checked=0
    for i in range(r.num_record_batches):
        b=r.get_batch(i);mask=pc.is_in(b.column('point_id'),value_set=keyset)
        if pc.any(mask).as_py():
            t=pa.Table.from_batches([b]).filter(mask);assert np.array_equal(point_key(t['x'].to_numpy(),t['y'].to_numpy(),t['z'].to_numpy()),t['point_id'].to_numpy());ntparts.append(t)
        checked+=b.num_rows
    nt=pa.concat_tables(ntparts);nt=strings(nt,['point_id','sv','body']);pq.write_table(nt,m/'data/presynaptic_site_nt.parquet',compression='zstd')
    # Verify the actual coordinate+body join, not only a packed-coordinate key.
    nd=nt.select(['point_id','x','y','z','body']).to_pandas();assert not nd.duplicated(['x','y','z','body']).any()
    joined=c[['x_pre','y_pre','z_pre','body_pre']].merge(nd,left_on=['x_pre','y_pre','z_pre','body_pre'],right_on=['x','y','z','body'],how='left',validate='many_to_one');missing=int(joined.point_id.isna().sum())
    ends=set(c.body_pre)|set(c.body_post);ntbody=reader(raw/'body-neurotransmitters-male-cns-v1.0.feather').read_all();mask=pc.is_in(ntbody['body'],value_set=pa.array(sorted(map(int,ends)),type=pa.int64()));pq.write_table(strings(ntbody.filter(mask),['body']),m/'registry/partner_neurotransmitters.parquet',compression='zstd')
    # Audit against the independently published aggregate edge table, both directions.
    weights=reader(raw/'connectome-weights-male-cns-v1.0-minconf-0.5.feather');print('weight schema',weights.schema.names,flush=True);wp=[]
    for i in range(weights.num_record_batches):
        b=weights.get_batch(i);mask=pc.or_(pc.is_in(b.column('body_pre'),value_set=ids),pc.is_in(b.column('body_post'),value_set=ids))
        if pc.any(mask).as_py():wp.append(pa.Table.from_batches([b]).filter(mask))
    w=strings(pa.concat_tables(wp),['body_pre','body_post']);pq.write_table(w,m/'data/source_aggregate_weights.parquet',compression='zstd');wd=w.to_pandas();weightcol=next(x for x in ['weight','synapses'] if x in wd.columns)
    actual=directed.set_index(['body_pre','body_post']).synapse_pairs.sort_index();expected=wd.groupby(['body_pre','body_post'])[weightcol].sum().sort_index();pd.testing.assert_series_equal(actual,expected,check_dtype=False,check_names=False)
    partner=pd.DataFrame({'bodyId':sorted(ends)}).merge(ann.assign(bodyId=ann.bodyId.astype(str)),on='bodyId',how='left',indicator='annotation_join');partner.to_parquet(m/'registry/partners.parquet',index=False)
    src=[]
    for p in [ap,path,ntpath,raw/'body-neurotransmitters-male-cns-v1.0.feather',raw/'connectome-weights-male-cns-v1.0-minconf-0.5.feather']:
        src.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
    for n in ['download_manifest.json','synaptic_download_manifest.json','release_listing.json']:shutil.copy2(raw/n,m/'audit'/n)
    status={'dataset_version':'MaleCNS v1.0 minconf0.5','KC_selection':'official type startswith KC','KC_count':len(kc),'left_KC':int(kc.somaSide.eq('L').sum()),'right_KC':int(kc.somaSide.eq('R').sum()),'original_synpartner_rows_scanned':offset,'incident_synapse_pairs':len(c),'KC_KC_synapses':int(c.body_pre.isin(set(kc.bodyId.astype(str))).mul(c.body_post.isin(set(kc.bodyId.astype(str)))).sum()),'directed_neuron_pairs':len(directed),'distinct_endpoints':len(ends),'partner_annotation_missing':int(partner.annotation_join.eq('left_only').sum()),'selected_presynaptic_sites':len(nt),'physical_contacts_without_site_nt':missing,'aggregate_count_conservation':'PASSED','coordinates':'original native xyz retained; unit evidence in COORDINATES.json','source_row_identifier':'0-based row in original syn-partners feather; not a biological synapse ID; polyadic pre sites repeat across contacts','sources':src,'elapsed_s':time.time()-start,'functional_groups':'known type/side only; no CM/SU homology or new functional validation'};dump(m/'audit/TABLES_STATUS.json',status);print(json.dumps({k:v for k,v in status.items() if k!='sources'}),flush=True)

def geometry(root,out):
    for dataset,src,pattern in [('flywire_v783',root/'artifacts/wu_20261030_all5177_structural_atlas/geometry','*.json'),('malecns_v1_0',root/'data/raw/male_cns_v1_0/kc_swc','*.swc')]:
        target=out/dataset/'geometry';target.mkdir(parents=True,exist_ok=True);paths=sorted(src.glob(pattern));rows=[]
        for shard in range(0,len(paths),128):
            name=f'geometry-{shard//128:04d}.tar.gz'
            with tarfile.open(target/name,'w:gz',compresslevel=3) as tar:
                for path in paths[shard:shard+128]:
                    tar.add(path,arcname=path.name);rows.append({'neuron_id':path.stem,'archive':'geometry/'+name,'member':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'format':'json' if dataset=='flywire_v783' else 'swc'})
        pd.DataFrame(rows).to_parquet(out/dataset/'registry/geometry_index.parquet',index=False)
        if dataset=='malecns_v1_0':shutil.copy2(src/'MANIFEST.json',out/dataset/'audit/SWC_DOWNLOAD_MANIFEST.json')
        print(dataset,'archived geometries',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tables','geometry']);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.source_root.resolve(),a.output.resolve())
