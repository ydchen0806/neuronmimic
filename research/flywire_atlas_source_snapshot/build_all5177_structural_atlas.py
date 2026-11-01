"""Complete KC geometry atlas with frozen stable8 and distinct consensus views."""
from pathlib import Path
import argparse,hashlib,json,shutil,struct,subprocess,time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from kc_tree_routes import TreeMetric
from analyze_kc_tree_routes import sha,choose_pair
from build_kc_structure_entries import partition

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'nature_methods_connectome/fine_structure_mechanism/runs/20260929_flywire_atlas_completion_v2'
PREV=ROOT/'artifacts/wu_20261029_structure_entries'
FROZEN=ROOT/'nature_methods_connectome/fine_structure_mechanism/runs/20261018_wu1018_frozen5177_review_v11/results_v9/all5177_evidence_atlas.parquet'
STABLE=ROOT/'artifacts/wu_20261019/registry/stable8_members.parquet'
ANN=ROOT/'external/flywire_annotations_upstream/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
WHOLE=ROOT/'artifacts/wu_20261026_spatial_units/registry/whole_brain_somata.parquet'
OUT=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
PROTOCOL='kc_all5177_structural_atlas_v1_20261030'
EXPECTED={'CM012':163,'CM014':152,'CM016':147,'CM019':132,'CM020':129,'CM022':114,'CM031':67,'CM033':61}


def cut_metrics(edges,core,parent):
    """Directed weights symmetrized by addition; loops excluded from conductance."""
    core=set(core)&set(parent);rest=set(parent)-core
    z=edges[edges.pre_id.isin(parent)&edges.post_id.isin(parent)].copy();loops=z[z.pre_id==z.post_id];z=z[z.pre_id!=z.post_id]
    aa=z[z.pre_id.isin(core)&z.post_id.isin(core)];bb=z[z.pre_id.isin(rest)&z.post_id.isin(rest)];ab=z[z.pre_id.isin(core)&z.post_id.isin(rest)];ba=z[z.pre_id.isin(rest)&z.post_id.isin(core)]
    w=lambda x:int(x.synapse_pair_count.sum());cut=w(ab)+w(ba);va=2*w(aa)+cut;vb=2*w(bb)+cut
    return dict(candidate_n=len(core),parent_n=len(parent),residual_n=len(rest),candidate_internal_synapses=w(aa),residual_internal_synapses=w(bb),candidate_to_residual_synapses=w(ab),residual_to_candidate_synapses=w(ba),candidate_internal_directed_pairs=len(aa),residual_internal_directed_pairs=len(bb),candidate_to_residual_directed_pairs=len(ab),residual_to_candidate_directed_pairs=len(ba),candidate_density=len(aa)/(len(core)*(len(core)-1)) if len(core)>1 else np.nan,residual_density=len(bb)/(len(rest)*(len(rest)-1)) if len(rest)>1 else np.nan,candidate_to_residual_density=len(ab)/(len(core)*len(rest)) if core and rest else np.nan,residual_to_candidate_density=len(ba)/(len(core)*len(rest)) if core and rest else np.nan,weighted_cut=cut,weighted_volume_candidate=va,weighted_volume_residual=vb,conductance=cut/min(va,vb) if min(va,vb)>0 else np.nan,excluded_autapse_synapses=w(loops))


def role_connections(edges,roles,all_kc):
    context=set.union(*roles.values());z=edges[edges.pre_id.isin(context)|edges.post_id.isin(context)].copy();lookup={r:k for k,v in roles.items() for r in v}
    def label(series):return series.map(lookup).fillna(pd.Series(np.where(series.isin(all_kc),'other_KC','non_KC'),index=series.index))
    z['pre_role']=label(z.pre_id);z['post_role']=label(z.post_id);z['autapse']=z.pre_id.eq(z.post_id)
    rows=[];sizes={k:len(v) for k,v in roles.items()};sizes['other_KC']=len(set(all_kc)-context)
    for (a,b),g in z.groupby(['pre_role','post_role']):
        nonloop=g[~g.autapse];den=sizes[a]*sizes[b]-(sizes[a] if a==b else 0) if a in sizes and b in sizes else np.nan
        rows.append(dict(pre_role=a,post_role=b,synapse_pairs=int(g.synapse_pair_count.sum()),directed_pairs_including_autapses=len(g),nonautapse_directed_pairs=len(nonloop),autapse_synapse_pairs=int(g.loc[g.autapse,'synapse_pair_count'].sum()),possible_nonautapse_directed_pairs=den,directed_density=len(nonloop)/den if pd.notna(den) and den>0 else np.nan,density_status='DEFINED_FOR_KC_SETS' if pd.notna(den) and den>0 else 'NON_KC_POPULATION_DENOMINATOR_UNSPECIFIED'))
    assert sum(r['synapse_pairs'] for r in rows)==int(z.synapse_pair_count.sum())
    return rows,len(z),int(z.synapse_pair_count.sum())


def build(out):
    started=time.time();protocol=json.loads((out/'audit/protocol.json').read_text());assert protocol['protocol_version']==PROTOCOL
    for sub in ['audit','registry','data','geometry','analysis','figures','report','validation','code']:(out/sub).mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(PREV/'registry/all5177_cells.parquet');assert len(meta)==5177 and meta.index.is_unique
    old=pd.read_parquet(FROZEN);stable=pd.read_parquet(STABLE);assert stable.groupby('candidate_id').size().to_dict()==EXPECTED and stable.root_id.nunique()==965
    assert set(old.loc[old.stable_backbone,'root_id'])==set(stable.root_id)
    assert old.evidence_level.value_counts().to_dict()=={'tentative':2207,'unresolved':1661,'stable_main':965,'coverage_fail':344}
    ann=pd.read_csv(ANN,sep='\t',dtype=str,keep_default_na=False).set_index('root_id');assert ann.index.is_unique
    meta=meta.join(ann[['hemibrain_type','cell_type','status','synonyms']].rename(columns=lambda c:'source_'+c))
    legacy=old.set_index('root_id')[['candidate_id','evidence_level','stable_backbone','member_retention','frozen_detailed_subtype']].rename(columns=lambda c:'legacy_'+c);meta=meta.join(legacy,validate='one_to_one')
    assert not meta.legacy_evidence_level.isna().any();meta.to_parquet(out/'registry/all5177_cells.parquet');shutil.copyfile(FROZEN,out/'registry/frozen_all5177_evidence.parquet');shutil.copyfile(STABLE,out/'registry/frozen_stable8_members.parquet')
    whole=pd.read_parquet(WHOLE);coords=ann.loc[whole.root_id,['soma_x','soma_y','soma_z']].apply(pd.to_numeric).to_numpy()*[.004,.004,.04]
    np.testing.assert_allclose(whole[['x_um','y_um','z_um']],coords,atol=1e-9,rtol=0);assert len(whole)==118102
    np.testing.assert_allclose(whole.set_index('root_id').loc[meta.index,['x_um','y_um','z_um']],meta[['soma_x_um','soma_y_um','soma_z_um']],atol=1e-9,rtol=0);whole.to_parquet(out/'registry/whole_brain_somata.parquet',index=False)
    oldroles=pd.read_parquet(PREV/'registry/entry_cells.parquet');prior_entries=pd.read_csv(PREV/'registry/entries.csv');oldcores={eid:set(g.loc[g.role.isin(['parent_hit','mixed_member']),'root_id']) for eid,g in oldroles.groupby('entry_id')}
    definitions=[(cid,set(g.root_id),'FROZEN_STABLE8','stable_main') for cid,g in stable.groupby('candidate_id')]+[(eid,core,'CROSS_METHOD_CONSENSUS','DESCRIPTIVE_ONLY') for eid,core in oldcores.items()]
    entries=[];roles_all=[];members=[];role_sets={};secondary={};cores={}
    for eid,core,kind,status in definitions:
        counts=meta.loc[sorted(core)].groupby(['official_KC_subtype','official_hemisphere']).size().sort_values(ascending=False);typ,side=counts.index[0]
        parent=set(meta.index[(meta.official_KC_subtype==typ)&(meta.official_hemisphere==side)]);opposite=set(meta.index[(meta.official_KC_subtype==typ)&(meta.official_hemisphere!=side)]);roles=partition(core,parent,opposite)
        labels=meta.loc[sorted(roles['parent_hit']),'source_hemibrain_type'];single=labels.nunique()==1 and labels.iloc[0]!='';mapped=labels.iloc[0] if single else ''
        mapped_parent={r for r in parent if meta.loc[r,'source_hemibrain_type']==mapped} if single else set();secondary[eid]=mapped_parent;role_sets[eid]=roles;cores[eid]=core
        for role,ids in roles.items():
            for root in sorted(ids):roles_all.append(dict(entry_id=eid,root_id=root,role=role,within_mapped_parent=root in mapped_parent,source_hemibrain_type=meta.loc[root,'source_hemibrain_type']))
        for root in sorted(core):members.append(dict(entry_id=eid,root_id=root,entry_kind=kind,evidence_status=status))
        entries.append(dict(entry_id=eid,entry_kind=kind,evidence_status=status,parent_type=typ,side=side,member_n=len(core),parent_hit_n=len(roles['parent_hit']),parent_n=len(parent),residual_n=len(roles['parent_residual']),mixed_n=len(roles['mixed_member']),opposite_parent_n=len(opposite),parent_coverage=len(roles['parent_hit'])/len(parent),purity=len(roles['parent_hit'])/len(core),mapped_type=mapped,mapped_parent_n=len(mapped_parent) if single else np.nan,mapped_residual_n=len(mapped_parent-core) if single else np.nan,mapped_parent_coverage=len(roles['parent_hit'])/len(mapped_parent) if mapped_parent else np.nan,secondary_parent_status='DEFINED_WITHIN_COARSE_PARENT' if single else 'MULTIPLE_OR_MISSING_MAPPED_TYPES'))
    entries=pd.DataFrame(entries);roles=pd.DataFrame(roles_all);entries.to_csv(out/'registry/entries.csv',index=False);roles.to_parquet(out/'registry/entry_roles.parquet',index=False);pd.DataFrame(members).to_parquet(out/'registry/entry_members.parquet',index=False)
    for eid,g in oldroles.groupby('entry_id'):
        assert set(zip(g.root_id,g.role))==set(zip(roles.loc[roles.entry_id==eid,'root_id'],roles.loc[roles.entry_id==eid,'role']))
    assert entries.loc[entries.entry_id=='CM016','mapped_parent_n'].iloc[0]==310 and entries.loc[entries.entry_id=='CM022','mapped_parent_n'].iloc[0]==311
    # All prior detailed contexts are a subset; no selected parent determines atlas acquisition coverage.
    coverage=meta[['official_KC_subtype','official_hemisphere','legacy_evidence_level','legacy_stable_backbone']].copy();coverage['prior_detailed_context']=coverage.index.isin(oldroles.root_id);coverage['detailed_geometry_available']=True;coverage['named_entry_context']=coverage.index.isin(roles.root_id);coverage.to_csv(out/'analysis/coverage_registry.csv')
    syn=pd.read_parquet(SRC/'results/all_KC_incident_synapses.parquet').rename(columns={'id':'synapse_id','pre_pt_root_id':'pre_id','post_pt_root_id':'post_id'})
    ep=pd.read_parquet(SRC/'results/synapse_branch_mapping.parquet').rename(columns={'neuron_id':'root_id'});br=pd.read_parquet(SRC/'results/topological_branches.parquet').rename(columns={'neuron_id':'root_id'})
    for df,cols in [(syn,['synapse_id','pre_id','post_id']),(ep,['root_id','synapse_id','pre_id','post_id']),(br,['root_id'])]:
        for col in cols:df[col]=df[col].astype(str)
    assert len(syn)==1881678 and syn.synapse_id.is_unique and len(ep)==2261016 and len(br)==1252322
    assert set(ep.root_id)==set(meta.index)==set(br.root_id)
    assert not ep.duplicated(['root_id','direction','synapse_id']).any();assert not br.duplicated(['root_id','branch_id']).any()
    raw=syn.set_index('synapse_id').loc[ep.synapse_id];assert np.array_equal(raw.pre_id,ep.pre_id) and np.array_equal(raw.post_id,ep.post_id)
    expectedxyz=np.where(ep.direction.eq('in').to_numpy()[:,None],raw[[f'post_pt_position_{a}' for a in 'xyz']],raw[[f'pre_pt_position_{a}' for a in 'xyz']])/1000
    np.testing.assert_allclose(expectedxyz,ep[['x_um','y_um','z_um']],atol=1e-9,rtol=0)
    for d,col in [('in','post_id'),('out','pre_id')]:assert ep.loc[ep.direction==d,col].equals(ep.loc[ep.direction==d,'root_id'])
    assert len(ep)==int(syn.pre_id.isin(meta.index).sum()+syn.post_id.isin(meta.index).sum())
    ep['partner_id']=np.where(ep.direction.eq('in'),ep.pre_id,ep.post_id)
    for col in ['super_class','cell_class','cell_type']:ep['partner_'+col]=ep.partner_id.map(ann[col]).fillna('')
    ep['partner_annotation_present']=ep.partner_id.isin(ann.index)
    ep['input_family']=np.select([ep.direction.eq('in')&ep.partner_cell_class.eq('ALPN'),ep.direction.eq('in')&ep.partner_super_class.eq('visual_projection'),ep.direction.eq('in')&ep.partner_cell_type.eq('APL')],['ALPN','VISUAL_PROJECTION','APL'],default='OTHER');ep['mbon_output']=ep.direction.eq('out')&ep.partner_cell_class.eq('MBON')
    ep.to_parquet(out/'data/endpoints.parquet',index=False);syn.to_parquet(out/'data/synapse_pairs.parquet',index=False);br.to_parquet(out/'data/branches.parquet',index=False)
    ann.loc[sorted(set(ep.partner_id)&set(ann.index))].reset_index().to_csv(out/'registry/partner_annotations.csv',index=False)
    edges=syn.groupby(['pre_id','post_id']).size().rename('synapse_pair_count').reset_index();edges.to_parquet(out/'data/directed_pairs.parquet',index=False);assert edges.synapse_pair_count.sum()==len(syn)
    groups=dict(tuple(ep.groupby('root_id')));bg=dict(tuple(br.groupby('root_id')));geometry_checks=[];geometry_sources=[]
    inputs=[FROZEN,STABLE,ANN,WHOLE,PREV/'registry/all5177_cells.parquet',PREV/'registry/entry_cells.parquet',PREV/'registry/entries.csv',SRC/'results/all_KC_incident_synapses.parquet',SRC/'results/synapse_branch_mapping.parquet',SRC/'results/topological_branches.parquet',out/'audit/protocol.json',Path(__file__)]
    manifest=dict(protocol_version=PROTOCOL,dataset_version='FlyWire v783',connectome_snapshot='frozen all-KC incident synapses and original geometry',annotation_snapshot=subprocess.check_output(['git','-C',str(ANN.parent.parent),'rev-parse','HEAD'],text=True).strip(),run_id='all5177_structural_atlas_20261030',seed=protocol['seed'],model_version='NONE_GEOMETRY_ONLY',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=[dict(path=str(p),sha256=sha(p)) for p in inputs])
    (out/'audit/provenance_manifest.json').write_text(json.dumps(manifest,indent=2))
    for k,root in enumerate(sorted(meta.index),1):
        rawpath=SRC/f'sources/skeletons/{root}.bin';blob=rawpath.read_bytes();digest=hashlib.sha256(blob).hexdigest();assert digest==meta.loc[root,'source_sha256']
        source=SRC/f'viewer/data/{root}_skeleton.json';g=json.loads(source.read_text());assert g['id']==root;v=np.asarray(g['vertices']);e=np.asarray(g['edges'],int);t=TreeMetric(v,e);nv,ne=struct.unpack_from('<II',blob);rv=np.frombuffer(blob,dtype='<f4',count=nv*3,offset=8).reshape(-1,3).astype(float)/1000;re=np.frombuffer(blob,dtype='<u4',count=ne*2,offset=8+nv*12).reshape(-1,2)
        np.testing.assert_array_equal(np.round(rv,6),v);np.testing.assert_array_equal(re,e)
        z=groups[root];indices=z.segment_index.to_numpy(int);f=z.segment_fraction.to_numpy(float);labels=np.asarray(g['branch_per_edge'],int);np.testing.assert_array_equal(labels[indices],z.branch_id)
        error=np.linalg.norm(z[['x_um','y_um','z_um']].to_numpy()-t.points(indices,f),axis=1);delta=float(np.max(np.abs(error-z.mapping_error_um)));assert delta<2e-6;assert np.array_equal(z.mapping_within_2um,z.mapping_error_um.le(2))
        lengths=np.linalg.norm(rv[e[:,0]]-rv[e[:,1]],axis=1);rawsum=pd.Series(lengths).groupby(labels).sum().sort_index();expect=bg[root].set_index('branch_id').cable_length_um.sort_index();np.testing.assert_allclose(rawsum,expect,atol=1e-12,rtol=1e-12)
        shutil.copyfile(source,out/'geometry'/f'{root}.json');geometry_sources.append(dict(root_id=root,raw_source=str(rawpath),raw_sha256=digest,geometry_source=str(source),geometry_sha256=sha(source)))
        geometry_checks.append(dict(root_id=root,n_vertices=len(v),n_edges=len(e),n_branches=len(expect),n_endpoints=len(z),max_projection_error_difference_um=delta,max_raw_branch_length_difference_um=float(np.max(np.abs(rawsum.to_numpy()-expect.to_numpy()))),connected_tree=True))
        if k%500==0:print(f'Geometry {k}/5177 verified;{time.time()-started:.1f}s',flush=True)
    pd.DataFrame(geometry_sources).to_csv(out/'audit/geometry_sources.csv',index=False);pd.DataFrame(geometry_checks).to_csv(out/'validation/all5177_geometry_checks.csv',index=False)
    # Keep zero-contact branches in the denominator.
    counts=ep.groupby(['root_id','branch_id','direction']).size().unstack(fill_value=0).rename(columns={'in':'n_input_endpoints','out':'n_output_endpoints'}).reset_index();bprofile=br.merge(counts,on=['root_id','branch_id'],how='left',validate='one_to_one');bprofile[['n_input_endpoints','n_output_endpoints']]=bprofile[['n_input_endpoints','n_output_endpoints']].fillna(0).astype(int)
    bprofile['occupied']=bprofile.n_input_endpoints.add(bprofile.n_output_endpoints).gt(0);bprofile['mixed']=bprofile.n_input_endpoints.gt(0)&bprofile.n_output_endpoints.gt(0);bprofile.to_parquet(out/'analysis/branch_contact_profiles.parquet',index=False)
    cell=bprofile.groupby('root_id').agg(n_branches=('branch_id','size'),n_occupied_branches=('occupied','sum'),n_mixed_branches=('mixed','sum'),n_input_endpoints=('n_input_endpoints','sum'),n_output_endpoints=('n_output_endpoints','sum'));cell['mixed_fraction_of_occupied']=cell.n_mixed_branches/cell.n_occupied_branches.replace(0,np.nan);cell['occupied_fraction']=cell.n_occupied_branches/cell.n_branches
    quality=ep.groupby('root_id').agg(mapping_error_median_um=('mapping_error_um','median'),mapping_error_p95_um=('mapping_error_um',lambda x:x.quantile(.95)),mapping_within_2um_fraction=('mapping_within_2um','mean'));cell=cell.join(quality).join(meta[['official_KC_subtype','official_hemisphere','source_hemibrain_type','legacy_evidence_level','cable_length_um','soma_nearest_distance_um']]);cell.to_csv(out/'analysis/cell_structural_profiles.csv')
    partner=ep.groupby(['root_id','direction','partner_id','partner_super_class','partner_cell_class','partner_cell_type'],dropna=False).size().rename('synapse_pairs').reset_index();partner.to_parquet(out/'analysis/partner_fingerprints.parquet',index=False)
    ep.groupby(['root_id','direction','neuropil'],dropna=False).size().rename('endpoint_records').reset_index().to_parquet(out/'analysis/cell_neuropil_profiles.parquet',index=False)
    relations=[];conservation=[];diag=[];representatives=[];branch_examples=[];routes=[];pathrows=[]
    oldreps=pd.read_csv(PREV/'tables/representative_cells.csv',dtype=str).set_index('entry_id');oldbranches=pd.read_csv(PREV/'tables/branch_examples.csv',dtype={'root_id':str}).set_index('entry_id')
    for r in entries.itertuples():
        eid=r.entry_id;roleset=role_sets[eid];rel,nedge,nsyn=role_connections(edges,roleset,set(meta.index));relations.extend([dict(entry_id=eid,**x) for x in rel]);conservation.append(dict(entry_id=eid,n_context_cells=len(set.union(*roleset.values())),n_directed_pairs=nedge,n_synapse_pairs=nsyn))
        for layer,parent in [('coarse',roleset['parent_hit']|roleset['parent_residual']),('mapped_within_coarse',secondary[eid])]:
            if not parent:continue
            metrics=cut_metrics(edges,cores[eid],parent);record=dict(entry_id=eid,parent_layer=layer,**metrics)
            for group,ids in [('candidate',cores[eid]&parent),('residual',parent-cores[eid])]:
                d=pdist(meta.loc[sorted(ids),['soma_x_um','soma_y_um','soma_z_um']])
                for q,label in [(.25,'q25'),(.5,'median'),(.75,'q75')]:record[group+'_soma_pair_distance_'+label+'_um']=float(np.quantile(d,q)) if len(d) else np.nan
            diag.append(record)
        hits=sorted(roleset['parent_hit']);lengths=meta.loc[hits,'cable_length_um'];root=oldreps.loc[eid,'root_id'] if eid in oldreps.index else min(hits,key=lambda x:(abs(lengths[x]-lengths.median()),x));representatives.append(dict(entry_id=eid,root_id=root,rule='Preserved existing median-cable representative' if eid in oldreps.index else 'Nearest median parent-hit cable length; lexical root tie'))
        z=groups[root];profile=bprofile[bprofile.root_id==root].copy();profile['min_io']=profile[['n_input_endpoints','n_output_endpoints']].min(axis=1);profile['total']=profile.n_input_endpoints+profile.n_output_endpoints
        branch=int(oldbranches.loc[eid,'branch_id']) if eid in oldbranches.index else int(profile.sort_values(['min_io','total','branch_id'],ascending=[False,False,True]).iloc[0].branch_id);branch_examples.append(dict(entry_id=eid,root_id=root,branch_id=branch,selection='Preserved frozen branch' if eid in oldbranches.index else 'Max min(input,output), then total, then branchID'))
        outputs=z[z.mbon_output].sort_values('synapse_id');family=next((f for f in ['ALPN','VISUAL_PROJECTION','APL'] if (z.input_family==f).any()),None)
        if family is None or outputs.empty:routes.append(dict(entry_id=eid,root_id=root,status='NO_VERIFIED_ROUTE'));continue
        inputs=z[z.input_family==family].sort_values('synapse_id');_,_,up,down,ni,no=choose_pair(inputs,outputs);a=inputs[inputs.partner_id==up];b=outputs[outputs.partner_id==down]
        g=json.loads((out/'geometry'/f'{root}.json').read_text());t=TreeMetric(g['vertices'],g['edges']);d=t.point_matrix(a.segment_index,a.segment_fraction,b.segment_index,b.segment_fraction);j=int(np.argmin(d[0]));path=t.point_path(int(a.iloc[0].segment_index),float(a.iloc[0].segment_fraction),int(b.iloc[j].segment_index),float(b.iloc[j].segment_fraction));assert abs(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()-d[0,j])<1e-7
        routes.append(dict(entry_id=eid,root_id=root,status='AVAILABLE',family=family,upstream_id=up,upstream_type=ann.loc[up,'cell_type'],downstream_id=down,downstream_type=ann.loc[down,'cell_type'],input_synapse_pairs=ni,output_synapse_pairs=no,endpoint_combinations=ni*no,median_cable_um=float(np.median(d)),example_path_cable_um=float(d[0,j])))
        pathrows.append(dict(entry_id=eid,root_id=root,family=family,upstream_id=up,downstream_id=down,input_synapse_ids=a.synapse_id.tolist(),output_synapse_ids=b.synapse_id.tolist(),path_input_synapse_id=a.iloc[0].synapse_id,path_output_synapse_id=b.iloc[j].synapse_id,path_vertices_um=path.tolist(),path_cable_um=float(d[0,j])))
    pd.DataFrame(relations).to_csv(out/'analysis/directed_role_connections.csv',index=False);pd.DataFrame(conservation).to_csv(out/'validation/entry_connection_conservation.csv',index=False);pd.DataFrame(diag).to_csv(out/'analysis/parent_layer_diagnostics.csv',index=False);pd.DataFrame(representatives).to_csv(out/'registry/representatives.csv',index=False);pd.DataFrame(branch_examples).to_csv(out/'registry/branch_examples.csv',index=False);pd.DataFrame(routes).to_csv(out/'analysis/representative_routes.csv',index=False);(out/'data/representative_paths.json').write_text(json.dumps(pathrows,indent=2))
    result=dict(status='PASS',n_all_KC=5177,n_whole_brain_registered_somata=118102,n_stable8=8,n_stable_members=965,n_prior_consensus_entries=9,n_atlas_views=17,n_named_entry_context_cells=roles.root_id.nunique(),n_unassigned_to_named_entry_context=5177-roles.root_id.nunique(),n_raw_synapse_pairs=len(syn),n_endpoints=len(ep),n_branches=len(br),n_directed_neuron_pairs=len(edges),n_vertices=sum(c['n_vertices'] for c in geometry_checks),n_original_stable_members_added_to_detailed_atlas=int((~stable.root_id.isin(oldroles.root_id)).sum()),max_projection_error_difference_um=max(c['max_projection_error_difference_um'] for c in geometry_checks),max_raw_branch_length_difference_um=max(c['max_raw_branch_length_difference_um'] for c in geometry_checks),all_original_states_preserved=True,n_representative_routes=sum(r['status']=='AVAILABLE' for r in routes),new_stable_promotions=0,runtime_seconds=time.time()-started)
    (out/'validation/build_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();build(a.out)
