"""Validate complete anatomy, identity, coordinate and physical-contact conservation."""
from pathlib import Path
import argparse,json,hashlib,tarfile,io
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();b=a.bundle;reports={}
    for dataset in ['flywire_v783','malecns_v1_0']:
        d=b/dataset;cells=pd.read_parquet(d/'registry/cells.parquet');idcol='root_id' if dataset=='flywire_v783' else 'bodyId';ids=set(cells[idcol]);assert len(ids)==len(cells) and all(isinstance(x,str) for x in ids)
        groups=pd.read_parquet(d/'registry/group_members.parquet');assert set(groups.root_id)<=ids
        s=pd.read_parquet(d/'data/synapse_pairs.parquet');pre,post=('pre_id','post_id') if dataset=='flywire_v783' else ('body_pre','body_post');assert s.synapse_id.is_unique
        for col in [pre,post,'synapse_id']:assert s[col].map(lambda v:isinstance(v,str)).all()
        ins=s[post].isin(ids);outs=s[pre].isin(ids);assert (ins|outs).all()
        pairs=pd.read_parquet(d/'data/directed_pairs.parquet');paircount='synapse_pair_count' if dataset=='flywire_v783' else 'synapse_pairs'
        assert int(pairs[paircount].sum())==len(s)
        idx=pd.read_parquet(d/'registry/geometry_index.parquet');assert idx.neuron_id.is_unique and set(idx.neuron_id)==ids;checked=0;nodes=0;swc_stats=[]
        for archive,g in idx.groupby('archive'):
            with tarfile.open(d/archive,'r:gz') as tar:
                for r in g.itertuples(index=False):
                    data=tar.extractfile(r.member).read();assert hashlib.sha256(data).hexdigest()==r.sha256;checked+=1
                    if dataset=='malecns_v1_0':
                        v=np.loadtxt(io.BytesIO(data));assert np.isfinite(v).all();node=v[:,0].astype(int);parent=v[:,6].astype(int);assert np.array_equal(node,v[:,0]) and np.array_equal(parent,v[:,6]);assert len(set(node))==len(node)
                        ix=pd.Index(node);has=parent!=-1;pi=ix.get_indexer(parent[has]);assert (pi>=0).all();ci=np.flatnonzero(has);assert (pi!=ci).all();mat=sparse.coo_matrix((np.ones(len(pi)),(ci,pi)),shape=(len(v),len(v))).tocsr();components=connected_components(mat,directed=True,connection='weak',return_labels=False);rank=len(pi)-len(v)+components;assert rank==0
                        swc_stats.append({'neuron_id':r.neuron_id,'nodes':len(v),'edges':len(pi),'components':components,'cycle_rank':rank,'root_nodes':int((parent==-1).sum()),'nonpositive_radius_nodes':int((v[:,5]<=0).sum()),'native_coordinate_unit':'8nm voxel','downsampling':'original NeuTu coarse header retained'});nodes+=len(v)
                    else:
                        v=json.loads(data);assert str(v['id'])==r.neuron_id;nodes+=len(v['vertices'])
            if checked%1024==0:print(dataset,'verified geometries',checked,flush=True)
        extra={}
        if dataset=='flywire_v783':
            e=pd.read_parquet(d/'data/endpoints.parquet');assert len(e)==int(ins.sum()+outs.sum());assert set(e.direction)=={'in','out'};mapped=s.set_index('synapse_id')
            for direction,prefix,col in [('in','post',post),('out','pre',pre)]:
                sub=e[e.direction.eq(direction)];raw=mapped.loc[sub.synapse_id];assert np.array_equal(sub.root_id.to_numpy(),raw[col].to_numpy())
                expected=raw[[f'{prefix}_pt_position_{axis}' for axis in 'xyz']].to_numpy(float)*.001;actual=sub[['x_um','y_um','z_um']].to_numpy(float);np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
            frozen=pd.read_parquet(d/'registry/frozen_stable8_members.parquet');assert len(frozen)==965
            extra={'endpoints':len(e),'coordinate_direction_conservation':'PASSED_ALL_ENDPOINTS','stable8_members':len(frozen)}
        else:pd.DataFrame(swc_stats).to_parquet(d/'audit/skeleton_topology_QC.parquet',index=False)
        reports[dataset]={'cells':len(cells),'group_views':groups.group_id.nunique(),'unique_group_members':groups.root_id.nunique(),'geometry_files_hash_verified':checked,'skeleton_nodes':nodes,'physical_contact_pairs':len(s),'input_contacts':int(ins.sum()),'output_contacts':int(outs.sum()),'internal_contacts':int((ins&outs).sum()),'distinct_directed_body_pairs':len(pairs),**extra}
    out=b/'validation';out.mkdir(exist_ok=True);(out/'ANATOMY_CONSERVATION.json').write_text(json.dumps({'passed':True,'datasets':reports},indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
