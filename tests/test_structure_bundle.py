"""Fixture-only contracts; no synthetic anatomy is published as real data."""
import hashlib,io,json,tarfile
import pandas as pd
import pytest
from neuronmimic.structure_bundle import groups,export,read_geometry,select

@pytest.fixture
def bundle(tmp_path):
    b=tmp_path/'bundle';d=b/'flywire_v783'
    for part in ['registry','data','geometry']:(d/part).mkdir(parents=True)
    a='720575940603174368';c='720575940603174369';outside='720575940603174370'
    pd.DataFrame({'root_id':[a,c]}).to_parquet(d/'registry/cells.parquet')
    pd.DataFrame({'root_id':[a,c],'group_id':['CM_TEST','CM_TEST'],'namespace':['fixture']*2,'entry_kind':['FIXTURE_TEST_ONLY']*2,'evidence_status':['FIXTURE_TEST_ONLY']*2}).to_parquet(d/'registry/group_members.parquet')
    pd.DataFrame({'pre_id':[outside,a,c],'post_id':[a,c,outside],'synapse_id':['s1','s2','s3'],'pre_pt_position_x':[10,20,30],'post_pt_position_x':[11,21,31]}).to_parquet(d/'data/synapse_pairs.parquet')
    for name in ['endpoints','branches']:pd.DataFrame({'root_id':[a,c]}).to_parquet(d/f'data/{name}.parquet')
    pd.DataFrame({'root_id':[outside]}).to_csv(d/'registry/partner_annotations.csv',index=False)
    entries=[]
    with tarfile.open(d/'geometry/test.tar.gz','w:gz') as t:
        for rid in [a,c]:
            data=json.dumps({'id':rid}).encode();i=tarfile.TarInfo(rid+'.json');i.size=len(data);t.addfile(i,io.BytesIO(data));entries.append({'neuron_id':rid,'archive':'geometry/test.tar.gz','member':rid+'.json','sha256':hashlib.sha256(data).hexdigest()})
    pd.DataFrame(entries).to_parquet(d/'registry/geometry_index.parquet')
    return b,a,c

def test_incident_contacts_keep_direction_without_double_counting_internal(bundle,tmp_path):
    b,a,c=bundle;out=tmp_path/'export';q=export(b,'flywire_v783',out,group='CM_TEST')
    assert q['unique_physical_contacts']==3 and q['input_contacts']==2 and q['output_contacts']==2 and q['internal_contacts']==1
    contacts=pd.read_parquet(out/'synapse_pairs.parquet');assert contacts.synapse_id.tolist()==['s1','s2','s3'];assert contacts.pre_id.iloc[1]==a
    assert contacts.pre_pt_position_x.tolist()==[10,20,30] and contacts.post_pt_position_x.tolist()==[11,21,31]
    assert json.loads((out/'geometry'/f'{a}.json').read_text())['id']==a
    with pytest.raises(FileExistsError):export(b,'flywire_v783',out,group='CM_TEST')

def test_precise_identity_and_dataset_boundary(bundle):
    b,a,c=bundle;assert groups(b,'flywire_v783').n_members.tolist()==[2]
    assert select(b,'flywire_v783',neuron=a)[1]=={a}
    with pytest.raises(TypeError):select(b,'flywire_v783',neuron=int(a))
    with pytest.raises(ValueError):select(b,'flywire_v783',group='CM_OTHER')
    with pytest.raises(ValueError):select(b,'unknown_dataset',neuron=a)

def test_corrupt_geometry_is_rejected(bundle):
    b,a,c=bundle;p=b/'flywire_v783/registry/geometry_index.parquet';d=pd.read_parquet(p);d['sha256']='invalid';d.to_parquet(p)
    with pytest.raises(ValueError,match='SHA256'):read_geometry(b,'flywire_v783',a)
