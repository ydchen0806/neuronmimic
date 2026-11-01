"""FIXTURE_TEST_ONLY mathematical contracts, not biological evidence."""
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
from bio_fly.wu_20261019.arbor_opportunity import subdivide, SegmentIndex, projection_distances, cable_kernels


def test_cable_conservation_and_branch_identity():
    c=subdivide([[0,0,0],[3,0,0],[3,0,0],[3,2,0]],[[0,1],[1,2],[2,3]],[7,8,9],.7)
    np.testing.assert_allclose(c['weight'].sum(),5.)
    assert c['weight'].max()<=.7 and set(c['original_edge'])=={0,2}
    np.testing.assert_array_equal(c['branch_id'],np.where(c['original_edge']==0,7,9))
    np.testing.assert_allclose(np.linalg.norm(c['b']-c['a'],axis=1),c['weight'])


def test_exact_search_recovers_long_segment_not_nearest_midpoint():
    # The nearest segment has its midpoint far away; naive kNN would miss it.
    a=np.vstack([np.array([[0.,0,0]]),np.column_stack([np.arange(1,40),np.ones(39),np.zeros(39)])])
    b=a.copy();b[0]=[1000,0,0]
    p=np.array([[.1,.01,0],[600,5,0],[2000,0,0]])
    ix=SegmentIndex(a,b)
    brute=projection_distances(p[:,None,:],a[None,:,:],b[None,:,:]).min(axis=1)
    np.testing.assert_allclose(ix.distances(p,initial_k=1),brute,atol=1e-12)


def test_random_exact_search_and_degenerate_segments():
    rng=np.random.default_rng(2026102909);a=rng.normal(size=(121,3));b=a+rng.normal(size=(121,3));b[:5]=a[:5]
    p=rng.normal(size=(47,3))*3
    expected=projection_distances(p[:,None,:],a[None,:,:],b[None,:,:]).min(axis=1)
    np.testing.assert_allclose(SegmentIndex(a,b).distances(p),expected,atol=1e-12)


def test_empty_region_is_missing_and_integration_direction_differs():
    empty=SegmentIndex(np.empty((0,3)),np.empty((0,3)))
    k,d=cable_kernels(np.array([[0.,0,0]]),np.array([1.]),empty)
    assert np.isnan(k).all() and np.isnan(d)
    short=subdivide([[0,0,0],[1,0,0]],[[0,1]],[0],.1)
    long=subdivide([[0,0,0],[10,0,0]],[[0,1]],[0],.1)
    f,_=cable_kernels(short['midpoint'],short['weight'],SegmentIndex(long['a'],long['b']))
    r,_=cable_kernels(long['midpoint'],long['weight'],SegmentIndex(short['a'],short['b']))
    np.testing.assert_allclose(f,1)
    assert (r<f).all()


def test_empty_cable_and_uint64_id_roundtrip(tmp_path):
    c=subdivide([[0,0,0]],np.empty((0,2),int),[],1.)
    assert len(c['weight'])==0
    ids=['720575940623118976','720575940623118977']
    file=tmp_path/'ids.parquet';pd.DataFrame({'root_id':ids}).to_parquet(file,index=False)
    assert json.loads(pd.read_parquet(file).to_json(orient='records'))==[{'root_id':i} for i in ids]


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wu_20261029_ap2_arbor_opportunity'


def test_real_complete_parent_geometry_and_numerics():
    if not (OUT/'validation/pair_geometry_checks.json').exists():pytest.skip('Complete real geometry not yet computed')
    c=pd.read_csv(OUT/'data/cell_geometry.csv',dtype={'root_id':str})
    assert c.groupby('side').size().to_dict()=={'left':138,'right':160}
    assert (abs(c.json_rounding_delta_um)<=c.json_rounding_bound_um+1e-9).all()
    assert (c.full_cable_um>0).all()
    g=pd.read_parquet(OUT/'data/directional_arbor_features.parquet')
    assert len(g)==160*159+138*137 and not g.duplicated(['source_id','target_id']).any()
    assert (g.source_id!=g.target_id).all() and all(isinstance(x,str) for x in g.source_id)
    for col in [x for x in g if '_gaussian_' in x]:assert g[col].dropna().between(0,1).all()
    audit=json.loads((OUT/'validation/numerical_checks.json').read_text())
    assert audit['passed'] and audit['max_exact_distance_error_um']<1e-9
    assert json.loads((OUT/'audit/geometry_manifest.json').read_text())['contact_data_used_for_geometry'] is False


def test_real_left_target_contacts_removed_and_geometry_order():
    if not (OUT/'data/directional_arbor_features.parquet').exists():pytest.skip('Real inputs not yet computed')
    from fit_ap2_arbor import geometry_design
    p=pd.read_parquet(OUT/'data/left_pair_features.parquet');m=pd.read_parquet(OUT/'data/left_training_contact_endpoints.parquet')
    counts=m.groupby(['root_id','direction']).size()
    np.testing.assert_array_equal(p.pre_cloud_n_pair_removed,p.pre_id.map(counts.xs('out',level='direction'))-p.train_synapse_count)
    np.testing.assert_array_equal(p.post_cloud_n_pair_removed,p.post_id.map(counts.xs('in',level='direction'))-p.train_synapse_count)
    assert len(p)==138*137 and p.pre_id.nunique()==138
    assert p.loc[p.train_synapse_count>0,'split'].eq('train').all()
    pair=p.iloc[[0]].copy();reverse=p[(p.pre_id==pair.post_id.iloc[0])&(p.post_id==pair.pre_id.iloc[0])]
    a,_,_=geometry_design(pair);b,_,_=geometry_design(reverse)
    np.testing.assert_array_equal(a[:,:3],b[:,3:]);np.testing.assert_array_equal(a[:,3:],b[:,:3])


def test_real_model_replay_and_exact_control_labels():
    if not (OUT/'validation/model_checks.json').exists():pytest.skip('Real model fits not yet completed')
    from fit_ap2_arbor import designs,OLD,label_block
    from scipy.special import expit
    right,left,rd,ld=designs();cells=pd.read_csv(OLD/'data/parent_cells.csv',dtype={'root_id':str})
    h=label_block(cells.label.to_numpy(),right.pre_index.to_numpy(),right.post_index.to_numpy())
    prediction=pd.read_parquet(OUT/'analysis/pair_predictions.parquet')
    for name in ['ARBOR','REGION','SITE_REGION']:
        for kind in ['baseline','plus_label']:
            z=np.load(OUT/f'models/{name}_{kind}.npz');design=rd[name] if kind=='baseline' else np.c_[rd[name],h]
            for domain,pair,x in [('right_fit_heldout',right,design)]+([('left_transfer',left,ld[name])] if kind=='baseline' else []):
                keep=pair.split.eq('test').to_numpy();expected=prediction[(prediction.domain==domain)&(prediction.baseline==name)&(prediction.kind==kind)]
                assert expected.pre_id.tolist()==pair.loc[keep,'pre_id'].tolist() and expected.post_id.tolist()==pair.loc[keep,'post_id'].tolist()
                actual=expit(((x[keep]-z['mean'])/z['scale'])@z['coef'].ravel()+z['intercept'][0])
                np.testing.assert_allclose(actual,expected.probability,rtol=1e-12,atol=1e-12)
    null=pd.read_csv(OUT/'analysis/matched_label_controls.csv');old=pd.read_csv(OLD/'analysis/matched_label_controls.csv')
    assert len(null)==100 and null.label_sha256.tolist()==old.label_sha256.tolist()
    assert pd.read_csv(OUT/'validation/fit_convergence.csv').converged.all()
