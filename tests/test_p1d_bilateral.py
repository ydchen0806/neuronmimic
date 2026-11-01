"""Geometry-blocking and portable-prediction contracts; fixtures are not science."""
import importlib.util
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bilateral_runner',ROOT/'scripts/run_kc_p1d_bilateral.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
OUT=Path(os.environ.get('P1D_BILATERAL_DATA',str(ROOT/'data/p1d_v3_bilateral')))

def test_contiguous_blocks_and_pair_separation():
    xyz=np.c_[np.arange(50),np.zeros(50),np.zeros(50)]
    blocks,score,axis=runner.spatial_blocks(xyz)
    assert (np.diff(blocks[np.argsort(score)])>=0).all()
    assert np.bincount(blocks).tolist()==[10]*5
    i,j=np.where(~np.eye(50,dtype=bool))
    for fold in range(5):
        split=runner.spatial_pair_fold(blocks,i,j,fold)
        groups=[set(i[split==s])|set(j[split==s]) for s in [0,1,2]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        assert len(groups[2])==10 and (split==-1).any()

def test_portable_prediction_matches_sklearn():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,6));y=(x[:,0]>0).astype(int)
    scale=StandardScaler().fit(x[:70]);z=scale.transform(x)/np.sqrt(6)
    rff=RBFSampler(gamma=.5,n_components=128,random_state=8).fit(z[:70]);features=rff.transform(z)
    model=LogisticRegression().fit(features[:70],y[:70])
    state={'mean':scale.mean_,'scale':scale.scale_,'n_features':6,'rff_weights':rff.random_weights_,'rff_offset':rff.random_offset_,'coef':model.coef_.ravel(),'intercept':model.intercept_}
    np.testing.assert_allclose(runner.predict(state,x[70:]),model.predict_proba(features[70:])[:,1],rtol=1e-12,atol=1e-12)

@pytest.fixture
def output():
    if not (OUT/'audit/MODELS_STATUS.json').exists():pytest.skip('bilateral real-data run not available')
    return OUT

def test_real_complete_parent_and_nonshared_cells(output):
    cohorts=[]
    for side,n in [('left',148),('right',133)]:
        p=pd.read_parquet(output/f'data/{side}_pairs.parquet')
        assert len(p)==n*(n-1) and p.pre_id.nunique()==n
        cohorts.append(set(p.pre_id));testcells=[]
        for fold in range(5):
            col=f'spatial_fold_{fold}';groups=[set(p.loc[p[col].eq(s),'pre_id'])|set(p.loc[p[col].eq(s),'post_id']) for s in [0,1,2]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            testcells.extend(groups[2])
        assert len(testcells)==len(set(testcells))==n
    assert not cohorts[0]&cohorts[1]

def test_real_budget_and_source_validation_only(output):
    m=pd.read_csv(output/'models/manifest.csv');t=pd.read_csv(output/'analysis/tuning.csv');scores=pd.read_csv(output/'analysis/model_comparison.csv')
    assert len(m)==108 and t.groupby('model_id').size().eq(4).all()
    assert scores.nominal_parameters.eq(129).all()
    assert (t.n_iter<1000).all()
    transfer=scores[scores.evaluation.eq('cross_side')]
    assert transfer.source_side.ne(transfer.target_side).all()
    assert len(transfer)==18

def test_saved_real_models_reproduce_scores(output):
    pred=pd.read_parquet(output/'data/test_predictions.parquet')
    manifests=pd.read_csv(output/'models/manifest.csv')
    for source in ['left','right']:
        row=manifests[(manifests.source_side==source)&(manifests.evaluation=='pair_grouped')&(manifests.model=='M2')].iloc[0]
        target='right' if source=='left' else 'left'
        x=np.load(output/f'data/{target}_designs.npz')['M2']
        state=dict(np.load(output/row.path));actual=runner.predict(state,x)
        expected=pred[(pred.model_id==row.model_id)&pred.evaluation.eq('cross_side')].probability.to_numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
