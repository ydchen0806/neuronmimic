"""Scientific contracts for the executed focused P1D-v3 experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nature_kc_20261030/p1d_v3'

@pytest.fixture
def output():
    if not (OUT/'audit/MODELS_STATUS.json').exists():pytest.skip('real P1D-v3 run not available')
    return OUT

def test_complete_parent_all_positive_and_negative_pairs(output):
    p=pd.read_parquet(output/'data/complete_parent_prediction_pairs.parquet')
    assert len(p)==148*147 and not p.duplicated(['pre_id','post_id']).any()
    assert (p.pre_id!=p.post_id).all() and p.pre_id.nunique()==148
    assert set(p.observed_edge)=={0,1}
    rev=p[['pre_id','post_id','pair_split']].rename(columns={'pre_id':'post_id','post_id':'pre_id','pair_split':'reverse_split'})
    joined=p.merge(rev,on=['pre_id','post_id'],validate='one_to_one')
    assert joined.pair_split.eq(joined.reverse_split).all()

def test_cell_holdout_no_shared_neuron(output):
    p=pd.read_parquet(output/'data/complete_parent_prediction_pairs.parquet')
    sets=[set(g.pre_id)|set(g.post_id) for _,g in p.loc[p.cell_disjoint_split>=0].groupby('cell_disjoint_split')]
    assert len(sets)==3 and not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    m=pd.read_csv(output/'analysis/MODEL_COMPARISON.csv')
    assert not m.loc[m.evaluation.eq('cell_disjoint_geometry'),'model'].str.startswith(('M3','M4')).any()

def test_geometry_is_bounded_and_numerically_checked(output):
    g=pd.read_parquet(output/'data/regional_pair_opportunities.parquet')
    assert len(g)==148*147
    for c in g.filter(like='_gaussian_'):
        assert g[c].dropna().between(0,1).all()
    q=json.loads((output/'validation/regional_geometry_numerics.json').read_text())
    assert q['passed'] and q['max_exact_distance_error_um']<1e-9
    assert json.loads((output/'audit/GEOMETRY_STATUS.json').read_text())['contact_records_used']==0

def test_matching_failure_cannot_produce_significant_result(output):
    r=pd.read_csv(output/'analysis/routing_matched_null_results.csv')
    for _,row in r.iterrows():
        if row.n_balanced_null<100:
            assert pd.isna(row.matched_p) and row.state=='MATCHING_FAILED_NO_INFERENCE'
    frozen=pd.read_parquet(output/'registry/frozen18_with_history.parquet')
    null=pd.read_parquet(output/'registry/matched_null_groups.parquet')
    assert not set(null.root_id)&set(frozen.root_id)
    assert null.groupby('replicate').root_id.nunique().eq(18).all()

def test_malecns_release_scope_identity_is_not_fabricated(output):
    q=json.loads((output/'audit/CROSSBRAIN_SOURCE_QC.json').read_text())
    assert q['release_hashes_match'] and q['exact_original_subset_conflicts']==0
    assert q['unannotated_but_in_original_NT_table']+q['unannotated_and_absent_NT']==q['unannotated_endpoints']
    r=pd.read_parquet(output/'registry/malecns_missing_annotation_identity.parquet')
    assert r.body_id.map(lambda x:isinstance(x,str)).all()
    assert r.annotation_status.eq('not_in_released_body_annotation_table').all()

def test_historical_residual_identity_and_no_promotion(output):
    r=pd.read_parquet(output/'registry/frozen18_with_history.parquet')
    assert r.candidate_id.value_counts().to_dict()=={'CM030':16,'CM019':2}
    status=json.loads((output/'audit/CM019_STATUS.json').read_text())
    assert status['new_stable_units']==0

def test_density_control_retains_pairs_to_declared_tolerance(output):
    d=pd.read_csv(output/'analysis/density_matched_dropout.csv')
    assert (d.directed_pairs-d.target_directed_pairs).abs().le(1).all()
    assert d.status.eq('POSTHOC_EDGE_COUNT_MATCHED_RANDOM_CONTROL').all()

def test_budget_and_all_conditions_retained(output):
    model=pd.read_csv(output/'analysis/MODEL_COMPARISON.csv')
    tuning=pd.read_csv(output/'analysis/model_tuning.csv')
    assert model.n_trainable_coefficients.eq(129).all()
    assert tuning.groupby(['evaluation','model']).size().eq(4).all()
    perturb=pd.read_csv(output/'analysis/CM019_18_perturbations.csv')
    assert len(perturb)==27*3
    assert perturb.jaccard.between(0,1).all() and perturb.retention.between(0,1).all()
