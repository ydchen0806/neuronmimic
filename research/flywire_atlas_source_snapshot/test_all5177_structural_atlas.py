"""Real atlas invariants; no fixture data is mixed into scientific outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from query_all5177_entry import query
OUT=Path('artifacts/wu_20261030_all5177_structural_atlas')

def require_real():
    if not (OUT/'validation/build_checks.json').exists():pytest.skip('Full atlas not built')

def test_full_population_and_historical_states_are_frozen():
    require_real();c=json.loads((OUT/'validation/build_checks.json').read_text());assert c['status']=='PASS';assert c['n_all_KC']==5177;assert c['n_stable_members']==965;assert c['n_raw_synapse_pairs']==1881678;assert c['n_endpoints']==2261016;assert c['n_branches']==1252322;assert c['n_original_stable_members_added_to_detailed_atlas']==261;assert c['new_stable_promotions']==0
    e=pd.read_csv(OUT/'registry/entries.csv');assert len(e)==17;assert e.entry_id.is_unique;assert e[e.entry_id=='CM016'].mapped_parent_n.iloc[0]==310;assert e[e.entry_id=='CM022'].mapped_parent_n.iloc[0]==311
    atlas=pd.read_parquet(OUT/'registry/frozen_all5177_evidence.parquet');assert atlas.evidence_level.value_counts().to_dict()=={'tentative':2207,'unresolved':1661,'stable_main':965,'coverage_fail':344}

def test_all_geometry_and_contact_lineage_are_conserved():
    require_real();g=pd.read_csv(OUT/'validation/all5177_geometry_checks.csv');assert len(g)==5177;assert g.connected_tree.all();assert g.max_projection_error_difference_um.max()<1e-7;assert g.max_raw_branch_length_difference_um.max()<1e-7
    ep=pd.read_parquet(OUT/'data/endpoints.parquet');syn=pd.read_parquet(OUT/'data/synapse_pairs.parquet');br=pd.read_parquet(OUT/'data/branches.parquet');assert ep.synapse_id.nunique()==syn.synapse_id.nunique()==1881678;assert len(ep)==2261016;assert len(br)==1252322;assert ep.root_id.nunique()==5177
    assert (ep.loc[ep.direction.eq('in'),'post_id']==ep.loc[ep.direction.eq('in'),'root_id']).all();assert (ep.loc[ep.direction.eq('out'),'pre_id']==ep.loc[ep.direction.eq('out'),'root_id']).all()
    pairs=syn.groupby(['pre_id','post_id']).size();assert pairs.sum()==len(syn);assert pairs.size==888641

def test_roles_partitions_and_connection_conservation():
    require_real();roles=pd.read_parquet(OUT/'registry/entry_roles.parquet');e=pd.read_csv(OUT/'registry/entries.csv');con=pd.read_csv(OUT/'validation/entry_connection_conservation.csv');assert roles.root_id.nunique()==2990;assert roles.entry_id.nunique()==17
    for eid,z in roles.groupby('entry_id'):
        assert not z.duplicated('root_id').any();assert {'parent_hit','opposite_parent'}.issubset(set(z.role));assert set(z.role).issubset({'parent_hit','parent_residual','mixed_member','opposite_parent'})
    assert len(con)==17;assert (con.n_synapse_pairs>=con.n_directed_pairs).all()
    d=pd.read_csv(OUT/'analysis/directed_role_connections.csv');assert d.entry_id.nunique()==17

def test_query_returns_full_entry_and_root_provenance():
    require_real();r=query(OUT,'CM016','720575940626044924');assert r['entry']['entry_id']=='CM016';assert r['cell']['role']=='parent_hit';assert len(r['endpoints'])>0;assert isinstance(r['cell']['root_id'],str)

def test_fine_parent_role_audit_is_descriptive_and_complete():
    require_real();s=pd.read_csv(OUT/'analysis/fine_parent_role_summary.csv');o=pd.read_csv(OUT/'analysis/fine_parent_role_partner_overlap.csv')
    assert set(s.entry_id)=={'CM016','CM022'};assert set(s.role)=={'candidate','fine_residual'};assert s.groupby('entry_id').n_cells.sum().to_dict()=={'CM016':310,'CM022':311}
    assert len(o)==4;assert o.jaccard.between(0,1).all()
    assert json.loads((OUT/'validation/fine_parent_role_checks.json').read_text())['new_stable_promotions']==0
