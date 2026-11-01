"""Descriptive candidate versus fine-parent residual profiles for CM016/CM022.
No p values, threshold changes, clustering or promotion.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
TARGET=['CM016','CM022']

def run(out=OUT):
 out=Path(out);ep=pd.read_parquet(out/'data/endpoints.parquet');roles=pd.read_parquet(out/'registry/entry_roles.parquet');meta=pd.read_parquet(out/'registry/all5177_cells.parquet');bp=pd.read_parquet(out/'analysis/branch_contact_profiles.parquet');pairs=pd.read_parquet(out/'data/directed_pairs.parquet')
 summary=[];features=[];partner=[];overlap=[];entry_table=pd.read_csv(out/'registry/entries.csv').set_index('entry_id')
 for eid in TARGET:
  r=roles[roles.entry_id.eq(eid)].set_index('root_id');fine=set(r.index[r.within_mapped_parent]);cand=set(r.index[r.role.eq('parent_hit')]);res=fine-cand;assert len(cand)+len(res)==len(fine)
  z=ep[ep.root_id.isin(fine)].copy();z['role']=np.where(z.root_id.isin(cand),'candidate','fine_residual')
  for root,g in z.groupby('root_id'):
   typ='candidate' if root in cand else 'fine_residual';in0=g[g.direction.eq('in')];out0=g[g.direction.eq('out')];allb=bp[bp.root_id.eq(root)]
   row=dict(entry_id=eid,root_id=root,role=typ,n_input=len(in0),n_output=len(out0),n_endpoints=len(g),n_input_branches=in0.branch_id.nunique(),n_output_branches=out0.branch_id.nunique(),n_mixed_branches=int(set(in0.branch_id)&set(out0.branch_id).__len__() if False else len(set(in0.branch_id)&set(out0.branch_id))),n_total_branches=len(allb),occupied_branches=int(allb.occupied.sum()),mixed_fraction=float(allb.loc[allb.occupied,'mixed'].mean()) if allb.occupied.any() else np.nan,input_mapping_error_median_um=in0.mapping_error_um.median(),output_mapping_error_median_um=out0.mapping_error_um.median())
   for fam in ['ALPN','VISUAL_PROJECTION','APL','OTHER']:
    row['input_'+fam.lower()+'_fraction']=float((in0.input_family.eq(fam)).mean()) if len(in0) else np.nan
   row['input_unknown_annotation_fraction']=float(in0.partner_cell_class.eq('').mean()) if len(in0) else np.nan
   for cls in ['MBON','DAN','MBIN','Kenyon_Cell']:
    row['output_'+cls.lower()+'_fraction']=float((out0.partner_cell_class.eq(cls)).mean()) if len(out0) else np.nan
   features.append(row)
  for direction,own in [('in','pre_id'),('out','post_id')]:
   q=z[z.direction.eq(direction)].groupby(['role','partner_id','partner_super_class','partner_cell_class','partner_cell_type']).size().rename('synapse_pairs').reset_index()
   totals=q.groupby('role').synapse_pairs.sum().rename('role_total');q=q.join(totals,on='role');q['fraction_of_role_endpoints']=q.synapse_pairs/q.role_total;q['direction']=direction;partner.append(q.assign(entry_id=eid))
  for direction in ['in','out']:
   q=z[z.direction.eq(direction)];a=set(q.loc[q.role.eq('candidate'),'partner_id']);b=set(q.loc[q.role.eq('fine_residual'),'partner_id']);overlap.append(dict(entry_id=eid,direction=direction,candidate_partner_n=len(a),residual_partner_n=len(b),intersection_n=len(a&b),jaccard=len(a&b)/len(a|b) if a|b else np.nan))
  # Entry-level structural summary with coarse comparison from stored diagnostics.
  for role,ids in [('candidate',cand),('fine_residual',res)]:
   q=pd.DataFrame([x for x in features if x['entry_id']==eid and x['role']==role]);
   row=dict(entry_id=eid,role=role,n_cells=len(ids),fine_parent_n=len(fine),coarse_parent_n=int(entry_table.loc[eid,'parent_n']),coarse_candidate_n=int(entry_table.loc[eid,'parent_hit_n']))
   for col in q.columns:
    if col in ['entry_id','root_id','role']:continue
    if pd.api.types.is_numeric_dtype(q[col]):
     for quant,label in [(.25,'q25'),(.5,'median'),(.75,'q75')]:row[col+'_'+label]=q[col].quantile(quant)
   summary.append(row)
 pd.DataFrame(features).to_csv(out/'analysis/fine_parent_role_cell_profiles.csv',index=False);pd.DataFrame(summary).to_csv(out/'analysis/fine_parent_role_summary.csv',index=False);pd.concat(partner,ignore_index=True).to_parquet(out/'analysis/fine_parent_role_partner_fingerprints.parquet',index=False);pd.DataFrame(overlap).to_csv(out/'analysis/fine_parent_role_partner_overlap.csv',index=False)
 result=dict(status='PASS',entries=TARGET,n_cells_profiles=len(features),partner_rows=sum(len(x) for x in partner),all_fine_parent_members=sum(len(set(roles[roles.entry_id.eq(e)].loc[roles[roles.entry_id.eq(e)].within_mapped_parent,'root_id'])) for e in TARGET),new_stable_promotions=0)
 (out/'validation/fine_parent_role_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
