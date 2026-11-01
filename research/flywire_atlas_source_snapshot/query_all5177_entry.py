"""Query complete-parent entry roles, real geometry and directed connection summaries."""
from pathlib import Path
import argparse,json
import pandas as pd

def query(bundle,entry_id,root_id=None):
    bundle=Path(bundle);entries=pd.read_csv(bundle/'registry/entries.csv');row=entries[entries.entry_id.eq(entry_id)]
    if row.empty:raise ValueError(f'Unknown entry: {entry_id}')
    roles=pd.read_parquet(bundle/'registry/entry_roles.parquet');r=roles[roles.entry_id.eq(entry_id)]
    conn=pd.read_csv(bundle/'analysis/directed_role_connections.csv');conn=conn[conn.entry_id.eq(entry_id)]
    diag=pd.read_csv(bundle/'analysis/parent_layer_diagnostics.csv');diag=diag[diag.entry_id.eq(entry_id)]
    result=dict(entry=row.iloc[0].to_dict(),role_counts=r.role.value_counts().to_dict(),roles=r.to_dict('records'),directed_role_connections=conn.to_dict('records'),parent_layer_diagnostics=diag.to_dict('records'),scope='Complete 5,177-KC registry; entry role geometry/endpoints available for all 5,177 cells; role views overlap across entries.',evidence_tier='DESCRIPTIVE_ONLY',interpretation='Frozen candidate/status context plus real EM skeleton/contact data; no new cluster/function promotion.')
    if root_id is not None:
        root=str(root_id);z=r[r.root_id.eq(root)]
        if z.empty:raise ValueError(f'Root {root} is not in {entry_id}')
        ep=pd.read_parquet(bundle/'data/endpoints.parquet',filters=[('root_id','==',root)])
        br=pd.read_parquet(bundle/'data/branches.parquet',filters=[('root_id','==',root)])
        input_branches=set(ep.loc[ep.direction.eq('in'),'branch_id'])
        output_branches=set(ep.loc[ep.direction.eq('out'),'branch_id'])
        result['cell']=z.iloc[0].to_dict()
        result['endpoint_counts']=ep.groupby('direction').size().to_dict()
        result['branch_counts']=dict(n_branches=len(br),n_input_branches=len(input_branches),n_output_branches=len(output_branches),n_mixed_branches=len(input_branches & output_branches))
        result['endpoints']=ep.to_dict('records')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--entry',required=True);p.add_argument('--root');p.add_argument('--out',type=Path);a=p.parse_args();r=query(a.bundle,a.entry,a.root);s=json.dumps(r,ensure_ascii=False,indent=2)
    if a.out:a.out.write_text(s)
    else:print(s)
