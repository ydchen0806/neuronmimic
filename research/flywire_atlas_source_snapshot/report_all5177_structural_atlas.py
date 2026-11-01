"""Render a full 5,177-KC structural entry atlas from frozen real EM outputs."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from analyze_kc_tree_routes import sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wu_20261030_all5177_structural_atlas'
COLORS={'parent_hit':'#007c91','parent_residual':'#c87552','mixed_member':'#9267a4','opposite_parent':'#87929b'}
STATUS_COLORS={'stable_main':'#007c91','coverage_fail':'#d28b55','tentative':'#9696a2','unresolved':'#c8cbd0'}


def scale(ax):
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();span=x1-x0
    length=50 if span>200 else 10 if span>40 else 2
    x=x0+.06*span;y=y0+.05*(y1-y0);ax.plot([x,x+length],[y,y],color='#333',lw=2);ax.text(x+length/2,y+.02*(y1-y0),f'{length:g} µm',ha='center',fontsize=6)


def line_segments(v,e,axes=(0,2)):
    return v[e][:,:,list(axes)]


def report(out):
    plt.rcParams.update({'font.size':8,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    meta=pd.read_parquet(out/'registry/all5177_cells.parquet');roles=pd.read_parquet(out/'registry/entry_roles.parquet');entries=pd.read_csv(out/'registry/entries.csv');evidence=pd.read_parquet(out/'registry/frozen_all5177_evidence.parquet');ep=pd.read_parquet(out/'data/endpoints.parquet');br=pd.read_parquet(out/'data/branches.parquet');profiles=pd.read_parquet(out/'analysis/branch_contact_profiles.parquet');diag=pd.read_csv(out/'analysis/parent_layer_diagnostics.csv');conn=pd.read_csv(out/'analysis/directed_role_connections.csv');routes=pd.read_csv(out/'analysis/representative_routes.csv');reps=pd.read_csv(out/'registry/representatives.csv',dtype={'root_id':str}).set_index('entry_id');bex=pd.read_csv(out/'registry/branch_examples.csv',dtype={'root_id':str}).set_index('entry_id');paths=json.loads((out/'data/representative_paths.json').read_text());paths={(x['entry_id'],x.get('family','')):x for x in paths}
    cells=set(meta.index);meta.index=meta.index.astype(str);groups=dict(tuple(ep.groupby('root_id')));brgroups=dict(tuple(br.groupby('root_id')))
    pdf=PdfPages(out/'figures/all5177_structural_atlas.pdf');ledger=[];page=0
    def save(fig,name,source,metric,n):
        nonlocal page
        page+=1;fig.text(.025,.015,f'FlyWire v783 | kc_all5177_structural_atlas_v1_20261030 | run all5177_structural_atlas_20261030 | DESCRIPTIVE_ONLY EM anatomy\nSource: {source} | n={n} | no physiological propagation or independent animal replication',fontsize=5.8,color='#555',va='bottom')
        for ext in ['pdf','svg','png']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=100 if ext=='png' else 170)
        pdf.savefig(fig,dpi=170);plt.close(fig);ledger.append(dict(page=page,figure=name,source_file=source,dataset_version='FlyWire v783',protocol='kc_all5177_structural_atlas_v1_20261030',run_id='all5177_structural_atlas_20261030',n_denominator=n,metric_definition=metric,evidence_tier='DESCRIPTIVE_ONLY',simulation_vs_biology_label='EM_STRUCTURE_NOT_DYNAMICS'))
    # Full-population overview.
    fig,ax=plt.subplots(1,3,figsize=(16,6));fig.subplots_adjust(left=.10,right=.97,bottom=.22,top=.83,wspace=.38);fig.suptitle('The complete 5,177-KC registry retains stable, tentative, unresolved and coverage-fail states',fontsize=14)
    counts=evidence.evidence_level.value_counts().reindex(['stable_main','coverage_fail','tentative','unresolved']).fillna(0);ax[0].bar(counts.index,counts.values,color=[STATUS_COLORS[x] for x in counts.index]);ax[0].set_ylabel('KC count');ax[0].tick_params(axis='x',rotation=25)
    for i,v in enumerate(counts):ax[0].text(i,v+50,f'{int(v)}\n{v/len(evidence):.1%}',ha='center',fontsize=8)
    ax[1].scatter(meta.soma_x_um,meta.soma_z_um,s=.35,color='#d6d9dc',rasterized=True);stable=meta[meta.legacy_evidence_level=='stable_main'];ax[1].scatter(stable.soma_x_um,stable.soma_z_um,s=2,color='#007c91',alpha=.7,rasterized=True);ax[1].set(xlabel='X (µm)',ylabel='Z (µm)',title='All KC soma positions; stable8 overlay');ax[1].set_aspect('equal');scale(ax[1])
    atlas_counts=entries.entry_kind.value_counts();ax[2].bar(['stable8 views','consensus views'],[int((entries.entry_kind=='FROZEN_STABLE8').sum()),int((entries.entry_kind=='CROSS_METHOD_CONSENSUS').sum())],color=['#007c91','#9168a4']);ax[2].set_ylabel('Entry pages');ax[2].set_ylim(0,10);ax[2].text(0,8.4,'8 frozen candidates\n965 KC',ha='center');ax[2].text(1,8.4,'9 prior consensus views\nnot new stable modules',ha='center')
    fig.text(.10,.10,'All 5,177 KC retain original evidence state. The 17 atlas views are representations of frozen sets, not 17 promotions.\nDetailed skeleton/contact data cover all 5,177; named entry roles cover 2,990 cells and leave 2,187 cells in the registry without a named entry.',fontsize=8)
    save(fig,'fig01_full_population','registry/frozen_all5177_evidence.parquet;registry/all5177_cells.parquet;registry/entries.csv','Evidence state counts and soma coordinates; atlas-view counts are bookkeeping, not module counts','5,177 KC; 965 stable; 4,212 non-stable')
    # Parent mapping overview.
    fig,ax=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(left=.26,right=.97,bottom=.18,top=.84,wspace=.4);fig.suptitle('Each entry is reported against its complete coarse parent and, where available, a mapped fine type',fontsize=14)
    order=entries.sort_values(['entry_kind','parent_type','side']).entry_id.tolist();y=np.arange(len(order));z=entries.set_index('entry_id').loc[order].reset_index()
    ax[0].barh(y,z.parent_n,color='#d4d9dc',label='coarse parent');ax[0].barh(y,z.parent_hit_n,color='#007c91',label='candidate parent-hit');ax[0].barh(y,z.mapped_parent_n.fillna(0),color='#9168a4',alpha=.7,label='mapped fine parent');ax[0].set(yticks=y,yticklabels=order,xlabel='Cells');ax[0].legend(fontsize=7,loc='lower right');ax[0].invert_yaxis()
    ax[1].scatter(z.parent_coverage,z.purity,s=35,c=np.where(z.entry_kind.eq('FROZEN_STABLE8'),'#007c91','#9168a4'));ax[1].axvline(.5,color='#bbb',lw=.8);ax[1].set(xlabel='Candidate / coarse parent',ylabel='Parent-hit / candidate',title='Coverage and purity');
    for i,r in enumerate(z.itertuples()):ax[1].text(r.parent_coverage+.01,r.purity, r.entry_id.replace('J25_K30_',''),fontsize=6)
    fig.text(.26,.08,'CM016 and CM022 are 147/310 and 114/311 within the mapped KCab-s parent, while their coarse KCab coverage is 147/799 and 114/844.\nThis distinction prevents a known fine type from being mislabeled as a new subtype. Residuals remain explicit.',fontsize=8)
    save(fig,'fig02_parent_hierarchy','registry/entries.csv;registry/entry_roles.parquet','Candidate/coarse and candidate/mapped-type denominators; no threshold-based promotion','17 frozen views; 2,990 named-context cells')
    # diagnostics.
    q=diag[diag.parent_layer.eq('coarse')].copy();fig,ax=plt.subplots(1,3,figsize=(16,6));fig.subplots_adjust(left=.10,right=.97,bottom=.20,top=.83,wspace=.38);fig.suptitle('Candidate–residual structure must be read with both geometry and directed connectivity',fontsize=14)
    col=np.where(q.entry_id.isin(entries.entry_id[entries.entry_kind.eq('FROZEN_STABLE8')]),'#007c91','#9168a4');ax[0].scatter(q.candidate_soma_pair_distance_median_um,q.residual_soma_pair_distance_median_um,c=col,s=30);ax[0].plot([0,100],[0,100],'--',color='#aaa');ax[0].set(xlabel='Candidate soma pair distance median (µm)',ylabel='Residual soma pair distance median (µm)',title='Within-parent geometry')
    ax[1].scatter(q.candidate_density,q.residual_density,c=col,s=30);ax[1].plot([0,.8],[0,.8],'--',color='#aaa');ax[1].set(xlabel='Candidate directed pair density',ylabel='Residual directed pair density',title='Within-parent KC→KC density')
    ax[2].scatter(q.weighted_cut,q.conductance,c=col,s=30);ax[2].set(xlabel='Candidate↔residual weighted cut',ylabel='Conductance',title='Boundary scale and conductance');
    for i,r in q.iterrows():ax[2].text(r.weighted_cut,r.conductance,r.entry_id.replace('J25_K30_',''),fontsize=6)
    fig.text(.10,.08,'Points are descriptive entry-level summaries; no new null model or significance threshold is used. Large coarse residuals of CM016/CM022 are not comparable to near-complete parent cores.\nConductance uses directed synapse counts symmetrized by addition and excludes autapses; it is not a biological boundary probability.',fontsize=8)
    save(fig,'fig03_candidate_residual_diagnostics','analysis/parent_layer_diagnostics.csv','Soma pair-distance quantiles, directed KC pair densities and weighted cut/conductance','17 coarse entries; candidate/residual roles')
    # Connections summary.
    mat=conn[(conn.pre_role.isin(['parent_hit','parent_residual','mixed_member']))&(conn.post_role.isin(['parent_hit','parent_residual','mixed_member']))].groupby(['pre_role','post_role']).synapse_pairs.sum().unstack(fill_value=0).reindex(index=['parent_hit','parent_residual','mixed_member'],columns=['parent_hit','parent_residual','mixed_member'],fill_value=0)
    fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(np.log10(mat.to_numpy()+1),cmap='viridis');ax.set(xticks=range(3),xticklabels=mat.columns,yticks=range(3),yticklabels=mat.index,xlabel='Post role',ylabel='Pre role',title='Aggregated within-entry KC directed synapse counts\n(log10[count + 1], summed across 17 views)');fig.colorbar(im,ax=ax,label='log10 synapse pairs + 1')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(int(mat.iloc[i,j])),ha='center',va='center',color='white' if im.norm(mat.iloc[i,j])>.5 else 'black',fontsize=9)
    fig.text(.10,.07,'Role matrices are contextual summaries and may count a root in multiple overlapping entry views. Raw directed pairs and synapse records remain in data/directed_pairs.parquet and data/endpoints.parquet.',fontsize=8)
    save(fig,'fig04_role_connections','analysis/directed_role_connections.csv','Directed synapse-pair counts by role; overlapping atlas views retained','17 entry views; 888,641 directed pairs globally')
    # Each entry.
    xyz=['soma_x_um','soma_y_um','soma_z_um'];limits={d:(meta[d].min()-10,meta[d].max()+10) for d in xyz};soma=meta
    for erow in entries.itertuples():
        eid=erow.entry_id;er=roles[roles.entry_id.eq(eid)].set_index('root_id');ids=set(er.index);root=str(reps.loc[eid,'root_id']);z=groups[root];g=json.loads((out/'geometry'/f'{root}.json').read_text());v=np.asarray(g['vertices']);e=np.asarray(g['edges'],int);bp=np.asarray(g['branch_per_edge'],int);branch=int(bex.loc[eid,'branch_id']);be=brgroups[root];
        fig,ax=plt.subplots(3,3,figsize=(16,13));fig.subplots_adjust(left=.055,right=.98,bottom=.09,top=.89,wspace=.34,hspace=.52);fig.suptitle(f'{eid} | {erow.entry_kind} | {erow.evidence_status}\n{erow.side} {erow.parent_type}; candidate {erow.member_n} | coarse parent {erow.parent_hit_n}/{erow.parent_n} | mapped parent {erow.mapped_parent_n if pd.notna(erow.mapped_parent_n) else "NA"}',fontsize=14)
        for a,(i,j) in zip(ax[0],[(0,1),(0,2),(1,2)]):
            a.scatter(soma[xyz[i]],soma[xyz[j]],s=.22,color='#dedfe0',alpha=.45,linewidths=0,rasterized=True)
            for role,color in COLORS.items():
                r=er[er.role.eq(role)];
                if len(r):a.scatter(soma.loc[r.index,xyz[i]],soma.loc[r.index,xyz[j]],s=8,color=color,label=f'{role} n={len(r)}',linewidths=0,alpha=.9,rasterized=True)
            a.scatter([soma.loc[root,xyz[i]]],[soma.loc[root,xyz[j]]],s=45,facecolors='none',edgecolors='black');a.set(xlim=limits[xyz[i]],ylim=limits[xyz[j]],xlabel='XYZ'[i]+' (µm)',ylabel='XYZ'[j]+' (µm)',title='All 5,177 KC somata');a.set_aspect('equal');scale(a)
        ax[0,0].legend(fontsize=6,loc='upper left')
        # Complete coarse parent skeleton in representative plane.
        parent_ids=er.index[er.role.isin(['parent_hit','parent_residual'])];parts=[]
        for rid in parent_ids:
            gg=json.loads((out/'geometry'/f'{rid}.json').read_text());parts.append(line_segments(np.asarray(gg['vertices']),np.asarray(gg['edges'])))
        a=ax[1,0];a.add_collection(LineCollection(np.concatenate(parts),colors='#c4cbd0',lw=.12,alpha=.20,rasterized=True));a.add_collection(LineCollection(line_segments(v,e),colors='#007c91',lw=.55,alpha=.85,rasterized=True));a.autoscale();a.set_aspect('equal');a.set(xlabel='X (µm)',ylabel='Z (µm)',title=f'Complete same-side parent\nrepresentative root {root}');scale(a)
        # Contacts and branch.
        a=ax[1,1];a.add_collection(LineCollection(line_segments(v,e),colors='#c1c7cc',lw=.35,alpha=.7,rasterized=True));bpseg=z
        for mask,color,label in [(z.direction.eq('in'),'#2779b6','input'),(z.direction.eq('out'),'#da812f','output')]:
            pz=z[mask];a.scatter(pz.x_um,pz.z_um,s=5,color=color,label=f'{label} n={len(pz)}',alpha=.7,linewidths=0,rasterized=True)
        seg=v[e[bp==branch]][:,:,[0,2]];a.add_collection(LineCollection(seg,colors='#8f5ba3',lw=2));a.autoscale();a.set_aspect('equal');a.set(xlabel='X (µm)',ylabel='Z (µm)',title=f'Contacts + frozen/display branch {branch}');scale(a);a.legend(fontsize=6)
        # Entry network matrix from role table.
        c=conn[conn.entry_id.eq(eid)].pivot_table(index='pre_role',columns='post_role',values='synapse_pairs',aggfunc='sum',fill_value=0).reindex(index=['parent_hit','parent_residual','mixed_member','opposite_parent'],columns=['parent_hit','parent_residual','mixed_member','opposite_parent'],fill_value=0)
        a=ax[1,2];im=a.imshow(np.log10(c.to_numpy()+1),cmap='magma');a.set(xticks=range(len(c.columns)),xticklabels=c.columns,yticks=range(len(c.index)),yticklabels=c.index,title='Entry directed synapses\nlog10(count+1)',xlabel='post role',ylabel='pre role');
        for i in range(len(c.index)):
            for j in range(len(c.columns)):a.text(j,i,str(int(c.iloc[i,j])),ha='center',va='center',fontsize=7,color='white' if im.norm(c.iloc[i,j])>.5 else 'black')
        # structural profile distributions.
        for a,metric,title in [(ax[2,0],'n_input_endpoints','Input contacts / branch'),(ax[2,1],'n_output_endpoints','Output contacts / branch')]:
            for role,color in COLORS.items():
                roots=er.index[er.role.eq(role)];vals=profiles[profiles.root_id.isin(roots)].groupby('root_id')[metric].sum();
                if len(vals):a.scatter(np.full(len(vals),list(COLORS).index(role)),vals,s=8,color=color,alpha=.35);a.plot([list(COLORS).index(role)]*2,np.quantile(vals,[.25,.75]),color='black',lw=2);a.scatter([list(COLORS).index(role)],[np.median(vals)],color='black',s=18)
            a.set(xticks=range(4),xticklabels=['hit','residual','mixed','opposite'],ylabel=title,title='Cell distribution');a.grid(axis='y',alpha=.15)
        a=ax[2,2];a.axis('off');route=routes[routes.entry_id.eq(eid)].iloc[0];lines=[f'Entry status: {erow.evidence_status}',f'Coarse parent: {erow.parent_hit_n}/{erow.parent_n}',f'Coarse residual: {erow.residual_n}',f'Mixed: {erow.mixed_n}',f'Fine parent label: {erow.mapped_type or "multiple/NA"}',f'Fine parent coverage: {erow.mapped_parent_coverage if pd.notna(erow.mapped_parent_coverage) else "NA"}',f'Example route: {route.status}']
        if route.status=='AVAILABLE':lines.extend([f'{route.family}: {route.upstream_type} → KC → {route.downstream_type}',f'input/output pairs: {route.input_synapse_pairs}/{route.output_synapse_pairs}',f'cable median: {route.median_cable_um:.1f} µm',f'example path: {route.example_path_cable_um:.1f} µm'])
        lines.extend(['','IDs, endpoints, branches and parent roles:','registry/, data/, geometry/','No new cluster or function claim.'])
        a.text(0,1,'\n'.join(lines),va='top',fontsize=7,linespacing=1.5)
        save(fig,f'entry_{eid}','registry/entry_roles.parquet;registry/all5177_cells.parquet;data/endpoints.parquet;analysis/directed_role_connections.csv;geometry/<root>.json','Full parent/residual/opposite/mixed soma context, real skeleton/contact positions, directed role matrix and branch contact profiles',f'candidate={erow.member_n};parent={erow.parent_n};residual={erow.residual_n};full KC=5177')
    pdf.close();pd.DataFrame(ledger).to_csv(out/'report/figure_source_ledger.csv',index=False);print(json.dumps(dict(status='PASS',pages=len(ledger),pdf=str(out/'figures/all5177_structural_atlas.pdf')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();report(a.out)
