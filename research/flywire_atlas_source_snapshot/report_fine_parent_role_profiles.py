"""Plot fine-parent candidate/residual profiles for the two coarse KCab stable candidates."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
OUT=Path('artifacts/wu_20261030_all5177_structural_atlas')

def report(out=OUT):
 out=Path(out);s=pd.read_csv(out/'analysis/fine_parent_role_summary.csv');over=pd.read_csv(out/'analysis/fine_parent_role_partner_overlap.csv');fig,ax=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(left=.12,right=.96,bottom=.18,top=.86,wspace=.32,hspace=.43);fig.suptitle('CM016/CM022 candidate and fine-parent residual show similar aggregate input/output classes',fontsize=14)
 labels=['ALPN','VISUAL_PROJECTION','APL','unknown_annotation'];cols=['input_alpn_fraction_median','input_visual_projection_fraction_median','input_apl_fraction_median','input_unknown_annotation_fraction_median'];x=np.arange(len(labels));width=.18
 for i,eid in enumerate(['CM016','CM022']):
  for j,role in enumerate(['candidate','fine_residual']):
   q=s[(s.entry_id==eid)&(s.role==role)].iloc[0];vals=[q[c] for c in cols];ax[0,0].bar(x+(i*2+j)*width,vals,width,label=f'{eid} {"candidate" if role=="candidate" else "residual"}',color=['#007c91','#c87552','#1b9e77','#9168a4'][i*2+j],alpha=.9)
 ax[0,0].set(xticks=x+1.5*width,xticklabels=labels,ylabel='Median fraction of incoming endpoints',title='Input identities per cell');ax[0,0].legend(fontsize=7);ax[0,0].grid(axis='y',alpha=.15)
 labels=['MBON','DAN','MBIN','Kenyon_Cell'];cols=['output_mbon_fraction_median','output_dan_fraction_median','output_mbin_fraction_median','output_kenyon_cell_fraction_median'];x=np.arange(len(labels))
 for i,eid in enumerate(['CM016','CM022']):
  for j,role in enumerate(['candidate','fine_residual']):
   q=s[(s.entry_id==eid)&(s.role==role)].iloc[0];ax[0,1].bar(x+(i*2+j)*width,[q[c] for c in cols],width,color=['#007c91','#c87552','#1b9e77','#9168a4'][i*2+j],alpha=.9,label=f'{eid} {"candidate" if role=="candidate" else "residual"}')
 ax[0,1].set(xticks=x+1.5*width,xticklabels=labels,ylabel='Median fraction of outgoing endpoints',title='Output classes per cell');ax[0,1].grid(axis='y',alpha=.15)
 x=np.arange(4);groups=[];vals=[]
 for eid in ['CM016','CM022']:
  for direction in ['in','out']:
   q=over[(over.entry_id==eid)&(over.direction==direction)].iloc[0];groups.append(f'{eid}\n{direction}');vals.append(q.jaccard)
 ax[1,0].bar(x,vals,color=['#007c91','#007c91','#9168a4','#9168a4']);ax[1,0].set(xticks=x,xticklabels=groups,ylabel='Jaccard of partner-root sets',title='Candidate vs fine residual partner overlap');ax[1,0].set_ylim(0,1);ax[1,0].grid(axis='y',alpha=.15)
 # Branch/contact distributions: medians with IQR for each role.
 colors={'candidate':'#007c91','fine_residual':'#c87552'};positions=[];labels=[]
 for i,eid in enumerate(['CM016','CM022']):
  for j,role in enumerate(['candidate','fine_residual']):
   q=s[(s.entry_id==eid)&(s.role==role)].iloc[0];pos=i*3+j;positions.append(pos);labels.append(f'{eid}\n'+('candidate' if role=='candidate' else 'residual'));ax[1,1].errorbar(pos,q.n_mixed_branches_median,yerr=[[q.n_mixed_branches_median-q.n_mixed_branches_q25],[q.n_mixed_branches_q75-q.n_mixed_branches_median]],fmt='o',color=colors[role],capsize=4);ax[1,1].errorbar(pos+.12,q.n_input_median,yerr=[[q.n_input_median-q.n_input_q25],[q.n_input_q75-q.n_input_median]],fmt='s',color='#444',capsize=4)
 ax[1,1].set(xticks=positions,xticklabels=labels,ylabel='Median ± cell IQR',title='Mixed branches (circles) and input endpoints (squares)');ax[1,1].grid(axis='y',alpha=.15)
 fig.text(.12,.08,'All summaries use the complete mapped fine parent: CM016 147 candidate / 163 residual; CM022 114 candidate / 197 residual.\nNo partner-count matching, null test or binary subtype threshold was added. Similar aggregate class fractions are descriptive evidence against a simple coarse class separation; low partner-set Jaccard motivates a pre-registered held-out partner test.',fontsize=8)
 fig.savefig(out/'figures/fine_parent_role_profiles.pdf');fig.savefig(out/'figures/fine_parent_role_profiles.svg');fig.savefig(out/'figures/fine_parent_role_profiles.png',dpi=150);print(out/'figures/fine_parent_role_profiles.pdf')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();report(a.out)
