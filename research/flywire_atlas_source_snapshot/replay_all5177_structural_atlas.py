"""Portable integrity replay for the complete structural atlas."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import pandas as pd

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def replay(bundle):
 bundle=Path(bundle);manifest=json.loads((bundle/'audit/package_hashes.json').read_text());
 for name,value in manifest.items():assert sha(bundle/name)==value,name
 checks=json.loads((bundle/'validation/build_checks.json').read_text());assert checks['status']=='PASS';assert checks['n_all_KC']==5177;assert checks['n_stable_members']==965;assert checks['n_original_stable_members_added_to_detailed_atlas']==261
 entries=pd.read_csv(bundle/'registry/entries.csv');roles=pd.read_parquet(bundle/'registry/entry_roles.parquet');assert len(entries)==17 and roles.root_id.nunique()==2990
 ep=pd.read_parquet(bundle/'data/endpoints.parquet');syn=pd.read_parquet(bundle/'data/synapse_pairs.parquet');pairs=pd.read_parquet(bundle/'data/directed_pairs.parquet')
 assert len(ep)==2261016 and len(syn)==1881678 and len(pairs)==888641;assert ep.synapse_id.nunique()==len(syn);assert pairs.synapse_pair_count.sum()==len(syn)
 return dict(status='PASS',hashes_verified=len(manifest),entries=len(entries),named_context_roots=int(roles.root_id.nunique()),synapse_pairs=len(syn),endpoint_records=len(ep),directed_pairs=len(pairs),scientific_payload_replayed=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--save',type=Path);a=p.parse_args();r=replay(a.bundle);print(json.dumps(r,indent=2));
 if a.save:a.save.write_text(json.dumps(r,indent=2))
