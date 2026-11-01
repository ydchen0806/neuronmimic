"""Query published KC structures without merging candidate namespaces or brain IDs."""
from pathlib import Path
import hashlib,json,tarfile
import pandas as pd

DATASETS=('flywire_v783','malecns_v1_0')

def base(bundle,dataset):
    if dataset not in DATASETS:raise ValueError(f'Choose dataset from {DATASETS}')
    p=Path(bundle)/dataset
    if not p.is_dir():raise FileNotFoundError(p)
    return p

def groups(bundle,dataset):
    m=pd.read_parquet(base(bundle,dataset)/'registry/group_members.parquet')
    return m.groupby(['namespace','group_id','entry_kind','evidence_status'],dropna=False).root_id.nunique().rename('n_members').reset_index()

def select(bundle,dataset,group=None,neuron=None):
    b=base(bundle,dataset);cells=pd.read_parquet(b/'registry/cells.parquet')
    key='root_id' if dataset=='flywire_v783' else 'bodyId'
    assert cells[key].map(lambda v:isinstance(v,str)).all()
    if neuron is not None:
        if not isinstance(neuron,str):raise TypeError('neuron ID must be a string')
        if neuron not in set(cells[key]):raise ValueError('Neuron not in the declared KC population')
        return cells[cells[key].eq(neuron)],{neuron}
    if group=='ALL_KC':return cells,set(cells[key])
    members=pd.read_parquet(b/'registry/group_members.parquet');m=members[members.group_id.eq(group)]
    if m.empty:raise ValueError(f'Unknown group {group!r}; use --list-groups')
    if m.namespace.nunique()!=1:raise ValueError('Ambiguous group namespace; require a versioned group name')
    ids=set(m.root_id);return cells[cells[key].isin(ids)],ids

def read_geometry(bundle,dataset,neuron):
    if not isinstance(neuron,str):raise TypeError('neuron ID must be a string')
    b=base(bundle,dataset);index=pd.read_parquet(b/'registry/geometry_index.parquet');row=index[index.neuron_id.eq(neuron)]
    if len(row)!=1:raise ValueError('Missing or duplicated geometry identity')
    r=row.iloc[0]
    with tarfile.open(b/r.archive,'r:gz') as tar:
        stream=tar.extractfile(r.member)
        if stream is None:raise ValueError('Geometry archive member missing')
        content=stream.read()
    if hashlib.sha256(content).hexdigest()!=r.sha256:raise ValueError('Geometry SHA256 mismatch')
    return content,r.member

def export(bundle,dataset,output,group=None,neuron=None,geometry=True):
    out=Path(output)
    if out.exists():raise FileExistsError('Choose a new export directory; existing results are not overwritten')
    b=base(bundle,dataset);cells,ids=select(bundle,dataset,group,neuron);pre,post=('pre_id','post_id') if dataset=='flywire_v783' else ('body_pre','body_post')
    syn=pd.read_parquet(b/'data/synapse_pairs.parquet',filters=[[(pre,'in',sorted(ids))],[(post,'in',sorted(ids))]])
    assert syn.synapse_id.map(lambda v:isinstance(v,str)).all() and syn.synapse_id.is_unique
    out.mkdir(parents=True);cells.to_parquet(out/'cells.parquet',index=False);syn.to_parquet(out/'synapse_pairs.parquet',index=False)
    syn.groupby([pre,post]).size().reset_index(name='synapse_pairs').to_parquet(out/'directed_pairs.parquet',index=False)
    memberships=pd.read_parquet(b/'registry/group_members.parquet');memberships[memberships.root_id.isin(ids)].to_parquet(out/'all_membership_versions.parquet',index=False)
    if dataset=='flywire_v783':
        for name in ['endpoints','branches']:
            pd.read_parquet(b/f'data/{name}.parquet',filters=[('root_id','in',sorted(ids))]).to_parquet(out/f'{name}.parquet',index=False)
        # Original annotation table accompanies the exported structure, including unknown partners.
        partners=pd.read_csv(b/'registry/partner_annotations.csv',dtype=str);partners.to_parquet(out/'source_partner_annotations.parquet',index=False)
    else:
        nt=pd.read_parquet(b/'data/presynaptic_site_nt.parquet',filters=[('body','in',sorted(set(syn[pre])))])
        keys=syn[['x_pre','y_pre','z_pre',pre]].drop_duplicates()
        nt=nt.merge(keys,left_on=['x','y','z','body'],right_on=['x_pre','y_pre','z_pre',pre],how='inner',validate='one_to_one').drop(columns=['x_pre','y_pre','z_pre',pre]);nt.to_parquet(out/'presynaptic_site_nt.parquet',index=False)
        partners=pd.read_parquet(b/'registry/partners.parquet');partners[partners.bodyId.isin(set(syn[pre])|set(syn[post]))].to_parquet(out/'partners.parquet',index=False)
    if geometry:
        (out/'geometry').mkdir()
        index=pd.read_parquet(b/'registry/geometry_index.parquet');selected=index[index.neuron_id.isin(ids)]
        if set(selected.neuron_id)!=ids or not selected.neuron_id.is_unique:raise ValueError('Incomplete geometry index')
        for archive,part in selected.groupby('archive'):
            with tarfile.open(b/archive,'r:gz') as tar:
                for row in part.itertuples(index=False):
                    data=tar.extractfile(row.member).read()
                    if hashlib.sha256(data).hexdigest()!=row.sha256:raise ValueError('Geometry SHA256 mismatch')
                    (out/'geometry'/Path(row.member).name).write_bytes(data)
    summary={'dataset':dataset,'group':group,'neuron':neuron,'members':len(ids),'unique_physical_contacts':len(syn),'input_contacts':int(syn[post].isin(ids).sum()),'output_contacts':int(syn[pre].isin(ids).sum()),'internal_contacts':int((syn[pre].isin(ids)&syn[post].isin(ids)).sum()),'external_partner_geometry':'not included; selected KC geometries plus incident partners/annotations','functional_validation':'NOT_ESTABLISHED_BY_STRUCTURE_EXPORT','coordinate_contract':'See original bundle COORDINATES.json; preserve native coordinates'}
    assert summary['input_contacts']+summary['output_contacts']-summary['internal_contacts']==len(syn)
    (out/'QUERY.json').write_text(json.dumps(summary,indent=2));coordinate=Path(bundle)/'COORDINATES.json'
    if coordinate.exists():(out/'COORDINATES.json').write_bytes(coordinate.read_bytes())
    return summary
