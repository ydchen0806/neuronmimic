"""List groups or export every physical contact and skeleton of one group/cell."""
from pathlib import Path
import argparse,json
from neuronmimic.structure_bundle import DATASETS,groups,export
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--list-groups',action='store_true');q=p.add_mutually_exclusive_group();q.add_argument('--group');q.add_argument('--neuron');p.add_argument('--output',type=Path);p.add_argument('--no-geometry',action='store_true');a=p.parse_args()
if a.list_groups:print(groups(a.bundle,a.dataset).to_string(index=False))
else:
    if a.output is None or (a.group is None and a.neuron is None):p.error('Export requires --output and --group or --neuron')
    print(json.dumps(export(a.bundle,a.dataset,a.output,a.group,a.neuron,not a.no_geometry),indent=2))
