"""Check an independent compact replay against released tables and weights."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    tables=['analysis/model_comparison.csv','analysis/tuning.csv','analysis/spatial_separation_audit.csv','registry/spatial_blocks.csv','analysis/paired_geometry_contrasts.csv','analysis/fold_contrasts.csv']
    for name in tables:
        pd.testing.assert_frame_equal(pd.read_csv(a.reference/name),pd.read_csv(a.replay/name),rtol=1e-12,atol=1e-12)
    original=pd.read_parquet(a.reference/'data/test_predictions.parquet');repeat=pd.read_parquet(a.replay/'data/test_predictions.parquet')
    pd.testing.assert_frame_equal(original,repeat,rtol=1e-12,atol=1e-12)
    assert original.pre_id.map(lambda v:isinstance(v,str)).all()
    for side in ['left','right']:
        pd.testing.assert_frame_equal(pd.read_parquet(a.reference/f'data/{side}_pairs.parquet'),pd.read_parquet(a.replay/f'data/{side}_pairs.parquet'),check_exact=True)
    inventory=pd.read_csv(a.reference/'models/manifest.csv')
    for name in inventory.path:
        with np.load(a.reference/name) as x,np.load(a.replay/name) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:np.testing.assert_allclose(x[key],y[key],rtol=1e-12,atol=1e-12)
    result={'passed':True,'scope':'Independent-process compact model refit/evaluation; original raw-source and geometry audits NOT rerun','prediction_rows':len(original),'models_compared':len(inventory),'max_absolute_prediction_difference':float(np.max(np.abs(original.probability-repeat.probability))),'tolerance':1e-12,'exact_ID_and_fold_preservation':True}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
