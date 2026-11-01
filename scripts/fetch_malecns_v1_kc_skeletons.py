"""Fetch original v1.0 KC skeletons with server MD5 and source-generation receipts."""
from pathlib import Path
import argparse,base64,hashlib,json,time,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--proxy');p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    ann=pd.read_feather(a.annotations);ids=sorted(ann.loc[ann.type.fillna('').str.startswith('KC'),'bodyId'].astype(str));a.output.mkdir(parents=True,exist_ok=True)
    tls=threading.local()
    def get(rid):
        if not hasattr(tls,'session'):
            tls.session=requests.Session();tls.session.trust_env=False
            if a.proxy:tls.session.proxies={'http':a.proxy,'https':a.proxy}
        url=f'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{rid}.swc';path=a.output/(rid+'.swc');receipt=a.output/(rid+'.source.json')
        if path.exists() and receipt.exists():
            d=json.loads(receipt.read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==d['sha256'];return d
        for attempt in range(4):
            try:
                r=tls.session.get(url,timeout=50);r.raise_for_status();raw=r.content
                parts=[x.strip() for x in r.headers.get('x-goog-hash','').split(',')];expected=next(x[4:] for x in parts if x.startswith('md5='))
                assert base64.b64encode(hashlib.md5(raw).digest()).decode()==expected
                assert raw.startswith(b'#') and b'\n1 ' in raw[:800]
                path.write_bytes(raw);d={'neuron_id':rid,'dataset_version':'MaleCNS v1.0','url':url,'generation':r.headers.get('x-goog-generation'),'bytes':len(raw),'md5_base64':expected,'sha256':hashlib.sha256(raw).hexdigest(),'status':'VERIFIED_ORIGINAL_SWC'};receipt.write_text(json.dumps(d,indent=2));return d
            except Exception as e:
                if attempt==3:return {'neuron_id':rid,'url':url,'status':'BLOCKED_MISSING_DATA','error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}
                time.sleep(attempt+1)
    rows=[];start=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for n,f in enumerate(as_completed([pool.submit(get,r) for r in ids]),1):
            rows.append(f.result())
            if n%200==0:print('SWCs',n,'/',len(ids),'elapsed_s',round(time.time()-start),flush=True)
    rows.sort(key=lambda x:x['neuron_id']);d={'dataset_version':'MaleCNS v1.0','selection':'official type startswith KC; complete current annotation population','requested':len(ids),'verified':sum(x['status']=='VERIFIED_ORIGINAL_SWC' for x in rows),'elapsed_s':time.time()-start,'records':rows};(a.output/'MANIFEST.json').write_text(json.dumps(d,indent=2));print('Finished:',d['verified'],'/',d['requested'],flush=True)
if __name__=='__main__':main()
