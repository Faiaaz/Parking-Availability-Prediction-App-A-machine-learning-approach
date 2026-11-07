"""Read-only audit of original project artifacts. Outputs only to research/."""
import hashlib
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    raw=pd.read_csv(ROOT/'data/raw/dataset.csv')
    raw['LastUpdated']=pd.to_datetime(raw.LastUpdated)
    features=pd.read_parquet(ROOT/'data/processed/birmingham_features.parquet')
    ordered=features.sort_values('date_time').reset_index(drop=True)
    cut=int(len(ordered)*.8)
    train,test=ordered.iloc[:cut],ordered.iloc[cut:]
    keys=['parking_zone','date_time']
    original_files=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'app').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),ROOT/'README.md',ROOT/'requirements.txt',*sorted((ROOT/'models').glob('*.joblib')),*sorted((ROOT/'reports').glob('*')),ROOT/'data/processed/birmingham_features.parquet',*sorted((ROOT/'data/raw').glob('*'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files if p.is_file()}
    result={'raw_rows':len(raw),'zones':raw.SystemCodeNumber.nunique(),'raw_exact_duplicate_excess':int(raw.duplicated().sum()),'raw_key_duplicate_excess':int(raw.duplicated(['SystemCodeNumber','LastUpdated']).sum()),'invalid_occupancy':int(((raw.Occupancy<0)|(raw.Occupancy>raw.Capacity)).sum()),'processed_rows':len(features),'processed_key_duplicate_excess':int(features.duplicated(keys).sum()),'train_rows':len(train),'test_rows':len(test),'train_last':str(train.date_time.max()),'test_first':str(test.date_time.min()),'shared_timestamps':len(set(train.date_time)&set(test.date_time)),'shared_keys':len(pd.merge(train[keys],test[keys],on=keys)),'test_duplicate_key_excess':int(test.duplicated(keys).sum()),'sha256':hashes}
    path=ROOT/'research/legacy_audit.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior!=result:raise RuntimeError('Original artifacts changed since initial audit')
        print('Original artifact hashes and audit unchanged')
    else:
        path.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
