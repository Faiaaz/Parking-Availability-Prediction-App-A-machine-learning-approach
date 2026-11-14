"""Bounded public source acquisition. Downloads remain ignored by Git."""
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path

SOURCES = {
 'harvard_metadata.json':('https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/YLWCSU',100000,None),
 'description.pdf':('https://dataverse.harvard.edu/api/access/datafile/3128637',300000,None),
 'segments.tab':('https://dataverse.harvard.edu/api/access/datafile/3426356',100000,None),
 'harvard_prefix.tab':('https://dataverse.harvard.edu/api/access/datafile/3128639',4194304,'bytes=0-4194303'),
 'melbourne_prefix.zip':('https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip',2097152,'bytes=0-2097151')
}


def main():
    root=Path('data/external/v2')
    root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,(url,limit,byte_range) in SOURCES.items():
        path=root/name
        if not path.exists():
            request=urllib.request.Request(url,headers={'Range':byte_range} if byte_range else {})
            with urllib.request.urlopen(request,timeout=60) as response:
                if byte_range and response.status!=206:
                    raise ValueError('Server ignored bounded range')
                data=response.read(limit+1)
                if len(data)>limit:
                    raise ValueError('Response exceeds configured limit')
                path.write_bytes(data)
        data=path.read_bytes()
        manifest.append(dict(file=name,url=url,bytes=len(data),range=byte_range,
                             sha256=hashlib.sha256(data).hexdigest()))
    prefix=(root/'melbourne_prefix.zip').read_bytes()
    header=struct.unpack('<4s5H3I2H',prefix[:30])
    if header[0]!=b'PK\x03\x04' or header[3]!=8:
        raise ValueError('Unsupported ZIP layout')
    offset=30+header[-2]+header[-1]
    sample=zlib.decompressobj(-15).decompress(prefix[offset:],2000000)
    sample=sample[:sample.rfind(b'\n')+1]
    (root/'melbourne_sample.csv').write_bytes(sample)
    manifest.append(dict(file='melbourne_sample.csv',bytes=len(sample),
        sha256=hashlib.sha256(sample).hexdigest(),derived_from='melbourne_prefix.zip; bounded 2 MB deflate output'))
    Path('research/results/v2/download_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
