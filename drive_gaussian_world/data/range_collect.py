"""Bounded public ZIP-range collection; fetch selected members, not multi-GB archives."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path,PurePosixPath
import struct
import threading
import time
import zipfile
import zlib
import hashlib
import requests
from drive_gaussian_world.common import write_json,sha256
from drive_gaussian_world.data.collect import HOST,DATE,DRIVES,fetch,extract,download_session

_LOCAL=threading.local()
def session():
    if not hasattr(_LOCAL,'session'):
        _LOCAL.session=download_session()
    return _LOCAL.session

def get_range(url,start,end):
    if end-start>8_000_000:raise ValueError('range budget exceeded')
    for attempt in range(4):
        try:
            r=session().get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,90))
            r.raise_for_status()
            if r.status_code!=206 or len(r.content)!=end-start+1:
                raise ValueError('server must honor bounded HTTP ranges')
            return r.content,r.headers
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(attempt+1)

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0
        _,headers=get_range(url,0,0)
        self.size=int(headers['Content-Range'].split('/')[-1]);self.etag=headers.get('ETag')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        size=min(size,self.size-self.pos)
        if size<=0:return b''
        b,_=get_range(self.url,self.pos,self.pos+size-1);self.pos+=len(b);return b

def member(url,info,root):
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe member')
    target=Path(root).joinpath(*p.parts)
    if target.exists():
        content=target.read_bytes()
        if (zlib.crc32(content)&0xffffffff)!=info.CRC:raise ValueError('changed cached member')
    else:
        # Local header's extra length can differ from central directory; read bounded header first.
        header,_=get_range(url,info.header_offset,info.header_offset+29)
        if header[:4]!=b'PK\x03\x04':raise ValueError('bad local header')
        name_len,extra_len=struct.unpack_from('<HH',header,26)
        start=info.header_offset+30+name_len+extra_len
        body,_=get_range(url,start,start+info.compress_size-1)
        if info.compress_type==zipfile.ZIP_DEFLATED:content=zlib.decompress(body,-15)
        elif info.compress_type==zipfile.ZIP_STORED:content=body
        else:raise ValueError('unsupported compression')
        if len(content)!=info.file_size or (zlib.crc32(content)&0xffffffff)!=info.CRC:
            raise ValueError('member checksum mismatch')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    return {'path':str(p),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'crc32':info.CRC,
            'compressed_bytes':info.compress_size}

def collect(root,frames=128,stride=4,workers=6,max_bytes=49_000_000_000):
    if not 32<=frames<=300 or stride<1 or not 1<=workers<=8:raise ValueError('invalid collection budget')
    if not 1_000_000<=max_bytes<50_000_000_000:raise ValueError('collection limit must be at least 1 MB and below 50 GB')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    calib=fetch(HOST+DATE+'_calib.zip',root/'archives'/f'{DATE}_calib.zip')
    cm=extract(calib,root/'raw',frames)
    archives=[{'split':'calibration','url':HOST+DATE+'_calib.zip','bytes':calib.stat().st_size,
               'sha256':sha256(calib),'members':cm}]
    for split,drive in DRIVES.items():
        name=f'{DATE}_drive_{drive}';url=HOST+f'{name}/{name}_sync.zip'
        rz=RemoteZip(url)
        with zipfile.ZipFile(rz) as z:infos=z.infolist()
        selected=[]
        for info in infos:
            p=PurePosixPath(info.filename)
            if info.is_dir():continue
            keep=p.name=='timestamps.txt'
            if '/data/' in str(p) and p.stem.isdigit():
                fid=int(p.stem)
                keep=fid<frames and fid%stride==0 and any(s in p.parts for s in ['image_02','image_03','oxts'])
                # A few real LiDAR frames are collected for diagnostic projection, not world-model inputs.
                keep=keep or ('velodyne_points' in p.parts and fid in [0,stride,2*stride])
            if keep:selected.append(info)
        existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        additional=sum(i.file_size for i in selected if not (root/'raw'/i.filename).exists())
        if existing+additional>max_bytes:raise ValueError('collection exceeds total disk byte budget')
        # Inspect real OXTS before fetching images: a road dataset can contain stationary intervals.
        motion_infos=[i for i in selected if '/oxts/data/' in i.filename]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            motion=list(pool.map(lambda i:member(url,i,root/'raw'),motion_infos))
        speeds=[float((root/'raw'/m['path']).read_text().split()[8]) for m in motion]
        if not speeds or sum(speeds)/len(speeds)<1.0:
            raise ValueError(f'{split}: selected interval is stationary; choose a moving drive/interval')
        print(f'{split}: OXTS mean speed {sum(speeds)/len(speeds):.2f} m/s',flush=True)
        print(f'{split}: fetching {len(selected)} members ({sum(i.compress_size for i in selected)/1e6:.1f} MB)',flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records=list(pool.map(lambda i:member(url,i,root/'raw'),selected))
        archives.append({'split':split,'url':url,'remote_archive_bytes':rz.size,'remote_etag':rz.etag,
                         'members':records,'selected_compressed_bytes':sum(i.compress_size for i in selected)})
        write_json(root/f'{split}_collection.json',archives[-1])
        print(f'{split}: complete',flush=True)
    report={'schema_version':1,'dataset':'KITTI raw','scenario':'road_driving',
            'source':'https://www.cvlibs.net/datasets/kitti/raw_data.php',
            'license':'CC BY-NC-SA 3.0; research/non-commercial; retain KITTI attribution',
            'license_url':'https://www.cvlibs.net/datasets/kitti/','frame_limit':frames,'source_stride':stride,
            'max_bytes':max_bytes,'drives':DRIVES,'method':'public S3 ZIP ranges, CRC32 + per-member SHA256',
            'archives':archives,'selected_compressed_bytes':sum(a.get('selected_compressed_bytes',0) for a in archives),
            'notes':['Whole-log splits, possible geographical overlap; engineering subset only.',
                     'Motion-derived conditioning labels are not issued throttle/steering controls.']}
    write_json(root/'collection.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/kitti')
    p.add_argument('--frames',type=int,default=128);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--max-bytes',type=int,default=49_000_000_000);a=p.parse_args()
    m=collect(a.root,a.frames,a.stride,a.workers,a.max_bytes)
    print('COLLECTED_BYTES',m['selected_compressed_bytes'])
if __name__=='__main__':main()
