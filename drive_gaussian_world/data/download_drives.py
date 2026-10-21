"""Download a bounded multi-drive KITTI subset; retain raw sensors, discard verified ZIPs."""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time
import zipfile
import numpy as np
import requests
from drive_gaussian_world.common import read_json,write_json,sha256
from .collect import HOST,download_session
from .range_collect import RemoteZip,member

LIMIT=49_000_000_000
STREAMS={'image_02','image_03','oxts','velodyne_points'}


def selected(info):
    from pathlib import PurePosixPath
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe ZIP path')
    return not info.is_dir() and (p.name=='timestamps.txt' or ('data' in p.parts and any(s in p.parts for s in STREAMS)))


def plan_download(plan,root,max_bytes=LIMIT):
    if not 1_000_000<=max_bytes<50_000_000_000:raise ValueError('budget must be below 50 GB')
    if len({(e['date'],e['drive']) for e in plan})!=len(plan):raise ValueError('duplicate drive/split')
    if {e['split'] for e in plan}!={'train','val','test'}:raise ValueError('train/val/test splits required')
    jobs=[]
    for e in plan:
        date=e['date'];drive=e['drive']
        if not date.replace('_','').isdigit() or len(date)!=10 or not drive.isdigit() or len(drive)!=4:
            raise ValueError('invalid date/drive')
        name=f'{date}_drive_{drive}';url=HOST+f'{name}/{name}_sync.zip'
        remote=RemoteZip(url)
        with zipfile.ZipFile(remote) as z:infos=[i for i in z.infolist() if selected(i)]
        frames=sorted(int(Path(i.filename).stem) for i in infos if '/image_02/data/' in i.filename)
        motion=[i for i in infos if '/oxts/data/' in i.filename]
        if not frames or not motion:raise ValueError('missing driving observations')
        sample=[motion[i] for i in np.linspace(0,len(motion)-1,min(8,len(motion))).astype(int)]
        speeds=[]
        for info in sample:
            record=member(url,info,Path(root)/'raw')
            speeds.append(float((Path(root)/'raw'/record['path']).read_text().split()[8]))
        mean=float(np.mean(speeds))
        if mean<1.:raise ValueError(f'{name}: sampled motion is stationary ({mean:.3f} m/s)')
        jobs.append({**e,'url':url,'archive_bytes':remote.size,'selected_raw_bytes':sum(i.file_size for i in infos),
                     'frame_ids':frames,'sampled_speed_m_s':mean,'remote_etag':remote.etag})
        print(f"plan {name}: {len(frames)} times, ZIP {remote.size/1e9:.2f} GB, moving {mean:.2f} m/s",flush=True)
    archive_bytes=sum(j['archive_bytes'] for j in jobs)
    raw_bytes=sum(j['selected_raw_bytes'] for j in jobs)
    # Two concurrent transfers: all final raw files + the two largest ZIPs + metadata.
    peak=raw_bytes+sum(sorted((j['archive_bytes'] for j in jobs),reverse=True)[:2])+100_000_000
    if max(archive_bytes,peak)>max_bytes:raise ValueError(f'plan exceeds budget: ZIP={archive_bytes}, peak={peak}, limit={max_bytes}')
    return {'schema_version':1,'dataset':'KITTI raw','scenario':'road_driving','max_bytes':max_bytes,
            'archive_bytes':archive_bytes,'selected_raw_bytes':raw_bytes,'conservative_peak_bytes':peak,'episodes':jobs}


def _fetch_stream(url,path,expected_bytes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=expected_bytes:raise ValueError('cached ZIP size mismatch')
        return path
    partial=path.with_suffix('.part')
    session=download_session()
    for attempt in range(6):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==expected_bytes:partial.replace(path);return path
        if offset>expected_bytes:raise ValueError('oversized partial')
        try:
            with session.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                if offset:
                    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('resume requires a correct Content-Range')
                total=offset
                with partial.open('ab' if offset else 'wb') as f:
                    for chunk in response.iter_content(4*1024*1024):
                        total+=len(chunk)
                        if total>expected_bytes:raise ValueError('archive exceeds declared byte budget')
                        f.write(chunk)
            if total!=expected_bytes:raise IOError('incomplete ZIP transfer')
            partial.replace(path);return path
        except (OSError, requests.RequestException):
            if attempt==5:raise
            time.sleep(min(2**attempt,10))
    raise RuntimeError('download failed')


def fetch_parallel(url,path,expected_bytes,block_bytes=8*1024*1024,workers=8):
    """Bounded range transfers, with a resumable prefix and verified-length chunks."""
    import shutil
    import threading
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=expected_bytes:raise ValueError('cached ZIP size mismatch')
        return path
    partial=path.with_suffix('.part')
    prefix=partial.stat().st_size if partial.exists() else 0
    if prefix>expected_bytes:raise ValueError('oversized partial')
    chunks=path.with_suffix('.chunks');chunks.mkdir(exist_ok=True)
    identity={'url':url,'expected_bytes':expected_bytes,'block_bytes':block_bytes}
    metadata=chunks/'source.json'
    if metadata.exists() and read_json(metadata)!=identity:raise ValueError('range source provenance mismatch')
    write_json(metadata,identity)
    for previous in [*chunks.glob('*.chunk'),*chunks.glob('*.tmp')]:
        try:finished_end=int(previous.stem.split('-')[1])
        except (ValueError,IndexError):raise ValueError('unexpected range cache file')
        if finished_end<prefix:previous.unlink()
    local=threading.local()
    ranges=[(start,min(start+block_bytes,expected_bytes)-1) for start in range(prefix,expected_bytes,block_bytes)]
    def chunk_name(start,end):return chunks/f'{start:012d}-{end:012d}.chunk'
    expected_paths={chunk_name(start,end).name for start,end in ranges}
    # A crash during prefix assembly can change the range grid. Drop obsolete
    # cached ranges before downloading, avoiding duplicates and stale files.
    for previous in [*chunks.glob('*.chunk'),*chunks.glob('*.tmp')]:
        if previous.with_suffix('.chunk').name not in expected_paths:previous.unlink()
    def transfer(bounds):
        start,end=bounds;target=chunk_name(start,end);length=end-start+1
        if target.exists():
            if target.stat().st_size!=length:raise ValueError('changed cached range')
            return
        if not hasattr(local,'session'):local.session=download_session()
        session=local.session
        for attempt in range(6):
            try:
                with session.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,60)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{expected_bytes}':
                        raise ValueError('server must honor exact bounded ranges')
                    temporary=target.with_suffix('.tmp');count=0
                    with temporary.open('wb') as output:
                        for data in response.iter_content(1024*1024):
                            count+=len(data)
                            if count>length:raise ValueError('oversized range')
                            output.write(data)
                    if count!=length:raise IOError('truncated range')
                    temporary.replace(target)
                return
            except (OSError,requests.RequestException):
                if attempt==5:raise
                time.sleep(min(2**attempt,10))
        raise RuntimeError('range download failed')
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(transfer,ranges))
    with partial.open('ab') as output:
        for start,end in ranges:
            part=chunk_name(start,end)
            with part.open('rb') as source:shutil.copyfileobj(source,output,1024*1024)
            output.flush();part.unlink()
    if partial.stat().st_size!=expected_bytes:raise ValueError('assembled ZIP size mismatch')
    partial.replace(path)
    metadata.unlink();chunks.rmdir()
    return path


def fetch_resume(url,path,expected_bytes):
    workers=int(os.environ.get('DGW_TRANSFER_WORKERS','8'))
    if not 1<=workers<=32:raise ValueError('DGW_TRANSFER_WORKERS must be within 1..32')
    if expected_bytes>64*1024*1024:return fetch_parallel(url,path,expected_bytes,workers=workers)
    return _fetch_stream(url,path,expected_bytes)


def extract_checked(archive,root):
    records=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not selected(info):continue
            content=z.read(info)  # zipfile validates member CRC.
            path=Path(root)/info.filename
            if path.exists() and path.read_bytes()!=content:raise ValueError(f'changed raw file: {path}')
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=path.with_suffix(path.suffix+'.part');partial.write_bytes(content);partial.replace(path)
            records.append({'path':info.filename,'bytes':len(content),'sha256':sha256(path),'crc32':info.CRC})
    return records


def collect(root,plan='configs/data/kitti_driving.json',max_bytes=LIMIT,preflight=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    report_path=root/'download_plan.json'
    specification=read_json(plan)
    fingerprint=sha256(plan)
    if report_path.exists():
        report=read_json(report_path)
        if report.get('plan_sha256')!=fingerprint or report['max_bytes']!=max_bytes:raise ValueError('existing plan provenance mismatch')
    else:
        report=plan_download(specification,root,max_bytes);report['plan_sha256']=fingerprint
        write_json(report_path,report)
    if preflight:return report
    # Same date calibration is shared by every episode, copied as a small verified ZIP.
    from .collect import fetch,extract
    for date in sorted({j['date'] for j in report['episodes']}):
        calibration=fetch(HOST+date+'_calib.zip',root/'archives'/f'{date}_calib.zip')
        extract(calibration,root/'raw',0)
    archives=[];status={'status':'RUNNING','completed':[],'plan_sha256':fingerprint}
    write_json(root/'status.json',status)
    def collect_job(job):
        name=f"{job['date']}_drive_{job['drive']}_sync"
        metadata=root/f'{name}.collection.json'
        if metadata.exists():
            record=read_json(metadata)
            for m in record['members']:
                if sha256(root/'raw'/m['path'])!=m['sha256']:raise ValueError('changed previously collected member')
        else:
            path=root/'archives'/f'{name}.zip'
            print(f'downloading {name} ({job["archive_bytes"]/1e9:.2f} GB)',flush=True)
            fetch_resume(job['url'],path,job['archive_bytes'])
            digest=sha256(path)
            records=extract_checked(path,root/'raw')
            record={**job,'archive_sha256':digest,'members':records}
            write_json(metadata,record)
            # Only the newly downloaded, verified temporary source ZIP is discarded.
            path.unlink()
        return name,record
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(collect_job,job) for job in report['episodes']]
            for task in as_completed(tasks):
                name,record=task.result();archives.append(record);status['completed'].append(name)
                write_json(root/'status.json',status)
                print(f'complete {name}: {len(record["members"])} files',flush=True)
    except Exception as error:
        status.update(status='FAILED',error=str(error));write_json(root/'status.json',status)
        raise
    archives.sort(key=lambda j:(j['date'],j['drive']))
    collection={**report,'archives':archives,'source':'https://registry.opendata.aws/kitti/',
                'license':'CC BY-NC-SA 3.0','method':'official public ZIPs; per-member CRC32 and SHA256; whole-log splits'}
    write_json(root/'collection.json',collection)
    status['status']='COMPLETED';status['raw_disk_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if status['raw_disk_bytes']>=max_bytes:raise ValueError('final collection exceeds budget')
    write_json(root/'status.json',status)
    return collection


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/kitti_driving')
    p.add_argument('--plan',default='configs/data/kitti_driving.json');p.add_argument('--max-bytes',type=int,default=LIMIT)
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    result=collect(**vars(args));print('planned archive bytes',result['archive_bytes'])

if __name__=='__main__':main()
