"""Collect bounded KITTI road-driving samples from the official public S3 host.

No account credentials or alternate gated mirrors. Hashes identify exact fetched
bytes; they are local provenance hashes, not upstream authenticity signatures.
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import time
import zipfile
import requests
from drive_gaussian_world.common import sha256, write_json

HOST = 'https://s3-eu-central-1.amazonaws.com/avg-kitti/raw_data/'
DRIVES = {'train': '0005', 'val': '0013', 'test': '0009'}
DATE = '2011_09_26'

def download_session():
    session = requests.Session()
    # Explicit project proxy; ignore unrelated/dead shell proxies by default.
    session.trust_env = False
    proxy = os.environ.get('DGW_HTTP_PROXY')
    if proxy:session.proxies.update({'http':proxy,'https':proxy})
    return session

def fetch(url, target, max_bytes=1_500_000_000):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Refuse a corrupted cached ZIP; never silently treat a partial as complete.
    if target.exists():
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:
                raise ValueError(f'corrupt archive: {target}')
        return target
    partial = target.with_suffix('.part')
    session = download_session()
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                if int(response.headers.get('Content-Length', 0)) > max_bytes:
                    raise ValueError('download exceeds per-archive byte budget')
                total = 0
                with partial.open('wb') as f:
                    for chunk in response.iter_content(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError('download exceeds byte budget')
                        f.write(chunk)
            with zipfile.ZipFile(partial) as z:
                if z.testzip() is not None:
                    raise ValueError('ZIP checksum mismatch')
            partial.replace(target)
            return target
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError('download failed')

def extract(archive, root, frame_limit):
    selected = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('unsafe ZIP member')
            if info.is_dir():
                continue
            keep = ('calib_' in p.name or p.name == 'timestamps.txt')
            if '/data/' in str(p) and any(x in p.parts for x in ['image_02', 'image_03', 'oxts', 'velodyne_points']):
                keep = p.stem.isdigit() and int(p.stem) < frame_limit
            if keep:
                target = Path(root).joinpath(*p.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(info)  # CRC verified by zipfile.
                if target.exists() and target.read_bytes() != content:
                    raise ValueError(f'refusing to overwrite changed data: {target}')
                target.write_bytes(content)
                selected.append({'path': str(p), 'bytes': len(content), 'crc32': info.CRC})
    return selected

def collect(root, frame_limit=128):
    if not 16 <= frame_limit <= 300:
        raise ValueError('collection budget: 16..300 frames per drive')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [('calibration', DATE + '_calib.zip', DATE + '_calib.zip')]
    for split, drive in DRIVES.items():
        name = f'{DATE}_drive_{drive}'
        jobs.append((split, f'{name}/{name}_sync.zip', f'{name}_sync.zip'))
    def run(job):
        split, relative, name = job
        print(f'fetching {split}: {name}', flush=True)
        archive = fetch(HOST + relative, root/'archives'/name)
        members = extract(archive, root/'raw', frame_limit)
        result = {'split': split, 'url': HOST + relative, 'archive': str(archive.relative_to(root)),
                  'sha256': sha256(archive), 'bytes': archive.stat().st_size, 'members': members}
        print(f'collected {split}: {len(members)} files', flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run, jobs))
    manifest = {'schema_version': 1, 'dataset': 'KITTI raw', 'scenario': 'road_driving',
                'source': 'https://www.cvlibs.net/datasets/kitti/raw_data.php',
                'license': 'CC BY-NC-SA 3.0; research/non-commercial; retain KITTI attribution',
                'license_url': 'https://www.cvlibs.net/datasets/kitti/',
                'frame_limit': frame_limit, 'drives': DRIVES, 'archives': records,
                'total_archive_bytes': sum(r['bytes'] for r in records),
                'notes': ['Small engineering subset, not a benchmark or geographically disjoint validation.',
                          'OXTS measurements are not issued throttle/steering controls.']}
    write_json(root/'collection.json', manifest)
    return manifest

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/kitti')
    p.add_argument('--frames', type=int, default=128)
    args = p.parse_args()
    m = collect(args.root, args.frames)
    print({k: m[k] for k in ['dataset', 'drives', 'total_archive_bytes']})

if __name__ == '__main__':
    main()
