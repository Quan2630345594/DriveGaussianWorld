"""Validate collected KITTI driving sources without running a model or loading point clouds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def validate(root):
    root=Path(root)
    plan=json.loads((root/'download_plan.json').read_text())
    rows=[]
    for job in plan['episodes']:
        scene=f"{job['date']}_drive_{job['drive']}_sync"
        metadata=root/f'{scene}.collection.json'
        if not metadata.exists():continue
        source=json.loads(metadata.read_text())
        base=root/'raw'/job['date']/scene;ids=job['frame_ids']
        for sensor,extension in [('image_02','png'),('image_03','png'),('oxts','txt')]:
            actual=sorted(int(p.stem) for p in (base/sensor/'data').glob('*.'+extension))
            require(actual==ids,f'{scene}/{sensor}: frame IDs differ from plan')
        lidar_ids=sorted(int(p.stem) for p in (base/'velodyne_points/data').glob('*.bin'))
        declared_lidar_ids=sorted(int(Path(m['path']).stem) for m in source['members'] if '/velodyne_points/data/' in m['path'])
        require(lidar_ids==declared_lidar_ids,f'{scene}: downloaded LiDAR differs from source ZIP members')
        require(set(lidar_ids).issubset(ids),f'{scene}: LiDAR IDs outside image sequence')
        times=[]
        for sensor in ('image_02','image_03','oxts'):
            ts=np.array([np.datetime64(v.strip(),'ns').astype(np.int64)
                         for v in (base/sensor/'timestamps.txt').read_text().splitlines()])[ids]
            require(bool((np.diff(ts)>0).all()),f'{scene}: timestamps are not increasing')
            times.append(ts)
        mismatch=max(np.abs(ts-times[0]).max() for ts in times[1:])/1e6
        require(mismatch<=50,f'{scene}: sensor clocks differ by {mismatch:.3f} ms')
        poses=np.stack([np.loadtxt(base/'oxts/data'/f'{i:010d}.txt') for i in ids])
        require(bool(np.isfinite(poses).all()),f'{scene}: non-finite OXTS measurements')
        speed=poses[:,8]
        require(speed.mean()>=1.,f'{scene}: full-log mean motion is below 1 m/s')
        shapes=set()
        for fid in ids:
            for sensor in ('image_02','image_03'):
                with (base/sensor/'data'/f'{fid:010d}.png').open('rb') as stream:header=stream.read(24)
                require(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR',f'{scene}: invalid PNG header')
                width,height=struct.unpack('>II',header[16:24]);require(width>0 and height>0,f'{scene}: empty image')
                shapes.add((width,height))
        for fid in lidar_ids:
            size=(base/'velodyne_points/data'/f'{fid:010d}.bin').stat().st_size
            require(size>0 and size%16==0,f'{scene}: invalid XYZI point-cloud record size')
        rows.append({'scene_id':scene,'split':job['split'],'frames':len(ids),
                     'mean_forward_speed_m_s':float(speed.mean()),
                     'moving_fraction_over_0_5_m_s':float((np.abs(speed)>.5).mean()),
                     'max_sensor_clock_difference_ms':float(mismatch),'image_width_height':sorted(shapes),
                     'lidar_frames':len(lidar_ids),'source_missing_lidar_ids':sorted(set(ids)-set(lidar_ids))})
        print(f'{scene}: {len(ids)} times, mean forward speed {speed.mean():.3f} m/s',flush=True)
    report={'status':'PASS' if len(rows)==len(plan['episodes']) else 'PARTIAL','episodes':rows,
            'frames':sum(row['frames'] for row in rows),'lidar_frames':sum(row['lidar_frames'] for row in rows),
            'scope':'RGB/OXTS IDs and LiDAR source-member availability, increasing synchronized timestamps, finite OXTS, road motion, PNG headers/dimensions and LiDAR record sizes. Collection separately records per-member CRC32/SHA256.',
            'model_inference_performed':False,'gpu_execution':False}
    if (root/'collection.json').exists():
        report['collection_sha256']=hashlib.sha256((root/'collection.json').read_bytes()).hexdigest()
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/kitti_driving')
    p.add_argument('--output');p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    result=validate(args.root)
    output=Path(args.output) if args.output else Path(args.root)/'raw_validation.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],len(result['episodes']),result['frames'])
    if result['status']!='PASS' and not args.allow_partial:raise SystemExit('Collection is still incomplete; report saved.')

if __name__=='__main__':main()
