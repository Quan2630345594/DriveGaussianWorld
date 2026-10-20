"""Audit downloaded bytes against stored provenance and independent LiDAR depth samples."""
from __future__ import annotations
import argparse
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
from drive_gaussian_world.common import read_json,write_json,sha256
from drive_gaussian_world.geometry.stereo import grid_pixels
from drive_gaussian_world.data.prepare import DATE,calibration

def audit_collection(root):
    root=Path(root);manifest=read_json(root/'collection.json');n=0;total=0
    for archive in manifest['archives']:
        for member in archive['members']:
            path=(root/'raw'/member['path']).resolve()
            if not path.is_relative_to((root/'raw').resolve()):raise ValueError('unsafe source path')
            if path.stat().st_size!=member['bytes']:raise ValueError(f'size mismatch: {path}')
            if 'sha256' in member and sha256(path)!=member['sha256']:raise ValueError(f'corrupt source: {path}')
            if (zlib.crc32(path.read_bytes())&0xffffffff)!=member['crc32']:raise ValueError(f'CRC mismatch: {path}')
            n+=1;total+=path.stat().st_size
    return {'status':'PASS','checked_files':n,'uncompressed_bytes':total,
            'collection_sha256':sha256(root/'collection.json'),'scenario':'road_driving'}

def parse_calibration(path):
    result={}
    for line in Path(path).read_text().splitlines():
        key,_,rest=line.partition(':')
        try:result[key]=np.array([float(v) for v in rest.split()])
        except ValueError:continue
    return result

def lidar_diagnostic(dataset,raw):
    dataset=Path(dataset);raw=Path(raw);index=read_json(dataset/'dataset.json')
    cal=parse_calibration(raw/DATE/'calib_cam_to_cam.txt')
    velo=parse_calibration(raw/DATE/'calib_velo_to_cam.txt')
    R=cal['R_rect_00'].reshape(3,3);K,baseline=calibration(raw/DATE/'calib_cam_to_cam.txt')
    p2=cal['P_rect_02'].reshape(3,4);offset=np.linalg.solve(K,p2[:,3])
    reports=[]
    for episode in index['episodes']:
        with np.load(dataset/episode['path'],allow_pickle=False) as f:a=dict(f)
        source=read_json(dataset/episode['sources']);errors=[];rels=[];frames=[]
        for i,fid in enumerate(a['frame_ids']):
            path=raw/DATE/episode['scene_id']/'velodyne_points/data'/f'{fid:010d}.bin'
            if not path.exists():continue
            points=np.fromfile(path,np.float32).reshape(-1,4)[:,:3]
            cam=(points@velo['R'].reshape(3,3).T+velo['T'])@R.T+offset
            cam=cam[cam[:,2]>1]
            image=Image.open(raw/source['frames'][i]['left']);w,h=image.size
            uvh=cam@K.T;uv=uvh[:,:2]/uvh[:,2:]
            centers=grid_pixels(h,w);matched=0
            for j,(u,v) in enumerate(centers):
                if not a['token_mask'][i,j]:continue
                chosen=(np.abs(uv[:,0]-u)<w/100)&(np.abs(uv[:,1]-v)<h/32)
                depths=cam[chosen,2]
                if len(depths)<2:continue
                truth=float(np.median(depths));pred=float(a['tokens'][i,j,2]*40)
                errors.append(abs(pred-truth));rels.append(abs(pred-truth)/truth);matched+=1
            frames.append({'frame_id':int(fid),'matched_grid_patches':matched})
        reports.append({'scene_id':episode['scene_id'],'split':episode['split'],
                        'geometry_source':episode['geometry_source'],'matched_grid_patches':len(errors),
                        'patch_depth_mae_m':float(np.mean(errors)) if errors else None,
                        'patch_depth_abs_rel':float(np.mean(rels)) if rels else None,'frames':frames})
    return {'status':'DIAGNOSTIC_ONLY','episodes':reports,
            'interpretation':'Independent LiDAR depth in sparse image patches, not dense benchmark depth or occupancy.',
            'limitations':['Only three LiDAR frames per drive; patch medians can mix occlusion layers.',
                          'DGGT uses stereo scale anchoring; camera intrinsics/pose remain predicted.',
                          'Camera/IMU reference points and local frames are approximate in the CPU dynamics prior.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/kitti');p.add_argument('--dataset')
    p.add_argument('--output',required=True);a=p.parse_args()
    report={'collection':audit_collection(a.root)}
    if a.dataset:report['lidar']=lidar_diagnostic(a.dataset,Path(a.root)/'raw')
    write_json(a.output,report);print(report)
if __name__=='__main__':main()
