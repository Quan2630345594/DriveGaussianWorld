from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from drive_gaussian_world.common import fresh_dir,write_json,sha256,read_json
from drive_gaussian_world.data.collect import DATE,DRIVES
from drive_gaussian_world.data.schema import TOKEN_FIELDS,validate_dataset
from drive_gaussian_world.geometry.stereo import stereo_tokens

def calibration(path):
    values={}
    for line in Path(path).read_text().splitlines():
        key,_,rest=line.partition(':')
        try:values[key]=np.array([float(v) for v in rest.split()])
        except ValueError:continue
    p2=values['P_rect_02'].reshape(3,4);p3=values['P_rect_03'].reshape(3,4)
    baseline=abs(p2[0,3]/p2[0,0]-p3[0,3]/p3[0,0])
    if not 0.1<baseline<2:raise ValueError('invalid stereo calibration')
    return p2[:,:3].copy(),float(baseline)

def timestamps(path):
    # Retain KITTI nanosecond fractions, remove the large epoch before float conversion.
    ts=np.array([np.datetime64(t.strip(),'ns').astype(np.int64) for t in Path(path).read_text().splitlines()])
    return (ts-ts[0]).astype(np.float64)*1e-9

def oxts_position(oxts):
    # Standard KITTI Mercator approximation, in metres; local ENU relative to first pose.
    lat,lon,alt=oxts[:,:3].T
    scale=np.cos(np.deg2rad(lat[0]));earth=6378137.
    xyz=np.stack([scale*earth*np.deg2rad(lon),scale*earth*np.log(np.tan(np.pi/4+np.deg2rad(lat)/2)),alt],-1)
    return xyz-xyz[0]

def transition_labels(position,yaw,speed,ts):
    dt=np.diff(ts)
    delta=np.diff(position,axis=0)
    forward=delta[:,0]*np.cos(yaw[:-1])+delta[:,1]*np.sin(yaw[:-1])
    left=-delta[:,0]*np.sin(yaw[:-1])+delta[:,1]*np.cos(yaw[:-1])
    dyaw=np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)))
    # These are future-motion-derived conditioning labels, not issued vehicle controls.
    actions=np.stack([np.diff(speed)/dt,dyaw/dt],-1)
    ego=np.stack([forward,left,dyaw],-1)
    return actions.astype(np.float32),ego.astype(np.float32),dt.astype(np.float32)

def prepare(raw,output,stride=4,frames=128):
    if stride<1 or frames<32:raise ValueError('invalid sample budget')
    raw=Path(raw);output=fresh_dir(output)
    k,baseline=calibration(raw/DATE/'calib_cam_to_cam.txt')
    episodes=[]
    for split,drive in DRIVES.items():
        scene=f'{DATE}_drive_{drive}_sync';base=raw/DATE/scene
        candidates=[p for p in sorted((base/'image_02/data').glob('*.png')) if int(p.stem)<frames and int(p.stem)%stride==0]
        ids=np.array([int(p.stem) for p in candidates])
        if ids.tolist()!=list(range(0,frames,stride)):
            raise ValueError(f'incomplete source transfer/collection: {scene}; expected all selected frames')
        if len(ids)<8:raise ValueError(f'not enough synchronized frames: {scene}')
        image_times=timestamps(base/'image_02/timestamps.txt')
        right_times=timestamps(base/'image_03/timestamps.txt')
        oxts_times=timestamps(base/'oxts/timestamps.txt')
        # Compare absolute timestamps as well: each sensor's first epoch can differ.
        absolutes=[]
        for stream in ['image_02','image_03','oxts']:
            a=np.array([np.datetime64(s.strip(),'ns').astype(np.int64) for s in (base/stream/'timestamps.txt').read_text().splitlines()])
            absolutes.append(a[ids])
        for a in absolutes[1:]:
            if np.max(np.abs(a-absolutes[0]))>50_000_000:raise ValueError('sensor time misalignment >50 ms')
        measurements=np.stack([np.loadtxt(base/'oxts/data'/f'{i:010d}.txt') for i in ids])
        ts=image_times[ids];ts-=ts[0]
        pos=oxts_position(measurements);yaw=measurements[:,5];speed=measurements[:,8]
        actions,ego,dt=transition_labels(pos,yaw,speed,ts)
        rgbs=[];tokens=[];masks=[];sources=[]
        for p in candidates:
            right=base/'image_03/data'/p.name
            tok,mask,_=stereo_tokens(p,right,k,baseline)
            rgbs.append(np.asarray(Image.open(p).convert('RGB').resize((96,32))).transpose(2,0,1))
            tokens.append(tok);masks.append(mask)
            sources.append({'frame_id':int(p.stem),'left':str(p.relative_to(raw)),
                            'left_sha256':sha256(p),'right_sha256':sha256(right)})
        np.savez_compressed(output/f'{scene}.npz',rgb=np.stack(rgbs),tokens=np.stack(tokens),
            token_mask=np.stack(masks),timestamps=ts,speed=speed.astype(np.float32),
            yaw=yaw.astype(np.float32),position=pos.astype(np.float32),actions=actions,
            ego_delta=ego,dt=dt,frame_ids=ids)
        write_json(output/f'{scene}.sources.json',{'frames':sources,'K':k.tolist(),
            'baseline_m':baseline,'geometry_source':'stereo_sgbm_reference','metric':True,
            'dynamic_probability_available':False,'input_causal':True})
        episodes.append({'scene_id':scene,'split':split,'path':f'{scene}.npz',
            'sha256':sha256(output/f'{scene}.npz'),'geometry_source':'stereo_sgbm_reference',
            'sources':f'{scene}.sources.json'})
        print(f'prepared {split}: {len(ids)} frames, mean speed {speed.mean():.2f} m/s',flush=True)
    index={'schema_version':1,'project':'DriveGaussianWorld','scenario':'road_driving',
           'token_fields':TOKEN_FIELDS,'coordinates':'camera optical: right, down, forward',
           'action_semantics':'motion-derived [acceleration_m_s2, yaw_rate_rad_s]; NOT issued controls',
           'conditioning':'Future action labels are scenario conditions; unavailable to unconditional inference.',
           'split_policy':'whole KITTI drive per split; nearby routes may overlap geographically',
           'image_shape':[3,32,96],'token_shape':[64,16],'stride':stride,'episodes':episodes,
           'collection_sha256':sha256(raw.parent/'collection.json') if (raw.parent/'collection.json').exists() else None}
    write_json(output/'dataset.json',index)
    report=validate_dataset(output)
    write_json(output/'validation.json',{'status':'PASS','episodes':report})
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/kitti/raw')
    p.add_argument('--output',required=True);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--frames',type=int,default=128);a=p.parse_args()
    print(prepare(a.raw,a.output,a.stride,a.frames))
if __name__=='__main__':main()
