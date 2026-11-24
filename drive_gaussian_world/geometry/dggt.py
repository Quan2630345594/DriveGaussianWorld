"""Native DGGT checkpoint inference, with causal histories and configurable GPU inference.

Each observation is independently reconstructed from at most the preceding C
frames. Never reconstruct a whole log and slice it into predictive examples:
DGGT temporal attention would expose the prediction target to the encoder.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from drive_gaussian_world.common import configure_runtime,precision_context,fresh_dir,read_json,write_json,sha256
from drive_gaussian_world.geometry.stereo import grid_pixels
from drive_gaussian_world.data.schema import validate_dataset

def prediction_tokens(pred,extri,reference,reference_mask):
    pts=np.asarray(pred['world_points'])[0,-1]
    gs=np.asarray(pred['gs_map'])[0,-1]
    conf=np.asarray(pred['world_points_conf'])[0,-1]
    dyn=np.asarray(pred['dynamic_conf'])[0,-1].reshape(pts.shape[:2])
    r,t=extri[:,:3],extri[:,3]
    cam=pts@r.T+t
    h,w=pts.shape[:2];uv=grid_pixels(h,w)
    z=cam[uv[:,1],uv[:,0],2]
    scale_mask=reference_mask&(z>1e-5)&np.isfinite(z)
    if scale_mask.sum()<6:raise ValueError('insufficient historical stereo anchors for metric scale')
    ratios=reference[scale_mask,2]*40/z[scale_mask]
    scale=float(np.median(ratios))
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid scale')
    tokens=np.zeros((64,16),np.float32);mask=np.zeros(64,bool)
    for i,(u,v) in enumerate(uv):
        point=cam[v,u]*scale
        if not np.isfinite(point).all() or not 1<point[2]<80:continue
        q=gs[v,u,7:11].copy()
        if not np.isfinite(q).all() or np.linalg.norm(q)<1e-6:continue
        # Upstream Gaussian quaternions are wxyz in reconstruction-world gauge.
        q/=np.linalg.norm(q)
        qr=(Rotation.from_matrix(r)*Rotation.from_quat(q[[1,2,3,0]])).as_quat()
        q=qr[[3,0,1,2]]
        color=np.clip(gs[v,u,:3],0,1);opacity=np.clip(gs[v,u,3],0,1)
        sizes=np.maximum(gs[v,u,4:7]*scale,1e-5)
        prob=1/(1+np.exp(-np.clip(dyn[v,u],-30,30)))
        confidence=float(np.clip(conf[v,u],0,1))
        tokens[i]=[* (point/40),*color,opacity,*(sizes/5),*q,prob,confidence]
        mask[i]=True
    residual=float(np.median(np.abs(z[scale_mask]*scale-reference[scale_mask,2]*40)))
    return tokens,mask,{'stereo_scale':scale,'scale_anchor_count':int(scale_mask.sum()),
                       'median_anchor_depth_error_m':residual,
                       'metric_status':'stereo_median_scale_only; not full extrinsic registration'}

class NativeDGGT:
    def __init__(self,upstream,checkpoint,width=518,threads=2,device='cuda',precision='fp32'):
        self.device=configure_runtime(42,threads,device)
        self.precision=precision
        precision_context(self.device,precision)
        import torch
        if width%14 or not 98<=width<=518:raise ValueError('width must be a multiple of 14 in [98,518]')
        self.width=width;self.torch=torch
        sys.path.insert(0,str(Path(upstream).resolve()))
        from dggt.models.vggt import VGGT
        from dggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.decode_pose=pose_encoding_to_extri_intri
        from dggt.utils.geometry import unproject_depth_map_to_point_map
        self.unproject=unproject_depth_map_to_point_map
        # Meta initialization avoids allocating and initializing a second 5-GB parameter copy.
        from accelerate import init_empty_weights
        with init_empty_weights():
            self.model=VGGT()
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
        self.model.load_state_dict(state,strict=True,assign=True)
        self.model=self.model.to(self.device).eval()
        self.model.requires_grad_(False)
    def __call__(self,paths):
        torch=self.torch;images=[]
        for p in paths:
            img=Image.open(p).convert('RGB');w,h=img.size
            nh=round(h*self.width/w/14)*14
            img=img.resize((self.width,nh),Image.Resampling.BICUBIC)
            images.append(np.asarray(img).copy().transpose(2,0,1)/255.)
        x=torch.tensor(np.stack(images),dtype=torch.float32,device=self.device)
        with torch.inference_mode(), precision_context(self.device,self.precision):
            p=self.model(x)
            e,k=self.decode_pose(p['pose_enc'],x.shape[-2:])
        # Follow upstream inference.py use_depth=True. The checkpoint's point head is not the
        # production reconstruction path; use predicted depth and predicted camera parameters.
        names=['gs_map','dynamic_conf','depth','depth_conf']
        result={n:p[n].float().cpu().numpy() for n in names}
        result['world_points']=self.unproject(result['depth'][0],e[0].float().cpu().numpy(),k[0].float().cpu().numpy())[None]
        result['world_points_conf']=result['depth_conf']
        return result,e[0,-1].float().cpu().numpy(),k[0,-1].float().cpu().numpy()

def cache(dataset,raw,output,upstream,checkpoint,width=518,context=4,threads=2,device='cuda',precision='fp32',resume=False,memory_fraction=0.35):
    if not 1<=context<=64:raise ValueError('context must be in [1,64]')
    dataset=Path(dataset);raw=Path(raw)
    if not 0<memory_fraction<=1:raise ValueError('invalid memory fraction')
    output=Path(output)
    if resume:output.mkdir(parents=True,exist_ok=True)
    else:output=fresh_dir(output)
    index=copy.deepcopy(read_json(dataset/'dataset.json'))
    import torch
    if str(device).startswith('cuda'):
        if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('DGGT caching requires exactly one CUDA_VISIBLE_DEVICES entry')
        torch.cuda.set_per_process_memory_fraction(memory_fraction,torch.device(device))
    checkpoint_hash=sha256(checkpoint)
    protocol={'input_dataset_sha256':sha256(dataset/'dataset.json'),
              'checkpoint_sha256':checkpoint_hash,'width':width,'context':context,'precision':precision}
    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=protocol:
        raise ValueError('resume protocol mismatch')
    write_json(output/'protocol.json',protocol)
    model=NativeDGGT(upstream,checkpoint,width,threads,device,precision)
    import subprocess
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    completed=0;total=0
    for episode in index['episodes']:
        with np.load(dataset/episode['path'],allow_pickle=False) as arrays:total+=len(arrays['frame_ids'])
    for e in index['episodes']:
        finalmeta=output/f'{e["scene_id"]}.dggt.json'
        if resume and finalmeta.exists() and (output/e['path']).exists():
            meta=read_json(finalmeta)
            if meta.get('output_sha256')==sha256(output/e['path']):
                e.update(sha256=meta['output_sha256'],geometry_source='dggt',geometry_metadata=finalmeta.name)
                completed+=len(meta['frames'])
                continue
        progress=output/'.progress'/e['scene_id'];progress.mkdir(parents=True,exist_ok=True)
        with np.load(dataset/e['path'],allow_pickle=False) as f:a=dict(f)
        source=read_json(dataset/e['sources'])
        history=[];tok=[];mask=[]
        for i,fid in enumerate(a['frame_ids']):
            start=max(0,i-context+1)
            paths=[raw/source['frames'][j]['left'] for j in range(start,i+1)]
            framefile=progress/f'{int(fid):010d}.npz'
            if resume and framefile.exists():
                with np.load(framefile,allow_pickle=False) as saved:
                    t=saved['tokens'];m=saved['mask'];record=json.loads(str(saved['record']))
            else:
                began=time.monotonic()
                pred,extri,k=model(paths)
                try:
                    t,m,scale=prediction_tokens(pred,extri,a['tokens'][i],a['token_mask'][i])
                except ValueError as error:
                    if str(error) not in ('insufficient historical stereo anchors for metric scale','invalid scale'):raise
                    t=np.zeros((64,16),np.float32);t[:,10]=1;t[:,7:10]=1e-5
                    m=np.zeros(64,bool)
                    scale={'metric_status':'INVALID','reason':str(error)}
                record={'target_frame_id':int(fid),'source_frame_ids':a['frame_ids'][start:i+1].tolist(),
                    'wall_seconds':time.monotonic()-began,'valid_tokens':int(m.sum()),**scale}
                temporary=framefile.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,tokens=t,mask=m,record=json.dumps(record))
                temporary.replace(framefile)
            tok.append(t);mask.append(m);history.append(record);completed+=1
            write_json(output/'status.json',{'status':'RUNNING','completed_frames':completed,'total_frames':total,
                       'scene_id':e['scene_id'],'frame_id':int(fid),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
            print(f'DGGT {e["scene_id"]} {i+1}/{len(a["frame_ids"])} total={completed}/{total}: {record["wall_seconds"]:.2f}s',flush=True)
        a['tokens']=np.stack(tok);a['token_mask']=np.stack(mask)
        np.savez_compressed(output/e['path'],**a)
        source.update({'geometry_source':'dggt','scale_reference':'historical_stereo_sgbm',
                       'dynamic_probability_available':True})
        write_json(output/e['sources'],source)
        meta=f'{e["scene_id"]}.dggt.json'
        write_json(output/meta,{'upstream':'https://github.com/xiaomi-research/dggt','upstream_commit':commit,
            'checkpoint_sha256':checkpoint_hash,'checkpoint_name':Path(checkpoint).name,
            'device':str(model.device),'precision':precision,'causal':True,'max_context_frames':context,'input_width':width,
            'resolution_note':'Native width' if width==518 else 'Reduced input width; not a native-resolution benchmark',
            'frames':history,'input_dataset_sha256':sha256(dataset/'dataset.json'),
            'output_sha256':sha256(output/e['path'])})
        shutil.rmtree(progress)
        e.update({'sha256':sha256(output/e['path']),'geometry_source':'dggt','geometry_metadata':meta})
    index['geometry_note']='Native pretrained DGGT, historical stereo scale anchoring; no road/drivability ground truth'
    write_json(output/'dataset.json',index)
    report=validate_dataset(output)
    write_json(output/'validation.json',{'status':'PASS','episodes':report})
    shutil.rmtree(output/'.progress',ignore_errors=True)
    write_json(output/'status.json',{'status':'COMPLETED','completed_frames':completed,'total_frames':total})
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    p.add_argument('--raw',default='data/kitti/raw');p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='third_party/dggt');p.add_argument('--checkpoint',default='checkpoints/model_latest_waymo.pth')
    p.add_argument('--width',type=int,default=518);p.add_argument('--context',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=0.35)
    a=p.parse_args()
    print(cache(**vars(a)))
if __name__=='__main__':main()
