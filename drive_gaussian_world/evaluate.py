"""Held-out conditional and no-future-action evaluations, plus analytic baselines."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from drive_gaussian_world.common import configure_runtime,move_to,write_json,read_json,sha256

def trajectory(ego):
    import torch
    x=ego.new_zeros(ego.shape[0]);y=x.clone();yaw=x.clone();out=[]
    for i in range(ego.shape[1]):
        dx,dy,da=ego[:,i].unbind(-1)
        x,y=x+torch.cos(yaw)*dx-torch.sin(yaw)*dy,y+torch.sin(yaw)*dx+torch.cos(yaw)*dy
        yaw=yaw+da;out.append(torch.stack([x,y],-1))
    return torch.stack(out,1)

def baseline(batch,kind='persistence'):
    import torch
    from drive_gaussian_world.models.world import warp_gaussians
    o=batch['observation'];a=batch['condition']['actions'];dt=batch['condition']['dt']
    image=o['rgb'][:,-1];tokens=o['tokens'][:,-1];speed=o['speed'][:,-1]
    gs=[];eg=[]
    for i in range(a.shape[1]):
        warped,speed,ego=warp_gaussians(tokens,speed,a[:,i],dt[:,i])
        if kind=='action_warp':tokens=warped
        gs.append(tokens);eg.append(ego)
    return {'rgb':image[:,None].expand(-1,a.shape[1],-1,-1,-1),
            'tokens':torch.stack(gs,1),'ego':torch.stack(eg,1),
            'mask_logits':torch.logit(o['mask'][:,-1].float().clamp(.001,.999))[:,None].expand(-1,a.shape[1],-1)}

def scores(pred,target):
    import torch
    mse=(pred['rgb']-target['rgb']).square().mean((2,3,4))
    l1=(pred['rgb']-target['rgb']).abs().mean((2,3,4))
    psnr=-10*torch.log10(mse.clamp_min(1e-10))
    errors=torch.linalg.vector_norm(trajectory(pred['ego'])-trajectory(target['ego']),dim=-1)
    gtmask=target['mask'];pmask=pred['mask_logits'].sigmoid()>.5
    chamfer=[];coverage=[]
    # Masked symmetric surface distance in metres; these are pseudo-geometry targets.
    for b in range(len(gtmask)):
        distances=[]
        for t in range(gtmask.shape[1]):
            gt=target['tokens'][b,t,gtmask[b,t],:3]*40
            pr=pred['tokens'][b,t,pmask[b,t],:3]*40
            if len(gt) and len(pr):
                d=torch.cdist(pr[None],gt[None])[0]
                distances.append((d.min(0).values.mean()+d.min(1).values.mean()).item()/2)
            else:distances.append(80.)
        chamfer.append(np.mean(distances))
        intersection=(gtmask[b]&pmask[b]).sum().item();union=(gtmask[b]|pmask[b]).sum().item()
        coverage.append(intersection/max(union,1))
    return {'rgb_l1':l1.mean(1).tolist(),'psnr_db':psnr.mean(1).tolist(),
            'ego_ade_m':errors.mean(1).tolist(),'ego_fde_m':errors[:,-1].tolist(),
            'gaussian_chamfer_m':chamfer,'surface_mask_iou':coverage,
            **{f'psnr_step_{i+1}_db':psnr[:,i].tolist() for i in range(psnr.shape[1])}}

def evaluate(dataset,checkpoint,output,split='test',batch_size=4,threads=2,device='cuda',samples=1,seed=42):
    if samples<1:raise ValueError('samples must be positive')
    selected=configure_runtime(seed,threads,device)
    import torch
    from torch.utils.data import DataLoader
    from drive_gaussian_world.data.windows import DrivingWindows
    from drive_gaussian_world.train import load_model
    model,ckpt=load_model(checkpoint,selected);cfg=ckpt['config']
    if sha256(Path(dataset)/'dataset.json')!=ckpt['dataset_sha256']:
        raise ValueError('checkpoint dataset provenance mismatch')
    ds=DrivingWindows(dataset,split,cfg['history'],cfg['horizon'])
    results={name:{} for name in ['learned_logged_motion','learned_current_motion','persistence','action_warp','constant_speed']}
    stochastic={}
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            batch=move_to(batch,selected)
            actions=batch['condition']['actions'];dt=batch['condition']['dt']
            used=torch.zeros_like(actions) if cfg['no_actions'] else actions
            state=model.observe(batch['observation'])
            logged=model.rollout(state,used,dt)
            # An available-now scenario uses zero acceleration and yaw rate estimated from history is unavailable
            # in this observation contract; keep a straight constant-speed condition, with no target-derived action.
            current=model.rollout(state,torch.zeros_like(actions),dt)
            if samples>1:
                candidates=[logged]+[model(batch['observation'],used,dt) for _ in range(samples-1)]
                errors=torch.stack([torch.linalg.vector_norm(trajectory(p['ego'])-trajectory(batch['target']['ego']),dim=-1).mean(1) for p in candidates])
                xy=torch.stack([trajectory(p['ego']) for p in candidates])
                values={'mean_sample_ade_m':errors.mean(0).tolist(),'best_of_k_ade_m':errors.min(0).values.tolist(),
                        'trajectory_std_m':xy.std(0,unbiased=False).norm(dim=-1).mean(1).tolist()}
                for key,value in values.items():stochastic.setdefault(key,[]).extend(value)
            available_now={'observation':batch['observation'],
                           'condition':{'actions':torch.zeros_like(actions),'dt':dt}}
            predictions={'learned_logged_motion':logged,'learned_current_motion':current,
                         'persistence':baseline(batch),'action_warp':baseline(batch,'action_warp'),
                         'constant_speed':baseline(available_now,'action_warp')}
            for name,pred in predictions.items():
                for key,value in scores(pred,batch['target']).items():results[name].setdefault(key,[]).extend(value)
    summary={name:{k:float(np.mean(v)) for k,v in values.items()} for name,values in results.items()}
    report={'schema_version':2,'backend':cfg.get('backend','gru'),'sampling':{'count':samples,'seed':seed,'model_options':cfg.get('model_options',{})},
        'stochastic_metrics':{k:float(np.mean(v)) for k,v in stochastic.items()},'device':str(selected),'split':split,'windows':len(ds),'scene_ids':[e['scene_id'] for e,a in ds.episodes],
        'metrics':summary,'checkpoint_sha256':sha256(checkpoint),'dataset_sha256':ckpt['dataset_sha256'],
        'ablation':{'no_geometry':cfg['no_geometry'],'no_actions':cfg['no_actions']},
        'conditioning':{'learned_logged_motion':'Future motion-derived labels supplied as scenario; conditional prediction, not autonomous forecasting.',
                        'learned_current_motion':'Zero acceleration/yaw condition; no future motion labels supplied.',
                        'action_warp':'Analytic ego and Gaussian rigid-motion baseline, supplied logged motion.',
                        'constant_speed':'No future action labels; zero acceleration/yaw analytic ego and geometry, last image.',
                        'persistence':'Last image/surfaces; analytic ego using supplied logged motion.'},
        'limitations':['Held-out windows overlap and are correlated; see scene_ids for the evaluated drives.',
                      'Gaussian metrics compare DGGT/stereo pseudo-geometry, not independent 3D ground truth.',
                      'No closed-loop driving, occupancy truth, safety or benchmark claim.']}
    write_json(output,report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--samples',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    print(evaluate(**vars(p.parse_args())))
if __name__=='__main__':main()
