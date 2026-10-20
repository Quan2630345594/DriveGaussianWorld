"""Bounded CPU training entry point, full provenance, validation-only model selection."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from drive_gaussian_world.common import cpu_runtime,fresh_dir,read_json,write_json,sha256

def train(dataset,output,steps=200,batch_size=4,history=3,horizon=4,hidden=96,lr=0.0003,
          seed=42,threads=2,no_geometry=False,no_actions=False):
    if not 1<=steps<=2000 or not 1<=batch_size<=16 or not 1<=threads<=4 or not 16<=hidden<=256:
        raise ValueError('CPU development budget exceeded; use a separately reviewed training plan')
    cpu_runtime(seed,threads)
    import torch
    from torch.utils.data import DataLoader
    from drive_gaussian_world.data.schema import validate_dataset
    from drive_gaussian_world.data.windows import DrivingWindows
    from drive_gaussian_world.models.world import GaussianWorldModel,world_loss
    dataset=Path(dataset);output=fresh_dir(output)
    validate_dataset(dataset)
    configuration={'dataset':str(dataset),'steps':steps,'batch_size':batch_size,'history':history,
        'horizon':horizon,'hidden':hidden,'lr':lr,'seed':seed,'threads':threads,
        'no_geometry':no_geometry,'no_actions':no_actions,'device':'cpu'}
    write_json(output/'config.json',configuration)
    trainset=DrivingWindows(dataset,'train',history,horizon);valset=DrivingWindows(dataset,'val',history,horizon)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0,generator=generator)
    validation=DataLoader(valset,batch_size=batch_size,shuffle=False,num_workers=0)
    model=GaussianWorldModel(hidden,not no_geometry).cpu();optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    best=float('inf');records=[];start=time.monotonic();iterator=iter(loader)
    for step in range(1,steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        model.train();optimizer.zero_grad(set_to_none=True)
        actions=batch['condition']['actions']
        if no_actions:actions=torch.zeros_like(actions)
        prediction=model(batch['observation'],actions,batch['condition']['dt'])
        loss,losses=world_loss(prediction,batch['target'])
        if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%25==0 or step==steps:
            model.eval();vtotal=0.;count=0
            with torch.inference_mode():
                for vb in validation:
                    va=vb['condition']['actions']
                    if no_actions:va=torch.zeros_like(va)
                    vp=model(vb['observation'],va,vb['condition']['dt'])
                    vl,_=world_loss(vp,vb['target']);n=len(va)
                    vtotal+=vl.item()*n;count+=n
            validation_loss=vtotal/count
            record={'step':step,'train':losses,'val_total':validation_loss,
                    'gradient_norm':float(norm),'elapsed_s':time.monotonic()-start}
            records.append(record)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if validation_loss<best:
                best=validation_loss
                torch.save({'model':model.state_dict(),'config':configuration,'step':step,
                    'dataset_sha256':sha256(dataset/'dataset.json'),'selection_split':'val'},output/'best.pt')
            print(f'step {step}/{steps} train={loss.item():.4f} val={validation_loss:.4f} elapsed={record["elapsed_s"]:.1f}s',flush=True)
    write_json(output/'training_summary.json',{'status':'COMPLETED','device':'cpu','seed':seed,
        'steps':steps,'train_windows':len(trainset),'val_windows':len(valset),
        'parameters':sum(p.numel() for p in model.parameters()),'best_val_loss':best,
        'wall_seconds':time.monotonic()-start,'checkpoint_sha256':sha256(output/'best.pt'),
        'torch':torch.__version__,'python':platform.python_version(),
        'geometry_sources':sorted({e['geometry_source'] for e in read_json(dataset/'dataset.json')['episodes']}),
        'claim_scope':'Small CPU engineering run. No large-scale training or benchmark claims.'})
    return str(output/'best.pt')

def load_model(checkpoint):
    import torch
    from drive_gaussian_world.models.world import GaussianWorldModel
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ckpt['config'];model=GaussianWorldModel(cfg['hidden'],not cfg['no_geometry']).cpu()
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    return model,ckpt

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    for name,default in [('steps',200),('batch-size',4),('history',3),('horizon',4),('hidden',96),('seed',42),('threads',2)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=0.0003);p.add_argument('--no-geometry',action='store_true');p.add_argument('--no-actions',action='store_true')
    print(train(**vars(p.parse_args())))
if __name__=='__main__':main()
