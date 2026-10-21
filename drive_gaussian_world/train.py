"""Action-conditioned world-model training with explicit device and precision control."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
from drive_gaussian_world.common import configure_runtime,resolve_device,move_to,precision_context,fresh_dir,read_json,write_json,sha256

def train(dataset,output,steps=200,batch_size=4,history=3,horizon=4,hidden=96,lr=0.0003,
          seed=42,threads=2,no_geometry=False,no_actions=False,device='cuda',precision='fp32',num_workers=0,backend='gru',model_options=None):
    if min(steps,batch_size,history,horizon,hidden,threads)<=0 or num_workers<0 or lr<=0:
        raise ValueError('invalid training hyperparameters')
    selected=configure_runtime(seed,threads,device)
    # Validate precision before creating artifacts or loading the data.
    precision_context(selected,precision)
    import torch
    from torch.utils.data import DataLoader
    from drive_gaussian_world.data.schema import validate_dataset
    from drive_gaussian_world.data.windows import DrivingWindows
    from drive_gaussian_world.models import build_world_model,world_loss
    dataset=Path(dataset)
    validate_dataset(dataset)
    output=fresh_dir(output)
    configuration={'dataset':str(dataset),'steps':steps,'batch_size':batch_size,'history':history,
        'horizon':horizon,'hidden':hidden,'lr':lr,'seed':seed,'threads':threads,
        'no_geometry':no_geometry,'no_actions':no_actions,'device':str(selected),
        'precision':precision,'num_workers':num_workers,'backend':backend,'model_options':model_options or {}}
    write_json(output/'config.json',configuration)
    trainset=DrivingWindows(dataset,'train',history,horizon)
    valset=DrivingWindows(dataset,'val',history,horizon)
    generator=torch.Generator().manual_seed(seed)
    options={'batch_size':batch_size,'num_workers':num_workers,'pin_memory':selected.type=='cuda'}
    loader=DataLoader(trainset,shuffle=True,generator=generator,**options)
    validation=DataLoader(valset,shuffle=False,**options)
    model=build_world_model(backend,hidden,not no_geometry,model_options).to(selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    scaler=torch.amp.GradScaler('cuda',enabled=selected.type=='cuda' and precision=='fp16')
    best=float('inf');start=time.monotonic();iterator=iter(loader)
    for step in range(1,steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch=move_to(batch,selected)
        model.train();optimizer.zero_grad(set_to_none=True)
        actions=batch['condition']['actions']
        if no_actions:actions=torch.zeros_like(actions)
        with precision_context(selected,precision):
            loss,losses=model.training_loss(batch['observation'],actions,batch['condition']['dt'],batch['target'])
        if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        if step==1 or step%25==0 or step==steps:
            model.eval();vtotal=0.;count=0
            # Fixed validation RNG gives each checkpoint the same stochastic draws.
            rng_devices=[selected.index if selected.index is not None else torch.cuda.current_device()] if selected.type=='cuda' else []
            with torch.random.fork_rng(devices=rng_devices),torch.inference_mode():
                torch.random.default_generator.manual_seed(seed)
                if selected.type=='cuda':
                    with torch.cuda.device(selected):torch.cuda.manual_seed(seed)
                for vb in validation:
                    vb=move_to(vb,selected);va=vb['condition']['actions']
                    if no_actions:va=torch.zeros_like(va)
                    with precision_context(selected,precision):
                        vp=model(vb['observation'],va,vb['condition']['dt'])
                        vl,_=world_loss(vp,vb['target'])
                    n=len(va);vtotal+=vl.item()*n;count+=n
            validation_loss=vtotal/count
            record={'step':step,'train':losses,'val_total':validation_loss,
                    'gradient_norm':float(norm),'elapsed_s':time.monotonic()-start}
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if validation_loss<best:
                best=validation_loss
                torch.save({'model':model.state_dict(),'config':configuration,'step':step,
                    'dataset_sha256':sha256(dataset/'dataset.json'),'selection_split':'val'},output/'best.pt')
            print(f'step {step}/{steps} train={loss.item():.4f} val={validation_loss:.4f} device={selected}',flush=True)
    if selected.type=='cuda':torch.cuda.synchronize(selected)
    write_json(output/'training_summary.json',{'status':'COMPLETED','device':str(selected),'seed':seed,
        'precision':precision,'backend':backend,'model_options':model_options or {},'steps':steps,'train_windows':len(trainset),'val_windows':len(valset),
        'parameters':sum(p.numel() for p in model.parameters()),'best_val_loss':best,
        'wall_seconds':time.monotonic()-start,'checkpoint_sha256':sha256(output/'best.pt'),
        'torch':str(torch.__version__),'python':platform.python_version(),
        'geometry_sources':sorted({e['geometry_source'] for e in read_json(dataset/'dataset.json')['episodes']})})
    return str(output/'best.pt')

def load_model(checkpoint,device='cuda'):
    import torch
    from drive_gaussian_world.models import build_world_model
    selected=resolve_device(device)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ckpt['config'];model=build_world_model(cfg.get('backend','gru'),cfg['hidden'],not cfg['no_geometry'],cfg.get('model_options'))
    model.load_state_dict(ckpt['model'],strict=True);model.to(selected).eval()
    return model,ckpt

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    for name,default in [('steps',20000),('batch-size',32),('history',3),('horizon',4),('hidden',96),('seed',42),('threads',2),('num-workers',0)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--backend',choices=['gru','rssm','diffusion'],default='gru')
    p.add_argument('--model-options',type=json.loads,default={},help='Backend options as a JSON object')
    p.add_argument('--device',default='cuda',help='cuda, cuda:N, cpu or auto')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--no-geometry',action='store_true');p.add_argument('--no-actions',action='store_true')
    print(train(**vars(p.parse_args())))
if __name__=='__main__':main()
