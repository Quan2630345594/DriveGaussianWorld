"""Bounded single-GPU RSSM pilot; select on validation, evaluate test once."""
import json
import os
import time
from pathlib import Path

if not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
    raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one physical GPU')

import torch
from torch.utils.data import DataLoader
from drive_gaussian_world.common import configure_runtime, move_to, write_json, sha256
from drive_gaussian_world.data.schema import validate_dataset
from drive_gaussian_world.data.windows import DrivingWindows
from drive_gaussian_world.models import build_world_model, world_loss
from drive_gaussian_world.evaluate import evaluate


def main():
    device = configure_runtime(42, 2, 'cuda:0')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(0.12, 0)
    root = Path('data/processed/kitti_dggt')
    validate_dataset(root)
    out = Path(os.environ['DGW_RUN_OUTPUT'])
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(dataset=str(root), hidden=256, history=4, horizon=8,
               backend='rssm', model_options={'stochastic':64}, no_geometry=False,
               no_actions=False, seed=42, batch_size=8, lr=0.0003,
               device='cuda:0', precision='fp32')
    splits = {s: DrivingWindows(root, s, 4, 8) for s in ('train','val','test')}
    ids = {s: [e['scene_id'] for e,a in ds.episodes] for s,ds in splits.items()}
    assert all(not set(ids[a]) & set(ids[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    write_json(out/'protocol.json', dict(config=cfg, split_scenes=ids,
        windows={s:len(d) for s,d in splits.items()}, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        memory_budget_fraction=0.12, max_steps=2000, max_seconds=1800,
        patience_evaluations=15, min_delta=0.0001, min_steps=500,
        limitations=['Small cached DGGT pilot, not expanded dataset or cross-dataset evaluation.',
                      'Motion-derived future conditions, not issued controls; pseudo-geometry targets.']))
    model = build_world_model('rssm',256,True,cfg['model_options']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    loader = DataLoader(splits['train'],batch_size=8,shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    validation = DataLoader(splits['val'],batch_size=8)
    best = float('inf'); reference = float('inf'); stale=0; start=time.monotonic()
    reason='max_steps'; iterator=iter(loader); rows=[]
    for step in range(1,2001):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        batch=move_to(batch,device); model.train(); optimizer.zero_grad(set_to_none=True)
        loss, logs=model.training_loss(batch['observation'],batch['condition']['actions'],batch['condition']['dt'],batch['target'])
        if not torch.isfinite(loss): reason='nonfinite_loss'; break
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): reason='nonfinite_gradient'; break
        optimizer.step()
        if step==1 or step%25==0:
            model.eval(); total=0.; count=0
            with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                torch.manual_seed(42)
                for vb in validation:
                    vb=move_to(vb,device)
                    pred=model(vb['observation'],vb['condition']['actions'],vb['condition']['dt'])
                    vl,_=world_loss(pred,vb['target']); n=len(vb['condition']['actions'])
                    total+=vl.item()*n; count+=n
            val=total/count
            row=dict(step=step,train=logs,val_total=val,elapsed_s=time.monotonic()-start,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm))
            rows.append(row)
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if val<best:
                best=val
                torch.save(dict(model=model.state_dict(),config=cfg,step=step,
                    dataset_sha256=sha256(root/'dataset.json'),selection_split='val'),out/'best.pt')
            if val<reference-0.0001: reference=val; stale=0
            else: stale+=1
            if step>=500 and stale>=15: reason='validation_plateau'; break
        if time.monotonic()-start>=1800: reason='time_budget'; break
    write_json(out/'training_summary.json',dict(status='STOPPED',reason=reason,steps=step,
        best_val_loss=best,wall_seconds=time.monotonic()-start,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    del model, optimizer
    torch.cuda.empty_cache()
    evaluate(root,out/'best.pt',out/'test_evaluation.json',batch_size=4,device='cuda:0',samples=5)
    # Save a real held-out RGB rollout without temporary prediction tensors/files.
    from drive_gaussian_world.train import load_model
    from PIL import Image, ImageDraw
    model,checkpoint=load_model(out/'best.pt',device)
    batch=move_to(next(iter(DataLoader(splits['test'],batch_size=1))),device)
    torch.manual_seed(42)
    with torch.inference_mode():
        pred=model(batch['observation'],batch['condition']['actions'],batch['condition']['dt'])
    frames=[]
    for t in range(8):
        frame=Image.new('RGB',(768,160)); draw=ImageDraw.Draw(frame)
        for j,(name,rgb) in enumerate([('Ground truth',batch['target']['rgb'][0,t]),('RSSM prediction',pred['rgb'][0,t])]):
            im=Image.fromarray((rgb.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype('uint8'))
            frame.paste(im.resize((384,128)),(384*j,28)); draw.text((384*j+8,6),name,fill='white')
        frames.append(frame)
    frames[0].save(out/'test_rollout.gif',save_all=True,append_images=frames[1:],duration=400,loop=0)
    # Portable learning curve, no plotting dependency or intermediate files.
    from html import escape
    low=min(r['val_total'] for r in rows); high=max(r['val_total'] for r in rows)
    points=' '.join(f'{40+720*r["step"]/rows[-1]["step"]:.1f},{260-220*(r["val_total"]-low)/max(high-low,1e-8):.1f}' for r in rows)
    (out/'validation_curve.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300"><rect width="800" height="300" fill="white"/><text x="30" y="20">RSSM validation loss: {high:.4f} to {low:.4f}; {escape(reason)}</text><polyline points="{points}" fill="none" stroke="blue"/><text x="40" y="290">Optimizer steps (0 to {rows[-1]["step"]})</text></svg>')
    write_json(out/'status.json',dict(status='COMPLETED',stop_reason=reason,best_step=checkpoint['step']))


if __name__=='__main__': main()
