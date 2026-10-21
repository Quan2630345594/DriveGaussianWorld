"""Explicit opt-in inference profiling; one backend per process for allocator isolation."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from drive_gaussian_world.common import configure_runtime,precision_context,write_json


def profile(backend='gru',output=None,device='cuda',precision='fp32',hidden=96,
            batch_size=4,history=3,horizon=4,warmup=3,iterations=20,seed=42,model_options=None):
    if min(hidden,batch_size,history,horizon,iterations)<1 or warmup<0:raise ValueError('invalid profiling dimensions')
    selected=configure_runtime(seed,2,device)
    precision_context(selected,precision)
    import torch
    from drive_gaussian_world.models import build_world_model
    model=build_world_model(backend,hidden,model_options=model_options).to(selected).eval()
    tokens=torch.zeros(batch_size,history,64,16,device=selected)
    tokens[...,2]=.5;tokens[...,3:7]=.5;tokens[...,7:10]=.01;tokens[...,10]=1;tokens[...,15]=.9
    observation={'rgb':torch.rand(batch_size,history,3,32,96,device=selected),'tokens':tokens,
                 'mask':torch.ones(batch_size,history,64,dtype=torch.bool,device=selected),
                 'speed':torch.full((batch_size,history),10.,device=selected)}
    actions=torch.zeros(batch_size,horizon,2,device=selected);dt=torch.full((batch_size,horizon),.1,device=selected)
    def synchronize():
        if selected.type=='cuda':torch.cuda.synchronize(selected)
    with torch.inference_mode():
        for _ in range(warmup):
            with precision_context(selected,precision):prediction=model(observation,actions,dt)
            del prediction
        synchronize()
        if selected.type=='cuda':torch.cuda.reset_peak_memory_stats(selected)
        timings=[]
        for _ in range(iterations):
            synchronize();start=time.perf_counter()
            with precision_context(selected,precision):prediction=model(observation,actions,dt)
            synchronize();timings.append(time.perf_counter()-start)
            del prediction
    mean=statistics.mean(timings)
    report={'schema_version':1,'backend':backend,'device':str(selected),'precision':precision,
            'parameters':sum(p.numel() for p in model.parameters()),'model_options':model_options or {},
            'batch_size':batch_size,'history':history,'horizon':horizon,'warmup':warmup,'iterations':iterations,'seed':seed,
            'mean_rollout_ms':mean*1000,'median_rollout_ms':statistics.median(timings)*1000,
            'rollouts_per_second':batch_size/mean,'predicted_frames_per_second':batch_size*horizon/mean,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(selected) if selected.type=='cuda' else None,
            'torch':str(torch.__version__),'torch_cuda_build':torch.version.cuda,
            'gpu_name':torch.cuda.get_device_name(selected) if selected.type=='cuda' else None,
            'scope':'Synthetic cached-scene inference, including history encoding and sampling; excludes DGGT, data loading and training.',
            'limitations':'Use one fresh process per backend/precision/sampling configuration. CUDA allocator statistics are not total board memory or training memory.'}
    if output:write_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['gru','rssm','diffusion'],default='gru')
    p.add_argument('--output');p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    for key,default in [('hidden',96),('batch-size',4),('history',3),('horizon',4),('warmup',3),('iterations',20),('seed',42)]:
        p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--model-options',type=json.loads,default={})
    print(json.dumps(profile(**vars(p.parse_args())),indent=2))

if __name__=='__main__':main()
