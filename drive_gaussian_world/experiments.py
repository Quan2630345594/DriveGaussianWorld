"""Run bounded CPU experiments from one config, with ablations and portable reports."""
from __future__ import annotations
import argparse
import traceback
from pathlib import Path
from drive_gaussian_world.common import cpu_runtime,read_json,write_json,fresh_dir,sha256

def run(dataset,output,config='configs/cpu_smoke.json',stereo_dataset=None):
    cpu_runtime()
    from drive_gaussian_world.train import train
    from drive_gaussian_world.evaluate import evaluate
    from drive_gaussian_world.demo import make_demo
    cfg=read_json(config);out=fresh_dir(output)
    if cfg['device']!='cpu':raise ValueError('this runner only permits CPU')
    write_json(out/'config.json',cfg)
    status={'status':'RUNNING','dataset_sha256':sha256(Path(dataset)/'dataset.json'),'stages':[]}
    write_json(out/'status.json',status)
    experiments=[('dggt_world',dataset,False,False),('rgb_only',dataset,True,False),
                 ('no_actions',dataset,False,True)]
    if stereo_dataset:experiments.append(('stereo_world',stereo_dataset,False,False))
    summaries={}
    try:
        for name,data,no_geom,no_action in experiments:
            status['active_stage']=name;write_json(out/'status.json',status)
            checkpoint=train(data,out/name,**cfg['training'],**cfg['model'],seed=cfg['seed'],
                threads=cfg['threads'],no_geometry=no_geom,no_actions=no_action)
            evaluation=evaluate(data,checkpoint,out/name/'test.json')
            summaries[name]=evaluation
            status['stages'].append({'name':name,'status':'PASS','checkpoint':f'{name}/best.pt'})
            write_json(out/'status.json',status)
        make_demo(dataset,out/'dggt_world/best.pt',out/'demo')
        status.update({'status':'COMPLETED','active_stage':None})
        write_json(out/'comparison.json',summaries)
        write_json(out/'status.json',status)
    except Exception as e:
        status.update({'status':'FAILED','error':str(e),'traceback':traceback.format_exc()})
        write_json(out/'status.json',status);raise
    return str(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/cpu_smoke.json');p.add_argument('--stereo-dataset')
    print(run(**vars(p.parse_args())))
if __name__=='__main__':main()
