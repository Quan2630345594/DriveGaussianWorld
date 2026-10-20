"""Local inference API. Binds loopback; only historical observations reach model."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
from pathlib import Path
from drive_gaussian_world.common import configure_runtime,move_to,to_cpu,sha256

def serve(dataset,checkpoint,port=8765,device='cuda'):
    selected=configure_runtime(device=device)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from drive_gaussian_world.train import load_model
    from drive_gaussian_world.data.windows import DrivingWindows
    from drive_gaussian_world.demo import png_data
    model,ckpt=load_model(checkpoint,selected);cfg=ckpt['config']
    if sha256(Path(dataset)/'dataset.json')!=ckpt['dataset_sha256']:raise ValueError('dataset provenance mismatch')
    ds=DrivingWindows(dataset,'test',cfg['history'],cfg['horizon']);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,value):
            body=json.dumps(value,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/health':self.reply(200,{'status':'ok','project':'DriveGaussianWorld','device':str(selected),'windows':len(ds)})
            else:self.reply(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/api/rollout':self.reply(404,{'error':'not found'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 1<=length<=65536:raise ValueError('request body must be <=64 KB')
                request=json.loads(self.rfile.read(length));window=int(request.get('window',0))
                if not 0<=window<len(ds):raise ValueError('window outside dataset')
                a=np.asarray(request['actions'],dtype=np.float32);dt=float(request.get('dt',.4))
                if a.shape!=(cfg['horizon'],2) or not np.isfinite(a).all():raise ValueError('invalid action shape/value')
                if (np.abs(a[:,0])>8).any() or (np.abs(a[:,1])>1).any() or not .01<=dt<=2:
                    raise ValueError('scenario outside development envelope')
                batch=move_to(default_collate([ds[window]]),selected)
                used=torch.from_numpy(a)[None].to(selected)
                if cfg['no_actions']:used=torch.zeros_like(used)
                with lock,torch.inference_mode():
                    pred=model(batch['observation'],used,torch.full((1,len(a)),dt,device=selected))
                pred=to_cpu(pred)
                self.reply(200,{'rgb':[png_data(x) for x in pred['rgb'][0].numpy()],
                    'gaussians':pred['tokens'][0].tolist(),'surface_mask_probability':pred['mask_logits'][0].sigmoid().tolist(),
                    'ego_delta':pred['ego'][0].tolist(),'scene_id':batch['scene_id'][0],
                    'anchor':int(batch['anchor'][0]),'action_semantics':'research scenario acceleration/yaw-rate; not controls',
                    'deployment_eligible':False})
            except (ValueError,KeyError,TypeError,UnicodeDecodeError,json.JSONDecodeError) as e:self.reply(400,{'error':str(e)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Inference API listening on http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--port',type=int,default=8765);serve(**vars(p.parse_args()))
if __name__=='__main__':main()
