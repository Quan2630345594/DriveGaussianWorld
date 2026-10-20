import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

COMMANDS={'collect':'data.range_collect','prepare':'data.prepare','dggt-cache':'geometry.dggt',
          'serve':'serve','train':'train','evaluate':'evaluate','demo':'demo','experiments':'experiments','audit':'data.audit'}

def main():
    parser=argparse.ArgumentParser(prog='python -m drive_gaussian_world',description='DriveGaussianWorld CPU tools')
    parser.add_argument('command',choices=[*COMMANDS,'validate','doctor'])
    args,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    if args.command in COMMANDS:
        importlib.import_module('drive_gaussian_world.'+COMMANDS[args.command]).main()
    elif args.command=='validate':
        p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
        from drive_gaussian_world.data.schema import validate_dataset
        print(json.dumps(validate_dataset(a.dataset),indent=2))
    else:
        import os,platform
        from drive_gaussian_world.common import cpu_runtime,ROOT
        cpu_runtime()
        import torch
        print(json.dumps({'project':'DriveGaussianWorld','root':str(ROOT),'python':platform.python_version(),
            'torch':torch.__version__,'device_policy':'cpu','CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
            'cuda_runtime_initialized':torch.cuda.is_initialized(),
            'dependencies':{k:bool(importlib.util.find_spec(k)) for k in ['numpy','PIL','cv2','scipy','requests','accelerate']},
            'dggt_checkout':(ROOT/'third_party/dggt/dggt/models/vggt.py').exists(),
            'checkpoint_exists':(ROOT/'checkpoints/model_latest_waymo.pth').exists()},indent=2))
if __name__=='__main__':main()
