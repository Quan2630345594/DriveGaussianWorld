import argparse
import importlib
import importlib.util
import json
import sys

COMMANDS={'profile':'profile_model','download-drives':'data.download_drives','collect':'data.range_collect','prepare':'data.prepare','dggt-cache':'geometry.dggt',
          'serve':'serve','train':'train','evaluate':'evaluate','demo':'demo','experiments':'experiments','audit':'data.audit'}

def main():
    # Dispatch before parsing so subcommand --help reaches its own argument parser.
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        command=sys.argv[1];sys.argv=[sys.argv[0],*sys.argv[2:]]
        importlib.import_module('drive_gaussian_world.'+COMMANDS[command]).main()
        return
    parser=argparse.ArgumentParser(prog='python -m drive_gaussian_world',description='DriveGaussianWorld tools')
    parser.add_argument('command',choices=[*COMMANDS,'validate','doctor'])
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    if args.command in COMMANDS:
        importlib.import_module('drive_gaussian_world.'+COMMANDS[args.command]).main()
    elif args.command=='validate':
        p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
        from drive_gaussian_world.data.schema import validate_dataset
        print(json.dumps(validate_dataset(a.dataset),indent=2))
    else:
        import platform
        import torch
        from drive_gaussian_world.common import ROOT
        print(json.dumps({'project':'DriveGaussianWorld','root':str(ROOT),'python':platform.python_version(),
            'torch':str(torch.__version__),'default_device':'cuda','cuda_available':torch.cuda.is_available(),
            'cuda_runtime_initialized':torch.cuda.is_initialized(),
            'dependencies':{k:bool(importlib.util.find_spec(k)) for k in ['numpy','PIL','cv2','scipy','requests','accelerate']},
            'dggt_checkout':(ROOT/'third_party/dggt/dggt/models/vggt.py').exists(),
            'checkpoint_exists':(ROOT/'checkpoints/model_latest_waymo.pth').exists()},indent=2))
if __name__=='__main__':main()
