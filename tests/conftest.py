import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['PYTHONNOUSERSITE']='1'
from pathlib import Path
import numpy as np
import pytest
from drive_gaussian_world.common import write_json,sha256

@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'fixture';root.mkdir();episodes=[]
    for split in ['train','val','test']:
        t=10;tokens=np.zeros((t,64,16),np.float32)
        tokens[:,:,:3]=[.01,.01,.5];tokens[:,:,3:7]=.5;tokens[:,:,7:10]=.01;tokens[:,:,10]=1;tokens[:,:,15]=.9
        a={'rgb':np.full((t,3,32,96),100,dtype=np.uint8),'tokens':tokens,
           'token_mask':np.ones((t,64),bool),'timestamps':np.arange(t)*.4,
           'speed':np.full(t,5,dtype=np.float32),'yaw':np.zeros(t,np.float32),
           'position':np.stack([np.arange(t)*2,np.zeros(t),np.zeros(t)],-1).astype(np.float32),
           'actions':np.zeros((t-1,2),np.float32),'dt':np.full(t-1,.4,np.float32),
           'ego_delta':np.tile(np.array([2,0,0],np.float32),(t-1,1)),'frame_ids':np.arange(t)*4}
        p=root/f'{split}.npz';np.savez_compressed(p,**a)
        episodes.append({'scene_id':split,'split':split,'path':p.name,'sha256':sha256(p),
                         'geometry_source':'unit_test_fixture'})
    write_json(root/'dataset.json',{'schema_version':1,'scenario':'road_driving','episodes':episodes})
    return root
