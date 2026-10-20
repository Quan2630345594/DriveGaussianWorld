import numpy as np
import pytest
from drive_gaussian_world.geometry.dggt import prediction_tokens

def prediction():
    h,w=28,98
    points=np.zeros((1,1,h,w,3),np.float32);points[...,2]=10
    gs=np.zeros((1,1,h,w,11),np.float32);gs[...,:4]=.5;gs[...,4:7]=.1;gs[...,7]=1
    return {'world_points':points,'gs_map':gs,'world_points_conf':np.ones((1,1,h,w))*2,
            'dynamic_conf':np.zeros((1,1,h,w,1))}

def test_stereo_scale_and_wxyz_rotation():
    reference=np.zeros((64,16),np.float32);reference[:,2]=.5
    e=np.eye(4)[:3];tok,mask,report=prediction_tokens(prediction(),e,reference,np.ones(64,bool))
    assert mask.all();assert report['stereo_scale']==2
    np.testing.assert_allclose(tok[:,2],.5)
    np.testing.assert_allclose(tok[:,10:14],np.tile([1,0,0,0],(64,1)))
    assert tok[0,6]==.5 # Activated opacity uses channel 3, not red channel.

def test_missing_metric_evidence_rejected():
    with pytest.raises(ValueError,match='insufficient historical'):
        prediction_tokens(prediction(),np.eye(4)[:3],np.zeros((64,16)),np.zeros(64,bool))
