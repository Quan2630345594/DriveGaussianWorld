import numpy as np
import pytest
from drive_gaussian_world.common import read_json,write_json
from drive_gaussian_world.data.schema import validate_dataset
from drive_gaussian_world.data.windows import DrivingWindows
from drive_gaussian_world.data.prepare import transition_labels,oxts_position,calibration,timestamps

def test_valid_dataset(dataset):
    assert len(validate_dataset(dataset))==3

def test_scene_split_leak_rejected(dataset):
    index=read_json(dataset/'dataset.json');index['episodes'][1]['scene_id']='train'
    write_json(dataset/'dataset.json',index)
    with pytest.raises(ValueError,match='multiple splits'):validate_dataset(dataset)

def test_corrupted_episode_rejected(dataset):
    p=dataset/'train.npz';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):validate_dataset(dataset)

def test_explicit_future_separation(dataset):
    ds=DrivingWindows(dataset,'train',3,4);batch=ds[0]
    assert set(batch['observation'])=={'rgb','tokens','mask','speed'}
    assert batch['observation']['rgb'].shape[0]==3
    assert batch['target']['rgb'].shape[0]==4
    assert batch['anchor']==8
    assert {e['split'] for e,a in ds.episodes}=={'train'}

def test_horizon_boundary(dataset):
    ds=DrivingWindows(dataset,'train',3,4)
    assert len(ds)==4
    assert ds[-1]['anchor']==20
    with pytest.raises(ValueError):DrivingWindows(dataset,'train',10,4)

def test_transition_geometry_and_wrap():
    pos=np.array([[0,0,0],[0,2,0],[0,4,0]],float);yaw=np.full(3,np.pi/2)
    actions,ego,dt=transition_labels(pos,yaw,np.ones(3)*5,np.array([0,.4,.8]))
    np.testing.assert_allclose(ego,[[2,0,0],[2,0,0]],atol=1e-6)
    np.testing.assert_allclose(actions,0,atol=1e-6)
    _,wrapped,_=transition_labels(pos,np.array([3.13,-3.13,-3.12]),np.ones(3)*5,np.array([0,.4,.8]))
    assert abs(wrapped[0,2])<.1

def test_calibration_and_nanosecond_clock(tmp_path):
    p=tmp_path/'calib.txt';p.write_text('P_rect_02: 100 0 50 0 0 100 20 0 0 0 1 0\nP_rect_03: 100 0 50 -54 0 100 20 0 0 0 1 0\n')
    k,b=calibration(p);assert b==.54;assert k[0,0]==100
    p.write_text('2011-09-26 13:00:00.000000001\n2011-09-26 13:00:00.100000002\n')
    np.testing.assert_allclose(timestamps(p),[0,.100000001],atol=1e-10)

def test_future_dggt_cache_rejected(dataset):
    idx=read_json(dataset/'dataset.json');e=idx['episodes'][0]
    e['geometry_source']='dggt';e['geometry_metadata']='bad.json'
    write_json(dataset/'bad.json',{'causal':True,'max_context_frames':2,
                                 'frames':[{'target_frame_id':4,'source_frame_ids':[0,8]}]})
    write_json(dataset/'dataset.json',idx)
    with pytest.raises(ValueError,match='future frame'):validate_dataset(dataset)
