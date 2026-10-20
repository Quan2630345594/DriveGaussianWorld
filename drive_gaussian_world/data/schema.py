"""Versioned, portable episode contract. Optical camera coordinates: right/down/forward."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from drive_gaussian_world.common import read_json, sha256

TOKEN_FIELDS = ['x/40m','y/40m','z/40m','red','green','blue','opacity',
                'scale_x/5m','scale_y/5m','scale_z/5m','qw','qx','qy','qz',
                'dynamic_probability','confidence']
TOKEN_DIM = len(TOKEN_FIELDS)

def validate_episode(path):
    with np.load(path, allow_pickle=False) as f:
        a = dict(f)
    t = len(a['rgb'])
    required = {'rgb':(t,3,32,96), 'tokens':(t,64,TOKEN_DIM), 'token_mask':(t,64),
                'timestamps':(t,), 'speed':(t,), 'yaw':(t,), 'position':(t,3),
                'actions':(t-1,2), 'dt':(t-1,), 'ego_delta':(t-1,3),
                'frame_ids':(t,)}
    for name, shape in required.items():
        if name not in a or a[name].shape != shape:
            raise ValueError(f'{name}: expected {shape}, got {a.get(name, np.empty(0)).shape}')
        if not np.isfinite(a[name]).all():
            raise ValueError(f'{name}: non-finite values')
    if a['rgb'].dtype != np.uint8 or a['token_mask'].dtype != bool:
        raise ValueError('RGB must be uint8; token mask must be boolean')
    if t < 8 or np.any(np.diff(a['timestamps']) <= 0) or np.any(a['dt'] <= 0):
        raise ValueError('invalid temporal ordering')
    if not np.allclose(a['dt'], np.diff(a['timestamps']), atol=1e-5):
        raise ValueError('dt inconsistent with time')
    if np.any(np.diff(a['frame_ids']) <= 0):
        raise ValueError('frame indices must increase')
    valid = a['tokens'][a['token_mask']]
    if not len(valid) or np.any(valid[:,2] <= 0):
        raise ValueError('valid Gaussians must be in front of camera')
    if np.any((valid[:,3:7] < 0) | (valid[:,3:7] > 1)) or np.any(valid[:,7:10] <= 0):
        raise ValueError('invalid activated Gaussian attributes')
    if not np.allclose(np.linalg.norm(valid[:,10:14],axis=-1),1,atol=1e-3):
        raise ValueError('quaternions must be unit wxyz')
    if float(a['speed'].mean()) < 1.0:
        raise ValueError('selected log is not a moving driving sequence')
    return {'frames':t,'valid_token_fraction':float(a['token_mask'].mean()),
            'mean_speed_m_s':float(a['speed'].mean()),'duration_s':float(np.ptp(a['timestamps']))}

def validate_dataset(root, verify_hashes=True):
    root=Path(root)
    index=read_json(root/'dataset.json')
    seen=set(); report=[]
    if index['schema_version'] != 1 or index['scenario'] != 'road_driving':
        raise ValueError('unsupported dataset schema or scenario')
    for e in index['episodes']:
        if e['scene_id'] in seen:
            raise ValueError('scene appears in multiple splits')
        seen.add(e['scene_id'])
        p=(root/e['path']).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('episode path escapes dataset')
        if verify_hashes and sha256(p) != e['sha256']:
            raise ValueError('episode hash mismatch')
        if e['geometry_source']=='dggt':
            m=read_json(root/e['geometry_metadata'])
            if not m.get('causal') or m['max_context_frames'] < 1:
                raise ValueError('DGGT cache is not causal')
            if any(max(c['source_frame_ids']) > c['target_frame_id'] for c in m['frames']):
                raise ValueError('future frame present in DGGT observation cache')
        report.append({'scene_id':e['scene_id'],'split':e['split'],**validate_episode(p)})
    if {e['split'] for e in index['episodes']} != {'train','val','test'}:
        raise ValueError('all three scene-level splits are required')
    return report
