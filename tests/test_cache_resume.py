import json
from pathlib import Path

import numpy as np
import pytest

from drive_gaussian_world.common import read_json, write_json
from drive_gaussian_world.geometry import dggt


def test_frame_resume_preserves_causality_and_cleans_shards(dataset, tmp_path, monkeypatch):
    index=read_json(dataset/'dataset.json')
    for e in index['episodes']:
        e['sources']=e['scene_id']+'.sources.json'
        write_json(dataset/e['sources'],{'frames':[{'left':f'{i}.png','frame_id':i*4} for i in range(10)]})
    write_json(dataset/'dataset.json',index)
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'fixture')
    calls=[]; fail=[True]
    class FakeDGGT:
        device='cpu'
        def __init__(self,*args):pass
        def __call__(self,paths):
            calls.append(paths)
            if fail[0] and len(calls)==4:raise RuntimeError('interrupted')
            return {},None,None
    def tokens(pred,extri,reference,mask):
        return reference,mask,{'metric_status':'fixture'}
    monkeypatch.setattr(dggt,'NativeDGGT',FakeDGGT)
    monkeypatch.setattr(dggt,'prediction_tokens',tokens)
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'fixture\n')
    out=tmp_path/'cache'
    args=dict(dataset=dataset,raw=tmp_path,output=out,upstream=tmp_path,
              checkpoint=checkpoint,device='cpu',context=2,resume=True)
    with pytest.raises(RuntimeError,match='interrupted'):dggt.cache(**args)
    assert len(list((out/'.progress'/'train').glob('*.npz')))==3
    fail[0]=False
    dggt.cache(**args)
    assert len(calls)==31  # 30 frames plus the failed call; completed frames were reused.
    assert not (out/'.progress').exists()
    status=read_json(out/'status.json')
    assert status['completed_frames']==status['total_frames']==30
    import importlib.util
    spec=importlib.util.spec_from_file_location('verify_cache',Path('scripts/verify_dggt_cache.py'))
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    report=verifier.verify(dataset,out)
    assert report['frames']==30 and report['logs']==3
    assert report['invalid_geometry_frames']==0
    before=len(calls)
    dggt.cache(**args)
    assert len(calls)==before
    with pytest.raises(ValueError,match='protocol mismatch'):dggt.cache(**{**args,'context':3})


def test_cache_requires_one_visible_gpu(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='exactly one'):
        dggt.cache(dataset,tmp_path,tmp_path/'cache',tmp_path,tmp_path/'weights')
