import os
import pytest
import torch
from drive_gaussian_world.common import configure_runtime,resolve_device,move_to,to_cpu,precision_context

def test_cpu_runtime_preserves_cuda_visibility(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,5')
    assert configure_runtime(device='cpu').type=='cpu'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5'

def test_cuda_selection_and_explicit_fail_fast(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    assert str(resolve_device('cuda:2'))=='cuda:2'
    assert resolve_device('auto').type=='cuda'
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA is unavailable'):resolve_device('cuda')
    assert resolve_device('auto').type=='cpu'

def test_nested_batch_transfer_and_export():
    data={'tensor':torch.ones(2,requires_grad=True),'mask':torch.tensor([True]),'meta':['scene',4]}
    copied=move_to(data,torch.device('cpu'));result=to_cpu(copied)
    assert result['tensor'].device.type=='cpu'
    assert not result['tensor'].requires_grad
    assert result['meta']==['scene',4]
    assert result['mask'].dtype==torch.bool

def test_cpu_rejects_cuda_precision():
    with pytest.raises(ValueError,match='requires CUDA'):precision_context(torch.device('cpu'),'fp16')
    with pytest.raises(ValueError):precision_context(torch.device('cpu'),'invalid')

def test_default_cuda_is_not_silently_downgraded(dataset,tmp_path,monkeypatch):
    from drive_gaussian_world.train import train
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA is unavailable'):
        train(dataset,tmp_path/'must_not_exist',steps=1)
    assert not (tmp_path/'must_not_exist').exists()

def test_bf16_checks_the_selected_gpu(monkeypatch):
    from contextlib import contextmanager
    selected=[]
    @contextmanager
    def context(device):
        selected.append(str(device));yield
    monkeypatch.setattr(torch.cuda,'device',context)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    with pytest.raises(RuntimeError,match='BF16 is not supported'):
        precision_context(torch.device('cuda:2'),'bf16')
    assert selected==['cuda:2']
