import sys
from drive_gaussian_world.__main__ import main

def test_validate_dispatch(dataset,monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['drive_gaussian_world','validate','--dataset',str(dataset)])
    main();assert 'train' in capsys.readouterr().out

def test_doctor_reports_cuda_default(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['drive_gaussian_world','doctor'])
    main();out=capsys.readouterr().out
    assert '"default_device": "cuda"' in out
    assert '"cuda_runtime_initialized": false' in out
