"""Tiny tensor contracts only; no DGGT weights, dataset downloads or GPU work."""
import json
import pytest
import torch
from torch.utils.data import default_collate
from drive_gaussian_world.data.windows import DrivingWindows
from drive_gaussian_world.models import build_world_model
from drive_gaussian_world.models.world import world_loss


def make_model(name):
    options={'diffusion_steps':8,'sampling_steps':2} if name=='diffusion' else {'stochastic':8,'free_nats':0.} if name=='rssm' else {}
    return build_world_model(name,hidden=32,model_options=options)


def get_batch(dataset):
    return default_collate([DrivingWindows(dataset)[0],DrivingWindows(dataset)[1]])


@pytest.mark.parametrize('name',['gru','rssm','diffusion'])
def test_backend_loss_and_causal_rollout(name,dataset):
    batch=get_batch(dataset);model=make_model(name)
    loss,logs=model.training_loss(batch['observation'],**batch['condition'],target=batch['target'])
    assert torch.isfinite(loss);assert logs['total']==pytest.approx(loss.item())
    loss.backward()
    assert model.dynamics.weight_hh.grad.abs().sum()>0
    if name=='rssm':
        assert model.prior[-1].weight.grad.abs().sum()>0
        assert model.posterior[-1].weight.grad.abs().sum()>0
    if name=='diffusion':
        assert model.denoiser[-1].weight.grad.abs().sum()>0
        assert model.latent_encoder[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(17);first=model(batch['observation'],**batch['condition'])
        batch['target']['rgb'].fill_(1);batch['target']['tokens'].fill_(30)
        torch.manual_seed(17);second=model(batch['observation'],**batch['condition'])
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    assert first['rgb'].shape==(2,4,3,32,96)
    assert first['tokens'].shape==(2,4,64,16)
    assert first['ego'].shape==(2,4,3)
    bad={**batch['observation'],'future_rgb':batch['target']['rgb']}
    with pytest.raises(ValueError,match='unexpected'):model.observe(bad)


@pytest.mark.parametrize('name',['rssm','diffusion'])
def test_sampling_seed_diversity_and_action_response(name,dataset):
    b=get_batch(dataset);m=make_model(name).eval()
    with torch.no_grad():
        x=m.sample_predictions(b['observation'],**b['condition'],count=3,seed=19)
        y=m.sample_predictions(b['observation'],**b['condition'],count=3,seed=19)
        for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        assert not torch.allclose(x['tokens'][0],x['tokens'][1])
        p=m(b['observation'],**b['condition'],sample=False)
        q=m(b['observation'],**b['condition'],sample=False)
        torch.testing.assert_close(p['rgb'],q['rgb'],rtol=0,atol=0)
        a=b['condition']['actions'].clone();a[:,:,1]=.3
        q=m(b['observation'],a,b['condition']['dt'],sample=False)
        assert not torch.allclose(p['ego'],q['ego'])
        assert not torch.allclose(p['tokens'],q['tokens'])


@pytest.mark.parametrize('name',['rssm','diffusion'])
def test_stochastic_rgb_only_removes_geometry(name,dataset):
    b=get_batch(dataset);m=make_model(name).eval();m.use_geometry=False
    with torch.no_grad():
        first=m(b['observation'],**b['condition'],sample=False)
        b['observation']['tokens'].fill_(10);b['observation']['mask'].fill_(False)
        second=m(b['observation'],**b['condition'],sample=False)
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)


@pytest.mark.parametrize('name',['rssm','diffusion'])
def test_backend_checkpoint_evaluation_and_demo(name,dataset,tmp_path):
    from drive_gaussian_world.train import train,load_model
    from drive_gaussian_world.evaluate import evaluate
    from drive_gaussian_world.demo import make_demo
    options={'diffusion_steps':8,'sampling_steps':2} if name=='diffusion' else {'stochastic':8}
    path=train(dataset,tmp_path/name,steps=1,batch_size=2,hidden=32,device='cpu',backend=name,model_options=options)
    model,state=load_model(path,device='cpu')
    assert state['config']['backend']==name
    assert type(model) is type(make_model(name))
    report=evaluate(dataset,path,tmp_path/'evaluation.json',device='cpu',samples=2)
    assert report['backend']==name
    assert report['stochastic_metrics']['best_of_k_ade_m']<=report['stochastic_metrics']['mean_sample_ade_m']
    assert report['sampling']['count']==2
    make_demo(dataset,path,tmp_path/'demo',device='cpu')
    assert (tmp_path/'demo'/'rollout.gif').exists()


def test_old_gru_checkpoint_without_backend(dataset,tmp_path):
    from drive_gaussian_world.train import load_model
    model=make_model('gru');path=tmp_path/'old.pt'
    torch.save({'model':model.state_dict(),'config':{'hidden':32,'no_geometry':False}},path)
    loaded,_=load_model(path,device='cpu')
    for key,value in model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key])


def test_diffusion_noise_schedule_and_invalid_options():
    m=make_model('diffusion');alpha=m.alpha_bar
    assert (alpha>0).all() and (alpha<1).all() and (alpha[1:]<alpha[:-1]).all()
    clean=torch.randn(2,32);noise=torch.randn_like(clean);t=torch.tensor([0,7])
    noisy=m.q_sample(clean,t,noise)
    restored=(noisy-(1-alpha[t,None]).sqrt()*noise)/alpha[t,None].sqrt()
    torch.testing.assert_close(restored,clean,rtol=1e-4,atol=1e-4)
    with pytest.raises(ValueError):build_world_model('invalid')
    with pytest.raises(ValueError):build_world_model('diffusion',model_options={'sampling_steps':101})


def test_balanced_kl_is_zero_for_same_distribution():
    model=make_model('rssm')
    mean=torch.randn(2,8);std=torch.rand(2,8)+.1
    assert model.balanced_kl((mean,std),(mean,std)).item()==pytest.approx(0,abs=1e-6)


def test_comparison_suite_uses_all_backends(dataset,tmp_path):
    from pathlib import Path
    from drive_gaussian_world.experiments import run
    cfg=json.loads(Path('configs/comparison.json').read_text())
    cfg['device']='cpu';cfg['threads']=2
    cfg['training'].update(steps=1,batch_size=2)
    cfg['model']['hidden']=32
    cfg['backends']['diffusion']['model_options'].update(diffusion_steps=8,sampling_steps=2)
    cfg['evaluation']['samples']=2
    path=tmp_path/'comparison.json';path.write_text(json.dumps(cfg))
    output=Path(run(dataset,tmp_path/'suite',config=path))
    report=json.loads((output/'comparison.json').read_text())
    assert set(report)=={'gru','rssm','diffusion'}
    assert all(report[name]['backend']==name for name in report)
    assert json.loads((output/'status.json').read_text())['status']=='COMPLETED'
    assert not torch.cuda.is_initialized()


def test_train_subcommand_help_exposes_backend_options(monkeypatch,capsys):
    import sys
    from drive_gaussian_world.__main__ import main
    monkeypatch.setattr(sys,'argv',['drive_gaussian_world','train','--help'])
    with pytest.raises(SystemExit) as result:main()
    assert result.value.code==0
    help_text=capsys.readouterr().out
    assert '--backend' in help_text and '--model-options' in help_text


@pytest.mark.parametrize('name',['gru','rssm','diffusion'])
def test_profiler_cpu_contract_uses_cuda_build_without_gpu(name,tmp_path):
    from drive_gaussian_world.profile_model import profile
    options={'diffusion_steps':8,'sampling_steps':2} if name=='diffusion' else {}
    result=profile(name,output=tmp_path/'profile.json',device='cpu',hidden=32,batch_size=1,
                   history=2,horizon=2,warmup=0,iterations=1,model_options=options)
    assert result['parameters']>0 and result['mean_rollout_ms']>0
    assert result['peak_cuda_allocated_bytes'] is None
    assert not torch.cuda.is_initialized()
