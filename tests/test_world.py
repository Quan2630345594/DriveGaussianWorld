import torch
import pytest
from torch.utils.data import default_collate
from drive_gaussian_world.common import cpu_runtime
from drive_gaussian_world.data.windows import DrivingWindows
from drive_gaussian_world.models.world import GaussianWorldModel,world_loss,warp_gaussians
from drive_gaussian_world.evaluate import trajectory
from drive_gaussian_world.planning.counterfactual import rank_maneuvers

cpu_runtime(42,2)

def batch(dataset):return default_collate([DrivingWindows(dataset)[0]])

def test_open_loop_shape_and_gradients(dataset):
    b=batch(dataset);m=GaussianWorldModel(hidden=32)
    p=m(b['observation'],b['condition']['actions'],b['condition']['dt'])
    assert p['rgb'].shape==(1,4,3,32,96)
    assert p['tokens'].shape==(1,4,64,16)
    loss,_=world_loss(p,b['target']);loss.backward()
    assert torch.isfinite(loss)
    assert m.dynamics.weight_hh.grad.abs().sum()>0
    assert m.token_encoder[0].weight.grad.abs().sum()>0
    assert all(p.device.type=='cpu' for p in m.parameters())

def test_target_mutation_does_not_change_prediction(dataset):
    b=batch(dataset);m=GaussianWorldModel(32).eval()
    with torch.no_grad():
        x=m(b['observation'],**b['condition'])['rgb']
        b['target']['rgb'].fill_(1);b['target']['tokens'].fill_(100)
        y=m(b['observation'],**b['condition'])['rgb']
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    obs=dict(b['observation']);obs['future_rgb']=b['target']['rgb']
    with pytest.raises(ValueError,match='unexpected'):m.observe(obs)

def test_action_changes_rollout(dataset):
    b=batch(dataset);m=GaussianWorldModel(32).eval();state=m.observe(b['observation'])
    a=b['condition']['actions'];dt=b['condition']['dt']
    x=m.rollout(state,a,dt);changed=a.clone();changed[:,:,1]=.3
    y=m.rollout(state,changed,dt)
    assert not torch.allclose(x['ego'],y['ego'])
    assert not torch.allclose(x['tokens'],y['tokens'])

def test_analytic_warp_units():
    tokens=torch.zeros(1,64,16);tokens[:,:,2]=.5
    warped,speed,ego=warp_gaussians(tokens,torch.tensor([5.]),torch.zeros(1,2),torch.tensor([.4]))
    torch.testing.assert_close(warped[:,:,2],torch.full((1,64),.45))
    torch.testing.assert_close(ego,torch.tensor([[2.,0,0]]))
    torch.testing.assert_close(trajectory(ego[:,None].expand(-1,4,-1))[:,-1],torch.tensor([[8.,0]]))

def test_candidate_ranking_never_enables_actuation(dataset):
    b=batch(dataset);report=rank_maneuvers(GaussianWorldModel(32).eval(),b['observation'])
    assert len(report['ranked'])==4
    assert all(not c['deployment_eligible'] for c in report['ranked'])

def test_train_checkpoint_eval_and_demo(dataset,tmp_path):
    from drive_gaussian_world.train import train,load_model
    from drive_gaussian_world.evaluate import evaluate
    from drive_gaussian_world.demo import make_demo
    checkpoint=train(dataset,tmp_path/'run',steps=2,batch_size=2,hidden=32,device='cpu')
    model,state=load_model(checkpoint,device='cpu')
    assert state['selection_split']=='val'
    report=evaluate(dataset,checkpoint,tmp_path/'eval.json',device='cpu')
    assert report['split']=='test';assert report['windows']==4
    assert report['metrics']['persistence']['rgb_l1']==0
    path=make_demo(dataset,checkpoint,tmp_path/'demo',device='cpu')
    assert '__DATA__' not in open(path).read()
    assert (tmp_path/'demo'/'rollout.gif').exists()

def test_rgb_only_has_no_geometric_skip_leak(dataset):
    b=batch(dataset);m=GaussianWorldModel(32,use_geometry=False).eval()
    with torch.no_grad():
        first=m(b['observation'],**b['condition'])
        b['observation']['tokens'].fill_(9);b['observation']['mask'].fill_(False)
        second=m(b['observation'],**b['condition'])
    for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
