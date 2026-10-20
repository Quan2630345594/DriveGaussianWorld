"""Open-loop maneuver ranking for research demonstration; never vehicle actuation."""
from __future__ import annotations
import torch
from drive_gaussian_world.evaluate import trajectory

@torch.inference_mode()
def rank_maneuvers(model,observation,horizon=4,dt=0.4):
    if horizon<1 or horizon>20 or not .01<=dt<=2:raise ValueError('invalid planning horizon')
    if observation['rgb'].shape[0]!=1:raise ValueError('one context required')
    state=model.observe(observation);items=[]
    specs=[('cruise',0.,0.),('slow_down',-2.,0.),('left_arc',0.,.08),('right_arc',0.,-.08)]
    for name,acceleration,rate in specs:
        action=torch.tensor([acceleration,rate],dtype=torch.float32,device=state['hidden'].device)[None,None].expand(1,horizon,2)
        intervals=torch.full((1,horizon),dt,device=state['hidden'].device)
        prediction=model.rollout(state,action,intervals)
        path=trajectory(prediction['ego'])[0]
        tokens=prediction['tokens'][0];mask=prediction['mask_logits'][0].sigmoid()>.5
        # Proximity is only a sparse-surface diagnostic. No lane/occupancy/actor truth is available.
        xyz=tokens[:,:,:3]*40
        near=mask&(xyz[:,:,2]>0)&(xyz[:,:,2]<8)&(xyz[:,:,0].abs()<1.3)&(xyz[:,:,1]<.8)
        proximity=near.float().sum(1).mean().item()
        progress=path[-1,0].item();effort=acceleration**2*.05+rate**2*10
        score=-progress+effort+proximity*.1
        items.append({'maneuver':name,'acceleration_m_s2':acceleration,'yaw_rate_rad_s':rate,
                      'score':score,'progress_m':progress,'surface_proximity_proxy':proximity,
                      'trajectory_xy_m':path.tolist(),'deployment_eligible':False})
    return {'status':'RESEARCH_OPEN_LOOP_ONLY','ranked':sorted(items,key=lambda x:x['score']),
            'limitations':['No road/lane map or independent collision geometry.',
                          'Sparse Gaussian surfaces and learned opacity are not occupancy truth.',
                          'Counterfactual action effects have not been validated by intervention data.',
                          'No vehicle controls are issued.']}
