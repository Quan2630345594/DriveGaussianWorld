"""Observation encoders, residual prediction heads and losses for the RSSM."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def warp_gaussians(tokens,speed,actions,dt):
    acceleration=actions[:,0].clamp(-8,8);rate=actions[:,1].clamp(-1,1)
    dt=dt.clamp(1e-3,2)
    next_speed=(speed+acceleration*dt).clamp(0,45)
    distance=(speed+next_speed)*0.5*dt
    angle=rate*dt
    forward=distance*torch.cos(angle*0.5);left=distance*torch.sin(angle*0.5)
    # optical x right; vehicle's left translation has negative optical x.
    x=tokens[:,:,0]*40+left[:,None];z=tokens[:,:,2]*40-forward[:,None]
    c=torch.cos(angle)[:,None];s=torch.sin(angle)[:,None]
    xyz=torch.stack([(c*x+s*z)/40,tokens[:,:,1],(-s*x+c*z)/40],-1)
    prior=torch.cat([xyz,tokens[:,:,3:]],-1)
    ego=torch.stack([forward,left,angle],-1)
    return prior,next_speed,ego

class SceneDecoder(nn.Module):
    def __init__(self,hidden=96,use_geometry=True):
        super().__init__();self.hidden=hidden;self.use_geometry=use_geometry
        self.image_encoder=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),
            nn.SiLU(),nn.Conv2d(32,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,6)),nn.Flatten(),nn.Linear(384,64))
        self.token_encoder=nn.Sequential(nn.Linear(16,48),nn.SiLU(),nn.Linear(48,48))
        self.image_decoder=nn.Sequential(nn.Linear(hidden,32*4*12),nn.SiLU(),nn.Unflatten(1,(32,4,12)),
            nn.ConvTranspose2d(32,24,4,2,1),nn.SiLU(),nn.ConvTranspose2d(24,16,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(16,3,4,2,1))
        self.slots=nn.Embedding(64,16)
        self.gaussian_decoder=nn.Sequential(nn.Linear(hidden+16+16,96),nn.SiLU(),nn.Linear(96,17))
        self.ego_decoder=nn.Sequential(nn.Linear(hidden,32),nn.SiLU(),nn.Linear(32,3))
    def scene_state(self,observation,h):
        rgb,tokens,mask,speed=[observation[k] for k in ('rgb','tokens','mask','speed')]
        last_tokens=tokens[:,-1]
        last_mask=mask[:,-1].float()
        if not self.use_geometry:
            # A true RGB-only ablation also removes the geometric skip/warp input.
            last_tokens=torch.zeros_like(last_tokens)
            last_tokens[:,:,7:10]=1e-5
            last_tokens[:,:,10]=1
            last_mask=torch.zeros_like(last_mask)
        return {'hidden':h,'rgb':rgb[:,-1],'tokens':last_tokens,
                'mask':last_mask,'speed':speed[:,-1]}
    def decode_step(self,h,image,tokens,speed,a,delta):
        prior,speed,ego=warp_gaussians(tokens,speed,a,delta)
        # Shared residual head; inference rollout never re-encodes predicted RGB as observations.
        image=(image+0.15*torch.tanh(self.image_decoder(h))).clamp(0,1)
        emb=self.slots.weight[None].expand(h.shape[0],-1,-1)
        raw=self.gaussian_decoder(torch.cat([h[:,None].expand(-1,64,-1),emb,prior],-1))
        xyz=prior[:,:,:3]+0.1*torch.tanh(raw[:,:,:3])
        xyz=torch.cat([xyz[:,:,:2],xyz[:,:,2:].clamp_min(0.025)],-1)
        features=(prior[:,:,3:7]+0.1*torch.tanh(raw[:,:,3:7])).clamp(0,1)
        scales=(prior[:,:,7:10]+0.01*torch.tanh(raw[:,:,7:10])).clamp_min(1e-5)
        quat=F.normalize(prior[:,:,10:14]+0.1*raw[:,:,10:14],dim=-1)
        dynamic=(prior[:,:,14:16]+0.1*torch.tanh(raw[:,:,14:16])).clamp(0,1)
        tokens=torch.cat([xyz,features,scales,quat,dynamic],-1)
        ego=ego+torch.tanh(self.ego_decoder(h))*ego.new_tensor([1.,1.,0.1])
        return image,tokens,raw[:,:,16],ego,speed


def world_loss(pred,target):
    valid=target['mask'].float();denom=valid.sum().clamp_min(1)
    rgb=F.l1_loss(pred['rgb'],target['rgb'])
    # Spatial slots are sampling rays, not persistent actor identities. Match point sets.
    b,t,k,_=pred['tokens'].shape
    pp=pred['tokens'][...,:3].reshape(b*t,k,3).float()
    gt=target['tokens'][...,:3].reshape(b*t,k,3).float()
    vm=valid.reshape(b*t,k)
    distances=torch.cdist(pp,gt,p=2)
    to_gt=distances.masked_fill(~vm.bool()[:,None,:],10.).min(-1).values
    to_pred=distances.min(-2).values
    probability=pred['mask_logits'].sigmoid().reshape(b*t,k)
    forward=(to_gt*probability).sum()/probability.sum().clamp_min(1)
    backward=(to_pred*vm).sum()/denom
    geometry=(forward+backward)*0.5
    # Color/scale attributes are supervised at fixed ray slots; orientation is sign-invariant.
    attributes=(torch.abs(pred['tokens'][...,3:10]-target['tokens'][...,3:10]).mean(-1)*valid).sum()/denom
    quaternion=(1-(pred['tokens'][...,10:14]*target['tokens'][...,10:14]).sum(-1).abs()).clamp_min(0)
    attributes=attributes+0.1*(quaternion*valid).sum()/denom
    mask=F.binary_cross_entropy_with_logits(pred['mask_logits'],valid)
    ego=F.smooth_l1_loss(pred['ego'],target['ego'])
    total=rgb+2*geometry+0.2*attributes+0.05*mask+0.1*ego
    return total,{'rgb_l1':rgb.item(),'geometry_distance':geometry.item(),'attributes_l1':attributes.item(),
                  'mask_bce':mask.item(),'ego_huber':ego.item(),'total':total.item()}
