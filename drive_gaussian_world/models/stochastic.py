"""Shared causal encoders and output contract for stochastic dynamics backends."""
from __future__ import annotations
import torch
from .world import GaussianWorldModel


def control(actions,dt,speed):
    return torch.cat([actions[:,:1]/5,actions[:,1:2],dt[:,None],speed[:,None]/20],-1)


def stack_predictions(frames):
    return {key:torch.stack([frame[i] for frame in frames],1)
            for i,key in enumerate(('rgb','tokens','mask_logits','ego'))}


class StochasticWorldModel(GaussianWorldModel):
    def encode_frame(self,rgb,tokens,mask,speed):
        image=self.image_encoder(rgb)
        token=self.token_encoder(tokens)
        token=(token*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        if not self.use_geometry:token=torch.zeros_like(token)
        return torch.cat([image,token,speed[:,None]/20],-1)

    @staticmethod
    def validate_condition(actions,dt):
        if actions.ndim!=3 or actions.shape[-1]!=2 or dt.shape!=actions.shape[:2] or actions.shape[1]<1:
            raise ValueError('scenario must have actions [B,H,2], dt [B,H] with H >= 1')

    def forward(self,observation,actions,dt,*,sample=True,generator=None):
        return self.rollout(self.observe(observation,sample=sample,generator=generator),
                            actions,dt,sample=sample,generator=generator)

    def sample_predictions(self,observation,actions,dt,count=5,seed=42):
        if count<1:raise ValueError('count must be positive')
        generator=torch.Generator(device=actions.device).manual_seed(seed)
        samples=[self(observation,actions,dt,generator=generator) for _ in range(count)]
        return {key:torch.stack([item[key] for item in samples]) for key in samples[0]}
