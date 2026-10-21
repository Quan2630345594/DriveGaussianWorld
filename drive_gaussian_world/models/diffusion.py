"""Conditional latent DDPM training and DDIM sampling for driving scene transitions."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .stochastic import StochasticWorldModel,control,stack_predictions
from .world import world_loss


class DiffusionWorldModel(StochasticWorldModel):
    def __init__(self,hidden=96,use_geometry=True,diffusion_steps=100,sampling_steps=20,
                 denoising_weight=1.0,reconstruction_weight=1.0):
        super().__init__(hidden,use_geometry)
        if not 2<=diffusion_steps<=10000 or not 1<=sampling_steps<=diffusion_steps:
            raise ValueError('require 1 <= sampling_steps <= diffusion_steps, diffusion_steps >= 2')
        if denoising_weight<0 or reconstruction_weight<=0:raise ValueError('invalid diffusion loss weights')
        self.diffusion_steps=diffusion_steps;self.sampling_steps=sampling_steps
        self.denoising_weight=denoising_weight;self.reconstruction_weight=reconstruction_weight
        self.latent_encoder=nn.Sequential(nn.Linear(113,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.conditioner=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU())
        self.denoiser=nn.Sequential(nn.Linear(hidden*2+32,hidden*3),nn.SiLU(),
                                   nn.Linear(hidden*3,hidden*3),nn.SiLU(),nn.Linear(hidden*3,hidden))
        # Cosine alpha-bar avoids a high residual signal at the terminal noise level.
        grid=torch.linspace(0,1,diffusion_steps+1,dtype=torch.float64)
        curve=torch.cos((grid+.008)/1.008*math.pi/2).square();curve=curve/curve[0]
        betas=(1-curve[1:]/curve[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha_bar',(1-betas).cumprod(0).float())

    def observe(self,observation,*,sample=True,generator=None):
        state=super().observe(observation)
        feature=self.encode_frame(*(observation[k][:,-1] for k in ('rgb','tokens','mask','speed')))
        state['latent']=self.latent_encoder(feature)
        return state

    def predict_noise(self,noisy,timestep,condition):
        frequency=torch.exp(torch.arange(16,device=noisy.device,dtype=torch.float32)*(-math.log(10000)/15))
        phase=timestep.float()[:,None]*frequency[None]
        embedding=torch.cat([phase.sin(),phase.cos()],-1).to(noisy.dtype)
        return self.denoiser(torch.cat([noisy,condition,embedding],-1))

    def q_sample(self,clean,timestep,noise):
        alpha=self.alpha_bar[timestep,None]
        return alpha.sqrt()*clean+(1-alpha).sqrt()*noise

    def sample_latent(self,condition,*,sample=True,generator=None):
        # DDIM eta=0: diversity comes from the initial noise; sample=False uses zeros.
        x=torch.randn(condition.shape,device=condition.device,dtype=condition.dtype,generator=generator) if sample else torch.zeros_like(condition)
        schedule=torch.linspace(self.diffusion_steps-1,0,self.sampling_steps,device=x.device).round().long()
        for index,step in enumerate(schedule):
            t=step.expand(x.shape[0]);alpha=self.alpha_bar[step]
            epsilon=self.predict_noise(x,t,condition)
            clean=((x-(1-alpha).sqrt()*epsilon)/alpha.sqrt()).clamp(-5,5)
            if index+1==len(schedule):x=clean
            else:
                previous=self.alpha_bar[schedule[index+1]]
                x=previous.sqrt()*clean+(1-previous).sqrt()*epsilon
        return x

    def rollout(self,state,actions,dt,*,sample=True,generator=None):
        self.validate_condition(actions,dt)
        h=state['hidden'];z=state['latent'];image=state['rgb'];tokens=state['tokens'];speed=state['speed'];frames=[]
        for i in range(actions.shape[1]):
            a=actions[:,i];delta=dt[:,i]
            h=self.dynamics(control(a,delta,speed),h)
            condition=self.conditioner(torch.cat([h,z],-1))
            z=self.sample_latent(condition,sample=sample,generator=generator)
            image,tokens,mask,ego,speed=self.decode_step(z,image,tokens,speed,a,delta)
            frames.append((image,tokens,mask,ego))
        return stack_predictions(frames)

    def training_loss(self,observation,actions,dt,target):
        self.validate_condition(actions,dt)
        state=self.observe(observation)
        h=state['hidden'];z=state['latent'];image=state['rgb'];tokens=state['tokens'];speed=state['speed']
        predicted=[];reconstructed=[];noise_losses=[]
        for i in range(actions.shape[1]):
            a=actions[:,i];delta=dt[:,i]
            h=self.dynamics(control(a,delta,speed),h)
            condition=self.conditioner(torch.cat([h,z],-1))
            next_speed=(speed+a[:,0].clamp(-8,8)*delta.clamp(1e-3,2)).clamp(0,45)
            feature=self.encode_frame(target['rgb'][:,i],target['tokens'][:,i],target['mask'][:,i],next_speed)
            clean=self.latent_encoder(feature)
            t=torch.randint(self.diffusion_steps,(len(a),),device=a.device)
            noise=torch.randn_like(clean)
            noisy=self.q_sample(clean.detach(),t,noise)
            epsilon=self.predict_noise(noisy,t,condition)
            noise_losses.append(F.mse_loss(epsilon.float(),noise.float()))
            alpha=self.alpha_bar[t,None]
            estimate=((noisy-(1-alpha).sqrt()*epsilon)/alpha.sqrt()).clamp(-5,5)
            pi,pt,pm,pe,_=self.decode_step(estimate,image,tokens,speed,a,delta)
            ri,rt,rm,re,_=self.decode_step(clean,image,tokens,speed,a,delta)
            predicted.append((pi,pt,pm,pe));reconstructed.append((ri,rt,rm,re))
            # Teacher forcing is confined to this training method. Inference uses samples.
            image=target['rgb'][:,i];tokens=target['tokens'][:,i];speed=next_speed;z=clean
        prediction_loss,logs=world_loss(stack_predictions(predicted),target)
        codec_loss,_=world_loss(stack_predictions(reconstructed),target)
        noise_loss=torch.stack(noise_losses).mean()
        total=prediction_loss+self.reconstruction_weight*codec_loss+self.denoising_weight*noise_loss
        logs.update(denoising_mse=float(noise_loss.detach()),codec_reconstruction=float(codec_loss.detach()),
                    total=float(total.detach()))
        return total,logs
