"""Dreamer-style continuous RSSM dynamics, without actor/critic or reward learning."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .stochastic import StochasticWorldModel,control,stack_predictions
from .world import world_loss


class RSSMWorldModel(StochasticWorldModel):
    def __init__(self,hidden=96,use_geometry=True,stochastic=32,kl_weight=0.1,
                 kl_balance=0.8,free_nats=1.0):
        super().__init__(hidden,use_geometry)
        if stochastic<1 or kl_weight<0 or not 0<=kl_balance<=1 or free_nats<0:
            raise ValueError('invalid RSSM configuration')
        self.stochastic=stochastic;self.kl_weight=kl_weight
        self.kl_balance=kl_balance;self.free_nats=free_nats
        # Replace the deterministic future dynamics; retain common observation encoder/heads.
        self.dynamics=nn.GRUCell(stochastic+4,hidden)
        del self.observer
        self.prior=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,2*stochastic))
        self.posterior=nn.Sequential(nn.Linear(hidden+113,hidden),nn.SiLU(),nn.Linear(hidden,2*stochastic))
        self.feature=nn.Sequential(nn.Linear(hidden+stochastic,hidden),nn.SiLU())

    @staticmethod
    def distribution(raw):
        mean,scale=raw.float().chunk(2,-1)
        return mean, F.softplus(scale)+0.1

    @staticmethod
    def draw(stats,sample=True,generator=None):
        mean,std=stats
        if not sample:return mean
        noise=torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)
        return mean+std*noise

    def observe(self,observation,*,sample=True,generator=None):
        # Common observer strictly accepts historical fields only.
        if set(observation)!={'rgb','tokens','mask','speed'}:raise ValueError('unexpected observation fields')
        b,t=observation['rgb'].shape[:2]
        h=observation['rgb'].new_zeros(b,self.hidden)
        z=h.new_zeros(b,self.stochastic)
        for i in range(t):
            # Historical controls/dt are absent from the observation contract.
            # Use zero unknown controls and observed speed for the filtering recurrence.
            features=self.encode_frame(*(observation[k][:,i] for k in ('rgb','tokens','mask','speed')))
            unknown=h.new_zeros(b,4);unknown[:,-1]=observation['speed'][:,i]/20
            h=self.dynamics(torch.cat([z.to(h.dtype),unknown],-1),h)
            z=self.draw(self.distribution(self.posterior(torch.cat([h,features],-1))),sample,generator)
        state=self.scene_state(observation,h)
        state.update(stochastic=z)
        return state

    def _roll(self,state,actions,dt,*,sample=True,generator=None,target=None):
        self.validate_condition(actions,dt)
        h=state['hidden'];z=state['stochastic'];image=state['rgb'];tokens=state['tokens'];speed=state['speed']
        frames=[];kls=[]
        for i in range(actions.shape[1]):
            a=actions[:,i];delta=dt[:,i]
            h=self.dynamics(torch.cat([z.to(h.dtype),control(a,delta,speed)],-1),h)
            prior=self.distribution(self.prior(h))
            stats=prior
            if target is not None:
                next_speed=(speed+a[:,0].clamp(-8,8)*delta.clamp(1e-3,2)).clamp(0,45)
                features=self.encode_frame(target['rgb'][:,i],target['tokens'][:,i],target['mask'][:,i],next_speed)
                stats=self.distribution(self.posterior(torch.cat([h,features],-1)))
                kls.append(self.balanced_kl(stats,prior))
            z=self.draw(stats,sample,generator)
            feature=self.feature(torch.cat([h,z.to(h.dtype)],-1))
            image,tokens,mask,ego,speed=self.decode_step(feature,image,tokens,speed,a,delta)
            frames.append((image,tokens,mask,ego))
        return stack_predictions(frames),kls

    def rollout(self,state,actions,dt,*,sample=True,generator=None):
        # Future observations cannot be passed to this interface.
        return self._roll(state,actions,dt,sample=sample,generator=generator)[0]

    def balanced_kl(self,posterior,prior):
        qm,qs=posterior;pm,ps=prior
        def kl(mq,sq,mp,sp):
            return (torch.log(sp/sq)+(sq.square()+(mq-mp).square())/(2*sp.square())-.5).sum(-1)
        dynamics=kl(qm.detach(),qs.detach(),pm,ps).clamp_min(self.free_nats)
        representation=kl(qm,qs,pm.detach(),ps.detach()).clamp_min(self.free_nats)
        return (self.kl_balance*dynamics+(1-self.kl_balance)*representation).mean()

    def training_loss(self,observation,actions,dt,target):
        prediction,kls=self._roll(self.observe(observation),actions,dt,target=target)
        reconstruction,logs=world_loss(prediction,target)
        kl=torch.stack(kls).mean()
        total=reconstruction+self.kl_weight*kl
        logs.update(reconstruction=float(reconstruction.detach()),kl=float(kl.detach()),total=float(total.detach()))
        return total,logs
