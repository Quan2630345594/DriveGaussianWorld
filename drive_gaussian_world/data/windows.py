"""Scene-disjoint windows; inference inputs and training targets are explicit fields."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from drive_gaussian_world.common import read_json

class DrivingWindows(Dataset):
    def __init__(self,root,split='train',history=3,horizon=4):
        if history<1 or horizon<1:raise ValueError('history and horizon must be positive')
        self.root=Path(root);self.history=history;self.horizon=horizon
        self.index=read_json(self.root/'dataset.json');self.episodes=[];self.windows=[]
        for e in self.index['episodes']:
            if e['split']!=split:continue
            with np.load(self.root/e['path'],allow_pickle=False) as f:a=dict(f)
            ei=len(self.episodes);self.episodes.append((e,a))
            for end in range(history-1,len(a['rgb'])-horizon):self.windows.append((ei,end))
        if not self.windows:raise ValueError(f'no windows for split={split}')
    def __len__(self):return len(self.windows)
    def __getitem__(self,item):
        ei,end=self.windows[item];e,a=self.episodes[ei]
        h=slice(end-self.history+1,end+1);f=slice(end+1,end+1+self.horizon)
        # Only this dictionary may be passed to model.observe().
        observation={'rgb':torch.from_numpy(a['rgb'][h].copy()).float()/255,
            'tokens':torch.from_numpy(a['tokens'][h].copy()).float(),
            'mask':torch.from_numpy(a['token_mask'][h].copy()),
            'speed':torch.tensor(a['speed'][h].copy(),dtype=torch.float32)}
        # Actions are a separately supplied scenario, not a property of the observation.
        condition={'actions':torch.from_numpy(a['actions'][end:end+self.horizon].copy()),
                   'dt':torch.from_numpy(a['dt'][end:end+self.horizon].copy())}
        target={'rgb':torch.from_numpy(a['rgb'][f].copy()).float()/255,
                'tokens':torch.from_numpy(a['tokens'][f].copy()).float(),
                'mask':torch.from_numpy(a['token_mask'][f].copy()),
                'ego':torch.from_numpy(a['ego_delta'][end:end+self.horizon].copy())}
        return {'observation':observation,'condition':condition,'target':target,
                'scene_id':e['scene_id'],'anchor':int(a['frame_ids'][end])}
