"""World-model backend registry; old checkpoints default to deterministic GRU."""
from .world import GaussianWorldModel,world_loss
from .rssm import RSSMWorldModel
from .diffusion import DiffusionWorldModel

BACKENDS={'gru':GaussianWorldModel,'rssm':RSSMWorldModel,'diffusion':DiffusionWorldModel}

def build_world_model(backend='gru',hidden=96,use_geometry=True,model_options=None):
    if backend not in BACKENDS:raise ValueError(f'unknown backend {backend!r}; choose {tuple(BACKENDS)}')
    return BACKENDS[backend](hidden=hidden,use_geometry=use_geometry,**(model_options or {}))
