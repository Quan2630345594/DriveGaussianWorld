"""Dreamer-style RSSM world model."""
from .world import world_loss
from .rssm import RSSMWorldModel

def build_world_model(backend='rssm', hidden=96, use_geometry=True, model_options=None):
    if backend != 'rssm':
        raise ValueError(f'Only the RSSM world model is supported, got {backend!r}')
    return RSSMWorldModel(hidden=hidden, use_geometry=use_geometry, **(model_options or {}))
