from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVICE_CHOICES = ('cuda', 'cpu', 'auto')

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_device(device='cuda'):
    import torch
    if str(device) == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    selected = torch.device(device)
    if selected.type not in ('cuda', 'cpu'):
        raise ValueError('supported devices: cuda, cuda:N, cpu, auto')
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the driver, or pass --device cpu.')
    return selected

def configure_runtime(seed=42, threads=2, device='cuda'):
    if threads < 1:
        raise ValueError('threads must be positive')
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    return resolve_device(device)

def cpu_runtime(seed=42, threads=2):
    """Explicit CPU compatibility helper; never modifies CUDA visibility."""
    return configure_runtime(seed, threads, 'cpu')

def move_to(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=str(device).startswith('cuda'))
    if isinstance(value, dict):
        return {k: move_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to(v, device) for v in value)
    return value

def to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value

def precision_context(device, precision='fp32'):
    from contextlib import nullcontext
    import torch
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16 or bf16')
    if precision == 'fp32':
        return nullcontext()
    if device.type != 'cuda':
        raise ValueError('fp16/bf16 execution requires CUDA; use fp32 on CPU')
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    if precision == 'bf16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 is not supported by this CUDA device; use fp16 or fp32')
    return torch.autocast(device_type='cuda', dtype=dtype)

def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
