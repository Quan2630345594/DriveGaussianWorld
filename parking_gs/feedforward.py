"""Video-to-scene orchestration for feed-forward reconstruction models.

The project does not vendor model weights.  A video is decoded locally, then a
selected external feed-forward model is run once (DGGT) or through a supplied
command (Splatt3R).  The adapter normalizes the model artifact into the local
``Scene`` format, after which the same occupancy map and A* planner are used.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import numpy as np

from .scene import Scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DGGT_REPO = PROJECT_ROOT / 'third_party' / 'dggt'


@dataclass
class VideoManifest:
    video: str
    frames_dir: str
    frames: list
    fps: float
    width: int
    height: int


def extract_video(video, output_dir, stride=3, max_frames=96, max_width=960):
    """Decode a video into deterministic JPG frames and write a manifest."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Video reconstruction requires opencv-python; install requirements.txt') from exc
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if stride < 1 or max_frames < 2 or max_width < 160:
        raise ValueError('stride >= 1, max_frames >= 2 and max_width >= 160 are required')
    output_dir = Path(output_dir).resolve()
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames, index = [], 0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, max_width / max(w, 1))
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            path = frames_dir / f'{len(frames):06d}.jpg'
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise IOError(f'Cannot write frame {path}')
            frames.append(str(path))
            index += 1
    finally:
        cap.release()
    if len(frames) < 2:
        raise ValueError('Video must contain at least two decodable frames')
    manifest = VideoManifest(str(video), str(frames_dir), frames, fps, original_width, original_height)
    (output_dir / 'video_manifest.json').write_text(json.dumps(manifest.__dict__, indent=2), encoding='utf-8')
    return manifest


def _run(command, cwd=None, log_path=None):
    """Run a model command without shell interpolation."""
    if isinstance(command, str):
        command = shlex.split(command, posix=os.name != 'nt')
    if not command:
        raise ValueError('Feed-forward command is empty')
    proc = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_path:
        Path(log_path).write_text(proc.stdout, encoding='utf-8', errors='replace')
    if proc.returncode:
        raise RuntimeError(f'Feed-forward command failed ({proc.returncode}); see {log_path}\n{proc.stdout[-2000:]}')
    return proc.stdout


def _load_ply(path, metadata):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError('PLY import requires plyfile; install requirements.txt') from exc
    with Path(path).open('rb') as f:
        vertex = PlyData.read(f)['vertex'].data
    names = set(vertex.dtype.names or ())
    required = {'x', 'y', 'z'}
    if not required <= names:
        raise ValueError(f'PLY is missing {required - names}')
    means = np.stack([vertex[n] for n in ('x', 'y', 'z')], axis=1).astype(np.float64)
    n = len(means)

    def fields(prefix, default):
        keys = [f'{prefix}_{i}' for i in range(len(default))]
        return np.stack([vertex[k] for k in keys], axis=1).astype(np.float64) if set(keys) <= names else np.tile(default, (n, 1))

    if {'red', 'green', 'blue'} <= names:
        colors = np.stack([vertex[k] for k in ('red', 'green', 'blue')], 1).astype(np.float64) / 255.
    elif {'f_dc_0', 'f_dc_1', 'f_dc_2'} <= names:
        dc = np.stack([vertex[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float64)
        colors = .5 + .2820947918 * dc
    elif {'color_0', 'color_1', 'color_2'} <= names:
        colors = np.stack([vertex[f'color_{i}'] for i in range(3)], 1).astype(np.float64)
    else:
        colors = np.full((n, 3), .5)

    if 'opacity' in names:
        opacity = np.asarray(vertex['opacity'], dtype=np.float64)
        if np.nanmin(opacity) < 0 or np.nanmax(opacity) > 1:
            opacity = 1 / (1 + np.exp(-np.clip(opacity, -30, 30)))
    else:
        opacity = np.ones(n, dtype=np.float64)
    raw_scales = fields('scale', [.02, .02, .02])
    # Graphdeco PLY stores log-scales; DGGT-style exports generally store
    # activated positive scales. Detect the former conservatively.
    scales = np.exp(raw_scales) if np.nanmin(raw_scales) < 0 else raw_scales
    scales = np.maximum(np.abs(scales), 1e-4)
    quats = fields('rot', [1., 0., 0., 0.])
    return Scene(means, scales, quats, np.clip(colors, 0, 1), np.clip(opacity, 0, 1), metadata).validate()


def _load_npz(path, metadata):
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        def first(*names):
            for name in names:
                if name in keys:
                    return data[name]
            raise ValueError(f'NPZ must contain one of {names}')
        means = first('means', 'points', 'world_points')
        scales = first('scales')
        quats = first('quats', 'rotations', 'rotation')
        colors = first('colors', 'rgbs', 'features_dc')
        opacities = first('opacities', 'opacity')
    return Scene(means, np.maximum(np.abs(scales), 1e-4), quats,
                 np.clip(colors, 0, 1), np.clip(opacities.reshape(-1), 0, 1), metadata).validate()


def load_scene_artifact(output_dir, metadata):
    output_dir = Path(output_dir)
    direct = output_dir / 'scene.npz'
    if direct.is_file():
        with np.load(direct, allow_pickle=False) as data:
            has_metadata = 'metadata' in data.files
        return Scene.load(direct) if has_metadata else _load_npz(direct, metadata)
    candidates = sorted(output_dir.rglob('*.npz'))
    for candidate in candidates:
        try:
            return _load_npz(candidate, metadata)
        except (ValueError, KeyError, OSError):
            continue
    plys = sorted(output_dir.rglob('*.ply'))
    if plys:
        return _load_ply(plys[0], metadata)
    raise FileNotFoundError('Feed-forward output must contain scene.npz, a Gaussian NPZ, or a PLY')


def transform_scene(scene, scale, rotation, translation):
    """Return a similarity-transformed scene using a 3x3 R and wxyz quaternions.

    Means become ``scale * R @ mean + translation`` and covariances become
    ``scale**2 * R @ covariance @ R.T``. Scene metadata is retained unchanged.
    """
    scale = float(scale)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('metric_scale must be a positive finite number')
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0)):
        raise ValueError('world_rotation must be a finite proper 3x3 rotation (determinant +1)')
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('world_translation must contain three finite values')

    # The largest eigenvector gives a stable wxyz quaternion, including at pi.
    trace = np.trace(rotation)
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    matrix = np.empty((4, 4))
    matrix[0, 0] = trace
    matrix[0, 1:] = matrix[1:, 0] = skew
    matrix[1:, 1:] = rotation + rotation.T - trace * np.eye(3)
    _, vectors = np.linalg.eigh(matrix)
    w, xyz = vectors[0, -1], vectors[1:, -1]
    quats = np.asarray(scene.quats, dtype=float)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    # Left-compose the world rotation with each Gaussian's local orientation.
    composed = np.column_stack((w * quats[:, 0] - quats[:, 1:] @ xyz,
                                w * quats[:, 1:] + quats[:, :1] * xyz
                                + np.cross(xyz, quats[:, 1:])))
    return Scene(scale * (scene.means @ rotation.T) + translation,
                 scene.scales * scale, composed, scene.colors, scene.opacities,
                 dict(scene.metadata)).validate()


def _metadata(config, scene, backend, manifest):
    cfg = dict(config or {})
    bounds = cfg.get('bounds')
    if bounds is None:
        q = np.quantile(scene.means[:, :2], [0.01, .99], axis=0)
        margin = 2.0
        bounds = [float(q[0, 0] - margin), float(q[0, 1] - margin),
                  float(q[1, 0] + margin), float(q[1, 1] + margin)]
    if len(bounds) != 4 or not np.isfinite(bounds).all() or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('config.bounds must be [xmin,ymin,xmax,ymax]')
    cfg.update(units='meters', up_axis='z', source=f'feedforward_{backend}',
               video=manifest.video, frame_count=len(manifest.frames), bounds=list(map(float, bounds)))
    cfg.setdefault('known_free_polygon', [bounds[:2], [bounds[2], bounds[1]], bounds[2:], [bounds[0], bounds[3]]])
    cfg.setdefault('start', [float(bounds[0] + 2), float((bounds[1] + bounds[3]) / 2), 0.0])
    cfg.setdefault('goal', [float(bounds[2] - 2), float((bounds[1] + bounds[3]) / 2), 0.0])
    cfg.setdefault('slots', [])
    return cfg


def reconstruct_video(video, output_dir, backend='dggt', repo=None, checkpoint=None,
                      command=None, config=None, stride=3, max_frames=96, max_width=960,
                      sequence_length=8):
    """Run a feed-forward video backend and normalize its Gaussian output.

    DGGT is the supported video-native default.  It predicts poses and Gaussian
    maps from unposed driving images; the included bridge exports ``scene.npz``.
    Splatt3R is pair-based, so it is accepted through a user-supplied command
    that writes a single globally aligned PLY/NPZ artifact.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = extract_video(video, output_dir, stride, max_frames, max_width)
    backend = backend.lower()
    if backend not in {'dggt', 'splatt3r', 'external'}:
        raise ValueError('backend must be dggt, splatt3r or external')
    if backend == 'dggt':
        repo_path = Path(repo).expanduser().resolve() if repo else DEFAULT_DGGT_REPO
        if not repo_path.is_dir():
            raise ValueError(f'DGGT source not found at {repo_path}; initialize submodules or pass --repo')
        if not checkpoint:
            raise ValueError('DGGT requires --checkpoint (weights remain external to this repository)')
        bridge = Path(__file__).resolve().parents[1] / 'scripts' / 'dggt_export_scene.py'
        cmd = [sys.executable, str(bridge), '--dggt-repo', str(repo_path),
               '--checkpoint', str(Path(checkpoint).resolve()), '--frames', manifest.frames_dir,
               '--output', str(output_dir / 'scene.npz'), '--sequence-length', str(sequence_length)]
    elif command:
        values = {'video': str(Path(video).resolve()), 'frames': manifest.frames_dir,
                  'output': str(output_dir), 'checkpoint': str(checkpoint or ''), 'repo': str(repo or '')}
        cmd = command.format(**values)
    else:
        raise ValueError('Splatt3R/external backend requires --command with {frames} and {output} placeholders')
    _run(cmd, cwd=repo if backend != 'dggt' else None, log_path=output_dir / 'feedforward.log')
    # A bridge may write metadata before the final scene; load an artifact and
    # then attach the project-level task configuration.
    provisional = load_scene_artifact(output_dir, dict(units='meters', up_axis='z'))
    cfg = dict(config or {})
    provisional = transform_scene(provisional, cfg.get('metric_scale', 1.0),
                                  cfg.get('world_rotation', np.eye(3)),
                                  cfg.get('world_translation', [0, 0, 0]))
    metadata = _metadata(config, provisional, backend, manifest)
    scene = Scene(provisional.means, provisional.scales, provisional.quats,
                  provisional.colors, provisional.opacities, metadata).validate()
    scene.save(output_dir / 'scene.npz')
    report = dict(backend=backend, video=str(Path(video).resolve()), frames=len(manifest.frames),
                  gaussians=len(scene.means), output=str(output_dir / 'scene.npz'),
                  note='Feed-forward output; no per-scene photometric optimization')
    (output_dir / 'reconstruction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return scene, report
