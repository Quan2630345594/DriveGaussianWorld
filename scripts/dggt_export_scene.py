"""DGGT bridge: run the public model and export the project Scene NPZ.

This file intentionally keeps DGGT out of the main package.  Put a checkout of
https://github.com/xiaomi-research/dggt and its checkpoint on disk, then the
main project invokes this bridge.  No weights are downloaded automatically.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def load_images(paths, size=518):
    import torch
    from PIL import Image
    tensors = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        w, h = image.size
        new_h = round(h * (size / w) / 14) * 14
        image = image.resize((size, new_h), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.
        if new_h > size:
            top = (new_h - size) // 2
            tensor = tensor[:, top:top + size]
        tensors.append(tensor)
    return torch.stack(tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dggt-repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--frames')
    inputs.add_argument('--frame-manifest', help='JSON list of image paths in model input order')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--max-gaussians', type=int, default=250000)
    parser.add_argument('--prediction-output', help='Retain predicted cameras, depth and semantic labels')
    args = parser.parse_args()
    repo = Path(args.dggt_repo).resolve()
    if not (repo / 'dggt').is_dir():
        raise SystemExit('DGGT repo must contain the dggt/ package')
    sys.path.insert(0, str(repo))
    import torch
    from dggt.models.vggt import VGGT
    from dggt.utils.pose_enc import pose_encoding_to_extri_intri
    from dggt.utils.geometry import unproject_depth_map_to_point_map
    from dggt.utils.gs import get_split_gs

    paths = ([Path(p) for p in json.loads(Path(args.frame_manifest).read_text())]
             if args.frame_manifest else sorted(Path(args.frames).glob('*.jpg')))
    if len(paths) < 2:
        raise SystemExit('DGGT bridge requires at least two extracted frames')
    if args.sequence_length < 2 or args.max_gaussians < 1:
        raise SystemExit('--sequence-length >= 2 and --max-gaussians >= 1 are required')
    # Each invocation has its own coordinate gauge. Multi-window callers must
    # align overlapping windows before combining their Gaussian artifacts.
    paths = paths[:args.sequence_length]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    images = load_images(paths).to(device)
    model = VGGT().to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.eval()
    with torch.inference_mode():
        pred = model(images[None])
        h, w = images.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pred['pose_enc'], (h, w))
        # DGGT's public inference path uses its depth head plus predicted poses
        # to put each pixel into the predicted world frame.
        point_map = unproject_depth_map_to_point_map(pred['depth'][0], extrinsics[0], intrinsics[0])
        if not torch.is_tensor(point_map):
            point_map = torch.as_tensor(point_map, device=device, dtype=torch.float32)
        point_map = point_map[None, ...]
        gs_map = pred['gs_map']
        dynamic = pred.get('dynamic_conf')
        if dynamic is None:
            dynamic = torch.zeros(gs_map.shape[:-1] + (1,), device=device)
        dynamic = dynamic.squeeze(-1)
        mask = dynamic < .5
        # gs_conf is a temporal width, not a conventional quality score.
        mask &= torch.isfinite(point_map).all(-1) & torch.isfinite(gs_map).all(-1)
        mask &= (gs_map[..., 4:7] > 0).all(-1)
        mask &= torch.linalg.vector_norm(gs_map[..., 7:11], dim=-1) > 1e-8
        semantic_labels = pred['semantic_logits'].argmax(-1)
        mask &= semantic_labels != 9  # Upstream driving palette: sky.
        rgbs, opacity, scales, rotation = get_split_gs(gs_map, mask)
        opacity = opacity * (1 - dynamic[mask].sigmoid())
        points = point_map[mask].reshape(-1, 3)
        labels = semantic_labels[mask]
        count = min(args.max_gaussians, len(points))
        if count == 0:
            raise SystemExit('DGGT produced no static Gaussian points')
        if len(points) > count:
            keep = torch.linspace(0, len(points) - 1, count, device=device).long()
            points, rgbs, opacity, scales, rotation = [x[keep] for x in (points, rgbs, opacity, scales, rotation)]
            labels = labels[keep]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, means=points.detach().cpu().numpy(),
                        colors=rgbs.detach().cpu().numpy(),
                        opacities=opacity.detach().cpu().numpy().reshape(-1),
                        scales=scales.detach().cpu().numpy(),
                         quats=rotation.detach().cpu().numpy(), labels=labels.cpu().numpy())
    if args.prediction_output:
        prediction_path = Path(args.prediction_output)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, extrinsics=extrinsics[0].float().cpu().numpy(),
                            intrinsics=intrinsics[0].float().cpu().numpy(),
                            depth=pred['depth'][0].float().cpu().numpy(),
                            depth_conf=pred['depth_conf'][0].float().cpu().numpy(),
                            semantic_labels=pred['semantic_logits'][0].argmax(-1).cpu().numpy(),
                            dynamic_logits=dynamic[0].float().cpu().numpy(),
                            world_points=point_map[0].float().cpu().numpy(),
                            input_paths=np.array([str(p.resolve()) for p in paths]))
    if device == 'cuda':
        torch.cuda.synchronize()
    report = dict(device=device, input_frames=len(paths), image_size=[w, h], gaussians=count,
                  elapsed_s=time.perf_counter() - begin,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                  coordinate_frame='raw_model_gauge_not_yet_metric_or_z_up',
                  filtering='upstream static-logit threshold, finite attributes, uniform subsampling')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Exported {count} static DGGT Gaussians to {output}')


if __name__ == '__main__':
    main()
