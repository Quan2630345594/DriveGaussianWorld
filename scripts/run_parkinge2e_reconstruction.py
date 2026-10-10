"""Offline overlapping DGGT windows; calibration-only gauge and overlap alignment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parking_gs.feedforward import load_scene_artifact, transform_scene
from parking_gs.scene import Scene


def similarity(source, target):
    source, target = np.asarray(source), np.asarray(target)
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    u, values, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    r = (u * sign) @ vt
    variance = np.mean(np.sum(x * x, axis=1))
    if variance < 1e-8:
        raise ValueError('Camera centers are degenerate; metric alignment is not established')
    scale = float(values @ sign / variance)
    if scale <= 0:
        raise ValueError('Invalid camera-baseline similarity scale')
    t = b - scale * (r @ a)
    error = np.linalg.norm(scale * (source @ r.T) + t - target, axis=1)
    return scale, r, t, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/model_latest_waymo.pth')
    parser.add_argument('--stop-after', type=int, default=7)
    parser.add_argument('--max-gaussians', type=int, default=60000)
    parser.add_argument('--single-camera', choices=['front', 'back', 'left', 'right'])
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.preparation / 'manifest.json').read_text())
    if args.single_camera:
        if args.samples < 2:
            parser.error('--samples must be >= 2')
        indices = np.linspace(0, len(manifest['frames']) - 1, args.samples, dtype=int)
        paths = [str((args.preparation / manifest['frames'][i]['cameras'][args.single_camera]
                      ['rectified']).resolve()) for i in indices]
        frame_manifest = args.output / 'input_frames.json'
        frame_manifest.write_text(json.dumps(paths, indent=2))
        command = [sys.executable, str(ROOT / 'scripts/dggt_export_scene.py'),
                   '--dggt-repo', str(ROOT / 'third_party/dggt'), '--checkpoint',
                   str(args.checkpoint.resolve()), '--frame-manifest', str(frame_manifest.resolve()),
                   '--sequence-length', str(args.samples), '--max-gaussians', str(args.max_gaussians),
                   '--prediction-output', str((args.output / 'predictions.npz').resolve()),
                   '--output', str((args.output / 'scene.npz').resolve())]
        (args.output / 'command.json').write_text(json.dumps(command, indent=2))
        (args.output / 'selected_indices.json').write_text(json.dumps(indices.tolist()))
        if not (args.output / 'predictions.npz').exists():
            with (args.output / 'inference.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Single-camera inference failed; see {args.output / "inference.log"}')
        return
    windows = json.loads((args.preparation / 'sampled_windows.json').read_text())['windows']
    cameras = ['front', 'back', 'left', 'right']
    rig = np.stack([np.array(manifest['cameras'][c]['T_vehicle_from_optical'])[:3, 3]
                    for c in cameras])
    records, scenes, labels = [], [], []
    last_centers = None
    for index in range(min(len(windows) - 1, args.stop_after)):
        folder = args.output / f'window_{index:02d}'
        folder.mkdir(exist_ok=True)
        paths = [str((args.preparation / window['cameras'][c]['rectified']).resolve())
                 for window in windows[index:index + 2] for c in cameras]
        frame_manifest = folder / 'input_frames.json'
        if frame_manifest.exists() and json.loads(frame_manifest.read_text()) != paths:
            raise ValueError('Existing window contains different inputs')
        frame_manifest.write_text(json.dumps(paths, indent=2), encoding='utf-8')
        raw = folder / 'scene.npz'
        predictions = folder / 'predictions.npz'
        command = [sys.executable, str(ROOT / 'scripts/dggt_export_scene.py'),
                   '--dggt-repo', str(ROOT / 'third_party/dggt'), '--checkpoint',
                   str(args.checkpoint.resolve()), '--frame-manifest', str(frame_manifest.resolve()),
                   '--sequence-length', '8', '--max-gaussians', str(args.max_gaussians),
                   '--prediction-output', str(predictions.resolve()), '--output', str(raw.resolve())]
        (folder / 'command.json').write_text(json.dumps(command, indent=2))
        if not raw.exists() or not predictions.exists():
            begin = time.perf_counter()
            attempt = len(list(folder.glob('inference*.log'))) + 1
            log_path = folder / f'inference_{attempt:02d}.log'
            with log_path.open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failure = dict(window=index, returncode=result.returncode,
                               elapsed_s=time.perf_counter() - begin, log=str(log_path))
                (folder / f'failure_{attempt:02d}.json').write_text(json.dumps(failure, indent=2))
                raise RuntimeError(f'Window {index} failed; see {log_path}')
        scene = load_scene_artifact(folder, dict(units='meters', up_axis='z',
                                                scale_status='raw_model_gauge_not_metric'))
        with np.load(predictions) as data:
            extrinsics = data['extrinsics']
        rotations = extrinsics[:, :3, :3].transpose(0, 2, 1)
        centers = -np.einsum('nij,nj->ni', rotations, extrinsics[:, :3, 3])
        target = rig if last_centers is None else last_centers
        scale, rotation, translation, error = similarity(centers[:4], target)
        aligned = transform_scene(scene, scale, rotation, translation)
        aligned.metadata.update(source='dggt_parkinge2e_calibration_and_overlap_aligned',
                                scale_status='calibration_baseline_estimate_unvalidated',
                                metric_alignment='published rig baselines; no reference trajectory fitted',
                                coordinate_origin='published vehicle calibration frame at first snapshot',
                                localization_origin_equivalence='unverified')
        aligned.save(folder / 'aligned_scene.npz')
        global_centers = scale * (centers @ rotation.T) + translation
        global_rotations = np.einsum('ij,njk->nik', rotation, rotations)
        last_centers = global_centers[4:]
        np.savez_compressed(folder / 'alignment.npz', scale=scale, rotation=rotation,
                            translation=translation, camera_centers=global_centers,
                            camera_rotations=global_rotations)
        record = dict(window=index, snapshot_indices=[windows[index]['index'], windows[index+1]['index']],
                      elapsed_s=[windows[index]['elapsed_s'], windows[index+1]['elapsed_s']],
                      scale=scale, baseline_or_overlap_rmse_m=float(np.sqrt(np.mean(error**2))),
                      residuals_m=error.tolist(), gaussians=len(scene.means))
        (folder / 'alignment.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        records.append(record)
        scenes.append(aligned)
        with np.load(raw) as data:
            labels.append(data['labels'])
    if not scenes:
        raise ValueError('No windows requested')
    metadata = dict(scenes[0].metadata)
    metadata.update(window_count=len(scenes), offline_map=True,
                    coverage_seconds=[records[0]['elapsed_s'][0], records[-1]['elapsed_s'][-1]])
    fused = Scene(*(np.concatenate([getattr(s, k) for s in scenes])
                    for k in ['means', 'scales', 'quats', 'colors', 'opacities']), metadata).validate()
    fused.save(args.output / 'fused_scene.npz')
    np.savez_compressed(args.output / 'semantic_labels.npz', labels=np.concatenate(labels))
    report = dict(windows=records, gaussians=len(fused.means), metadata=metadata,
                  reference_used_for_alignment=False,
                  note='Camera-overlap stitching, not bundle adjustment; drift is uncorrected.')
    (args.output / 'reconstruction_summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
