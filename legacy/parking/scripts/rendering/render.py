#!/usr/bin/env python3
"""GaussianParkWorld offline full-image rendering and input-view evaluation."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gaussian_park_world.common.experiments import OUTPUT_ROOT, experiment_directory


def prepare_input_targets(paths):
    """Repeat the model's input preprocessing, without calibration defaults."""
    import numpy as np
    import torch
    from gaussian_park_world.rendering.preprocessing import load_images

    targets = {}
    for view, path in paths.items():
        folder = path["scene"].parent
        for name in ("scene.json", "input_intrinsics.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"{view}: input-view metrics require {folder / name}; "
                                        "use --no-input-metrics to render without metrics")
        report = json.loads((folder / "scene.json").read_text())
        if "horizontal_fov_deg" not in report:
            raise ValueError(f"{view}: scene.json must record horizontal_fov_deg (null if no crop)")
        fov = report["horizontal_fov_deg"]
        calibration = np.asarray(json.loads((folder / "input_intrinsics.json").read_text()), dtype=float)
        n = len(path["indices"])
        if (calibration.shape != (n, 3, 3) or not np.isfinite(calibration).all()
                or np.any(calibration[:, (0, 1), (0, 1)] <= 0)
                or not np.allclose(calibration[:, 2], [0, 0, 1], atol=1e-6, rtol=0)
                or (fov is not None and (not np.isfinite(fov) or not 30 <= fov <= 120))):
            raise ValueError(f"{view}: invalid metric input calibration or horizontal FOV")
        sources = [Path(name) for name in path["names"]]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"{view}: missing metric input frame: {source}")
        images, processed_K = load_images(sources, input_intrinsics=calibration, horizontal_fov_deg=fov)
        if (images.shape != (n, 3, path["height"], path["width"])
                or min(path["height"], path["width"]) < 11
                or not torch.isfinite(images).all() or torch.any((images < 0) | (images > 1))
                or processed_K.shape != (n, 3, 3) or not np.isfinite(processed_K).all()):
            raise ValueError(f"{view}: preprocessed metric inputs must match prediction depth dimensions")
        targets[view] = dict(images=images, K=processed_K, sources=sources, horizontal_fov_deg=fov)
    return targets


def input_image_metrics(rgb, target, alpha):
    """Quantized render vs float input; valid 11x11 box-window RGB SSIM."""
    import numpy as np
    import torch
    from torch.nn.functional import avg_pool2d

    if (rgb.shape != (*target.shape[1:], 3) or rgb.dtype != np.uint8
            or alpha.shape != rgb.shape[:2] or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("Metrics require raw uint8 RGB and finite alpha in [0, 1] at target dimensions")
    rendered = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.
    target = target[None]
    mse = (rendered - target).square().mean().clamp_min(1e-10)
    mu_x, mu_y = avg_pool2d(rendered, 11, 1), avg_pool2d(target, 11, 1)
    var_x = avg_pool2d(rendered.square(), 11, 1) - mu_x.square()
    var_y = avg_pool2d(target.square(), 11, 1) - mu_y.square()
    cov = avg_pool2d(rendered * target, 11, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + .01 ** 2) * (2 * cov + .03 ** 2)
            / ((mu_x.square() + mu_y.square() + .01 ** 2) * (var_x + var_y + .03 ** 2))).mean()
    return dict(psnr=float(-10 * torch.log10(mse)), ssim=float(ssim),
                alpha_mean=float(np.mean(alpha)), coverage_gt_095=float(np.mean(alpha > .95)),
                coverage_gt_001=float(np.mean(alpha > .01)))


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from gaussian_park_world.rendering.renderer import (
        CameraLoader, HEIGHT, VIEWS, WIDTH, Renderer, VideoWriter, interpolate_camera_path, load_cameras,
    )
    torch.set_num_threads(4)

    legacy = args.checkpoint is not None
    args.input_metrics = not legacy if args.input_metrics is None else args.input_metrics
    args.temporal_opacity = not legacy if args.temporal_opacity is None else args.temporal_opacity
    if legacy and args.input_metrics:
        raise ValueError("--input-metrics requires raw DGGT predictions and their exact preprocessed inputs; "
                         "legacy trained checkpoints have no matching input-view protocol")
    if legacy:
        if args.sfm is None or args.reconstruction_root is not None:
            raise ValueError("Explicit legacy --checkpoint requires --sfm and excludes --reconstruction-root")
        if args.temporal_opacity:
            raise ValueError("--temporal-opacity requires activated reconstruction scenes")
        checkpoint, sfm = args.checkpoint.resolve(), args.sfm.resolve()
        if not checkpoint.is_file() or not sfm.is_dir():
            raise FileNotFoundError("Legacy checkpoint file and full SfM directory must exist")
        inputs = [checkpoint, sfm]
        paths, _ = load_cameras(sfm)
        for path in paths.values():
            path.update(width=WIDTH, height=HEIGHT)
        offset = .10 if args.novel_offset_m is None else args.novel_offset_m
        offsets = dict.fromkeys(VIEWS, offset)
        units = "meters (legacy metric gauge assumption)"
    else:
        if args.sfm is not None:
            raise ValueError("--sfm is only valid with explicit legacy --checkpoint")
        if args.novel_offset_m is not None:
            raise ValueError("--novel-offset-m requires legacy metric mode; raw DGGT has no verified metric metadata")
        root = args.reconstruction_root
        if root is None:
            if args.experiment is None:
                raise ValueError("--reconstruction-root is required unless --experiment infers reconstruction/dggt")
            root = args.output_root / args.experiment / "reconstruction" / "dggt"
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Four-view reconstruction root does not exist: {root}")
        inputs = [root]
        paths = CameraLoader.load(root)
        offsets = {v: args.novel_offset_fraction * p["median_depth"] for v, p in paths.items()}
        units = "arbitrary per-view model units; NOT meters"
    if (type(args.fps) is not int or args.fps <= 0
            or not np.isfinite(args.seconds) or args.seconds <= 0
            or type(args.samples) is not int or args.samples < 2
            or not np.isfinite(args.novel_offset_fraction) or args.novel_offset_fraction <= 0
            or any(not np.isfinite(x) or x <= 0 for x in offsets.values())
            or (legacy and offset > .15)):
        raise ValueError("Require positive fps/seconds/fraction, samples >= 2; legacy offset must be in (0, .15] m")
    duration_frames = args.fps * args.seconds
    if not np.isfinite(duration_frames):
        raise ValueError("Video frame count must be finite")
    frames = round(duration_frames)
    if frames < 2 or args.samples > frames:
        raise ValueError("Video needs at least two frames and at least as many frames as samples")
    output_root = args.output_root.resolve()
    planned = output_root / args.experiment if args.experiment is not None else None
    for source in inputs:
        if output_root.is_relative_to(source):
            raise ValueError("Experiment output root must not be inside read-only inputs")
        if planned is not None:
            if planned.is_symlink():
                raise ValueError("Experiment directory must not be a symbolic link")
            output = planned / "render"
            archive = planned / "reconstruction" / "render_inputs"
            if any(destination.is_relative_to(source) or source.is_relative_to(destination)
                   for destination in (output.resolve(), archive.resolve())):
                raise ValueError("Render/archive outputs must not overlap read-only inputs")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise FileExistsError("Experiment render output is not empty; use a new experiment")
            if (planned / "experiment.json").is_symlink():
                raise ValueError("Experiment manifest must not be a symbolic link")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required")
    dimensions = {(p["width"], p["height"]) for p in paths.values()}
    if len(dimensions) != 1:
        raise ValueError("The 2x2 video requires equal processed dimensions across views")
    width, height = dimensions.pop()
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Processed prediction dimensions must be positive and even for MP4")
    start = max(p["indices"][0] for p in paths.values())
    end = min(p["indices"][-1] for p in paths.values())
    if start >= end:
        raise ValueError("Four camera paths have no shared source-frame interval")
    timestamps = np.linspace(start, end, frames)
    cameras = {v: interpolate_camera_path(p["indices"], p["poses"], timestamps, offset_m=0.)
               for v, p in paths.items()}
    intrinsics = {v: CameraLoader.intrinsics(p, timestamps) for v, p in paths.items()}
    novel = cameras["front"].copy()
    phase = (timestamps - start) / (end - start)
    novel[:, :3, 3] += (offsets["front"] * np.sin(2 * np.pi * phase))[:, None] * novel[:, :3, 0]
    selected = set(np.linspace(0, frames - 1, args.samples).round().astype(int))
    displacements = np.linalg.norm(novel[:, :3, 3] - cameras["front"][:, :3, 3], axis=1)
    eligible = np.flatnonzero((displacements > offsets["front"] * 1e-6)
                             & (timestamps > start) & (timestamps < end))
    if len(eligible) < args.samples:
        raise ValueError("Not enough displaced interior novel frames; increase video duration")
    novel_selected = set(eligible[np.linspace(0, len(eligible) - 1, args.samples).round().astype(int)])
    stills, verification = [], []
    for view, path in paths.items():
        candidates = np.flatnonzero((path["indices"] >= start) & (path["indices"] <= end))
        if len(candidates) < args.samples:
            raise ValueError(f"{view}: need {args.samples} original poses in the shared interval")
        chosen = candidates[np.linspace(0, len(candidates) - 1, args.samples).round().astype(int)]
        local_offsets = {"left": (-offsets[view], 0., 0.), "right": (offsets[view], 0., 0.)}
        if args.novel_upward:
            local_offsets["up"] = (0., -offsets[view], 0.)
        for row in chosen:
            support = path["poses"][row]
            for direction, local_offset in local_offsets.items():
                pose = support.copy()
                pose[:3, 3] += support[:3, :3] @ np.asarray(local_offset)
                stills.append((view, int(row), direction, support, pose))
    saved_poses = [(v, float(paths[v]["indices"][row]), support, pose,
                    f"novel/{v}/{row:06d}_{direction}.png")
                   for v, row, direction, support, pose in stills]
    saved_poses += [("front", float(timestamps[i]), cameras["front"][i], novel[i],
                     f"images/novel_front/{i:06d}.png") for i in sorted(novel_selected)]
    for view, timestamp, support, pose, artifact in saved_poses:
        displacement = pose[:3, 3] - support[:3, 3]
        if (np.linalg.norm(displacement) <= offsets[view] * 1e-6
                or not np.allclose(pose[:3, :3], support[:3, :3], atol=1e-10, rtol=0)
                or np.any(np.all(np.isclose(paths[view]["poses"], pose,
                                           atol=offsets[view] * 1e-7, rtol=0), axis=(1, 2)))):
            raise AssertionError("Novel pose must be displaced, fixed-orientation, and not a supported pose")
        verification.append(dict(camera=view, source_time=timestamp, artifact=artifact,
                                  displacement=displacement.tolist(), units=units,
                                  world_from_optical=pose.tolist()))
    targets = prepare_input_targets(paths) if args.input_metrics else {}
    needs_frame_archive = bool(targets) and any(
        not (paths[v]["scene"].parent / "input_frames" / f"{row:06d}.jpg").is_file()
        for v in VIEWS for row in range(len(paths[v]["indices"])))
    experiment = experiment_directory(args.experiment, args.output_root)
    output = experiment / "render"
    if any(output.iterdir()):
        raise FileExistsError("Experiment render output is not empty; refusing to overwrite")
    archive = experiment / "reconstruction" / "render_inputs"
    saved = []
    for source, name in zip(inputs, ("checkpoint.pt", "sfm") if legacy else ("dggt",)):
        destination = source
        if not source.is_relative_to(experiment) or needs_frame_archive:
            archive.mkdir(exist_ok=True)
            if archive.is_symlink():
                raise ValueError("Render input archive must not be a symbolic link")
            destination = archive / name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite archived input: {destination}")
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=False)
            else:
                shutil.copy2(source, destination)
        saved.append(destination)
    if targets:
        for view, target in targets.items():
            folder = saved[0] / f"reconstruction_single_{view}" / "input_frames"
            folder.mkdir(exist_ok=True)
            for row, source in enumerate(target["sources"]):
                destination = folder / f"{row:06d}.jpg"
                if not destination.is_file():
                    shutil.copy2(source, destination)
    # Use copied prediction arrays, never historical paths in copied JSON/config text.
    if legacy:
        paths, _ = load_cameras(saved[1])
        renderers = dict.fromkeys(VIEWS, Renderer(saved[0], width, height))
    else:
        paths = CameraLoader.load(saved[0])
        renderers = {v: Renderer(p["scene"], width, height, p["indices"], args.temporal_opacity)
                     for v, p in paths.items()}
    manifest_path = experiment / "experiment.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        experiment=experiment.name, created_at=datetime.now().isoformat())
    manifest["render"] = dict(
        status="running", backend="gsplat CUDA rasterization",
        validation="archivedtrainedbaseline" if legacy else "endtoendinputviewconsistency",
        independent_metric_accuracy=False, occupancy_validation=False, online_closed_loop=False,
        image_consistency_metrics_computed=False,
        input_metrics_enabled=args.input_metrics,
        gauge="legacy shared SfM gauge" if legacy else "four independent model gauges; fusion NOT proven",
        source_inputs=[str(p) for p in inputs],
        render_inputs=[str(p.relative_to(experiment)) for p in saved],
        copied_config_paths_may_reference_originals=True, renderer_camera_source="predictions.npz" if not legacy else "SfM",
        fps=args.fps, seconds=args.seconds, samples=args.samples, width=width, height=height,
        frames_per_video=frames, timestamps=timestamps.tolist(),
        novel_offset_fraction=None if legacy else args.novel_offset_fraction,
        novel_offset_m=offset if legacy else None, per_view_offsets=offsets, offset_units=units,
        temporal_opacity=args.temporal_opacity, temporal_widths_are_quality=False,
        temporal_opacity_formula="alpha * exp(log(0.1) / (width**2 + 1e-6) * (target_t - source_t)**2)" if args.temporal_opacity else None,
        novel_pose_verification=verification,
        camera_paths={v: dict(source_indices=paths[v]["indices"].tolist(),
                              input_paths=paths[v]["names"],
                              world_from_optical=cameras[v].tolist(), intrinsics=intrinsics[v].tolist(),
                              median_positive_depth=None if legacy else paths[v]["median_depth"])
                      for v in VIEWS})
    if legacy:
        digest = hashlib.sha256()
        with saved[0].open("rb") as stream:
            for chunk in iter(lambda: stream.read(2 ** 20), b""):
                digest.update(chunk)
        manifest["render"].update(checkpoint_sha256=digest.hexdigest(),
                                  sh_degree=renderers["front"].degree, training_step=renderers["front"].step)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Rendering {manifest['render']['validation']}: {width}x{height}; {units}. "
          f"Input-view metrics enabled: {args.input_metrics}; occupancy/independent metric validation: False.")
    if targets:
        per_view = {}
        metric_keys = ("psnr", "ssim", "alpha_mean", "coverage_gt_095", "coverage_gt_001")
        for view, target in targets.items():
            for kind in ("reference", "evaluation"):
                (output / kind / view).mkdir(parents=True)
            records = []
            for row, source_index in enumerate(paths[view]["indices"]):
                rgb, alpha = renderers[view].render(paths[view]["poses"][row], paths[view]["K"][row],
                                                    view=view, timestamp=float(source_index))
                values = input_image_metrics(rgb, target["images"][row], alpha)
                name = f"{int(source_index):06d}.png"
                reference = Path("reference") / view / name
                evaluation = Path("evaluation") / view / name
                # load_images produces byte RGB / 255; round-trip those bytes exactly for the PNG.
                reference_rgb = (target["images"][row].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(reference_rgb).save(output / reference)
                Image.fromarray(rgb).save(output / evaluation)
                records.append(dict(source_index=int(source_index), model_row=row,
                                    target_timestamp=float(source_index), reference=str(reference),
                                    evaluation=str(evaluation), **values))
            averages = {key: float(np.mean([record[key] for record in records])) for key in metric_keys}
            per_view[view] = dict(frame_count=len(records), averages=averages, frames=records,
                                  horizontal_fov_deg=target["horizontal_fov_deg"],
                                  preprocessed_input_intrinsics=target["K"].tolist())
            print(f"{view}: {len(records)} input poses; PSNR {averages['psnr']:.3f} dB; "
                  f"SSIM {averages['ssim']:.5f}; alpha_mean {averages['alpha_mean']:.5f}; "
                  f"coverage >.95 {averages['coverage_gt_095']:.2%}, >.01 {averages['coverage_gt_001']:.2%}")
        all_records = [record for result in per_view.values() for record in result["frames"]]
        manifest["render"].update(image_consistency_metrics_computed=True, input_view_metrics=dict(
            protocol="Input-view consistency ONLY, not held-out evaluation or geometric/occupancy accuracy. "
                     "Every selected model row at its exact predicted source pose, predicted K and source timestamp; "
                     "no interpolation, masks, overlays or alpha weighting. All RGB pixels enter PSNR; "
                     "SSIM averages valid 11x11 box windows, stride 1, no padding, population moments, "
                     "C1=0.01^2, C2=0.03^2. Quantized uint8 render / 255 vs float load_images input; "
                     "PSNR uses MSE floor 1e-10. Reference PNGs preserve preprocessed input bytes. "
                     "Alpha coverage is strictly alpha > threshold over all pixels, not geometric coverage. "
                     "Per-view and full averages are arithmetic means of per-frame metrics.",
            preprocessing="gaussian_park_world.rendering.preprocessing.load_images (default size=518), "
                          "scene.json horizontal_fov_deg and input_intrinsics.json; archived row JPEG preferred",
            frame_count=len(all_records), per_view=per_view,
            averages={key: float(np.mean([record[key] for record in all_records])) for key in metric_keys}))
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "videos").mkdir()
    for view in (*VIEWS, "novel_front"):
        (output / "images" / view).mkdir(parents=True)
    for view in VIEWS:
        (output / "novel" / view).mkdir(parents=True)
    with ExitStack() as stack:
        writers = {v: stack.enter_context(VideoWriter(output / "videos" / f"{v}.mp4", width, height, args.fps, frames))
                   for v in (*VIEWS, "novel_front")}
        grid_writer = stack.enter_context(VideoWriter(output / "videos" / "multiview2x2.mp4",
                                                      2 * width, 2 * height, args.fps, frames))
        for i, timestamp in enumerate(timestamps):
            images = {v: renderers[v].render(cameras[v][i], intrinsics[v][i], view=v, timestamp=timestamp)[0]
                      for v in VIEWS}
            for view, rgb in images.items():
                writers[view].write(rgb)
                if i in selected:
                    Image.fromarray(rgb).save(output / "images" / view / f"{i:06d}.png")
            grid_writer.write(np.concatenate((np.concatenate((images["front"], images["back"]), axis=1),
                                               np.concatenate((images["left"], images["right"]), axis=1)), axis=0))
            novel_rgb = renderers["front"].render(novel[i], intrinsics["front"][i], timestamp=timestamp)[0]
            writers["novel_front"].write(novel_rgb)
            if i in novel_selected:
                if np.array_equal(novel_rgb, images["front"]):
                    raise AssertionError("Novel video sample is pixel-identical to its supported view")
                Image.fromarray(novel_rgb).save(output / "images" / "novel_front" / f"{i:06d}.png")
    support_key, support_rgb = None, None
    for view, row, direction, support, pose in stills:
        timestamp = float(paths[view]["indices"][row])
        K = CameraLoader.intrinsics(paths[view], [timestamp])[0]
        if support_key != (view, row):
            support_rgb = renderers[view].render(support, K, view=view, timestamp=timestamp)[0]
            support_key = (view, row)
        rgb = renderers[view].render(pose, K, view=view, timestamp=timestamp)[0]
        if np.array_equal(rgb, support_rgb):
            raise AssertionError(f"{view}/{row}/{direction}: novel RGB is pixel-identical to support")
        Image.fromarray(rgb).save(output / "novel" / view / f"{row:06d}_{direction}.png")
    files = [p for p in output.rglob("*") if p.is_file()]
    total_metric_frames = sum(len(target["images"]) for target in targets.values())
    png_count = 5 * args.samples + len(stills) + 2 * total_metric_frames
    if (len(files) != 6 + png_count or sum(p.suffix == ".png" for p in files) != png_count
            or sum(p.suffix == ".mp4" for p in files) != 6):
        raise AssertionError("Render destination must contain only the expected raw PNGs and MP4s")
    for path in files:
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (width, height) or image.mode != "RGB":
                    raise AssertionError(f"Invalid raw RGB PNG: {path}")
    videos = {v: w.verification for v, w in {**writers, "multiview2x2": grid_writer}.items()}
    manifest["render"].update(status="complete", mp4_count=6, png_count=png_count,
                              videos=videos, all_saved_novel_poses_displaced=True,
                              all_novel_rgb_different_from_support=True, finished_at=datetime.now().isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(experiment=str(experiment), output=str(output), mp4_count=6,
                          png_count=png_count, frames_per_video=frames, videos=videos,
                          image_consistency_metrics_computed=bool(targets)), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-root", type=Path,
                        help="Root containing reconstruction_single_front/back/left/right; required unless inferred from --experiment or explicit legacy mode")
    parser.add_argument("--checkpoint", type=Path, help="Explicit archived trained SH baseline, NOT end-to-end validation")
    parser.add_argument("--sfm", type=Path, help="Full SfM input directory, required only with legacy --checkpoint")
    parser.add_argument("--experiment", help="Safe experiment name; infers reconstruction/dggt when root omitted")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--samples", type=int, default=8, help="PNG samples per camera and displacement (at least 2)")
    parser.add_argument("--novel-offset-fraction", type=float, default=.02,
                        help="Local displacement / per-view median positive predicted depth; NOT meters (default .02)")
    parser.add_argument("--novel-offset-m", type=float,
                        help="Legacy metric mode ONLY: displacement in (0, .15] meters (legacy default .10)")
    parser.add_argument("--input-metrics", action=argparse.BooleanOptionalAction, default=None,
                        help="Exact selected input-pose PSNR/SSIM and alpha coverage (default on for raw DGGT, unavailable for legacy)")
    parser.add_argument("--temporal-opacity", action=argparse.BooleanOptionalAction, default=None,
                        help="Apply upstream alpha_t at every target time (default on for raw DGGT, off for legacy)")
    parser.add_argument("--novel-upward", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.experiment is not None:
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment):
            parser.error("--experiment must be a single safe directory name")
    if args.checkpoint is not None:
        if args.sfm is None or args.reconstruction_root is not None:
            parser.error("Legacy --checkpoint requires --sfm and excludes --reconstruction-root")
    elif args.sfm is not None or args.novel_offset_m is not None:
        parser.error("--sfm and --novel-offset-m require explicit legacy --checkpoint")
    elif args.reconstruction_root is None and args.experiment is None:
        parser.error("--reconstruction-root or --experiment is required; no historical model is selected")
    run(args)


if __name__ == "__main__":
    main()
