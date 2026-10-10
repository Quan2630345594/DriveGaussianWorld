#!/usr/bin/env python3
"""Render an existing Gaussian NPZ only; never infer or modify reconstruction."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/parkinge2e_20261010_01/rendering"
SOURCE = ROOT / "outputs/parkinge2e_20261010_01/mapping/scene.npz"


def look_at(eye, target):
    """World-to-camera transform: OpenCV right, down, forward; world Z up."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) > .999
    assert (view[:3, :3] @ target + view[:3, 3])[2] > 0
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if args.frames < 8 or args.fps <= 0:
        parser.error("At least eight frames and positive FPS are required")
    source = args.scene.resolve()
    if OUTPUT in source.parents:
        parser.error("Input must not be inside the generated artifact directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "report.json"
    configuration = dict(input_filepath=str(source), frames=args.frames, fps=args.fps,
                         width=640, height=480, crop_lower=[-18., -18., -.3],
                         crop_upper=[10., 10., 5.])
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("status") == "complete":
            artifacts = previous.get("artifacts", [])
            if previous.get("configuration") == configuration and all(
                    (OUTPUT / p).is_file() and (OUTPUT / p).stat().st_size for p in artifacts):
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest == previous.get("input_sha256"):
                    print(f"Skipping existing complete artifacts: {report_path}")
                    return
            raise RuntimeError("Complete output exists but differs; refusing to overwrite")
    if any(OUTPUT.glob("*.png")) or any(OUTPUT.glob("*.mp4")):
        raise RuntimeError("Partial image/video artifacts exist; refusing to overwrite")
    # Capture native JIT compiler and ffmpeg output as well as Python diagnostics.
    log = open(OUTPUT / "render.log", "a", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    start = time.perf_counter()
    report = dict(configuration=configuration, status="running", input_filepath=str(source),
                  inference_performed=False, source_modified=False,
                  assumptions=["Assumed camera-height scale; no independent geometry truth",
                               "Independent temporal four-camera maps concatenated without deduplication",
                               "Preview crop is not metric validation",
                               "Model-map planning failed; obstacle-free planned control is diagnostic only",
                               "Not a pure full end-to-end success; ghosting/drift may remain"],
                  artifacts=[], environment={k: os.environ.get(k) for k in
                      ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
                       "MAX_JOBS", "PYTHONNOUSERSITE", "LD_PRELOAD")})
    encoder = None
    try:
        print("Loading existing NPZ (no weights or inference)", source, flush=True)
        report["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source, allow_pickle=False) as data:
            fields = {k: data[k].copy() for k in
                      ("means", "scales", "quats", "colors", "opacities")}
            metadata = json.loads(str(data["metadata"]))
        n = len(fields["means"])
        for key, size in (("means", 3), ("scales", 3), ("quats", 4), ("colors", 3),
                          ("opacities", None)):
            value = fields[key]
            if value.shape != ((n, size) if size else (n,)) or not np.isfinite(value).all():
                raise ValueError(f"Invalid saved field {key}: {value.shape}")
        if n == 0 or np.any(fields["scales"] <= 0):
            raise ValueError("Saved scales must be positive linear standard deviations")
        norms = np.linalg.norm(fields["quats"], axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Zero saved quaternion")
        for key in ("colors", "opacities"):
            if np.any((fields[key] < 0) | (fields[key] > 1)):
                raise ValueError(f"Saved {key} must already be activated in [0,1]")
        if metadata.get("units") != "meters" or metadata.get("up_axis") != "z":
            raise ValueError("Expected assumed metric, Z-up saved scene")
        report["input_metadata"] = metadata
        report["field_values"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
            min=np.min(v, axis=0).tolist(), max=np.max(v, axis=0).tolist(),
            quantiles=np.quantile(v, [0, .01, .5, .99, 1], axis=0).tolist())
            for k, v in fields.items()}
        lower = np.array(configuration["crop_lower"])
        upper = np.array(configuration["crop_upper"])
        means = fields["means"]
        keep = ((means >= lower) & (means <= upper)).all(axis=1)
        bounds = metadata.get("bounds")
        in_mapping = ((means[:, :2] >= np.array(bounds[:2])) &
                      (means[:, :2] <= np.array(bounds[2:]))).all(axis=1)
        report["crop"] = dict(label="PREVIEW CROP - NOT METRIC VALIDATION",
            selection="Gaussian centers inside inclusive XYZ bounds; no covariance clipping",
            lower_assumed_m=lower.tolist(), upper_assumed_m=upper.tolist(),
            input_gaussians=n, retained_gaussians=int(keep.sum()),
            excluded_gaussians=int((~keep).sum()), mapping_xy_bounds=bounds,
            centers_inside_mapping_xy=int(in_mapping.sum()),
            retained_zero_opacity=int(np.sum(fields["opacities"][keep] == 0)))
        if not keep.any():
            raise ValueError("Empty preview crop; do not produce an empty-camera video")
        fields = {k: np.ascontiguousarray(v[keep], dtype=np.float32) for k, v in fields.items()}
        fields["quats"] /= np.linalg.norm(fields["quats"], axis=1, keepdims=True)
        report["field_interpretation"] = dict(quats="normalized wxyz",
            scales="positive linear standard deviations, no exponentiation or clamping",
            colors="activated RGB [0,1], no SH decoding or extra sigmoid",
            opacities="activated [0,1], no extra sigmoid")
        width, height = 640, 480
        focal = width / (2 * math.tan(math.radians(70) / 2))
        K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float32)
        target = np.array([-4., -4., 1.])
        radius, altitude = 12., 8.
        views, eyes = [], []
        for i in range(args.frames):
            angle = 2 * math.pi * i / args.frames
            eye = target + np.array([radius * math.cos(angle), radius * math.sin(angle), altitude - target[2]])
            eyes.append(eye.tolist())
            views.append(look_at(eye, target))
        report["camera"] = dict(convention="OpenCV x-right y-down z-forward; world Z-up",
            target=target.tolist(), orbit_radius_assumed_m=radius, altitude_assumed_m=altitude,
            intrinsics=K.tolist(), eyes=eyes, viewmats=[v.tolist() for v in views],
            source="Synthetic orbit only; no GT cameras or trajectory fitting")
        backend_start = time.perf_counter()
        method = "gsplat_cuda_gaussian_rasterization"
        try:
            import torch
            from gsplat import rasterization
            import gsplat
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            report["renderer"] = dict(torch=torch.__version__, gsplat=gsplat.__version__,
                gpu=torch.cuda.get_device_name(0), logical_device="cuda:0", physical_gpu=5,
                packed=True, render_mode="RGB", rasterize_mode="classic", eps2d=.3,
                near_plane=.05, far_plane=200., background="transparent black")
            tensors = {k: torch.from_numpy(v).to("cuda:0") for k, v in fields.items()}
            kt = torch.from_numpy(K[None]).to("cuda:0")

            def render(view):
                with torch.inference_mode():
                    rgb, alpha, _ = rasterization(**tensors,
                        viewmats=torch.from_numpy(view[None]).to("cuda:0"), Ks=kt,
                        width=width, height=height, near_plane=.05, far_plane=200.,
                        packed=True, render_mode="RGB", rasterize_mode="classic")
                    rgba = torch.cat([rgb[0], alpha[0]], dim=-1).cpu().numpy()
                if not np.isfinite(rgba).all():
                    raise RuntimeError("Rasterizer returned non-finite RGBA")
                return rgba

            first = render(views[0])
        except Exception:
            error = traceback.format_exc()
            (OUTPUT / "gaussian_error.log").write_text(error)
            print(error, flush=True)
            method = "cpu_center_point_preview_NOT_gaussian_render"
            report["gaussian_attempt_error"] = error
            report["renderer"] = dict(method="CPU center-point z-buffer; no covariance splatting")

            def render(view):
                camera = fields["means"] @ view[:3, :3].T + view[:3, 3]
                valid = (camera[:, 2] > .05) & (camera[:, 2] < 200) & (fields["opacities"] > 0)
                ids = np.flatnonzero(valid)
                projected = camera[ids] @ K.T
                xy = np.rint(projected[:, :2] / projected[:, 2:]).astype(np.int64)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
                ids, xy = ids[inside], xy[inside]
                order = np.argsort(camera[ids, 2], kind="stable")
                pixels = xy[order, 1] * width + xy[order, 0]
                _, nearest = np.unique(pixels, return_index=True)
                ids = ids[order[nearest]]
                rgba = np.zeros((height * width, 4), np.float32)
                rgba[pixels[nearest], :3] = fields["colors"][ids] * fields["opacities"][ids, None]
                rgba[pixels[nearest], 3] = fields["opacities"][ids]
                return rgba.reshape(height, width, 4)

            first = render(views[0])
        report["method"] = method
        report["first_frame_wall_seconds_including_import_transfer_possible_jit"] = time.perf_counter() - backend_start
        gaussian = method == "gsplat_cuda_gaussian_rasterization"
        stem = "gaussian" if gaussian else "pointcloud_preview"
        video = "gaussian_orbit.mp4" if gaussian else "pointcloud_preview.mp4"
        temporary = OUTPUT / (video.removesuffix(".mp4") + ".partial.mp4")
        command = ["ffmpeg", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary)]
        report["ffmpeg_command"] = command
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        font = ImageFont.load_default()
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 13)
        samples = set(np.linspace(0, args.frames - 1, 8, dtype=int).tolist())
        report["frame_statistics"] = []
        for i, view in enumerate(views):
            frame_start = time.perf_counter()
            rgba = first if i == 0 else render(view)
            alpha = rgba[..., 3:4]
            report["frame_statistics"].append(dict(index=i,
                alpha_mean=float(alpha.mean()), alpha_max=float(alpha.max()),
                fraction_alpha_above_001=float((alpha > .01).mean()),
                wall_seconds_render_and_readback=None if i == 0 else time.perf_counter() - frame_start))
            # Rasterizer RGB is premultiplied; PNG requires straight-alpha RGB.
            straight = np.divide(rgba[..., :3], alpha, out=np.zeros_like(rgba[..., :3]), where=alpha > 1e-8)
            if i in samples:
                filename = f"{stem}_{i:03d}.png"
                Image.fromarray(np.rint(np.clip(np.concatenate([straight, alpha], -1), 0, 1) * 255).astype(np.uint8)).save(OUTPUT / filename)
                report["artifacts"].append(filename)
            composite = rgba[..., :3] + (1 - alpha) * np.array([.035, .04, .05])
            image = Image.fromarray(np.rint(np.clip(composite, 0, 1) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 77), fill=(12, 14, 18))
            title = "DGGT Gaussian preview - model-map planning failed" if gaussian else "CPU point-cloud preview - NOT a Gaussian render"
            for y, text in zip((5, 23, 41, 59), (title,
                    "Assumed camera-height scale; no independent geometry truth",
                    "PREVIEW CROP: X,Y [-18,10], Z [-0.3,5]; not metric validation",
                    f"{len(fields['means']):,}/{n:,} centers | synthetic orbit {i + 1}/{args.frames}")):
                draw.text((8, y), text, font=font, fill=(240, 240, 240))
            encoder.stdin.write(image.tobytes())
            print(f"Frame {i + 1}/{args.frames} alpha mean={alpha.mean():.4f}", flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=120) != 0:
            raise RuntimeError("ffmpeg encoding failed; see render.log")
        temporary.rename(OUTPUT / video)
        report["artifacts"].append(video)
        report["png_encoding"] = "Unannotated actual RGBA; straight alpha, 8-bit; no geometry repair"
        report["video_encoding"] = "H264 yuv420p; RGB alpha-composited on dark background; explicit overlays"
        report["performance_note"] = "Observed wall clock only, includes setup/JIT/IO/encoding; not a benchmark or real-time claim"
        report["status"] = "complete"
    except Exception:
        error = traceback.format_exc()
        (OUTPUT / "error.log").write_text(error)
        report["status"] = "failed"
        report["error"] = error
        print(error, flush=True)
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        raise
    finally:
        report["total_wall_seconds"] = time.perf_counter() - start
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Outcome: {report['status']}; report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
