"""Independent predicted-gauge and explicit archived-baseline Gaussian rendering."""

import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


WIDTH, HEIGHT = 640, 360
VIEWS = ("front", "back", "left", "right")


def interpolate_camera_path(frame_indices, world_from_optical, timestamps, offset_m=0.):
    """Interpolate original frame-index time; optionally shift sinusoidally in local x."""
    indices = np.asarray(frame_indices, dtype=np.float64)
    poses = np.asarray(world_from_optical, dtype=np.float64)
    query = np.asarray(timestamps, dtype=np.float64)
    if (indices.ndim != 1 or len(indices) < 2 or poses.shape != (len(indices), 4, 4)
            or query.ndim != 1 or not len(query)):
        raise ValueError("Need at least two poses, one index per pose, and 1D timestamps")
    if not all(np.isfinite(x).all() for x in (indices, poses, query)):
        raise ValueError("Camera path contains nonfinite values")
    if not np.isfinite(offset_m) or not 0 <= offset_m <= .15:
        raise ValueError("Novel-view offset must be between 0 and 0.15 meters")
    order = np.argsort(indices)
    indices, poses = indices[order], poses[order]
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Original frame indices must be distinct")
    if query.min() < indices[0] or query.max() > indices[-1]:
        raise ValueError("Camera timestamps would extrapolate beyond registered poses")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        raise ValueError("Expected rigid optical camera-to-world transforms")
    result = np.broadcast_to(np.eye(4), (len(query), 4, 4)).copy()
    result[:, :3, :3] = Slerp(indices, Rotation.from_matrix(rotations))(query).as_matrix()
    result[:, :3, 3] = np.column_stack([
        np.interp(query, indices, poses[:, axis, 3]) for axis in range(3)])
    if offset_m:
        phase = (query - indices[0]) / (indices[-1] - indices[0])
        result[:, :3, 3] += (offset_m * np.sin(2 * np.pi * phase))[:, None] * result[:, :3, 0]
    return result


def load_cameras(sfm):
    import pycolmap

    sfm = Path(sfm)
    rec = pycolmap.Reconstruction(str(sfm / "reconstruction"))
    with np.load(sfm / "poses_points.npz", allow_pickle=False) as data:
        indices = data["frame_indices"].copy()
        names = data["names"].copy()
        poses = data["world_from_optical"].copy()
        registered = data["registered"].copy()
        views = list(map(str, data["views"]))
    if len(views) != 4 or set(views) != set(VIEWS):
        raise ValueError(f"Expected four cameras {VIEWS}, found {views}")
    if (indices.ndim != 1 or not np.isfinite(indices).all()
            or names.shape != (len(indices), 4) or registered.shape != names.shape
            or poses.shape != (len(indices), 4, 4, 4) or len(np.unique(indices)) != len(indices)):
        raise ValueError("Inconsistent SfM NPZ shapes or duplicate source frame indices")
    paths, lookup = {}, {}
    for view in VIEWS:
        column = views.index(view)
        rows = np.flatnonzero(registered[:, column])
        if len(rows) < 2:
            raise ValueError(f"{view}: need at least two registered poses")
        K = None
        for row in rows:
            name = str(names[row, column])
            image = rec.find_image_with_name(name)
            if image is None or not image.has_pose:
                raise ValueError(f"{name}: registered NPZ image missing from reconstruction")
            camera = rec.camera(image.camera_id)
            if (camera.width, camera.height) != (WIDTH, HEIGHT):
                raise ValueError(f"{name}: original camera must be {WIDTH}x{HEIGHT}")
            if camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
                raise ValueError(f"{name}: distorted camera model requires explicit undistortion")
            intrinsic = np.asarray(camera.calibration_matrix())
            if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                    or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                    or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-9, rtol=0)):
                raise ValueError(f"{name}: invalid camera intrinsics")
            pose = np.eye(4)
            pose[:3] = image.cam_from_world().inverse().matrix()
            if not np.allclose(pose, poses[row, column], atol=1e-7, rtol=0):
                raise ValueError(f"{name}: reconstruction/NPZ coordinate mismatch")
            if K is not None and not np.allclose(K, intrinsic, atol=1e-9, rtol=0):
                raise ValueError(f"{view}: intrinsics change along path")
            K = intrinsic
            lookup[name] = (pose, K)
        order = rows[np.argsort(indices[rows])]
        ordered_indices, ordered_poses = indices[order], poses[order, column]
        interpolate_camera_path(ordered_indices, ordered_poses, ordered_indices)
        paths[view] = dict(indices=ordered_indices, poses=ordered_poses, K=K,
                           names=[str(names[row, column]) for row in order])
    return paths, lookup


class CameraLoader:
    """Predicted world-to-optical cameras, without SfM or metric assumptions."""

    @staticmethod
    def load(root):
        paths = {}
        for view in VIEWS:
            folder = Path(root) / f"reconstruction_single_{view}"
            indices = np.asarray(json.loads((folder / "selected_indices.json").read_text()), dtype=float)
            with np.load(folder / "predictions.npz", allow_pickle=False) as data:
                extrinsics = data["extrinsics"].astype(float)
                intrinsics = data["intrinsics"].astype(float)
                depth = data["depth"]
                names = data["input_paths"].copy()
                n = len(indices)
                if (indices.shape != (n,) or n < 2 or not np.isfinite(indices).all()
                        or np.any(indices != np.floor(indices))
                        or np.any(np.diff(indices) <= 0) or extrinsics.shape != (n, 3, 4)
                        or intrinsics.shape != (n, 3, 3) or names.shape != (n,)
                        or depth.ndim != 4 or depth.shape[0] != n or depth.shape[-1] != 1):
                    raise ValueError(f"{view}: inconsistent prediction shapes or source indices")
                if (not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all()
                        or np.any(intrinsics[:, (0, 1), (0, 1)] <= 0)
                        or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6)):
                    raise ValueError(f"{view}: invalid predicted cameras")
                positive = depth[np.isfinite(depth) & (depth > 0)]
                if not len(positive):
                    raise ValueError(f"{view}: no finite positive predicted depth")
                median_depth = float(np.median(positive))
                height, width = depth.shape[1:3]
            optical_from_world = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            optical_from_world[:, :3] = extrinsics
            poses = np.linalg.inv(optical_from_world)
            # Float32 predicted rotations need a float32-appropriate rigid check.
            rotations = poses[:, :3, :3]
            if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                    or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
                raise ValueError(f"{view}: nonrigid predicted extrinsics")
            if paths and not np.array_equal(indices, paths[VIEWS[0]]["indices"]):
                raise ValueError(f"{view}: all four views must have exactly the same selected source indices")
            interpolate_camera_path(indices, poses, indices, offset_m=0.)
            names = [str(folder / "input_frames" / f"{row:06d}.jpg")
                     if (folder / "input_frames" / f"{row:06d}.jpg").is_file() else str(name)
                     for row, name in enumerate(names)]
            paths[view] = dict(indices=indices, poses=poses, K=intrinsics, names=names,
                               width=width, height=height, median_depth=median_depth,
                               scene=folder / "scene.npz")
        return paths

    @staticmethod
    def intrinsics(path, timestamps):
        matrices = np.asarray(path["K"])
        query = np.asarray(timestamps)
        if query.min() < path["indices"][0] or query.max() > path["indices"][-1]:
            raise ValueError("Intrinsics timestamps would extrapolate")
        if matrices.shape == (3, 3):
            return np.broadcast_to(matrices, (len(query), 3, 3)).copy()
        return np.stack([np.interp(query, path["indices"], matrices[:, i, j])
                         for i in range(3) for j in range(3)], axis=1).reshape(-1, 3, 3)


class Renderer:
    def __init__(self, scene_or_checkpoint, width=WIDTH, height=HEIGHT, frame_times=None,
                 temporal_opacity=False):
        import torch
        from gsplat.rendering import rasterization

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU/point rendering fallback is provided")
        self.torch, self.rasterization = torch, rasterization
        self.width, self.height = width, height
        self.source_times = self.temporal_widths = None
        self.rig_corrections = {}
        if Path(scene_or_checkpoint).suffix == ".npz":
            self.degree, self.step = None, None
            with np.load(scene_or_checkpoint, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in ("means", "scales", "quats", "colors", "opacities")}
                n = len(arrays["means"])
                shapes = dict(means=(n, 3), scales=(n, 3), quats=(n, 4), colors=(n, 3))
                for key, value in arrays.items():
                    if (not n or not np.issubdtype(value.dtype, np.floating)
                            or not np.isfinite(value).all()
                            or (key in shapes and value.shape != shapes[key])
                            or (key == "opacities" and value.shape not in ((n,), (n, 1)))):
                        raise ValueError(f"Invalid activated scene attribute: {key}")
                if (np.any(arrays["scales"] <= 0)
                        or np.any(np.linalg.norm(arrays["quats"], axis=1) < 1e-8)
                        or np.any((arrays["opacities"] < 0) | (arrays["opacities"] > 1))
                        or np.any((arrays["colors"] < 0) | (arrays["colors"] > 1))):
                    raise ValueError("Scene requires positive scales, nonzero quaternions, and activated RGB/opacity")
                for key, value in arrays.items():
                    tensor = torch.as_tensor(value, device="cuda", dtype=torch.float32)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Scene {key} overflows float32")
                    setattr(self, key, tensor.reshape(n) if key == "opacities" else tensor)
                if (torch.any(self.scales <= 0)
                        or torch.any(torch.linalg.vector_norm(self.quats, dim=1) < 1e-8)):
                    raise ValueError("Scene scales or quaternions underflow in float32")
                if temporal_opacity:
                    if frame_times is None or len(frame_times) < 2:
                        raise ValueError("Temporal opacity requires the selected source-frame timeline")
                    source = data["source_frames"]
                    widths = data["temporal_widths"].reshape(-1)
                    count = len(frame_times)
                    if (source.shape != (n,) or widths.shape != (n,)
                            or not np.isfinite(source).all() or np.any(source != np.floor(source))
                            or np.any((source < 0) | (source >= count))
                            or not np.isfinite(widths).all() or np.any(widths <= 0)):
                        raise ValueError("Invalid source frames or temporal widths (must be finite and positive)")
                    self.frame_times = np.asarray(frame_times, dtype=float)
                    if not np.isfinite(self.frame_times).all() or np.any(np.diff(self.frame_times) <= 0):
                        raise ValueError("Temporal source timeline must be finite and increasing")
                    self.source_times = torch.as_tensor(source / (count - 1) * count / 4,
                                                        device="cuda", dtype=torch.float32)
                    self.temporal_widths = torch.as_tensor(widths, device="cuda", dtype=torch.float32)
                    if not torch.isfinite(self.temporal_widths).all() or torch.any(self.temporal_widths <= 0):
                        raise ValueError("Temporal widths overflow or underflow in float32")
            return
        if temporal_opacity:
            raise ValueError("Temporal opacity is only supported for activated NPZ scenes")
        data = torch.load(scene_or_checkpoint, map_location="cpu", weights_only=True)
        if (type(data.get("sh_degree")) is not int or data["sh_degree"] not in (0, 1, 2, 3)
                or type(data.get("step")) is not int):
            raise ValueError("Checkpoint requires sh_degree in [0,3] and integer step")
        self.degree, self.step = data["sh_degree"], data["step"]
        splats = data["splats"]
        n = splats["means"].shape[0]
        if n == 0:
            raise ValueError("Checkpoint contains no Gaussians")
        shapes = dict(means=(n, 3), scales=(n, 3), quats=(n, 4), sh0=(n, 1, 3))
        for key in (*shapes, "opacities", "shN"):
            value = splats[key]
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint splats[{key!r}]")
            shape = tuple(value.shape)
            if key in shapes and shape != shapes[key]:
                raise ValueError(f"splats[{key!r}] must have shape {shapes[key]}")
            if key == "opacities" and shape not in ((n,), (n, 1)):
                raise ValueError("Opacity logits must have shape [N] or [N,1]")
            if key == "shN" and (len(shape) != 3 or shape[0] != n or shape[2] != 3
                                  or shape[1] not in (0, 3, 8, 15)
                                  or shape[1] < (self.degree + 1) ** 2 - 1):
                raise ValueError("SH coefficients do not support checkpoint sh_degree")
        if torch.any(torch.linalg.vector_norm(splats["quats"], dim=1) < 1e-8):
            raise ValueError("Checkpoint contains zero-length wxyz quaternions")
        self.means = splats["means"].to(device="cuda", dtype=torch.float32)
        self.scales = splats["scales"].to(device="cuda", dtype=torch.float32).exp()
        if not torch.isfinite(self.scales).all() or torch.any(self.scales <= 0):
            raise ValueError("Log scales overflow or underflow in float32")
        self.quats = splats["quats"].to(device="cuda", dtype=torch.float32)
        self.opacities = splats["opacities"].to(device="cuda", dtype=torch.float32).sigmoid().reshape(n)
        self.colors = torch.cat((splats["sh0"], splats["shN"]), dim=1).to(device="cuda", dtype=torch.float32)
        if not all(torch.isfinite(x).all() for x in (self.means, self.quats, self.colors)):
            raise ValueError("Checkpoint values overflow in float32")
        self.rig_corrections = {}
        for view, correction in data.get("rig_camera_corrections", {}).items():
            value = correction.numpy() if torch.is_tensor(correction) else np.asarray(correction)
            if (view not in VIEWS or value.shape != (4, 4) or not np.isfinite(value).all()
                    or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                    or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                    or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)):
                raise ValueError("Invalid shared rig camera correction")
            self.rig_corrections[view] = value

    def render(self, pose, K, view="front", timestamp=None):
        if view not in VIEWS:
            raise ValueError(f"Unknown camera: {view}")
        torch = self.torch
        with torch.inference_mode():
            corrected_view = self.rig_corrections.get(view, np.eye(4)) @ np.linalg.inv(pose)
            viewmat = torch.as_tensor(corrected_view, device="cuda", dtype=torch.float32)[None]
            intrinsic = torch.as_tensor(K, device="cuda", dtype=torch.float32)[None]
            opacities = self.opacities
            if self.source_times is not None:
                if (timestamp is None or not np.isfinite(timestamp)
                        or not self.frame_times[0] <= timestamp <= self.frame_times[-1]):
                    raise ValueError("Temporal rendering requires an in-range source timestamp")
                count = len(self.frame_times)
                target = np.interp(timestamp, self.frame_times, np.arange(count)) / (count - 1) * count / 4
                # Upstream alpha_t, gamma1=0.1; widths are temporal support, not quality.
                opacities = opacities * torch.exp(
                    np.log(.1) / (self.temporal_widths ** 2 + 1e-6) * (target - self.source_times) ** 2)
            rgb, alpha, _ = self.rasterization(
                means=self.means, quats=self.quats, scales=self.scales,
                opacities=opacities, colors=self.colors, viewmats=viewmat,
                Ks=intrinsic, width=self.width, height=self.height, sh_degree=self.degree,
                render_mode="RGB", packed=True,
                near_plane=1e-5 if self.degree is None else .05,
                far_plane=1e10 if self.degree is None else 100.)
            if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
                raise RuntimeError("gsplat returned nonfinite RGB or alpha")
            return ((rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8),
                    alpha[0, ..., 0].cpu().numpy())


class VideoWriter:
    """Stream RGB to ffmpeg; publish only after decoded ffprobe frame checks."""

    def __init__(self, path, width, height, fps, frames):
        if (not np.isfinite(fps) or fps <= 0 or type(frames) is not int or frames < 1
                or width <= 0 or height <= 0 or width % 2 or height % 2):
            raise ValueError("Require positive fps/frames and even positive video dimensions")
        self.path, self.width, self.height = Path(path), width, height
        self.fps, self.frames, self.written = fps, frames, 0
        self.process = None
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")

    def __enter__(self):
        if self.path.exists() or self.temporary.exists():
            raise FileExistsError(f"Refusing to overwrite video: {self.path}")
        self.stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.temporary)], stdin=subprocess.PIPE, stderr=self.stderr)
        except BaseException:
            self.stderr.close()
            raise
        return self

    def write(self, rgb):
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Video frame must be uint8 RGB of the declared dimensions")
        if self.written >= self.frames:
            raise ValueError("Too many video frames")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.written += 1

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                self.process.kill()
                self.process.communicate()
                return False
            self.process.stdin.close()
            status = self.process.wait()
            self.stderr.seek(0)
            error = self.stderr.read().decode(errors="replace")
            if status or self.written != self.frames:
                raise RuntimeError(f"ffmpeg failed or incomplete frame stream: {error}")
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames,avg_frame_rate",
                "-of", "json", str(self.temporary)], check=True, capture_output=True, text=True)
            if result.stderr.strip():
                raise RuntimeError(f"ffprobe reported decoding errors: {result.stderr}")
            streams = json.loads(result.stdout)["streams"]
            if len(streams) != 1:
                raise RuntimeError("ffprobe did not find exactly one video stream")
            stream = streams[0]
            num, den = map(int, stream["avg_frame_rate"].split("/"))
            if (stream["codec_name"] != "h264" or int(stream["width"]) != self.width
                    or int(stream["height"]) != self.height
                    or int(stream["nb_read_frames"]) != self.frames
                    or not np.isclose(num / den, self.fps)):
                raise RuntimeError(f"ffprobe verification failed: {stream}")
            self.verification = stream
            if self.path.exists():
                raise FileExistsError(f"Refusing to overwrite video: {self.path}")
            self.temporary.replace(self.path)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.stderr.close()
            self.temporary.unlink(missing_ok=True)
