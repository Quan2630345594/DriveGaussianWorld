"""Version 1 offline scene contract.

Writers return the manifest dict and exclusively create a new output directory.
Scene NPZ files contain float64 activated arrays, RGB in [0, 1], and unit wxyz
quaternions. Cameras are small inline records with world-from-optical poses.
Unknown reconstruction gauge is useful for visualization, not metric planning.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import expit


def sha256(path):
    """Hash a file incrementally, including large archives without loading them."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    """Write strict JSON without overwriting any existing path."""
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def normalize_gaussians(means, scales, quats, colors, opacities, *,
                        scale_encoding="activated", quaternion_order="wxyz",
                        opacity_encoding="activated", covariance_relative_floor=1e-6):
    """Explicitly convert encodings, normalize quaternions, and derive SPD Sigma.

    The smallest axis is floored relative to each Gaussian's largest axis to
    avoid floating-point indefinite covariances for near-flat splats. Output
    scales reflect that floor; no hidden clipping of colors or opacities occurs.
    """
    if scale_encoding not in ("activated", "log"):
        raise ValueError("scale_encoding must be activated or log")
    if quaternion_order not in ("wxyz", "xyzw"):
        raise ValueError("quaternion_order must be wxyz or xyzw")
    if opacity_encoding not in ("activated", "logit"):
        raise ValueError("opacity_encoding must be activated or logit")
    if not np.isfinite(covariance_relative_floor) or not 1e-6 <= covariance_relative_floor <= 1:
        raise ValueError("covariance_relative_floor must be in [1e-6, 1]")
    arrays = {k: np.array(v, dtype=np.float64, copy=True) for k, v in
              zip(("means", "scales", "quats", "colors", "opacities"),
                  (means, scales, quats, colors, opacities))}
    n = len(arrays["means"])
    if arrays["opacities"].shape == (n, 1):
        arrays["opacities"] = arrays["opacities"][:, 0]
    for key, shape in {"means": (n, 3), "scales": (n, 3), "quats": (n, 4),
                       "colors": (n, 3), "opacities": (n,)}.items():
        if n == 0 or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key} shape or values")
    if scale_encoding == "log":
        with np.errstate(over="ignore", under="ignore"):
            arrays["scales"] = np.exp(arrays["scales"])
    s = arrays["scales"]
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("scales must be finite and positive after activation")
    arrays["scales"] = s = np.maximum(s, s.max(axis=1, keepdims=True) * covariance_relative_floor)
    if opacity_encoding == "logit":
        arrays["opacities"] = expit(arrays["opacities"])
    for key in ("colors", "opacities"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f"{key} must be in [0, 1]")
    q = arrays["quats"]
    if quaternion_order == "wxyz":
        q = q[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("invalid quaternion norm")
    rotation = Rotation.from_quat(q / norms)
    arrays["quats"] = rotation.as_quat()[:, [3, 0, 1, 2]]
    r = rotation.as_matrix()
    with np.errstate(over="ignore", under="ignore"):
        cov = (r * (s * s)[:, None, :]) @ r.transpose(0, 2, 1)
    arrays["covariances"] = (cov + cov.transpose(0, 2, 1)) / 2
    if not np.isfinite(cov).all() or np.any(np.linalg.eigvalsh(arrays["covariances"]) <= 0):
        raise ValueError("scales do not yield representable SPD covariances")
    return arrays


def validate_scene(manifest, arrays):
    """Raise ValueError on contract violations; return None on success."""
    if manifest.get("schema_version") != 1 or not manifest.get("scene_id"):
        raise ValueError("invalid schema_version or scene_id")
    for key in ("source_classification", "provenance", "world_gauge", "scene_time",
                "availability", "parameterization", "arrays", "cameras", "planning_status"):
        if key not in manifest:
            raise ValueError(f"missing manifest field {key}")
    expected = {"scale_encoding": "activated", "quaternion_order": "wxyz",
                "opacity_encoding": "activated", "color_encoding": "rgb_0_1"}
    if manifest["parameterization"] != expected:
        raise ValueError("noncanonical parameterization")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or any(
        not provenance.get(key) for key in ("source_path", "license", "reconstruction_version")
    ) or not isinstance(provenance.get("source_sha256"), dict):
        raise ValueError("invalid provenance")
    for digest in provenance["source_sha256"].values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source SHA256")
    if manifest["source_classification"] != "SYNTHETIC_PLANNING_FIXTURE" and not provenance["source_sha256"]:
        raise ValueError("imported scene requires source hashes")
    gauge = manifest["world_gauge"]
    if not gauge.get("id") or gauge.get("metric_scale") not in ("verified", "unknown"):
        raise ValueError("invalid world gauge")
    if gauge["metric_scale"] == "unknown" and manifest["planning_status"] != "BLOCKED_UNKNOWN_METRIC_GAUGE":
        raise ValueError("unknown metric gauge must block planning")
    if manifest["planning_status"] == "SYNTHETIC_FIXTURE_ONLY" and (
        manifest["source_classification"] != "SYNTHETIC_PLANNING_FIXTURE"
        or gauge["metric_scale"] != "verified" or gauge.get("axes") != "x_forward_y_left_z_up"
    ):
        raise ValueError("invalid synthetic planning claim")
    if manifest["planning_status"] not in ("BLOCKED_UNKNOWN_METRIC_GAUGE", "BLOCKED_GEOMETRY", "SYNTHETIC_FIXTURE_ONLY"):
        raise ValueError("unsupported planning status")
    if manifest.get("array_file") != "scene.npz":
        raise ValueError("array_file must be scene.npz")
    n = manifest.get("gaussian_count")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("invalid gaussian_count")
    for key, tail in {"means": (3,), "scales": (3,), "quats": (4,),
                      "covariances": (3, 3), "colors": (3,), "opacities": ()}.items():
        a = np.asarray(arrays.get(key))
        if a.shape != (n,) + tail or a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(f"invalid {key}")
        if manifest["arrays"].get(key) != {"shape": list(a.shape), "dtype": str(a.dtype)}:
            raise ValueError(f"array schema mismatch: {key}")
    s, q, cov = arrays["scales"], arrays["quats"], arrays["covariances"]
    if np.any(s <= 0) or not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("invalid scales or unit quaternions")
    for key in ("colors", "opacities"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f"invalid {key} range")
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    expected_cov = (r * (s * s)[:, None, :]) @ r.transpose(0, 2, 1)
    if (not np.allclose(cov, cov.transpose(0, 2, 1), atol=0, rtol=1e-10)
            or np.any(np.linalg.eigvalsh(cov) <= 0)
            or not np.allclose(cov, expected_cov, atol=1e-30, rtol=1e-8)):
        raise ValueError("covariances must be SPD and agree with scales/quats")
    if not manifest["cameras"]:
        raise ValueError("at least one camera required")
    for camera in manifest["cameras"]:
        k = np.asarray(camera.get("K"), dtype=float)
        t = np.asarray(camera.get("T_world_from_optical"), dtype=float)
        size = np.asarray(camera.get("image_size_wh"))
        timestamp = camera.get("timestamp")
        if not (isinstance(timestamp, str) and timestamp == "unknown") and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not np.isfinite(timestamp)
        ):
            raise ValueError("camera timestamp must be finite numeric or unknown")
        if (camera.get("frame") != "optical_x_right_y_down_z_forward"
                or size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0)
                or k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
                or k[0, 1] != 0 or k[1, 0] != 0
                or not (0 <= k[0, 2] < size[0] and 0 <= k[1, 2] < size[1])
                or t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
            raise ValueError("invalid optical camera")


def assert_same_gauge(*manifests):
    """Reject fusion across gauges; this utility does not estimate alignment."""
    if not manifests:
        raise ValueError("at least one scene required")
    reference = manifests[0]["world_gauge"]
    if any(m["world_gauge"] != reference for m in manifests[1:]):
        raise ValueError("different gauges require verified SE3+scale alignment before fusion")


def _write_scene(output_dir, scene_id, arrays, cameras, gauge, provenance, classification):
    manifest = {
        "schema_version": 1, "scene_id": scene_id, "array_file": "scene.npz",
        "gaussian_count": len(arrays["means"]), "source_classification": classification,
        "provenance": provenance, "world_gauge": gauge,
        "scene_time": {"model": "static_snapshot", "timestamp": "unknown"},
        "availability": {"visibility": "unknown", "dynamics": "unknown",
                         "occupancy_ground_truth": False},
        "parameterization": {"scale_encoding": "activated", "quaternion_order": "wxyz",
                             "opacity_encoding": "activated", "color_encoding": "rgb_0_1"},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "cameras": cameras,
        "planning_status": ("SYNTHETIC_FIXTURE_ONLY" if classification == "SYNTHETIC_PLANNING_FIXTURE"
                            else "BLOCKED_UNKNOWN_METRIC_GAUGE"),
    }
    validate_scene(manifest, arrays)
    output = Path(output_dir)
    output.mkdir(exist_ok=False)
    with (output / "scene.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    manifest["array_sha256"] = sha256(output / "scene.npz")
    write_json(output / "manifest.json", manifest)
    return manifest


def prepare_dggt(source_dir, output_dir):
    """Import the activated DGGT scene export, not raw prediction splats.

    Only camera extrinsics/intrinsics are loaded from predictions.npz. Historical
    absolute paths are retained as provenance files, never followed or rebased.
    K is predicted intrinsics, not physical input or image-warp intrinsics.
    """
    source = Path(source_dir)
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    with (source / "scene.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    files = ["scene.npz", "scene.json", "predictions.npz"]
    files += [name for name in ("command.json", "input_frames.json", "input_intrinsics.json",
                               "selected_indices.json") if (source / name).is_file()]
    hashes = {name: sha256(source / name) for name in files}
    with np.load(source / "scene.npz", allow_pickle=False) as data:
        raw = {k: data[k] for k in ("means", "scales", "quats", "colors", "opacities")}
        arrays = normalize_gaussians(**raw)
        available = {k: True for k in data.files if k not in raw}
        optional = {}
        for key in available:
            try:
                value = data[key]
            except ValueError:
                optional[key] = {"available": True, "inspection": "not loaded without pickle"}
                continue
            optional[key] = {"available": True, "shape": list(value.shape), "dtype": str(value.dtype)}
            if value.dtype.kind in "biufc":
                optional[key]["finite_count"] = int(np.isfinite(value).sum())
    with np.load(source / "predictions.npz", allow_pickle=False) as data:
        extrinsics = np.asarray(data["extrinsics"], dtype=float)
        intrinsics = np.asarray(data["intrinsics"], dtype=float)
    count = len(extrinsics)
    if extrinsics.shape != (count, 3, 4) or intrinsics.shape != (count, 3, 3) or count == 0:
        raise ValueError("invalid DGGT camera arrays")
    if metadata.get("gaussians", len(arrays["means"])) != len(arrays["means"]):
        raise ValueError("DGGT metadata Gaussian count mismatch")
    if metadata.get("input_frames", count) != count:
        raise ValueError("DGGT metadata camera count mismatch")
    source_frames = ["unknown"] * count
    if "selected_indices.json" in hashes:
        with (source / "selected_indices.json").open(encoding="utf-8") as stream:
            source_frames = json.load(stream)
        if (not isinstance(source_frames, list) or len(source_frames) != count
                or any(type(i) is not int or i < 0 for i in source_frames)
                or any(a >= b for a, b in zip(source_frames, source_frames[1:]))):
            raise ValueError("selected_indices must be increasing nonnegative integers matching camera count")
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    poses[:, :3] = extrinsics
    poses = np.linalg.inv(poses)
    cameras = [{"camera_id": str(i), "frame": "optical_x_right_y_down_z_forward",
                "image_size_wh": metadata["image_size"], "K": intrinsics[i].tolist(),
                "T_world_from_optical": poses[i].tolist(), "timestamp": "unknown",
                "timestamp_units": "seconds", "source_frame": source_frames[i],
                "source_frame_coordinate": ("selected_indices.json:source_frame_index"
                                            if "selected_indices.json" in hashes else "unknown")}
               for i in range(count)]
    regularization = "Numerical regularization changes geometry; not metrically meaningful in the unknown gauge."
    revision = {"repository_dggt_pin": "a3276d2bbe4cbb03bcc117830b1836110a27adeb",
                "evidence": "repository gitlink third_party/dggt; not evidence of export producer revision",
                "export_producer_revision": "unknown"}
    manifest = _write_scene(output_dir, source.name, arrays, cameras,
                        {"id": "dggt:" + hashes["predictions.npz"], "metric_scale": "unknown",
                         "units": "unknown", "axes": "unknown", "confidence": "unknown"},
                        {"source_path": str(source.resolve()), "source_sha256": hashes,
                          "license": "unknown", "reconstruction_version": "unknown",
                          "revision_evidence": revision,
                         "source_metadata": metadata, "source_optional_fields": available,
                         "normalization": {"input_scales": "activated", "input_quats": "wxyz",
                                            "input_opacities": "activated", "covariance_relative_scale_floor": 1e-6,
                                            "geometry_warning": regularization},
                         "camera_conversion": "inverse optical-from-world extrinsics",
                         "K_source": "predictions.npz:intrinsics",
                         "temporal_policy": "base opacity snapshot; not time-conditioned",
                         "historical_paths": "not followed"},
                         "DGGT_RECONSTRUCTION_UNKNOWN_METRIC_GAUGE")
    correction = arrays["scales"] - raw["scales"]
    report = {
        "source_sha256": hashes, "canonical_array_sha256": manifest["array_sha256"],
        "source_preservation": "Source files are unchanged; hashes cover exact raw file bytes.",
        "geometry_warning": regularization, "revision_evidence": revision,
        "scale_pre": {"min": float(raw["scales"].min()), "max": float(raw["scales"].max())},
        "scale_post": {"min": float(arrays["scales"].min()), "max": float(arrays["scales"].max())},
        "floored_splat_count": int(np.any(correction > 0, axis=1).sum()),
        "max_absolute_scale_correction": float(np.abs(correction).max()),
        "quaternion_norm_input_range": [float(v) for v in (
            np.linalg.norm(raw["quats"], axis=1).min(), np.linalg.norm(raw["quats"], axis=1).max())],
        "ranges": {stage: {key: {"min": float(values[key].min()), "max": float(values[key].max())}
                            for key in ("opacities", "colors")}
                   for stage, values in (("input", raw), ("output", arrays))},
        "covariance_eigenvalue_min": float(np.linalg.eigvalsh(arrays["covariances"]).min()),
        "finite_counts": {stage: {key: {"finite": int(np.isfinite(value).sum()), "total": int(value.size)}
                                  for key, value in values.items()}
                          for stage, values in (("input", raw), ("output", arrays),
                                                ("cameras", {"extrinsics": extrinsics, "intrinsics": intrinsics}))},
        "camera_count": count, "optional_fields": optional,
        "optional_field_policy": "Availability and numerical diagnostics only; no semantic truth claims.",
        "time": {"timestamp_seconds": "unknown", "source_frames": source_frames,
                 "source_frame_coordinate": cameras[0]["source_frame_coordinate"]},
    }
    # Project a deterministic subset of means, without compositing or inferring visibility.
    indices = np.linspace(0, len(arrays["means"]) - 1, min(2000, len(arrays["means"])), dtype=int)
    xyz = arrays["means"][indices] @ extrinsics[0, :, :3].T + extrinsics[0, :, 3]
    uvw = xyz @ intrinsics[0].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:]
    width, height = cameras[0]["image_size_wh"]
    visible = ((xyz[:, 2] > 0) & np.isfinite(uv).all(axis=1)
               & (uv >= 0).all(axis=1) & (uv < [width, height]).all(axis=1))
    label = "PROJECTED_CENTERS_NOT_RENDERED_RGB"
    report["diagnostic_preview"] = {
        "file": "diagnostic_preview.svg", "label": label, "camera_id": "0",
        "image_size_wh": [518, 294], "source_image_size_wh": [width, height],
        "sampled_count": len(indices), "visible_count": int(visible.sum()),
        "meaning": "In-frustum sampled means only; not an observation, not occupancy, not occlusion-tested.",
    }
    circles = [f'<circle cx="{u * 518 / width:.4f}" cy="{v * 294 / height:.4f}" r="1"/>'
               for u, v in uv[visible]]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="518" height="294" viewBox="0 0 518 294">'
           f'<title>{label}</title><rect width="518" height="294" fill="white"/>'
           '<g fill="#345678" fill-opacity="0.5">' + ''.join(circles) + '</g>'
           f'<text x="4" y="12" font-size="10">{label}</text>'
           f'<text x="4" y="26" font-size="10">Not an observation; not occupancy. '
           f'Visible sampled centers: {int(visible.sum())}/{len(indices)}</text></svg>\n')
    with (Path(output_dir) / "diagnostic_preview.svg").open("x", encoding="utf-8") as stream:
        stream.write(svg)
    write_json(Path(output_dir) / "conversion_report.json", report)
    return manifest


def make_fixture(output_dir, scene_id, variant=0):
    """Write 24 deterministic synthetic splats in a metric, Z-up world."""
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be nonempty")
    if not isinstance(variant, int) or variant < 0:
        raise ValueError("variant must be a nonnegative integer")
    rng = np.random.default_rng(variant)
    means = np.column_stack((rng.uniform(5, 9, 24), rng.uniform(-2, 2, 24), np.ones(24)))
    arrays = normalize_gaussians(means, rng.uniform(.08, .25, (24, 3)),
                                np.tile([1., 0., 0., 0.], (24, 1)),
                                rng.uniform(.15, .95, (24, 3)), np.full(24, .9))
    # Optical right = world -Y; optical down = world -Z; forward = world +X.
    pose = [[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 1.], [0., 0., 0., 1.]]
    cameras = [{"camera_id": "forward", "frame": "optical_x_right_y_down_z_forward",
                "image_size_wh": [64, 48], "K": [[48., 0., 32.], [0., 48., 24.], [0., 0., 1.]],
                "T_world_from_optical": pose, "timestamp": 0.}]
    return _write_scene(output_dir, scene_id, arrays, cameras,
                        {"id": f"synthetic:{scene_id}:{variant}", "metric_scale": "verified",
                         "units": "meters", "axes": "x_forward_y_left_z_up",
                         "confidence": "synthetic_by_construction"},
                        {"source_path": "generated", "source_sha256": {}, "license": "synthetic_fixture",
                         "reconstruction_version": "fixture_v1", "variant": variant,
                         "purpose": "pipeline fixture, not real free-space or obstacle evidence"},
                        "SYNTHETIC_PLANNING_FIXTURE")
