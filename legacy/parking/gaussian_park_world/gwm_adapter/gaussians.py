"""CPU-only vendor-neutral Gaussian packing and diagnostic selection.

This is NOT the official SH feed. upstream_spec.json records SH at 10:13,
where this format stores RGB. Matching width (including N=2048) establishes
neither upstream compatibility nor preservation of scene shape/appearance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


MAX_SOURCE_COUNT = 200_000
MAX_OUTPUT_COUNT = 8192
FEATURE_LAYOUT = ["means3", "activated_scales3", "wxyz_quaternion4", "rgb3",
                  "activated_opacity1"]
_FIELDS = {"means": (3,), "scales": (3,), "quats": (4,),
           "colors": (3,), "opacities": ()}


def pack_gaussians(arrays, *, dtype=np.float32):
    """Pack canonical activated arrays as [N,14]; never activate or convert SH.

    dtype=float64 retains canonical float64 values exactly, including quaternion
    sign. Default float32 is a lossy storage conversion. Inputs are not mutated.
    Covariance is optional and stays outside the feature vector.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    means = np.asarray(arrays["means"])
    if means.ndim != 2 or means.shape[1:] != (3,):
        raise ValueError("invalid means shape")
    n = len(means)
    if n > MAX_SOURCE_COUNT:
        raise ValueError("source count exceeds 200000")
    values = {}
    for name, tail in _FIELDS.items():
        value = np.asarray(arrays[name])
        if (value.shape != (n,) + tail or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()):
            raise ValueError(f"invalid {name} shape or finite numeric values")
        values[name] = value
    if np.any(values["scales"] <= 0):
        raise ValueError("activated scales must be positive")
    if not np.allclose(np.linalg.norm(values["quats"], axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("quats must be unit wxyz quaternions")
    for name in ("colors", "opacities"):
        if np.any((values[name] < 0) | (values[name] > 1)):
            raise ValueError(f"{name} must be in [0,1]")
    if "covariances" in arrays:
        cov = np.asarray(arrays["covariances"])
        if (cov.shape != (n, 3, 3) or cov.dtype.kind not in "fiu"
                or not np.isfinite(cov).all()):
            raise ValueError("invalid covariances")
    with np.errstate(over="ignore"):
        features = np.concatenate([values["means"], values["scales"], values["quats"],
                                   values["colors"], values["opacities"][:, None]], axis=1).astype(dtype)
    if not np.isfinite(features).all():
        raise ValueError("features not finite in requested dtype")
    if np.any(features[:, 3:6] <= 0):
        raise ValueError("positive scales not representable in requested dtype")
    return features


def unpack_gaussians(features, *, covariances=None):
    """Copy an unpadded [N,14] pack back to canonical arrays, without inference.

    Filter padded features using valid_mask first. Optional covariance is copied,
    not reconstructed, so independent canonical covariance precision is retained.
    """
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 14:
        raise ValueError("features must have shape [N,14]")
    arrays = {"means": features[:, :3].copy(), "scales": features[:, 3:6].copy(),
              "quats": features[:, 6:10].copy(), "colors": features[:, 10:13].copy(),
              "opacities": features[:, 13].copy()}
    if covariances is not None:
        arrays["covariances"] = np.array(covariances, copy=True)
    pack_gaussians(arrays, dtype=features.dtype)
    return arrays


def sample_gaussians(arrays, count=2048, allow_padding=False, *, dtype=np.float32):
    """Select the smallest SHA256 content hashes in canonical hash order.

    Hash input is each contiguous little-endian float64 feature row, with q/-q
    resolved by making the first nonzero wxyz component positive and all signed
    zeros made positive. Hash collisions tie-break by those same row bytes, never
    by IDs or row numbers. Identical rows are interchangeable as a feature
    multiset; their audit sidecars need not be permutation-invariant.

    Cost is O(N) hashing plus O(N log N) sorting and O(N) storage, no NxN arrays.
    canonical_arrays contains selected, unpadded ORIGINAL values (copies), not
    model inputs. audit_indices indexes the original rows for sidecar alignment.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= MAX_OUTPUT_COUNT:
        raise ValueError("count must be an integer in [1,8192]")
    if not isinstance(allow_padding, (bool, np.bool_)):
        raise ValueError("allow_padding must be bool")
    rows = pack_gaussians(arrays, dtype=np.float64)
    n = len(rows)
    if n < count and not allow_padding:
        raise ValueError(f"insufficient Gaussians: {n} < {count}; explicitly allow_padding for fixtures")
    q = rows[:, 6:10]
    first = np.argmax(q != 0, axis=1)
    q[q[np.arange(n), first] < 0] *= -1
    rows[rows == 0] = 0.0
    rows = np.ascontiguousarray(rows, dtype="<f8")
    content = [row.tobytes() for row in rows]
    hashes = [hashlib.sha256(row).digest() for row in content]
    indices = np.array(sorted(range(n), key=lambda i: (hashes[i], content[i]))[:count], dtype=np.int64)
    selected = len(indices)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    features = np.zeros((count, 14), dtype=dtype)
    with np.errstate(over="ignore"):
        features[:selected] = rows[indices]
    if not np.isfinite(features).all():
        raise ValueError("features not finite in requested dtype")
    if np.any(features[:selected, 3:6] <= 0):
        raise ValueError("positive scales not representable in requested dtype")
    valid_mask = np.arange(count) < selected
    canonical = {name: np.asarray(arrays[name])[indices].copy() for name in _FIELDS}
    if "covariances" in arrays:
        canonical["covariances"] = np.asarray(arrays["covariances"])[indices].copy()
    return {"features": features, "valid_mask": valid_mask,
            "audit_indices": indices, "canonical_arrays": canonical,
            "metadata": {"format": "vendor-neutral14", "layout": FEATURE_LAYOUT.copy(),
                         "dtype": str(dtype), "runtime_status": "PENDING",
                         "official_gaussian_binding": "PENDING", "official_sh_feed": False,
                         "source_count": n, "selected_count": selected, "output_count": int(count),
                         "padding_count": int(count - selected),
                         "algorithm": "sha256_canonical_float64_le_content_sort",
                         "ids_used": False, "early_rows_used": False, "is_fps": False,
                         "geometry_coverage_preserved": False, "opacity_preserved": False,
                         "causal_verified_model_input": False,
                         "caveats": ["Not FPS or official FPS equivalent.",
                                     "Geometry coverage, shape and composited opacity/appearance are not preserved.",
                                     "Width 14 and count 2048 do not verify runtime compatibility or shape preservation.",
                                     "RGB at 10:13 is not upstream SH; no official feed conversion is established.",
                                     "Padding is zero and invalid; indices and canonical sidecars are AUDIT ONLY.",
                                     "No causal visibility or occlusion verification."]}}


def frustum_candidates(arrays, K, T_world_from_optical, image_size,
                       source_time_s, available_at_s, query_time_s):
    """Diagnostic center-frustum/time masks ONLY, never observed/causal input.

    image_size is (width,height); optical axes are right/down/forward. Clip
    centers to z>0, 0<=u<width, 0<=v<height, not Gaussian support or occlusion.
    Source/availability times must be actual per-row [N] arrays, or None/the
    literal 'unknown'. Unknown times exclude all affected candidates; nonfinite
    numeric timestamps are errors. Future numeric times are explicitly masked.
    """
    features = pack_gaussians(arrays, dtype=np.float64)
    n = len(features)
    query = np.asarray(query_time_s)
    if query.shape != () or query.dtype.kind not in "fiu" or not np.isfinite(query):
        raise ValueError("query_time_s must be a finite scalar timestamp")
    masks, times = {}, {}
    for name, value in (("source_time_s", source_time_s), ("available_at_s", available_at_s)):
        if value is None or (isinstance(value, str) and value == "unknown"):
            times[name] = None
            masks[name] = np.zeros(n, dtype=bool)
        else:
            time = np.asarray(value)
            if time.shape != (n,) or time.dtype.kind not in "fiu" or not np.isfinite(time).all():
                raise ValueError(f"{name} must be finite per-row [N] timestamps or unknown")
            times[name] = time.copy()
            masks[name] = time <= query
    k, t, size = np.asarray(K, dtype=float), np.asarray(T_world_from_optical, dtype=float), np.asarray(image_size)
    if (size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0)
            or k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1])
            or k[1, 0] != 0
            or t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-7, rtol=0)):
        raise ValueError("invalid optical camera or image_size (width,height)")
    with np.errstate(over="ignore", invalid="ignore"):
        optical = (features[:, :3] - t[:3, 3]) @ t[:3, :3]
        projected = optical @ k.T
        depth = projected[:, 2]
        frustum = (np.isfinite(projected).all(axis=1) & (depth > 0)
                   & (projected[:, 0] >= 0) & (projected[:, 0] < size[0] * depth)
                   & (projected[:, 1] >= 0) & (projected[:, 1] < size[1] * depth))
    temporal = masks["source_time_s"] & masks["available_at_s"]
    return {"frustum_mask": frustum, "source_time_mask": masks["source_time_s"],
            "availability_mask": masks["available_at_s"], "temporal_mask": temporal,
            "candidate_mask": frustum & temporal, **times, "query_time_s": float(query),
            "metadata": {"status": "PARTIAL", "occlusion": "UNKNOWN",
                         "causal_verified_model_input": False, "diagnostic_only": True,
                         "timestamp_provenance": "Caller supplied; not independently verified",
                         "caveats": ["Center clipping only, not Gaussian-support visibility.",
                                     "Frustum membership does not establish observation or causal availability.",
                                     "Unknown timestamps exclude candidates; no verified data is produced."]}}


def main(argv=None):
    """Standalone sample command; existing CLIs can call sample_gaussians directly."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--input", required=True, help="Canonical scene NPZ, not raw SH")
    sample.add_argument("--output", required=True, help="New NPZ with metadata JSON and audit sidecars")
    sample.add_argument("--count", type=int, default=2048)
    sample.add_argument("--allow-padding", action="store_true")
    sample.add_argument("--report", help="Optional exclusively created numerical/provenance JSON report")
    diagnostic = commands.add_parser("diagnose-frustum", help="Diagnostics only, never an observed Gaussian feed")
    diagnostic.add_argument("--scene", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--query-time", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.command == "diagnose-frustum":
        from gaussian_park_world.dataset.offline import load_scene
        from gaussian_park_world.scenes.contract import sha256, write_json
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        manifest, arrays = load_scene(args.scene)
        camera = manifest["cameras"][0]
        result = frustum_candidates(arrays, camera["K"], camera["T_world_from_optical"],
                                    camera["image_size_wh"], None, None, args.query_time)
        report = dict(scene_id=manifest["scene_id"], gaussian_count=len(arrays["means"]),
                      source_manifest_sha256=sha256(Path(args.scene) / "manifest.json"),
                      source_arrays_sha256=manifest["array_sha256"],
                      counts={k: int(result[k].sum()) for k in ("frustum_mask", "temporal_mask", "candidate_mask")},
                      source_time="UNKNOWN", available_at="UNKNOWN", query_time_s=args.query_time,
                      query_time_policy="diagnostic cutoff only; source clock is UNKNOWN", **result["metadata"])
        write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if Path(args.output).exists() or (args.report and Path(args.report).exists()):
        raise FileExistsError("sampling output/report already exists")
    started = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as archive:
        result = sample_gaussians(archive, args.count, args.allow_padding)
    from gaussian_park_world.scenes.contract import sha256, write_json
    result["metadata"].update(source_path=str(Path(args.input).resolve()), source_sha256=sha256(args.input),
                              output_features_bytes=result["features"].nbytes,
                              output_mask_bytes=result["valid_mask"].nbytes,
                              elapsed_seconds=time.perf_counter() - started,
                              process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    with open(args.output, "xb") as stream:
        np.savez(stream, features=result["features"], valid_mask=result["valid_mask"],
                 audit_indices=result["audit_indices"],
                 metadata_json=np.array(json.dumps(result["metadata"], allow_nan=False)),
                  **{"audit_canonical_" + k: v for k, v in result["canonical_arrays"].items()})
    if args.report:
        write_json(args.report, dict(result["metadata"], output_sha256=sha256(args.output)))
    print(json.dumps(result["metadata"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
