#!/usr/bin/env python3
"""Offline genuine reconstruction evaluation; no trajectory-fitted alignment."""
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_gs.scene import Scene
from parking_gs.feedforward import transform_scene
from parking_gs.mapping import Grid, build_grid
from parking_gs.planner import Vehicle, collision, plan
from parking_gs.control import simulate
from analyze_parkinge2e import compare_paths, tracking_errors


def fit_floor(points, calibrated_rotation, camera_center, seed=14, camera_height=.65):
    """RANSAC with metric residuals inferred only from camera height per hypothesis."""
    q = (points - camera_center) @ calibrated_rotation.T
    rng = np.random.default_rng(seed)
    sample = q[rng.choice(len(q), min(len(q), 12000), replace=False)]
    best = None
    for _ in range(1200):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        d = -float(n @ a)
        if n[2] < np.cos(np.deg2rad(30)) or d <= 1e-6:
            continue
        residual = np.abs(sample @ n + d) * camera_height / d
        inliers = residual <= .04
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, n.copy(), d)
    if best is None:
        raise ValueError("No floor plane within 30 degrees of calibrated Z")
    _, n, d = best
    for _ in range(3):
        inliers = np.abs(q @ n+d)*camera_height/d <= .04
        center = q[inliers].mean(0)
        _, _, vh = np.linalg.svd(q[inliers]-center, full_matrices=False)
        n = vh[-1]
        if n[2] < 0:
            n = -n
        d = -float(n @ center)
        if d <= 0 or n[2] < np.cos(np.deg2rad(30)):
            raise ValueError("Refined plane violates height/normal constraint")
    residual = np.abs(q @ n+d)*camera_height/d
    inliers = residual <= .04
    if inliers.mean() < .35:
        raise ValueError("Floor ROI plane support below 35 percent")
    # Rodrigues' shortest proper rotation takes the calibrated floor normal to Z.
    v = np.cross(n, [0., 0., 1.])
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    leveling = np.eye(3) + k + k @ k / (1+n[2])
    return camera_height/d, leveling @ calibrated_rotation, dict(
        normal_calibrated=n.tolist(), offset_model_units=d,
        angle_to_calibrated_z_deg=float(np.rad2deg(np.arccos(n[2]))),
        roi_valid_points=len(q), inliers=int(inliers.sum()), inlier_fraction=float(inliers.mean()),
        inlier_residual_m=dict(rmse=float(np.sqrt(np.mean(residual[inliers]**2))),
                              p95=float(np.quantile(residual[inliers], .95)),
                              max=float(residual[inliers].max())),
        all_roi_residual_quantiles_m=np.quantile(residual, [0, .5, .9, .95, 1]).tolist(),
        residual_threshold_m=.04, minimum_support_fraction=.35,
        ransac_iterations=1200, seed=seed, maximum_normal_angle_deg=30)


def save_grid(path, grid, **extra):
    np.savez_compressed(path, risk=grid.risk, known=grid.known,
                        bounds=grid.bounds, resolution=grid.resolution,
                        threshold=grid.threshold, **extra)


def evaluate_case(name, source, grid, validation_grid, start, goal, reference, vehicle, timeout):
    case = dict(name=name, map_source=source, status="not_run", collision_checks=dict(
        start=collision(start, grid, vehicle), goal=collision(goal, grid, vehicle),
        validation_start=collision(start, validation_grid, vehicle),
        validation_goal=collision(goal, validation_grid, vehicle)))
    risk_only = Grid(grid.risk, np.ones_like(grid.known), grid.bounds, grid.resolution, grid.threshold)
    unknown_only = Grid(np.zeros_like(grid.risk), grid.known, grid.bounds, grid.resolution, grid.threshold)
    case["collision_checks"]["endpoint_attribution"] = {
        key: dict(risk_or_bounds=collision(pose, risk_only, vehicle),
                  unknown_or_bounds=collision(pose, unknown_only, vehicle))
        for key, pose in (("start", start), ("goal", goal))}
    begin = time.perf_counter()
    try:
        result = plan(start, goal, grid, vehicle, timeout=timeout, max_expansions=35000)
        case["plan"] = result
        planned = np.array([[p["x"], p["y"], p["yaw"]] for p in result["path"]])
        blocked = [collision(p, validation_grid, vehicle) for p in planned]
        case["collision_checks"].update(full_validation_path_blocked=bool(any(blocked)),
                                         full_validation_path_blocked_vertices=int(sum(blocked)))
        case["comparison"] = dict(planned_vs_reference=compare_paths(planned, reference))
        if any(blocked):
            case["status"] = "plan_rejected_on_full_covariance_grid"
        else:
            simulation = simulate(result, validation_grid)
            case["simulation"] = simulation
            executed = np.array([[p["x"], p["y"], p["yaw"]] for p in simulation["trace"]])
            case["comparison"].update(executed_vs_reference=compare_paths(executed, reference),
                                       executed_vs_planned=tracking_errors(executed, planned))
            case["status"] = "parked" if simulation["status"] == "parked" else "simulation_"+simulation["status"]
    except (ValueError, RuntimeError) as error:
        case.update(status="planning_failed", exception=dict(type=type(error).__name__, message=str(error)))
    case["elapsed_s"] = time.perf_counter()-begin
    return case


def run(args, report, log):
    root, prep = args.root, args.preparation
    mapping = root / "mapping"
    mapping.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prep / "manifest.json").read_text())
    with np.load(prep / "episode/reference_selected.npz", allow_pickle=False) as data:
        reference = np.column_stack((data["position_local_m"][:, :2], data["yaw_local_rad"]))
        times = data["time_ns"]
    start, goal = reference[0], reference[-1]
    report.update(sample_name="parkinge2e_20261010_01_candidate14_single_back_16frames",
                  reference=dict(source=manifest["source"], start=start.tolist(), goal=goal.tolist(),
                                 selection="first and last recorded poses in stationary margins",
                                 stationary_margin_metadata=manifest["selected_episode"],
                                 duration_s=float((times[-1]-times[0])/1e9), yaw_units="radians",
                                 localization_point_equivalent_to_rear_axle="UNVERIFIED"),
                  vehicle=dict(**asdict(Vehicle()), length_m=Vehicle().front+Vehicle().rear,
                               dimensions_source="core defaults ASSUMED, not measured vehicle"))
    recon = root / "reconstruction_single_back"
    report["inference"] = json.loads((recon / "scene.json").read_text())
    summary_path = root / "reconstruction/reconstruction_summary.json"
    report["earlier_multicamera_diagnostic"] = dict(
        status="alignment_failure_not_used_for_mapping",
        first_baseline_error_m_user_reported=1.56, drift="previously observed",
        artifact="reconstruction/fused_scene.npz",
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else None)
    with np.load(recon / "scene.npz", allow_pickle=False) as data:
        raw_arrays = {key: data[key] for key in data.files}
    np.savez_compressed(mapping / "raw_scene.npz", **raw_arrays)
    raw = Scene(**{k: raw_arrays[k] for k in ("means", "scales", "quats", "colors", "opacities")},
                metadata=dict(units="meters", up_axis="z", coordinate_frame="raw_model_gauge_NOT_metric_or_z_up")).validate()
    with np.load(recon / "predictions.npz", allow_pickle=False) as data:
        extrinsics, points = data["extrinsics"].astype(float), data["world_points"].astype(float)
        depth, conf = data["depth"][..., 0], data["depth_conf"]
        input_paths = data["input_paths"]
    camera_rotations = extrinsics[:, :3, :3].transpose(0, 2, 1)
    centers = -np.einsum("nij,nj->ni", camera_rotations, extrinsics[:, :3, 3])
    calibration = np.array(manifest["cameras"]["back"]["T_vehicle_from_optical"])
    calibrated_rotation = calibration[:3, :3] @ camera_rotations[0].T
    valid = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(conf) & (conf >= .99)
    h, w = conf.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    roi = (xx >= .25*w) & (xx < .75*w) & (yy >= .58*h) & (yy < .90*h)
    report["alignment"] = dict(status="fitting", method="initial camera calibration + floor plane + camera height, no motion fitting",
                               roi_normalized_xy=[.25, .75, .58, .90], depth_conf_threshold=.99,
                                roi_basis="image-reviewed visible floor, no ego hood; initial rear image only",
                               assumptions=["published rear-camera height 0.65 m above level floor",
                                            "initial calibration vehicle frame equals preparation initial local frame",
                                            "localization origin equivalent to vehicle/rear axle UNVERIFIED"],
                               initial_image=str(input_paths[0]), authoritative_video_source="input_paths back image timestamps; manifest nanoseconds",
                               future_reference_used_for_alignment=False)
    report["alignment"]["T_vehicle_from_optical"] = calibration.tolist()
    if (valid[0] & roi).sum() < 100:
        raise ValueError("Insufficient valid initial ROI depth")
    scale, rotation, floor = fit_floor(points[0][valid[0] & roi], calibrated_rotation, centers[0])
    translation = calibration[:3, 3] - scale * rotation @ centers[0]
    report["alignment"].update(status="valid_under_assumptions", scale_m_per_model_unit=scale,
                              rotation=rotation.tolist(), translation_m=translation.tolist(), floor_fit=floor)
    log(f"Floor scale={scale:.6f}, angle={floor['angle_to_calibrated_z_deg']:.3f} deg, support={floor['inlier_fraction']:.3f}")
    aligned = transform_scene(raw, scale, rotation, translation)
    transformed_points = scale * (points @ rotation.T) + translation
    transformed_centers = scale * (centers @ rotation.T) + translation
    transformed_rotations = rotation[None] @ camera_rotations
    report["alignment"]["episode_geometry_qc"] = dict(
        camera_height_range_m=[float(transformed_centers[:, 2].min()), float(transformed_centers[:, 2].max())],
        initial_camera_center_m=transformed_centers[0].tolist(),
        final_camera_center_m=transformed_centers[-1].tolist(),
        final_camera_height_change_m=float(transformed_centers[-1, 2]-.65),
        warning="Initial floor-fit validity does NOT validate future geometry; reconstructed camera height drifts under level-floor assumption")
    report["limitations"].append("Reconstructed camera height drifts over the episode; initial ROI plane fit is not global reconstruction validation.")
    np.savez_compressed(mapping / "cameras.npz", centers_m=transformed_centers,
                        R_local_from_optical=transformed_rotations, raw_centers=centers,
                        input_paths=input_paths, selected_indices=np.array(json.loads((recon / "selected_indices.json").read_text())))
    distance = np.linalg.norm(transformed_points-transformed_centers[:, None, None], axis=-1)
    reliable = valid & (distance >= .25) & (distance <= 12.)
    near_floor = reliable & (np.abs(transformed_points[..., 2]) <= .12)
    floor_points = transformed_points[near_floor]
    if len(floor_points) < 100:
        raise ValueError("Insufficient observed floor after metric alignment")
    # Bounds can contain endpoints, but endpoints never create observed/free cells.
    reliable_points = transformed_points[reliable]
    bound_points = np.concatenate((reliable_points[:, :2], reference[[0, -1], :2]))
    lower = np.floor((bound_points.min(0)-5.)/.25)*.25
    upper = np.ceil((bound_points.max(0)+5.)/.25)*.25
    bounds = np.r_[lower, upper].tolist()
    aligned.metadata.update(bounds=bounds, known_free_polygon=[], coordinate_frame="initial_local_vehicle_assumed",
                            alignment=report["alignment"], units="meters", up_axis="z")
    aligned.save(mapping / "scene.npz")
    near_distance, _ = cKDTree(floor_points).query(aligned.means, workers=1)
    excluded = (np.abs(aligned.means[:, 2]) <= .12) & (near_distance <= .20)
    fields = ("means", "scales", "quats", "colors", "opacities")
    obstacles = Scene(**{k: getattr(aligned, k)[~excluded] for k in fields}, metadata=dict(aligned.metadata)).validate()
    obstacles.save(mapping / "obstacle_scene.npz")
    np.savez_compressed(mapping / "floor_separation.npz", excluded_ground=excluded,
                        labels=raw_arrays["labels"], floor_points_m=floor_points,
                        **{k: getattr(aligned, k)[excluded] for k in fields})
    grid = build_grid(obstacles, resolution=.25, uncertainty=.12)
    counts = np.zeros_like(grid.known, dtype=np.int32)
    indices = grid.indices(floor_points[:, :2])
    np.add.at(counts, (indices[:, 1], indices[:, 0]), 1)
    dense = counts >= 3
    structure = np.ones((3, 3), bool)
    known = binary_dilation(binary_closing(dense, structure=structure, iterations=1),
                            structure=structure, iterations=1)
    grid.known = known
    save_grid(mapping / "grid.npz", grid, floor_point_counts=counts, dense_observed_floor=dense)
    ablation = build_grid(obstacles, resolution=.25, uncertainty=.12, use_covariance=False)
    ablation.known = known.copy()
    save_grid(mapping / "grid_centers_only.npz", ablation)
    report["mapping"] = dict(bounds=bounds, resolution_m=.25, uncertainty_m=.12, risk_threshold=grid.threshold,
                             known_cells=int(known.sum()), total_cells=int(known.size), known_coverage_fraction=float(known.mean()),
                             known_area_m2=float(known.sum()*.25**2), known_risky_cells=int((known & (grid.risk >= grid.threshold)).sum()),
                             point_conf_threshold=.99, metric_camera_range_m=[.25, 12.], floor_height_tolerance_m=.12,
                             minimum_floor_points_per_cell=3, closing_cells=1, dilation_cells=1,
                             known_source="confidence-valid reconstructed pointmaps near floor ONLY; no reference corridor",
                             geometric_ground_exclusion=dict(count=int(excluded.sum()), total=len(excluded),
                                 center_height_tolerance_m=.12, nearest_observed_floor_distance_m=.20,
                                 semantic_labels_used=False, uncertainty="Floor-like obstacle bases can be excluded; no semantic road truth; broad floor covariance retained in full scene"))
    report["mapping"].update(raw_gaussian_filtering="Existing upstream static/sky filtering retained; no additional semantic filtering",
                             reliable_point_count=int(reliable.sum()), observed_floor_point_count=len(floor_points),
                             aligned_scene_artifact="mapping/scene.npz (all 120000 Gaussians)",
                             planning_scene_artifact="mapping/obstacle_scene.npz",
                             full_covariance_grid_preserved=True)
    (mapping / "alignment.json").write_text(json.dumps(report["alignment"], indent=2)+"\n")
    (mapping / "parameters.json").write_text(json.dumps(report["mapping"], indent=2)+"\n")
    # QC stays under mapping, never in the analyzer's output directory.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ids = np.linspace(0, len(raw.means)-1, 12000, dtype=int)
    for ax, s, title in zip(axes, [raw, aligned], ["Before: raw model gauge", "After: camera-height / level-floor assumptions"]):
        ax.scatter(s.means[ids, 0], s.means[ids, 2], s=1, c=s.colors[ids])
        ax.set(title=title, xlabel="X", ylabel="Z", aspect="equal")
    fig.tight_layout(); fig.savefig(mapping / "geometry_qc.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#b6bcc5")
    image = ax.imshow(np.ma.masked_where(~known, grid.risk), origin="lower", extent=[bounds[0], bounds[2], bounds[1], bounds[3]], vmin=0, vmax=1, cmap="magma")
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=["cyan", "lime"], marker="x", label="Recorded endpoints (not used for known mask)")
    ax.set(title="Model covariance risk; gray = unknown (blocked)", xlabel="Local X (m)", ylabel="Local Y (m)", aspect="equal")
    ax.legend(); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(mapping / "known_risk.png", dpi=160); plt.close(fig)
    vehicle = Vehicle()
    for name, source, planning_grid in [("covariance_primary", "mapping/grid.npz", grid),
                                         ("centers_only_ablation", "mapping/grid_centers_only.npz; validation=mapping/grid.npz", ablation)]:
        case = evaluate_case(name, source, planning_grid, grid, start, goal, reference, vehicle, args.timeout)
        report["cases"].append(case)
        log(f"{name}: {case['status']}; collision checks={case['collision_checks']}")
    if report["cases"][0]["status"] != "parked":
        control = Grid(np.zeros_like(grid.risk), np.ones_like(known), bounds, .25, grid.threshold)
        save_grid(mapping / "grid_obstacle_free_control.npz", control)
        case = evaluate_case("obstacle_free_fully_known_CONTROL_DIAGNOSTIC", "mapping/grid_obstacle_free_control.npz (NOT model map)",
                             control, control, start, goal, reference, vehicle, args.timeout)
        case["interpretation"] = "Planner/controller exercise only; NOT full-pipeline validation or model-map fix"
        report["cases"].append(case)
        log(f"Control diagnostic: {case['status']}; simulation={(case.get('simulation') or {}).get('metrics')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/parkinge2e_20261010_01"))
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--timeout", type=float, choices=(25., 60.), default=25.)
    args = parser.parse_args()
    args.preparation = args.preparation or args.root / "preparation/selected_candidate14"
    report = dict(schema_version=1, cases=[], limitations=[
        "Recorded estimated localization is not independent ground truth.",
        "Camera-height and level-floor metric assumptions; no future-GT alignment or motion-scale fitting.",
        "Vehicle dimensions are default assumptions; localization/rear-axle equivalence UNVERIFIED.",
        "Indoor semantic palette unreliable; labels never establish free space.",
        "Sparse views, unknown-space blocking and geometric floor exclusion uncertainty remain.",
        "Offline kinematic simulation, not real vehicle testing or real-time inference."],
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
                         variables={k: os.environ.get(k) for k in ("LD_PRELOAD", "PYTHONNOUSERSITE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}),
        planner_budget=dict(timeout_s=args.timeout, max_expansions=35000))
    with (args.root / "evaluation.log").open("w") as stream:
        def log(message):
            print(message, flush=True)
            stream.write(message+"\n"); stream.flush()
        begin = time.perf_counter()
        try:
            run(args, report, log)
        except Exception as error:
            report["fatal_error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            report["cases"].append(dict(name="covariance_primary", status="geometry_or_evaluation_invalid", map_source="single_back_reconstruction", exception=report["fatal_error"]))
            log(traceback.format_exc())
        report["elapsed_s"] = time.perf_counter()-begin
        (args.root / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        log(f"Saved {args.root / 'evaluation.json'}; elapsed={report['elapsed_s']:.2f}s")
    return 1 if "fatal_error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
