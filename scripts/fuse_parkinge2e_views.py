#!/usr/bin/env python3
"""CPU-only fusion of four independent temporal DGGT scenes, without motion fitting."""
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
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
from parking_gs.planner import Vehicle
from evaluate_parkinge2e import evaluate_case, fit_floor


EXPERIMENT = "four_independent_temporal_dggt_height_floor_v1"
HEIGHTS = dict(front=.71, back=.65, left=1.02, right=1.0)
FIELDS = ("means", "scales", "quats", "colors", "opacities")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_scene(path, scene):
    scene.validate()
    atomic_npz(path, **{key: getattr(scene, key) for key in FIELDS},
               metadata=np.array(json.dumps(scene.metadata, allow_nan=False)))


def save_grid(path, grid, **extra):
    atomic_npz(path, risk=grid.risk, known=grid.known, bounds=grid.bounds,
               resolution=grid.resolution, threshold=grid.threshold, **extra)


def run(args, report, log):
    root, mapping = args.root, args.root / "mapping"
    manifest = json.loads((args.preparation / "manifest.json").read_text())
    # Reference positions never participate in camera alignment or floor observation.
    with np.load(args.preparation / "episode/reference_selected.npz", allow_pickle=False) as data:
        reference = np.column_stack((data["position_local_m"][:, :2], data["yaw_local_rad"]))
        times = data["time_ns"]
    start, goal = reference[0], reference[-1]
    vehicle = Vehicle()
    report.update(reference=dict(source=manifest["source"], start=start.tolist(), goal=goal.tolist(),
                                 duration_s=float((times[-1]-times[0])/1e9),
                                 usage="endpoints, finite map bounds, post-hoc comparisons ONLY",
                                 localization_point_equivalent_to_rear_axle="UNVERIFIED"),
                  vehicle=dict(**asdict(vehicle), length_m=vehicle.front+vehicle.rear,
                               dimensions_source="standard core defaults ASSUMED, never shrunk"),
                  alignment={}, inference={})
    aligned_scenes, raw_scenes, floors, reliable_xy, excluded_views = [], [], [], [], []
    for view, height in HEIGHTS.items():
        recon = root / ("reconstruction_single_" + view)
        destination = mapping / view
        destination.mkdir(exist_ok=True)
        with np.load(recon / "scene.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if len(arrays["means"]) != 120000:
            raise ValueError(f"{view}: expected raw 120000 Gaussians, got {len(arrays['means'])}")
        atomic_npz(destination / "raw_scene.npz", **arrays)
        raw = Scene(**{key: arrays[key] for key in FIELDS}, metadata=dict(
            units="meters", up_axis="z", coordinate_frame="raw_model_gauge_NOT_metric_or_z_up",
            source_view=view)).validate()
        with np.load(recon / "predictions.npz", allow_pickle=False) as data:
            extrinsics = data["extrinsics"].astype(float)
            points = data["world_points"].astype(float)
            depth, conf = data["depth"][..., 0], data["depth_conf"]
            input_paths = data["input_paths"]
            intrinsics = data["intrinsics"]
        if len(extrinsics) != 16 or len(points) != 16 or len(input_paths) != 16:
            raise ValueError(f"{view}: expected 16 temporal frames")
        camera_rotations = extrinsics[:, :3, :3].transpose(0, 2, 1)
        centers = -np.einsum("nij,nj->ni", camera_rotations, extrinsics[:, :3, 3])
        calibration = np.array(manifest["cameras"][view]["T_vehicle_from_optical"], dtype=float)
        calibration[2, 3] = height
        calibrated_rotation = calibration[:3, :3] @ camera_rotations[0].T
        valid = (np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0)
                 & np.isfinite(conf) & (conf >= .99))
        h, w = conf.shape[1:]
        yy, xx = np.mgrid[:h, :w]
        roi = (xx >= .25*w) & (xx < .75*w) & (yy >= .58*h) & (yy < .90*h)
        if (valid[0] & roi).sum() < 100:
            raise ValueError(f"{view}: insufficient valid initial floor ROI")
        scale, rotation, floor_fit = fit_floor(points[0][valid[0] & roi], calibrated_rotation,
                                               centers[0], seed=14, camera_height=height)
        translation = calibration[:3, 3] - scale * rotation @ centers[0]
        aligned = transform_scene(raw, scale, rotation, translation)
        transformed_points = scale * (points @ rotation.T) + translation
        transformed_centers = scale * (centers @ rotation.T) + translation
        transformed_rotations = rotation[None] @ camera_rotations
        distance = np.linalg.norm(transformed_points-transformed_centers[:, None, None], axis=-1)
        reliable = valid & (distance >= .25) & (distance <= 12.)
        near_floor = reliable & (np.abs(transformed_points[..., 2]) <= .12)
        floor_points = transformed_points[near_floor]
        if len(floor_points) < 100:
            raise ValueError(f"{view}: insufficient observed metric floor")
        near_distance, _ = cKDTree(floor_points).query(aligned.means, workers=1)
        excluded = (np.abs(aligned.means[:, 2]) <= .12) & (near_distance <= .20)
        record = dict(status="valid_under_assumptions_NOT_global_geometry_validation",
                      method="initial optical calibration + initial ROI level-floor RANSAC + published camera height",
                      camera_height_m=height, T_vehicle_from_optical=calibration.tolist(),
                      scale_m_per_model_unit=scale, rotation=rotation.tolist(), translation_m=translation.tolist(),
                      floor_fit=floor_fit, initial_image=str(input_paths[0]),
                      roi_normalized_xy=[.25, .75, .58, .90],
                      roi_basis="user-confirmed clear visible floor in each initial image",
                      future_reference_used_for_alignment=False, estimated_intrinsics=True,
                      camera_height_range_m=[float(transformed_centers[:, 2].min()),
                                             float(transformed_centers[:, 2].max())],
                      final_camera_height_change_m=float(transformed_centers[-1, 2]-height),
                      initial_camera_center_m=transformed_centers[0].tolist(),
                      final_camera_center_m=transformed_centers[-1].tolist(),
                      reliable_point_count=int(reliable.sum()), observed_floor_point_count=len(floor_points),
                      ground_excluded_gaussians=int(excluded.sum()),
                      per_frame_floor_point_count=near_floor.sum(axis=(1, 2)).tolist(),
                      warning="Estimated camera heads/intrinsics, indoor domain shift, low coverage and temporal drift remain")
        report["alignment"][view] = record
        report["inference"][view] = json.loads((recon / "scene.json").read_text())
        aligned.metadata.update(coordinate_frame="initial_local_vehicle_assumed", alignment=record,
                                known_free_polygon=[])
        save_scene(destination / "aligned_scene.npz", aligned)
        atomic_json(destination / "alignment.json", record)
        selected = json.loads((recon / "selected_indices.json").read_text())
        atomic_npz(destination / "cameras.npz", centers_m=transformed_centers,
                   R_local_from_optical=transformed_rotations, raw_centers=centers,
                   input_paths=input_paths, selected_indices=np.array(selected),
                   estimated_intrinsics=intrinsics)
        atomic_npz(destination / "floor_separation.npz", excluded_ground=excluded,
                   floor_points_m=floor_points,
                   **({"labels": arrays["labels"]} if "labels" in arrays else {}))
        raw_scenes.append(raw)
        aligned_scenes.append(aligned)
        floors.append(floor_points)
        reliable_xy.append(transformed_points[reliable][:, :2])
        excluded_views.append(excluded)
        log(f"{view}: scale={scale:.6f}, floor support={floor_fit['inlier_fraction']:.3f}, "
            f"height drift={record['final_camera_height_change_m']:.3f}m, floor points={len(floor_points)}")
        atomic_json(mapping / "alignment.json", report["alignment"])
    floor_points = np.concatenate(floors)
    # Endpoint bounds are storage/planner bounds only, never observed free-space evidence.
    bound_points = np.concatenate(reliable_xy + [reference[[0, -1], :2]])
    lower = np.floor((bound_points.min(0)-5.)/.25)*.25
    upper = np.ceil((bound_points.max(0)+5.)/.25)*.25
    bounds = np.r_[lower, upper].tolist()
    metadata = dict(units="meters", up_axis="z", bounds=bounds, known_free_polygon=[],
                    coordinate_frame="initial_local_vehicle_assumed", experiment_id=EXPERIMENT,
                    alignment=report["alignment"], fusion="concatenation; no ICP, motion fitting, or deduplication")
    fused = Scene(**{key: np.concatenate([getattr(scene, key) for scene in aligned_scenes])
                     for key in FIELDS}, metadata=metadata).validate()
    raw_fused = Scene(**{key: np.concatenate([getattr(scene, key) for scene in raw_scenes])
                         for key in FIELDS}, metadata=dict(
        units="meters", up_axis="z", coordinate_frame="four_SEPARATE_raw_model_gauges_NOT_metric",
        view_order=list(HEIGHTS), view_offsets=[0, 120000, 240000, 360000, 480000])).validate()
    save_scene(mapping / "raw_scene.npz", raw_fused)
    save_scene(mapping / "fused_scene.npz", fused)
    save_scene(mapping / "aligned_scene.npz", fused)
    save_scene(mapping / "scene.npz", fused)
    excluded = np.concatenate(excluded_views)
    obstacles = Scene(**{key: getattr(fused, key)[~excluded] for key in FIELDS},
                      metadata=dict(metadata)).validate()
    save_scene(mapping / "obstacle_scene.npz", obstacles)
    atomic_npz(mapping / "floor_separation.npz", excluded_ground=excluded,
               floor_points_m=floor_points, view_offsets=np.arange(5)*120000)
    grid = build_grid(obstacles, resolution=.25, uncertainty=.12)
    counts_by_view, known_by_view = [], []
    structure = np.ones((3, 3), bool)
    for observed in floors:
        counts = np.zeros_like(grid.known, dtype=np.int32)
        indices = grid.indices(observed[:, :2])
        np.add.at(counts, (indices[:, 1], indices[:, 0]), 1)
        dense = counts >= 3
        known_view = binary_dilation(binary_closing(dense, structure=structure, iterations=1),
                                     structure=structure, iterations=1)
        counts_by_view.append(counts)
        known_by_view.append(known_view)
    known = np.logical_or.reduce(known_by_view)
    grid.known = known
    save_grid(mapping / "grid.npz", grid, floor_point_counts=np.sum(counts_by_view, axis=0),
              floor_point_counts_by_view=np.stack(counts_by_view),
              known_by_view=np.stack(known_by_view), view_names=np.array(list(HEIGHTS)))
    ablation = build_grid(obstacles, resolution=.25, uncertainty=.12, use_covariance=False)
    ablation.known = known.copy()
    save_grid(mapping / "grid_centers_only.npz", ablation)
    report["mapping"] = dict(bounds=bounds, resolution_m=.25, uncertainty_m=.12, risk_threshold=grid.threshold,
                            known_cells=int(known.sum()), total_cells=int(known.size),
                            known_coverage_fraction=float(known.mean()), known_area_m2=float(known.sum()*.25**2),
                            known_risky_cells=int((known & (grid.risk >= grid.threshold)).sum()),
                            known_cells_by_view=dict(zip(HEIGHTS, [int(mask.sum()) for mask in known_by_view])),
                            known_source="union of confidence-valid visible-floor pointmap masks ONLY",
                            observed_floor_point_count=len(floor_points), raw_gaussians=len(fused.means),
                            obstacle_gaussians=len(obstacles.means), ground_excluded_gaussians=int(excluded.sum()),
                            geometric_ground_exclusion="abs(z)<=0.12m AND own-view nearest observed floor<=0.20m; no semantic labels",
                            warning="Low coverage, temporal drift and uncertain geometry; unknown space remains blocked")
    atomic_json(mapping / "summary.json", report["mapping"])
    log(f"Four-view known coverage={known.mean():.4%}; obstacles={len(obstacles.means)}")

    def evaluate(name, filename, planning, validation, validation_filename, interpretation=None):
        try:
            case = evaluate_case(name, "mapping/" + filename, planning, validation,
                                 start, goal, reference, vehicle, args.timeout)
        except Exception as error:
            case = dict(name=name, map_source="mapping/" + filename, status="evaluation_failed",
                        exception=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        case.update(grid_source="mapping/" + filename, validation_grid_source="mapping/" + validation_filename,
                    diagnostic_only=interpretation is not None)
        if interpretation:
            case["interpretation"] = interpretation
        report["cases"].append(case)
        atomic_json(root / "evaluation.json", report)
        log(f"{name}: {case['status']}; collision checks={case.get('collision_checks')}")

    evaluate("primary4_viewcov", "grid.npz", grid, grid, "grid.npz")
    evaluate("ablation4_view", "grid_centers_only.npz", ablation, grid, "grid.npz")
    if args.initial_ego_bootstrap:
        # Explicit initial occupancy assumption only. Raster pad matches collision()'s
        # conservative cell-overlap test; no goal footprint or future trajectory sweep.
        yy, xx = np.mgrid[:known.shape[0], :known.shape[1]]
        dx = bounds[0]+(xx+.5)*.25-start[0]
        dy = bounds[1]+(yy+.5)*.25-start[1]
        c, s = np.cos(start[2]), np.sin(start[2])
        lx, ly = dx*c+dy*s, -dx*s+dy*c
        pad = vehicle.margin+.25*.5*(abs(c)+abs(s))
        footprint = (lx >= -vehicle.rear-pad) & (lx <= vehicle.front+pad) & (np.abs(ly) <= vehicle.width/2+pad)
        bootstrap = Grid(grid.risk.copy(), known | footprint, bounds, .25, grid.threshold)
        save_grid(mapping / "grid_initial_ego_bootstrap.npz", bootstrap, initial_footprint=footprint)
        evaluate("initialego_bootstrap", "grid_initial_ego_bootstrap.npz", bootstrap, bootstrap,
                 "grid_initial_ego_bootstrap.npz")
        report["cases"][-1]["interpretation"] = "Conditional initial occupied ego footprint assumed known; same model risk, no future or goal knowledge; NOT strict primary"
        report["cases"][-1]["conditional_initial_occupancy_assumption"] = True
        report["cases"][-1]["initial_footprint_added_cells"] = int((footprint & ~known).sum())
    control = Grid(np.zeros_like(grid.risk), np.ones_like(known), bounds, .25, grid.threshold)
    save_grid(mapping / "grid_obstacle_free_control.npz", control)
    evaluate("obstacle_free_fully_known_CONTROL_DIAGNOSTIC", "grid_obstacle_free_control.npz",
             control, control, "grid_obstacle_free_control.npz",
             "Same-endpoint planner/controller exercise ONLY; NOT model mapping or full-pipeline success")
    report["primary_status"] = report["cases"][0]["status"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    for ax, risk, title in zip(axes, [grid.risk, ablation.risk],
                               ["Four-view full covariance (primary)", "Centers only (full covariance validation)"]):
        ax.set_facecolor("#b6bcc5")
        image = ax.imshow(np.ma.masked_where(~known, risk), origin="lower", extent=extent,
                          vmin=0, vmax=1, cmap="magma")
        ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=["cyan", "lime"], marker="x")
        ax.set(title=title+"; gray = unknown", xlabel="Local X (m)", ylabel="Local Y (m)", aspect="equal")
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(mapping / "known_risk.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/parkinge2e_20261010_01"))
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--timeout", type=float, choices=(25., 60.), default=25.)
    parser.add_argument("--initial-ego-bootstrap", action="store_true",
                        help="Add a separate conditional case; strict primary mask is unchanged")
    args = parser.parse_args()
    args.preparation = args.preparation or args.root / "preparation/selected_candidate14"
    # Validate complete ZIPs before changing outputs: predictions may still be written
    # by concurrent inference. No waiting, inference, or GPU access occurs here.
    try:
        sources = {}
        for view in HEIGHTS:
            recon = args.root / ("reconstruction_single_" + view)
            sources[view] = {}
            for filename in ("scene.npz", "predictions.npz", "scene.json", "selected_indices.json"):
                path = recon / filename
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as data:
                        for key in data.files:
                            data[key]
                else:
                    json.loads(path.read_text())
                sources[view][filename] = dict(path=str(path), bytes=path.stat().st_size,
                                               mtime_ns=path.stat().st_mtime_ns)
    except Exception as error:
        print(f"Inputs not ready/valid; no outputs changed. Main can rerun: {error}", file=sys.stderr)
        return 2
    mapping = args.root / "mapping"
    owned = False
    if (mapping / "parameters.json").exists():
        owned = json.loads((mapping / "parameters.json").read_text()).get("experiment_id") == EXPERIMENT
    if mapping.exists() and not owned and (args.root / "mapping_rear_only").exists():
        raise RuntimeError("Refusing to overwrite unowned mapping when rear-only backup already exists")
    # Preserve first rear-only reports and analyzer artifacts, never overwrite backups.
    for filename in ("evaluation.json", "evaluation.log", "analysis.md", "trajectory_summary.json"):
        source = args.root / filename
        backup = source.with_name(source.stem + "_rear_only" + source.suffix)
        if source.exists() and not backup.exists() and not owned:
            shutil.copy2(source, backup)
    if (args.root / "analysis").exists() and not (args.root / "analysis_rear_only").exists() and not owned:
        shutil.copytree(args.root / "analysis", args.root / "analysis_rear_only")
    if mapping.exists() and not owned:
        mapping.rename(args.root / "mapping_rear_only")
    mapping.mkdir(exist_ok=True)
    manifest = json.loads((args.preparation / "manifest.json").read_text())
    parameters = dict(experiment_id=EXPERIMENT, sources=sources, source_code=Path(__file__).read_text(),
                      floor_helper_source=Path(__file__).with_name("evaluate_parkinge2e.py").read_text(),
                      preparation=str(args.preparation), camera_heights_m=HEIGHTS,
                      published_T_vehicle_from_optical={view: manifest["cameras"][view]["T_vehicle_from_optical"]
                                                        for view in HEIGHTS},
                      initial_optical_extrinsics="preparation manifest T_vehicle_from_optical; Z from published heights",
                      roi_normalized_xy=[.25, .75, .58, .90], seed=14, floor_residual_m=.04,
                      minimum_floor_support=.35, maximum_floor_normal_angle_deg=30, ransac_iterations=1200,
                      confidence_threshold=.99, range_m=[.25, 12.], floor_tolerance_m=.12,
                      ground_exclusion_distance_m=.20, resolution_m=.25, uncertainty_m=.12,
                      floor_points_per_cell=3, closing_cells=1, dilation_cells=1,
                      known_union="per-view masks after one-cell closing and one-cell dilation",
                      bounds_padding_m=5., reference_usage="endpoints/bounds and post-hoc comparisons only",
                      initial_ego_bootstrap=args.initial_ego_bootstrap, vehicle=asdict(Vehicle()),
                      planner_timeout_s=args.timeout, planner_max_expansions=35000,
                      risk_threshold=.35, fusion="concatenate all transformed Gaussians; no motion/GT/ICP fitting")
    atomic_json(mapping / "parameters.json", parameters)
    report = dict(schema_version=2, experiment_id=EXPERIMENT, cases=[], parameters_source="mapping/parameters.json",
                  limitations=["Indoor domain shift; estimated raw camera heads/intrinsics, not calibrated projection.",
                               "Initial ROI floor-fit validity does not validate later geometry; temporal drift remains.",
                               "Low coverage; only observed floor is known; unknown space is blocked.",
                               "Initial vehicle frame/local frame and localization/rear-axle equivalence UNVERIFIED.",
                               "Standard vehicle dimensions assumed, not measured; offline kinematic simulation only.",
                               "Ground-like obstacle bases can be excluded; semantic labels never establish free space.",
                               "Recorded localization is not independent ground truth; no trajectory-based fitting.",
                               "Existing analysis is rear-only until analyzer is rerun with per-case grid_source; control is NOT model mapping."],
                  planner_budget=dict(timeout_s=args.timeout, max_expansions=35000))
    begin = time.perf_counter()
    temporary_log = args.root / "evaluation_four_view.tmp.log"
    with temporary_log.open("w") as stream:
        def log(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()
        try:
            run(args, report, log)
        except Exception as error:
            report["fatal_error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            report["primary_status"] = "geometry_or_evaluation_invalid"
            log(traceback.format_exc())
        report["elapsed_s"] = time.perf_counter()-begin
        atomic_json(args.root / "evaluation.json", report)
        log(f"Saved {args.root / 'evaluation.json'}; elapsed={report['elapsed_s']:.2f}s")
    temporary_log.replace(args.root / "evaluation.log")
    return 1 if "fatal_error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
