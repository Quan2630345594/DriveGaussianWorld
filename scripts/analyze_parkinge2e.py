#!/usr/bin/env python3
"""Offline artifact analysis only; no models, alignment, planning or simulation.

Pose arrays are [x_m, y_m, yaw_rad], in the preparation's existing local frame.
evaluation.json: {cases: [{name, status, plan: {path: [{x,y,yaw,...}]},
simulation: {status, trace: [{x,y,yaw,t,...}]}, comparison: ...}]}.
Missing/failed plans and simulations remain missing, never synthesized.
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import base64
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def _poses(path):
    a = np.asarray(path, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite Nx3 [x, y, yaw] array")
    return a


def resample_path(path, samples=256):
    """Equal arclength fractions, preserving direction/order and endpoint yaw.

    Repeated positions use the last yaw at that arclength (gear cusp convention).
    A wholly stationary path interpolates unwrapped yaw by vertex fraction.
    """
    a = _poses(path)
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1))]
    yaw = np.unwrap(a[:, 2])
    if s[-1] == 0:
        result = np.column_stack((np.repeat(a[:1, :2], samples, axis=0),
                                  np.interp(np.linspace(0, 1, samples),
                                            np.linspace(0, 1, len(a)), yaw)))
    else:
        keep = np.r_[np.diff(s) > 0, True]
        q = np.linspace(0, s[-1], samples)
        result = np.column_stack([np.interp(q, s[keep], v[keep])
                                  for v in (a[:, 0], a[:, 1], yaw)])
    result[0], result[-1] = a[0], a[-1]
    return result


def _stats(distances):
    return dict(rmse=float(np.sqrt(np.mean(distances**2))),
                mean=float(np.mean(distances)), p95=float(np.quantile(distances, .95)),
                max=float(np.max(distances)))


def compare_paths(predicted, reference, samples=256):
    """Geometry similarity, NOT time ADE; no fitting, rotation or reversal.

    Position statistics compare equal arclength fractions. Hausdorff and discrete
    Frechet use the resampled XY vertices (sampling-dependent approximations).
    Yaw differences are wrapped after unwrapped-yaw interpolation.
    """
    p, r = resample_path(predicted, samples), resample_path(reference, samples)
    distances = np.linalg.norm(p[:, None, :2] - r[None, :, :2], axis=2)
    previous = np.full(samples, np.inf)
    for i in range(samples):
        current = np.full(samples, np.inf)
        for j in range(samples):
            predecessor = 0. if i == j == 0 else min(
                previous[j], current[j-1] if j else np.inf,
                previous[j-1] if j else np.inf)
            current[j] = max(distances[i, j], predecessor)
        previous = current
    yaw = np.abs((p[:, 2]-r[:, 2]+np.pi) % (2*np.pi)-np.pi)
    return dict(label="geometry_similarity_not_time_ADE", samples=int(samples),
                position_m=_stats(np.linalg.norm(p[:, :2]-r[:, :2], axis=1)),
                yaw_rad=_stats(yaw),
                hausdorff_m=float(max(distances.min(0).max(), distances.min(1).max())),
                discrete_frechet_m=float(previous[-1]),
                start_position_error_m=float(distances[0, 0]),
                endpoint_position_error_m=float(distances[-1, -1]),
                endpoint_yaw_error_rad=float(yaw[-1]))


def tracking_errors(executed, planned):
    """Executed vertices to exact nearest planned XY segment, not time matched.

    Revisited/crossing segments choose the first nearest segment; yaw therefore
    is geometrically ambiguous at crossings. No monotonic matching is imposed.
    """
    e, p = _poses(executed), _poses(planned)
    start, delta = p[:-1, :2], np.diff(p[:, :2], axis=0)
    if not len(delta):
        start, delta = p[:1, :2], np.zeros((1, 2))
    denominator = np.sum(delta**2, axis=1)
    yaw = np.unwrap(p[:, 2])
    position_errors, yaw_errors = [], []
    # One vertex at a time bounds memory for long simulation traces.
    for pose in e:
        fraction = np.divide(np.sum((pose[:2]-start)*delta, axis=1), denominator,
                             out=np.zeros(len(delta)), where=denominator > 0).clip(0, 1)
        distance = np.linalg.norm(start+fraction[:, None]*delta-pose[:2], axis=1)
        index = int(np.argmin(distance))
        heading = yaw[index] + fraction[index]*(yaw[min(index+1, len(yaw)-1)]-yaw[index])
        position_errors.append(distance[index])
        yaw_errors.append(abs((pose[2]-heading+np.pi) % (2*np.pi)-np.pi))
    return dict(label="nearest_planned_polyline_not_time_tracking", count=len(e),
                position_m=_stats(np.asarray(position_errors)),
                yaw_rad=_stats(np.asarray(yaw_errors)),
                endpoint_position_error_m=float(np.linalg.norm(e[-1, :2]-p[-1, :2])),
                endpoint_yaw_error_rad=float(abs((e[-1, 2]-p[-1, 2]+np.pi) % (2*np.pi)-np.pi)))


def _case_path(value, key):
    rows = (value or {}).get(key)
    if rows is None or len(rows) == 0:
        return None
    return _poses([[row["x"], row["y"], row["yaw"]] for row in rows])


def _background(ax, grid):
    ax.set(xlabel="Local x (m)", ylabel="Local y (m)", aspect="equal")
    if grid is not None:
        bounds = grid["bounds"]
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        ax.set_facecolor("#b6bcc5")
        image = ax.imshow(np.ma.masked_where(~grid["known"].astype(bool), grid["risk"]),
                          origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
        ax.figure.colorbar(image, ax=ax, label="Covariance-aware risk; gray = unknown")
        if np.any(grid["risk"] >= float(grid["threshold"])):
            h, w = grid["risk"].shape
            x = bounds[0] + (np.arange(w)+.5)*float(grid["resolution"])
            y = bounds[1] + (np.arange(h)+.5)*float(grid["resolution"])
            ax.contour(x, y, np.ma.masked_where(~grid["known"].astype(bool), grid["risk"]),
                       levels=[float(grid["threshold"])], colors="white", linewidths=.6)
    ax.grid(alpha=.15)


def analyze(experiment, preparation, fps=10., samples=256, case_name=None, output_name='analysis'):
    """Write root summary/report and analysis/{figures, gallery, local videos}."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv2.setNumThreads(1)
    root, prep = Path(experiment), Path(preparation)
    manifest = json.loads((prep / "manifest.json").read_text())
    with np.load(prep / "episode/reference_selected.npz", allow_pickle=False) as data:
        reference = _poses(np.column_stack((data["position_local_m"][:, :2], data["yaw_local_rad"])))
        duration = float((data["time_ns"][-1]-data["time_ns"][0])/1e9)
    evaluation = root / "evaluation.json"
    cases = json.loads(evaluation.read_text()).get("cases", []) if evaluation.exists() else []
    grid_path = root / "mapping/grid.npz"
    grid = None
    if grid_path.exists():
        with np.load(grid_path, allow_pickle=False) as data:
            grid = {k: data[k] for k in ("risk", "known", "bounds", "resolution", "threshold")}
    out = root / output_name
    out.mkdir(parents=True, exist_ok=True)
    disclaimer = ("Reference: recorded estimated localization, NOT independent ground truth. "
                  "Offline inference; no real-time latency claim. No trajectory alignment/fitting.")
    summary = dict(schema_version=1, preparation=str(prep.resolve()), reference_source=manifest["source"],
                   disclaimer=disclaimer, evaluation_present=evaluation.exists(),
                   reference_duration_s=duration, animation="progress-normalized, NOT time matched",
                   cases=[], primary_case=None, artifacts=[])
    tracks = [("Reference (estimated)", reference, "#168c8c")]
    parsed = []
    for case in cases:
        record = dict(name=case["name"], status=case.get("status", "unspecified"),
                      map_source=case.get('map_source'),
                      simulation_status=(case.get("simulation") or {}).get("status"),
                      comparison=case.get("comparison"))
        paths = []
        for key, rows in (("plan", "path"), ("simulation", "trace")):
            try:
                paths.append(_case_path(case.get(key), rows))
            except (ValueError, KeyError, TypeError) as error:
                record[key+"_error"] = str(error)
                paths.append(None)
        planned, executed = paths
        record.update(plan_available=planned is not None, trace_available=executed is not None)
        if planned is not None:
            record["planned_vs_reference"] = compare_paths(planned, reference, samples)
        if executed is not None:
            record["executed_vs_reference"] = compare_paths(executed, reference, samples)
        if planned is not None and executed is not None:
            record["executed_vs_planned"] = tracking_errors(executed, planned)
        summary["cases"].append(record)
        parsed.append((case, planned, executed))
    # Prefer the covariance case even if it failed: failure is part of the result.
    primary = (next((c for c in parsed if c[0]['name'] == case_name), None) if case_name
               else next((c for c in parsed if "cov" in c[0]["name"].lower()), parsed[0] if parsed else None))
    if case_name and primary is None:
        raise ValueError(f'Case not found: {case_name}')
    if primary:
        summary["primary_case"] = primary[0]["name"]
        case_grid = primary[0].get('grid_source')
        if case_grid:
            with np.load(root / case_grid, allow_pickle=False) as data:
                grid = {k: data[k] for k in ("risk", "known", "bounds", "resolution", "threshold")}
        for label, path, color in (("Planned", primary[1], "#de891d"), ("Simulated", primary[2], "#cd4777")):
            if path is not None:
                tracks.append((label, path, color))
    status = "; ".join(f"{r['name']}: {r['status']}, simulation={r['simulation_status']} "
                       f"(plan={r['plan_available']}, trace={r['trace_available']})" for r in summary["cases"])
    status = status or "Reference-only phase: evaluation absent or no cases"

    def save_figure(fig, name):
        for extension in ("png", "pdf"):
            file = out / f"{name}.{extension}"
            fig.savefig(file, dpi=160, bbox_inches="tight")
            summary["artifacts"].append(str(file.relative_to(root)))

    fig, ax = plt.subplots(figsize=(9, 7))
    _background(ax, grid)
    for label, path, color in tracks:
        ax.plot(path[:, 0], path[:, 1], label=label, color=color, linewidth=2)
        indices = np.linspace(0, len(path)-1, min(8, len(path))).astype(int)
        ax.quiver(path[indices, 0], path[indices, 1], np.cos(path[indices, 2]), np.sin(path[indices, 2]),
                  color=color, angles="xy", scale_units="xy", scale=2, width=.004)
    ax.legend()
    ax.set_title("Reference / planned / simulated geometry" + ("; risk map unavailable" if grid is None else ""))
    fig.text(.02, .01, "Estimated reference, not independent GT; no alignment; offline inference", fontsize=8)
    save_figure(fig, "tracks_risk")
    plt.close(fig)
    # Separate case panels expose baseline failures and absent traces without fake lines.
    fig, axes = plt.subplots(1, max(1, len(parsed)), figsize=(6*max(1, len(parsed)), 5), squeeze=False)
    for ax, entry in zip(axes[0], parsed or [(None, None, None)]):
        ax.plot(reference[:, 0], reference[:, 1], label="Estimated reference", color=tracks[0][2])
        case, planned, executed = entry
        for label, path, color in (("Planned", planned, "#de891d"), ("Simulated", executed, "#cd4777")):
            if path is not None:
                ax.plot(path[:, 0], path[:, 1], label=label, color=color)
        ax.set_title(f"{case['name']}: {case.get('status')}" if case else "Reference only")
        ax.text(.02, .02, f"Plan {'available' if planned is not None else 'absent'}; trace {'available' if executed is not None else 'absent'}",
                transform=ax.transAxes, fontsize=9)
        ax.set(xlabel="Local x (m)", ylabel="Local y (m)", aspect="equal")
        ax.legend()
    save_figure(fig, "case_comparison")
    plt.close(fig)

    # Build eight synchronized input quads from manifest paths, not guessed names.
    quads = []
    frames = manifest["frames"]
    for number, index in enumerate(np.linspace(0, len(frames)-1, 8).round().astype(int)):
        views = [cv2.imread(str(prep / frames[index]["cameras"][camera]["rectified"]))
                 for camera in ("front", "back", "left", "right")]
        if any(v is None for v in views):
            raise RuntimeError(f"Missing manifest image at frame {index}")
        quad = np.vstack((np.hstack(views[:2]), np.hstack(views[2:])))
        cv2.putText(quad, f"Input frame {index}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        if not cv2.imwrite(str(out / f"input_quad_{number:02d}.jpg"), quad):
            raise RuntimeError("Could not write input quad")
        quads.append(cv2.resize(quad, (480, 320)))
    contact = np.vstack([np.hstack(quads[i:i+4]) for i in (0, 4)])
    if not cv2.imwrite(str(out / "input_contactsheet.jpg"), contact):
        raise RuntimeError("Could not write contact sheet")
    summary["artifacts"] += [f"{output_name}/input_quad_{i:02d}.jpg" for i in range(8)] + [f"{output_name}/input_contactsheet.jpg"]
    source_video = prep / "episode/quad.mp4"
    shutil.copyfile(source_video, out / "input_quad.mp4")
    capture = cv2.VideoCapture(str(source_video))
    source_fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or source_fps <= 0 or count < 1:
        capture.release()
        raise RuntimeError("Cannot read prepared quad video cadence/frame count")
    summary["video"] = dict(input_fps=source_fps, input_frames=count, output_fps=fps,
                             duration_s=count/source_fps, normalization="uniform progress fractions; not time matched")
    frame_count = max(2, round(count/source_fps*fps))
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    _background(ax, grid)
    progress = []
    for label, path, color in tracks:
        path = resample_path(path, frame_count)
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=.25)
        line, = ax.plot([], [], color=color, label=label, linewidth=2)
        arrow, = ax.plot([], [], color=color, linewidth=3)
        progress.append((path, line, arrow))
    ax.legend(loc="best", fontsize=8)
    title = ax.set_title("Progress normalized; NOT time matched", fontsize=10)
    fig.tight_layout()
    intermediate = out / "progress_comparison.encoding.mp4"
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    try:
        if not writer.isOpened():
            raise RuntimeError("MP4 video writer unavailable")
        last_index, quad = -1, None
        for i in range(frame_count):
            fraction = i/(frame_count-1)
            index = round(fraction*(count-1))
            if index != last_index:
                if index != last_index+1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, quad = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode quad frame {index}")
                last_index = index
            for path, line, arrow in progress:
                line.set_data(path[:i+1, 0], path[:i+1, 1])
                pose = path[i]
                arrow.set_data([pose[0], pose[0]+.6*np.cos(pose[2])], [pose[1], pose[1]+.6*np.sin(pose[2])])
            title.set_text(f"Progress {fraction:.0%}; NOT time matched")
            fig.canvas.draw()
            bev = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
            frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
            height, width = quad.shape[:2]
            scale = min(640/width, 640/height)
            width, height = round(width*scale), round(height*scale)
            left, top = (640-width)//2, 40+(640-height)//2
            frame[top:top+height, left:left+width] = cv2.resize(quad, (width, height))
            frame[40:680, 640:] = bev
            cv2.putText(frame, "Recorded RGB quad | Geometry progress comparison", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 1)
            cv2.putText(frame, "Estimated reference, NOT independent GT | Offline inference, NOT real-time", (15, 700), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1)
            selected_status = f"{primary[0]['name']}: {primary[0].get('status')}" if primary else status
            if 'CONTROL' in selected_status.upper():
                cv2.putText(frame, "OBSTACLE-FREE CONTROL: NOT MODEL-MAP VALIDATION", (15, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 200), 2)
            cv2.putText(frame, selected_status[:145], (15, 718), cv2.FONT_HERSHEY_SIMPLEX, .4, (25, 25, 25), 1)
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
        plt.close(fig)
    # OpenCV's mp4v output is not reliably browser-playable; deliver H.264.
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-threads", "1",
                    "-i", str(intermediate), "-c:v", "libx264", "-threads", "1",
                    "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out / "progress_comparison.mp4")], check=True)
    intermediate.unlink()
    summary["artifacts"] += [f"{output_name}/input_quad.mp4", f"{output_name}/progress_comparison.mp4", f"{output_name}/index.html"]
    summary_path = root / 'trajectory_summary.json' if output_name == 'analysis' else out / 'trajectory_summary.json'
    report_path = root / 'analysis.md' if output_name == 'analysis' else out / 'report.md'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    gallery_link = f'{output_name}/index.html' if output_name == 'analysis' else 'index.html'
    report = ("# ParkingE2E Artifact Analysis\n\n"+disclaimer+"\n\n"+status+"\n\n"
              "Position/yaw RMSE, mean and p95 compare equal arclength fractions, not time ADE. "
              "Hausdorff/discrete Frechet use resampled vertices. Endpoint errors retain path orientation. "
              "Tracking uses exact nearest planned segments; crossing/cusp yaw can be ambiguous. "
              "Videos normalize input preview fraction and each trajectory's arclength independently; "
              "they are not synchronized physical execution. Prepared video FPS is read dynamically. "
              "Risk is not a ground-truth occupancy probability; gray cells are unknown.\n\n"
               f"See [gallery]({gallery_link}) and [metrics](trajectory_summary.json).\n\n```json\n"+
              json.dumps(summary["cases"], indent=2, allow_nan=False)+"\n```\n")
    report_path.write_text(report)
    images = "".join(f'<figure><img src="data:image/{"jpeg" if p.suffix == ".jpg" else "png"};base64,{base64.b64encode(p.read_bytes()).decode()}"><figcaption>{html.escape(p.name)}</figcaption></figure>'
                     for p in [out / "tracks_risk.png", out / "case_comparison.png", out / "input_contactsheet.jpg"])
    (out / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>ParkingE2E analysis</title><style>body{font:16px sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem;background:#f4f4f0;color:#222}img,video{width:100%;height:auto}figure{margin:2rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>ParkingE2E artifact analysis</h1><p>'+html.escape(disclaimer)+'</p><p>'+html.escape(status)+
        '</p><p>Progress-normalized animation, NOT time matched. Local videos; embedded gallery; no CDN dependencies.</p>'
        '<h2>Input four views</h2><video controls preload="metadata" src="input_quad.mp4"></video>'
        '<h2>Input / trajectory progress</h2><video controls preload="metadata" src="progress_comparison.mp4"></video>'+images+
        '<h2>Metrics</h2><pre>'+html.escape(json.dumps(summary, indent=2, allow_nan=False))+'</pre></html>')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path, help="Selected preparation containing manifest.json and episode/")
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument('--case', help='Select a case explicitly, e.g. the separate control diagnostic')
    parser.add_argument('--output-name', default='analysis', help='Artifact directory name within the experiment')
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0 or args.samples < 2:
        parser.error("fps must be finite/positive and samples >= 2")
    if Path(args.output_name).name != args.output_name:
        parser.error('--output-name must be a directory name, not a path')
    analyze(args.experiment, args.preparation, args.fps, args.samples, args.case, args.output_name)


if __name__ == "__main__":
    main()
