import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gaussian_park_world.scenes import (
    assert_same_gauge, make_fixture, normalize_gaussians, prepare_dggt,
    sha256, validate_scene, write_json,
)


def load_arrays(path):
    with np.load(path / "scene.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_fixture_contract_determinism_and_projection(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    manifest = make_fixture(a, "parking")
    assert make_fixture(b, "parking") == manifest
    assert json.loads((a / "manifest.json").read_text()) == manifest
    arrays = load_arrays(a)
    validate_scene(manifest, arrays)
    assert manifest["source_classification"] == "SYNTHETIC_PLANNING_FIXTURE"
    assert manifest["planning_status"] == "SYNTHETIC_FIXTURE_ONLY"
    assert len(arrays["means"]) <= 32
    assert manifest["array_sha256"] == sha256(a / "scene.npz")
    for key, value in load_arrays(b).items():
        np.testing.assert_array_equal(value, arrays[key])
    camera = manifest["cameras"][0]
    xyz = (np.linalg.inv(camera["T_world_from_optical"]) @
           np.column_stack((arrays["means"], np.ones(24))).T).T[:, :3]
    uvw = xyz @ np.array(camera["K"]).T
    uv = uvw[:, :2] / uvw[:, 2:]
    assert np.all(xyz[:, 2] > 0)
    assert np.all((uv >= 0) & (uv < [64, 48]))
    other = make_fixture(tmp_path / "other", "parking", variant=1)
    assert not np.array_equal(load_arrays(tmp_path / "other")["means"], arrays["means"])
    assert_same_gauge(manifest, manifest)
    with pytest.raises(ValueError, match="different gauges"):
        assert_same_gauge(manifest, other)
    before = sha256(a / "scene.npz")
    with pytest.raises(FileExistsError):
        make_fixture(a, "parking")
    assert sha256(a / "scene.npz") == before


def test_explicit_activation_rotation_and_floor():
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()[None]
    a = normalize_gaussians([[0, 0, 0]], np.log([[1, 2, 3]]), q,
                            [[.2, .3, .4]], [0], scale_encoding="log",
                            quaternion_order="xyzw", opacity_encoding="logit")
    np.testing.assert_allclose(a["covariances"][0], np.diag([4, 1, 9]), atol=1e-14)
    np.testing.assert_allclose(a["opacities"], [.5])
    np.testing.assert_allclose(a["quats"][0], [2**-.5, 0, 0, 2**-.5])
    a = normalize_gaussians([[0, 0, 0]], [[1, 1e-14, 1]], [[1, 2, 3, 4]], [[0, 0, 0]], [1])
    assert a["scales"][0, 1] == 1e-6
    assert np.linalg.eigvalsh(a["covariances"]).min() > 0
    column = normalize_gaussians([[0, 0, 0]], [[1, 1, 1]], [[1, 0, 0, 0]],
                                 [[0, 0, 0]], [[0]], opacity_encoding="logit")
    assert column["opacities"].shape == (1,)
    np.testing.assert_array_equal(column["opacities"], [.5])


@pytest.mark.parametrize("field,value", [("scales", [[0, 1, 1]]),
    ("quats", [[0, 0, 0, 0]]), ("colors", [[2, 0, 0]]),
    ("opacities", [-1]), ("means", [[np.nan, 0, 0]])])
def test_bad_gaussians(field, value):
    args = dict(means=[[0, 0, 0]], scales=[[1, 1, 1]], quats=[[1, 0, 0, 0]],
                colors=[[0, 0, 0]], opacities=[1])
    args[field] = value
    with pytest.raises(ValueError):
        normalize_gaussians(**args)


@pytest.mark.parametrize("damage", ["covariance", "quaternion", "schema", "camera", "gauge", "encoding", "provenance"])
def test_validator_rejects_corruption(tmp_path, damage):
    manifest = make_fixture(tmp_path / "scene", "fixture")
    arrays = load_arrays(tmp_path / "scene")
    if damage == "covariance":
        arrays["covariances"][0] *= -1
    elif damage == "quaternion":
        arrays["quats"][0] *= 2
    elif damage == "schema":
        manifest["arrays"]["means"]["shape"] = [1, 3]
    elif damage == "camera":
        manifest["cameras"][0]["T_world_from_optical"][0][0] = 2
    elif damage == "gauge":
        manifest["world_gauge"]["metric_scale"] = "unknown"
    elif damage == "provenance":
        manifest["provenance"]["source_sha256"] = {"source.npz": "invalid"}
    else:
        manifest["parameterization"]["quaternion_order"] = "xyzw"
    with pytest.raises(ValueError):
        validate_scene(manifest, arrays)


def test_json_exclusive_and_strict(tmp_path):
    path = tmp_path / "test.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {})
    assert json.loads(path.read_text()) == {"a": 1}
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"a": float("nan")})
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("field,value", [
    ("skew", .1), ("lower_skew", .1), ("cx", -1), ("cx", 64),
    ("cy", -1), ("cy", 48), ("timestamp", None), ("timestamp", True),
    ("timestamp", "0"), ("timestamp", float("nan")), ("timestamp", float("inf")),
])
def test_camera_intrinsics_and_timestamp_rejected(tmp_path, field, value):
    manifest = make_fixture(tmp_path / "scene", "fixture")
    camera = manifest["cameras"][0]
    if field == "timestamp":
        camera[field] = value
    else:
        row, col = {"skew": (0, 1), "lower_skew": (1, 0), "cx": (0, 2), "cy": (1, 2)}[field]
        camera["K"][row][col] = value
    with pytest.raises(ValueError):
        validate_scene(manifest, load_arrays(tmp_path / "scene"))


@pytest.mark.parametrize("timestamp", ["unknown", 0, -1.5])
def test_camera_timestamp_valid(tmp_path, timestamp):
    manifest = make_fixture(tmp_path / "scene", "fixture")
    manifest["cameras"][0]["timestamp"] = timestamp
    validate_scene(manifest, load_arrays(tmp_path / "scene"))


def test_dggt_camera_conversion_provenance_and_planning_block(tmp_path):
    source = tmp_path / "source"
    make_fixture(source, "source")
    write_json(source / "scene.json", {"image_size": [64, 48], "coordinate_frame": "raw_model_gauge"})
    extrinsics = np.column_stack((Rotation.from_euler("z", .3).as_matrix(), [1, 2, 3]))[None]
    k = np.array([[[48, 0, 32], [0, 48, 24], [0, 0, 1]]])
    np.savez(source / "predictions.npz", extrinsics=extrinsics, intrinsics=k,
             image_intrinsics=k * 2, depth=np.array([object()], dtype=object))
    write_json(source / "command.json", ["/nonexistent/historical/checkpoint"])
    result = prepare_dggt(source, tmp_path / "canonical")
    validate_scene(result, load_arrays(tmp_path / "canonical"))
    assert result["planning_status"] == "BLOCKED_UNKNOWN_METRIC_GAUGE"
    assert result["world_gauge"]["units"] == "unknown"
    assert result["provenance"]["source_sha256"]["scene.npz"] == sha256(source / "scene.npz")
    camera = result["cameras"][0]
    assert camera["timestamp"] == "unknown"
    assert camera["timestamp_units"] == "seconds"
    assert camera["source_frame"] == "unknown"
    assert camera["source_frame_coordinate"] == "unknown"
    np.testing.assert_array_equal(camera["K"], k[0])
    pose = np.eye(4)
    pose[:3] = extrinsics[0]
    np.testing.assert_allclose(camera["T_world_from_optical"], np.linalg.inv(pose))
    with pytest.raises(FileExistsError):
        prepare_dggt(source, tmp_path / "canonical")
    bad = copy.deepcopy(result)
    bad["planning_status"] = "SYNTHETIC_FIXTURE_ONLY"
    with pytest.raises(ValueError, match="unknown metric gauge"):
        validate_scene(bad, load_arrays(tmp_path / "canonical"))


def test_dggt_conversion_report_and_projected_centers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    n = 2005
    means = np.tile([0., 0., 2.], (n, 1))
    means[0] = [0, 0, -2]
    means[-1] = [100, 0, 2]
    raw = dict(means=means, scales=np.tile([1., 1e-12, 2.], (n, 1)),
               quats=np.tile([2., 0, 0, 0], (n, 1)), colors=np.full((n, 3), .25),
               opacities=np.full((n, 1), .7), source_frames=np.arange(n),
               temporal_widths=np.ones(n))
    np.savez(source / "scene.npz", **raw)
    write_json(source / "scene.json", {"image_size": [518, 294], "input_frames": 16})
    k = np.array([[400., 0, 259], [0, 400, 147], [0, 0, 1]])
    extrinsics = np.broadcast_to(np.eye(4)[:3], (16, 3, 4)).copy()
    extrinsics[1:, 0, 3] = 100  # Preview must use the first camera, not another view.
    np.savez(source / "predictions.npz", extrinsics=extrinsics,
             intrinsics=np.broadcast_to(k, (16, 3, 3)))
    selected = list(range(0, 224, 14))
    write_json(source / "selected_indices.json", selected)
    before = {p.name: sha256(p) for p in source.iterdir()}
    output = tmp_path / "canonical"
    manifest = prepare_dggt(source, output)
    arrays = load_arrays(output)
    validate_scene(manifest, arrays)
    assert arrays["opacities"].shape == (n,)
    assert before == {p.name: sha256(p) for p in source.iterdir()}
    report = json.loads((output / "conversion_report.json").read_text())
    assert report["source_sha256"] == before == manifest["provenance"]["source_sha256"]
    assert report["canonical_array_sha256"] == sha256(output / "scene.npz")
    assert report["scale_pre"] == {"min": 1e-12, "max": 2.}
    assert report["scale_post"] == {"min": 2e-6, "max": 2.}
    assert report["floored_splat_count"] == n
    assert report["max_absolute_scale_correction"] == 2e-6 - 1e-12
    assert report["quaternion_norm_input_range"] == [2., 2.]
    assert report["covariance_eigenvalue_min"] == np.linalg.eigvalsh(arrays["covariances"]).min()
    assert "changes geometry" in report["geometry_warning"]
    assert "not metrically meaningful" in report["geometry_warning"]
    for stage in ("input", "output"):
        assert report["ranges"][stage]["opacities"] == {"min": .7, "max": .7}
        assert report["ranges"][stage]["colors"] == {"min": .25, "max": .25}
    for counts in report["finite_counts"].values():
        assert all(item["finite"] == item["total"] for item in counts.values())
    assert report["finite_counts"]["input"]["means"]["finite"] == n * 3
    assert report["camera_count"] == 16
    assert set(report["optional_fields"]) == {"source_frames", "temporal_widths"}
    assert report["optional_fields"]["source_frames"]["shape"] == [n]
    assert report["revision_evidence"]["repository_dggt_pin"] == "a3276d2bbe4cbb03bcc117830b1836110a27adeb"
    assert report["revision_evidence"]["export_producer_revision"] == "unknown"
    assert manifest["provenance"]["reconstruction_version"] == "unknown"
    assert manifest["scene_time"]["timestamp"] == "unknown"
    assert report["time"]["timestamp_seconds"] == "unknown"
    assert [c["source_frame"] for c in manifest["cameras"]] == selected
    assert all(c["timestamp"] == "unknown" for c in manifest["cameras"])
    assert all(c["source_frame_coordinate"] == "selected_indices.json:source_frame_index"
               for c in manifest["cameras"])
    preview = report["diagnostic_preview"]
    assert preview["sampled_count"] == 2000
    assert preview["visible_count"] == 1998
    assert preview["image_size_wh"] == [518, 294]
    svg = ET.parse(output / preview["file"]).getroot()
    assert svg.attrib["width"] == "518" and svg.attrib["height"] == "294"
    circles = svg.findall(".//{http://www.w3.org/2000/svg}circle")
    assert len(circles) == preview["visible_count"]
    assert all(float(c.attrib["cx"]) == 259 and float(c.attrib["cy"]) == 147 for c in circles)
    text = ''.join(svg.itertext())
    assert "PROJECTED_CENTERS_NOT_RENDERED_RGB" in text
    assert "Not an observation; not occupancy" in text
    with pytest.raises(FileExistsError):
        prepare_dggt(source, output)


@pytest.mark.parametrize("indices", [[0] * 16, list(range(15)), list(range(16))[::-1],
                                     list(range(15)) + [15.5], list(range(15)) + [True],
                                     [-1] + list(range(15)), {"indices": list(range(16))}])
def test_dggt_selected_indices_rejected(tmp_path, indices):
    source = tmp_path / "source"
    make_fixture(source, "source")
    write_json(source / "scene.json", {"image_size": [64, 48]})
    np.savez(source / "predictions.npz", extrinsics=np.broadcast_to(np.eye(4)[:3], (16, 3, 4)),
             intrinsics=np.broadcast_to([[48, 0, 32], [0, 48, 24], [0, 0, 1]], (16, 3, 3)))
    write_json(source / "selected_indices.json", indices)
    with pytest.raises(ValueError, match="selected_indices"):
        prepare_dggt(source, tmp_path / "canonical")
    assert not (tmp_path / "canonical").exists()


def test_actual_authorized_sample(tmp_path):
    source = Path(__file__).resolve().parents[1] / (
        "output/experiment_20261012_endtoend_validation/reconstruction/dggt/reconstruction_single_left")
    if not source.is_dir():
        pytest.skip("authorized local DGGT sample unavailable")
    manifest = prepare_dggt(source, tmp_path / "actual")
    arrays = load_arrays(tmp_path / "actual")
    validate_scene(manifest, arrays)
    assert manifest["gaussian_count"] == 120000
    assert len(manifest["cameras"]) == 16
    assert manifest["cameras"][0]["image_size_wh"] == [518, 294]
    assert manifest["planning_status"] == "BLOCKED_UNKNOWN_METRIC_GAUGE"
    assert np.linalg.eigvalsh(arrays["covariances"]).min() > 0
    assert arrays["opacities"].shape == (120000,)
    assert [camera["source_frame"] for camera in manifest["cameras"]] == json.loads(
        (source / "selected_indices.json").read_text())
    assert all(camera["timestamp"] == "unknown" for camera in manifest["cameras"])
    report = json.loads((tmp_path / "actual" / "conversion_report.json").read_text())
    assert report["camera_count"] == 16
    assert report["floored_splat_count"] > 0
    assert report["max_absolute_scale_correction"] > 0
    assert report["diagnostic_preview"]["image_size_wh"] == [518, 294]
    assert 0 <= report["diagnostic_preview"]["visible_count"] <= report["diagnostic_preview"]["sampled_count"] <= 2000
    assert report["source_sha256"]["scene.npz"] == sha256(source / "scene.npz")
