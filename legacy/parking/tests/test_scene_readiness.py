"""Synthetic metadata tests are not real-scene geometry certification."""

import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from gaussian_park_world.scenes import readiness as r
from gaussian_park_world.scenes.contract import make_fixture, sha256, write_json


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)
    path = tmp_path / "scene"
    make_fixture(path, "synthetic_metadata_test")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["cameras"][0].update(source_frame=0, source_frame_coordinate="test:frame_index")
    # This is a synthetic test manifest, not a changed real artifact.
    (path / "manifest.json").unlink()
    write_json(path / "manifest.json", manifest)
    return path


def metadata(scene):
    data = r.template(scene)
    gauge = data["binding"]["gauge"]["id"]
    geometry = dict(gauge_id=gauge, units="meters", frame="metric_x_forward_y_left_z_up",
                    independent_source="synthetic survey fixture", confidence=.99,
                    unknown_space_policy="occupied")
    values = dict(
        scale_to_meters=1.,
        world_up_ground=dict(world_up=[0, 0, 1], ground_plane=[0, 0, 1, 0]),
        T_metric_from_gauge=dict(scale=1., rigid=np.eye(4).tolist(), from_gauge=gauge,
                                 to_frame="metric_x_forward_y_left_z_up"),
        source_frame_alignment=dict(gauge_id=gauge, camera_frames=[dict(
            camera_id="forward", source_frame=0, source_frame_coordinate="test:frame_index")]),
        drivable_geometry=dict(geometry, polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]),
        obstacles=dict(geometry, rectangles=[[2, 2, 3, 3]]),
        vehicle_footprint=dict(units="meters", frame="vehicle_x_forward_y_left_z_up",
                               polygon=[[-1, -1], [3, -1], [3, 1], [-1, 1]]),
        T_vehicle_from_optical=dict(units="meters", camera_ids=["forward"],
                                    from_frame="optical_x_right_y_down_z_forward",
                                    to_frame="vehicle_x_forward_y_left_z_up", rigid=[
                                        [0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]]),
        rights=dict(allowed_use=["local_planning"], rights_holder="synthetic test author"))
    for name, value in values.items():
        proof = dict(binding=data["binding"], requirement=name, value=value,
                     provenance="synthetic test only", verification_method="test fixture construction",
                     reviewer="synthetic test reviewer, not authenticated")
        path = scene / f"{name}.json"
        write_json(path, proof)
        data["requirements"][name] = dict(status="VERIFIED", value=value,
                                          evidence=[dict(path=path.name, sha256=sha256(path))])
    return data


def save(scene, data):
    path = scene / "metadata.json"
    write_json(path, data)
    return path


def test_no_metadata_and_template(scene):
    report = r.check(scene)
    assert report.status == "BLOCKED_GEOMETRY"
    assert report.real_expert_trajectories == 0
    assert all(f.status == r.EvidenceStatus.UNKNOWN and not f.evidence_validated for f in report.findings)
    template = r.template(scene)
    assert template["binding"]["manifest_sha256"] == sha256(scene / "manifest.json")
    assert all(v["value"] is None for v in template["requirements"].values())
    checked = r.check(scene, save(scene, template))
    assert checked == report
    json.dumps(report.to_dict(), allow_nan=False)


def test_complete_local_evidence_is_not_authenticated_or_planner_ready(scene):
    report = r.check(scene, save(scene, metadata(scene)))
    assert all(f.evidence_validated for f in report.findings)
    assert report.status == "BLOCKED_GEOMETRY" and report.real_expert_trajectories == 0
    assert "UNSUPPORTED_REAL_IMPORTER_PENDING" in report.limitations[0]
    assert "not authenticate" in report.limitations[1]


@pytest.mark.parametrize("missing", ["scale_to_meters", "drivable_geometry"])
def test_missing_proof_blocks(scene, missing):
    data = metadata(scene)
    del data["requirements"][missing]
    report = r.check(scene, save(scene, data))
    finding = next(f for f in report.findings if f.requirement == missing)
    assert finding.status == r.EvidenceStatus.UNKNOWN and not finding.evidence_validated
    assert report.status == "BLOCKED_GEOMETRY"


def test_user_assertion_never_verified(scene):
    data = metadata(scene)
    data["requirements"]["scale_to_meters"]["status"] = "USER_ASSERTED"
    report = r.check(scene, save(scene, data))
    assert report.findings[0].status == r.EvidenceStatus.USER_ASSERTED
    assert not report.findings[0].evidence_validated
    assert report.status == "BLOCKED_GEOMETRY"


@pytest.mark.parametrize("damage", ["gauge", "camera", "rights", "units", "confidence", "policy",
                                   "polygon", "obstacles", "scale", "sim3_scale", "up"])
def test_invalid_values_rejected_before_evidence(scene, damage):
    data = metadata(scene)
    values = {k: v["value"] for k, v in data["requirements"].items()}
    if damage == "gauge":
        values["drivable_geometry"]["gauge_id"] = "different-view"
    elif damage == "camera":
        values["T_vehicle_from_optical"]["rigid"][0][2] = -1  # reflection
    elif damage == "rights":
        values["rights"]["allowed_use"] = ["visualization_only"]
    elif damage == "units":
        values["obstacles"]["units"] = "gauge_units"
    elif damage == "confidence":
        values["drivable_geometry"]["confidence"] = "unknown"
    elif damage == "policy":
        values["drivable_geometry"]["unknown_space_policy"] = "free"
    elif damage == "polygon":
        values["drivable_geometry"]["polygon"] = [[0, 0], [1, 1], [0, 1], [1, 0]]
    elif damage == "obstacles":
        values["obstacles"]["rectangles"] = []
    elif damage == "scale":
        data["requirements"]["scale_to_meters"]["value"] = True
    elif damage == "sim3_scale":
        values["T_metric_from_gauge"]["rigid"][0][0] = 2
    else:
        values["world_up_ground"]["world_up"] = [0, 0, 2]
    with pytest.raises(ValueError):
        r.check(scene, save(scene, data))


@pytest.mark.parametrize("damage", ["corrupt", "unbound", "reviewer", "provenance", "method", "no_evidence", "no_value"])
def test_verified_requires_integrity_binding_and_provenance(scene, damage):
    data = metadata(scene)
    entry = data["requirements"]["scale_to_meters"]
    path = scene / entry["evidence"][0]["path"]
    if damage == "no_evidence":
        entry["evidence"] = []
    elif damage == "no_value":
        entry["value"] = None
    else:
        proof = json.loads(path.read_text())
        if damage in ("corrupt", "unbound"):
            proof["binding"]["scene_id"] = "wrong-scene"
        else:
            del proof[{"reviewer": "reviewer", "provenance": "provenance", "method": "verification_method"}[damage]]
        path.unlink()
        write_json(path, proof)
        if damage != "corrupt":
            entry["evidence"][0]["sha256"] = sha256(path)
    with pytest.raises(ValueError):
        r.check(scene, save(scene, data))


@pytest.mark.parametrize("escape", ["../other.json", "/etc/passwd", "https://example.org/proof", "link.json"])
def test_evidence_paths_task_scoped(scene, escape):
    data = metadata(scene)
    if escape == "link.json":
        outside = scene.parent / "other.json"
        write_json(outside, {})
        (scene / escape).symlink_to(outside)
    data["requirements"]["scale_to_meters"]["evidence"][0]["path"] = escape
    with pytest.raises((ValueError, OSError)):
        r.check(scene, save(scene, data))


def test_wrong_binding_and_manifest_or_array_corruption(scene):
    data = r.template(scene)
    data["binding"]["gauge"]["id"] = "other"
    with pytest.raises(ValueError, match="binding"):
        r.check(scene, save(scene, data))
    with (scene / "scene.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SCENE_HASH_MISMATCH"):
        r.check(scene)


def test_cli_exclusive_blocked_and_malformed(scene):
    output = scene / "template.json"
    args = ["--scene", str(scene), "--output", str(output)]
    assert r.main(["template", *args]) == 0
    before = output.read_bytes()
    assert r.main(["template", *args]) != 0
    assert output.read_bytes() == before
    result = scene / "report.json"
    assert r.main(["check", "--scene", str(scene), "--metadata", str(output), "--output", str(result)]) == 0
    assert json.loads(result.read_text())["status"] == "BLOCKED_GEOMETRY"
    bad = r.template(scene)
    bad["requirements"]["rights"]["status"] = "TRUST_ME"
    bad_path = save(scene, bad)
    failed = scene / "failed.json"
    assert r.main(["check", "--scene", str(scene), "--metadata", str(bad_path), "--output", str(failed)]) != 0
    assert not failed.exists()


@pytest.mark.parametrize("text", ['{"schema_version": 1, "schema_version": 1}',
                                '{"value": NaN}', '{"value": 1e999}', '[]'])
def test_strict_json(scene, text):
    path = scene / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        r.check(scene, path)


@pytest.mark.parametrize("damage", ["scale", "up", "ground"])
def test_cross_field_consistency_with_intact_evidence(scene, damage):
    data = metadata(scene)
    name = "T_metric_from_gauge"
    entry = data["requirements"][name]
    if damage == "scale":
        entry["value"]["scale"] = 2
    elif damage == "up":
        entry["value"]["rigid"] = np.diag([1, -1, -1, 1]).tolist()
    else:
        entry["value"]["rigid"][2][3] = 1
    path = scene / entry["evidence"][0]["path"]
    proof = json.loads(path.read_text())
    proof["value"] = entry["value"]
    path.unlink()
    write_json(path, proof)
    entry["evidence"][0]["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="Sim3"):
        r.check(scene, save(scene, data))


def test_outside_repo_and_unverified_label_rejected(scene):
    with pytest.raises(ValueError, match="inside repository"):
        r.check(scene.parent.parent)
    data = r.template(scene)
    data["requirements"]["rights"]["status"] = "VERIFIED"
    with pytest.raises(ValueError, match="VERIFIED requires"):
        r.check(scene, save(scene, data))


def test_actual_converted_left_scene():
    scene = Path(__file__).resolve().parents[1] / "output/offline_dggt_left_20261016_r2"
    if not scene.is_dir():
        pytest.skip("task-scoped converted left scene unavailable")
    before = sha256(scene / "manifest.json")
    report = r.check(scene)
    assert report.status == "BLOCKED_GEOMETRY" and report.real_expert_trajectories == 0
    assert report.gauge["metric_scale"] == "unknown"
    assert all(f.status == r.EvidenceStatus.UNKNOWN for f in report.findings)
    assert before == sha256(scene / "manifest.json")


def test_actual_module_cli():
    root = Path(__file__).resolve().parents[1]
    scene = root / "output/offline_dggt_left_20261016_r2"
    if not scene.is_dir():
        pytest.skip("task-scoped converted left scene unavailable")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    command = [sys.executable, "-B", "-m", "gaussian_park_world.scenes.readiness"]
    with tempfile.TemporaryDirectory(prefix="readiness_test_", dir=root / "output") as directory:
        template = Path(directory) / "template.json"
        report = Path(directory) / "report.json"
        run = subprocess.run(command + ["template", "--scene", str(scene), "--output", str(template)],
                             cwd=root, env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        run = subprocess.run(command + ["check", "--scene", str(scene), "--metadata", str(template),
                                        "--output", str(report)],
                             cwd=root, env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert json.loads(report.read_text())["real_expert_trajectories"] == 0
        assert json.loads(report.read_text())["status"] == "BLOCKED_GEOMETRY"
