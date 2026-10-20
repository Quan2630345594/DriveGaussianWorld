"""Stage18 E1 metadata validation, not a real-scene planner or trust authority.

Evidence paths are relative to the scene directory and cannot escape it (including
via symlinks). Evidence JSON contains binding, requirement, and value equal to
the metadata plus nonempty provenance, verification_method and reviewer strings.
Hashes establish byte integrity and binding, not reviewer identity or truth.
Only convex planar polygons and axis-aligned obstacle rectangles are supported.
Even complete evidence remains blocked: a real planning importer is pending.
"""

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

from .contract import sha256, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = (
    "scale_to_meters", "world_up_ground", "T_metric_from_gauge",
    "source_frame_alignment", "drivable_geometry", "obstacles",
    "vehicle_footprint", "T_vehicle_from_optical", "rights",
)


class EvidenceStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    VERIFIED = "VERIFIED"
    USER_ASSERTED = "USER_ASSERTED"


@dataclass(frozen=True)
class RequirementFinding:
    requirement: str
    status: EvidenceStatus
    evidence_validated: bool
    reason: str


@dataclass(frozen=True)
class ReadinessReport:
    scene_id: str
    gauge: dict[str, Any]
    manifest_sha256: str
    status: Literal["BLOCKED_GEOMETRY"]
    real_expert_trajectories: int
    findings: tuple[RequirementFinding, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _local(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("path must be a local task-scoped file")
    return resolved


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("JSON exceeds bounded 2 MiB limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def floating(value):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs, parse_constant=constant,
                          parse_float=floating)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _scene(scene_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    scene = Path(scene_dir).resolve(strict=True)
    if not scene.is_dir() or not scene.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("scene must be inside repository")
    path = _local(scene / "manifest.json", scene)
    manifest = _read(path)
    if (manifest.get("schema_version") != 1 or not manifest.get("scene_id")
            or not isinstance(manifest.get("world_gauge"), dict)
            or not manifest["world_gauge"].get("id")
            or manifest.get("array_file") != "scene.npz"):
        raise ValueError("invalid canonical scene manifest")
    if sha256(_local(scene / "scene.npz", scene)) != manifest.get("array_sha256"):
        raise ValueError("SCENE_HASH_MISMATCH")
    binding = dict(scene_id=manifest["scene_id"], gauge=manifest["world_gauge"],
                   manifest_sha256=sha256(path))
    return scene, manifest, binding


def template(scene_dir: str | Path) -> dict[str, Any]:
    """Return strict-JSON-compatible fillable metadata; never invent geometry.

    Values: scale is a positive number; up/ground contains world_up [3] and
    ground_plane [nx,ny,nz,d] in gauge units. Sim3 contains scale, rigid [4,4],
    from_gauge and to_frame='metric_x_forward_y_left_z_up'. Alignment contains
    gauge_id and camera_frames [{camera_id,source_frame,source_frame_coordinate}]
    matching the manifest exactly. Geometry contains gauge_id, units='meters',
    frame='metric_x_forward_y_left_z_up', independent_source, confidence (0,1],
    unknown_space_policy='occupied', and polygon [[x,y],...] or rectangles
    [[xmin,ymin,xmax,ymax],...]. Empty obstacles require measured_empty=true.
    Footprint contains units='meters', frame='vehicle_x_forward_y_left_z_up',
    polygon. Camera extrinsics contain units='meters', camera_ids (all manifest
    cameras), from_frame='optical_x_right_y_down_z_forward',
    to_frame='vehicle_x_forward_y_left_z_up', rigid [4,4]. Rights contains
    allowed_use (list including 'local_planning') and rights_holder (string).
    Each evidence entry is {path: scene-relative JSON, sha256: lowercase hash}.
    """
    _, _, binding = _scene(scene_dir)
    return dict(schema_version=1, binding=binding, requirements={
        name: dict(status=EvidenceStatus.UNKNOWN.value, value=None, evidence=[])
        for name in REQUIREMENTS
    })


def _number(value: Any) -> bool:
    return type(value) in (int, float) and np.isfinite(value)


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"expected finite numeric array {shape}")
    return array.astype(float)


def _rigid(value: Any) -> np.ndarray:
    t = _array(value, (4, 4))
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("expected proper rigid transform (no scale/reflection)")
    return t


def _text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in ("", "unknown", "pending")


def _polygon(value: Any) -> None:
    if not isinstance(value, list) or not 3 <= len(value) <= 1000:
        raise ValueError("polygon requires 3..1000 vertices")
    p = _array(value, (len(value), 2))
    # Strict convexity with every other vertex on one side also rejects stars.
    signs = []
    for i in range(len(p)):
        edge = p[(i + 1) % len(p)] - p[i]
        others = p[[j for j in range(len(p)) if j not in (i, (i + 1) % len(p))]] - p[i]
        cross = edge[0] * others[:, 1] - edge[1] * others[:, 0]
        signs.append(np.all(cross > 1e-10))
        if not (np.all(cross > 1e-10) or np.all(cross < -1e-10)):
            raise ValueError("only nondegenerate simple convex polygons supported")
    if not (all(signs) or not any(signs)):
        raise ValueError("inconsistent polygon winding")


def _value(name: str, value: Any, manifest: dict[str, Any]) -> None:
    gauge = manifest["world_gauge"]["id"]
    if name == "scale_to_meters":
        valid = _number(value) and value > 0
    else:
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected object")
        if name == "world_up_ground":
            up = _array(value.get("world_up"), (3,))
            plane = _array(value.get("ground_plane"), (4,))
            valid = (np.isclose(np.linalg.norm(up), 1, atol=1e-6, rtol=0)
                     and np.allclose(up, plane[:3], atol=1e-6, rtol=0))
        elif name == "T_metric_from_gauge":
            _rigid(value.get("rigid"))
            valid = (value.get("from_gauge") == gauge
                     and value.get("to_frame") == "metric_x_forward_y_left_z_up"
                     and _number(value.get("scale")) and value["scale"] > 0)
        elif name == "source_frame_alignment":
            expected = [{k: c.get(k, "unknown") for k in
                         ("camera_id", "source_frame", "source_frame_coordinate")}
                        for c in manifest["cameras"]]
            valid = (value.get("gauge_id") == gauge and value.get("camera_frames") == expected
                     and all(c["source_frame"] != "unknown" and
                             c["source_frame_coordinate"] != "unknown" for c in expected))
        elif name in ("drivable_geometry", "obstacles"):
            valid = (value.get("gauge_id") == gauge and value.get("units") == "meters"
                     and value.get("frame") == "metric_x_forward_y_left_z_up"
                     and _text(value.get("independent_source"))
                     and _number(value.get("confidence")) and 0 < value["confidence"] <= 1
                     and value.get("unknown_space_policy") == "occupied")
            if name == "drivable_geometry":
                _polygon(value.get("polygon"))
            else:
                rectangles = value.get("rectangles")
                if not isinstance(rectangles, list) or len(rectangles) > 10000:
                    raise ValueError("expected bounded obstacle rectangles list")
                for rectangle in rectangles:
                    r = _array(rectangle, (4,))
                    if not (r[0] < r[2] and r[1] < r[3]):
                        raise ValueError("degenerate obstacle rectangle")
                valid = valid and (bool(rectangles) or value.get("measured_empty") is True)
        elif name == "vehicle_footprint":
            _polygon(value.get("polygon"))
            valid = (value.get("units") == "meters"
                     and value.get("frame") == "vehicle_x_forward_y_left_z_up")
        elif name == "T_vehicle_from_optical":
            _rigid(value.get("rigid"))
            valid = (value.get("units") == "meters"
                     and value.get("from_frame") == "optical_x_right_y_down_z_forward"
                     and value.get("to_frame") == "vehicle_x_forward_y_left_z_up"
                     and value.get("camera_ids") == [c["camera_id"] for c in manifest["cameras"]])
        else:
            uses = value.get("allowed_use")
            valid = (isinstance(uses, list) and all(_text(u) for u in uses)
                     and "local_planning" in uses and _text(value.get("rights_holder")))
    if not valid:
        raise ValueError(f"{name}: invalid value, units, gauge, frame, or required policy")


def check(scene_dir: str | Path, metadata: str | Path | None = None) -> ReadinessReport:
    """Validate local metadata, raising ValueError/OSError on malformed input.

    Missing/UNKNOWN/USER_ASSERTED proofs are well-formed blocked findings.
    VERIFIED proofs must have valid values and exact bound evidence; they are
    integrity-checked claims, not independently authenticated measurements.
    No arrays are loaded, no provenance paths from the manifest are followed.
    """
    scene, manifest, binding = _scene(scene_dir)
    requirements = {}
    if metadata is not None:
        data = _read(_local(Path(metadata), REPO_ROOT))
        if data.get("schema_version") != 1 or data.get("binding") != binding:
            raise ValueError("metadata scene/gauge/manifest binding mismatch")
        requirements = data.get("requirements")
        if not isinstance(requirements, dict) or set(requirements) - set(REQUIREMENTS):
            raise ValueError("invalid requirements object")
    findings = []
    for name in REQUIREMENTS:
        entry = requirements.get(name, dict(status="UNKNOWN", value=None, evidence=[]))
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: invalid requirement")
        status = EvidenceStatus(entry.get("status"))
        value = entry.get("value")
        if value is not None:
            _value(name, value, manifest)
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"{name}: evidence must be list")
        for ref in evidence:
            if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                raise ValueError("invalid evidence reference")
            relative = Path(ref["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("evidence must be scene-relative task-scoped path")
            path = _local(scene / relative, scene)
            if sha256(path) != ref.get("sha256"):
                raise ValueError(f"{name}: EVIDENCE_HASH_MISMATCH")
            proof = _read(path)
            if (proof.get("binding") != binding or proof.get("requirement") != name
                    or proof.get("value") != value or not all(_text(proof.get(k)) for k in
                    ("provenance", "verification_method", "reviewer"))):
                raise ValueError(f"{name}: incomplete or unbound evidence")
        if status == EvidenceStatus.VERIFIED and (value is None or not evidence):
            raise ValueError(f"{name}: VERIFIED requires value and bound local evidence")
        validated = status == EvidenceStatus.VERIFIED
        reason = {EvidenceStatus.UNKNOWN: "Required proof absent or UNKNOWN",
                  EvidenceStatus.USER_ASSERTED: "User assertion is not measured ground truth",
                  EvidenceStatus.VERIFIED: "Value and bound evidence integrity checked; identity not authenticated"}[status]
        findings.append(RequirementFinding(name, status, validated, reason))
    values = {k: v.get("value") for k, v in requirements.items()}
    sim = values.get("T_metric_from_gauge")
    up = values.get("world_up_ground")
    if sim and values.get("scale_to_meters") is not None:
        if not np.isclose(sim["scale"], values["scale_to_meters"], rtol=1e-9, atol=0):
            raise ValueError("Sim3 scale disagrees with scale_to_meters")
    if sim and up:
        rigid = np.asarray(sim["rigid"])
        rotated = rigid[:3, :3] @ up["world_up"]
        if not np.allclose(rotated, [0, 0, 1], atol=1e-6, rtol=0):
            raise ValueError("Sim3 rotation does not align world_up to metric Z")
        # Polygons are metric XY at Z=0, so the supplied ground must map there.
        offset = sim["scale"] * up["ground_plane"][3] - rotated @ rigid[:3, 3]
        if not np.isclose(offset, 0, atol=1e-6, rtol=0):
            raise ValueError("Sim3 ground plane does not map to metric Z=0")
    return ReadinessReport(**binding, status="BLOCKED_GEOMETRY", real_expert_trajectories=0,
                           findings=tuple(findings), limitations=(
                               "UNSUPPORTED_REAL_IMPORTER_PENDING: metadata validation only; old planning gate unchanged",
                               "Local hashes and reviewer text do not authenticate measurements or rights",
                               "Only convex planar polygons and axis-aligned rectangles validated; no geometry inferred",
                               "Canonical array hash checked; Gaussian arrays not loaded or semantically validated",
                           ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("template", "check"):
        command = commands.add_parser(name)
        command.add_argument("--scene", required=True)
        command.add_argument("--output", required=True)
        if name == "check":
            command.add_argument("--metadata")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output)
        if not output.parent.resolve(strict=True).is_relative_to(REPO_ROOT.resolve()):
            raise ValueError("output must be inside repository")
        result = (template(args.scene) if args.command == "template"
                  else check(args.scene, args.metadata).to_dict())
        write_json(output, result)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"readiness: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
