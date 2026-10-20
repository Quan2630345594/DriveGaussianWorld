"""Training-free offline scene, expert and episode CLI."""

import argparse
from pathlib import Path
import json

import numpy as np

from .scenes.contract import make_fixture, prepare_dggt, write_json
from .planning.delivery import generate, validate_experts
from .dataset.offline import read_json, write_episode, validate_dataset


def prepare_scenes(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    for i in range(3):
        make_fixture(root / f"fixture_{i}", f"fixture_{i}", variant=2 * i)


def generate_experts(scenes, output, batch=False, budget=6000, retries=2):
    return generate(scenes, output, batch=batch, budget=budget, retries=retries)


def build_episodes(experts, output, backend="cpu_reference"):
    experts, output = Path(experts), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    index = read_json(experts / "experts.json")
    scene_ids = sorted({entry["scene_id"] for entry in index["trajectories"]})
    splits = {sid: ("train", "val", "test")[i % 3] for i, sid in enumerate(scene_ids)}
    episodes, failures = [], list(index["failures"])
    for entry in index["trajectories"]:
        name = entry["scene_id"] + "_" + entry["trajectory_id"]
        with np.load(experts / entry["file"], allow_pickle=False) as data:
            trajectory = {k: data[k] for k in ("states", "actions", "dt")}
        trajectory.update(status="SUCCESS", metadata=entry["metadata"])
        try:
            write_episode(output / name, entry["scene_dir"], entry["planning_map"], trajectory,
                          entry["trajectory_id"], splits[entry["scene_id"]], entry["goal"], backend=backend)
        except ValueError as error:
            failures.append(dict(scene_id=entry["scene_id"], trajectory_id=entry["trajectory_id"], reason=str(error)))
            continue
        episodes.append(name)
    write_json(output / "dataset.json", dict(schema_version=1, episodes=episodes, failures=failures,
                                             scene_splits=splits, source_classification="SYNTHETIC_PLANNING_FIXTURE",
                                             fixture_limitations=index.get("limitations", "Appearance seeds may share planning geometry; synthetic engineering fixtures only, not real planning evidence."),
                                             gwm_adapter="PENDING", training="NOT_RUN"))
    summary = validate_dataset(output)
    write_json(output / "validation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scenes = commands.add_parser("prepare-scenes")
    scenes.add_argument("--output", required=True)
    scenes.add_argument("--dggt-source", help="Existing single-gauge activated scene export; no inference")
    experts = commands.add_parser("generate-experts")
    experts.add_argument("--scenes", required=True)
    experts.add_argument("--output", required=True)
    experts.add_argument("--batch", action="store_true")
    experts.add_argument("--budget", type=int, default=6000)
    experts.add_argument("--retries", type=int, default=2)
    validate_expert = commands.add_parser("validate-experts")
    validate_expert.add_argument("--experts", required=True)
    episodes = commands.add_parser("build-episodes")
    episodes.add_argument("--experts", required=True)
    episodes.add_argument("--output", required=True)
    episodes.add_argument("--renderer", choices=("cpu_reference", "gsplat"), default="cpu_reference")
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if args.command == "prepare-scenes":
        if args.dggt_source:
            result = prepare_dggt(args.dggt_source, args.output)
        else:
            prepare_scenes(args.output)
            result = dict(scenes=3, source_classification="SYNTHETIC_PLANNING_FIXTURE")
    elif args.command == "generate-experts":
        result = generate_experts(args.scenes, args.output, args.batch, args.budget, args.retries)
    elif args.command == "validate-experts":
        result = validate_experts(args.experts)
    elif args.command == "build-episodes":
        result = build_episodes(args.experts, args.output, args.renderer)
    else:
        result = validate_dataset(args.dataset)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
