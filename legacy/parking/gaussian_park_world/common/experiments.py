"""Independent rendering experiment roots and archived input directories."""
from datetime import datetime
from pathlib import Path
import re


OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output"


def experiment_directory(name=None, output_root=OUTPUT_ROOT):
    if name is not None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in (".", "..")):
        raise ValueError("Experiment name must be a single safe directory name")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if name is not None:
        experiment = root / name
        if experiment.is_symlink():
            raise ValueError("Experiment directory must not be a symbolic link")
        experiment.mkdir(exist_ok=True)
    else:
        while True:
            experiment = root / datetime.now().strftime("experiment_%Y%m%d_%H%M%S_%f")
            try:
                experiment.mkdir()
                break
            except FileExistsError:
                continue
    for stage in ("reconstruction", "render"):
        destination = experiment / stage
        if destination.is_symlink():
            raise ValueError("Experiment stages must not be symbolic links")
        destination.mkdir(exist_ok=True)
    return experiment
