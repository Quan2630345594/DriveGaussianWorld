"""Snapshot executable sources, environment metadata and local result checksums."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('outputs/parkinge2e_20261010_01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment = args.experiment.resolve()
    if not (experiment / 'evaluation.json').is_file():
        parser.error('Experiment evaluation.json is required')
    snapshot = experiment / 'source_snapshot'
    sources = [*sorted((root / 'scripts').glob('*.py')), *sorted((root / 'parking_gs').glob('*.py')),
               *sorted((root / 'tests').glob('*.py')), root / 'requirements.txt',
               root / 'requirements-dggt.txt', root / 'README.md', root / 'LICENSE']
    for source in sources:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    packages = sorted({d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                       if d.metadata['Name']}.items())
    def git_revision(path):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       packages=dict(packages), project_git_revision=git_revision(root),
                       dggt_git_revision=git_revision(root / 'third_party/dggt'),
                       package_source='parkings isolated interpreter; no user site packages',
                       source_snapshot='Includes working sources; Git revision alone does not describe edits',
                       inference_gpus=dict(front=5, back=5, left=7, right=4),
                       reference_type='recorded estimated localization, not independent ground truth')
    (experiment / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    inputs = dict(bag=dict(path='data/raw/parkinge2e/demo_bag.bag',
                          sha256='28abde1050e8b0d5d04caf7aa7ef7845103f9a29c24c8e70b3d747fe934f010f',
                          checksum_source='computed locally; no publisher checksum supplied',
                          url='https://drive.google.com/file/d/1jIG1iRMeW9XXdWP7eEJKnZP1gC0xvG7o/view'),
                  checkpoint=dict(path='checkpoints/model_latest_waymo.pth',
                                  sha256='352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def',
                                  checksum_source='verified against publisher'),
                  calibration_commit='acec907b9e09b052120f7861a48d261c093ab5bb',
                  archive_excludes='Raw 9GB bag and model weights; both remain in local input directories',
                  distribution='Local archive only; no upload or public dataset redistribution')
    for item in (inputs['bag'], inputs['checkpoint']):
        actual = digest(root / item['path'])
        if actual != item['sha256']:
            raise RuntimeError(f"Input changed since experiment preparation: {item['path']}")
        item['verified_at_packaging'] = True
    (experiment / 'inputs.json').write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    files = sorted(p for p in experiment.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.name != 'SHA256SUMS')
    (experiment / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(experiment)}\n'
                                                 for p in files), encoding='utf-8')
    archive = experiment.with_name(experiment.name + '_results.zip')
    if archive.exists():
        parser.error(f'Archive already exists: {archive}')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for source in [*files, experiment / 'SHA256SUMS']:
            bundle.write(source, Path(experiment.name) / source.relative_to(experiment))
    print(json.dumps(dict(archive=str(archive), files=len(files) + 1,
                          bytes=archive.stat().st_size, sha256=digest(archive)), indent=2))


if __name__ == '__main__':
    main()
