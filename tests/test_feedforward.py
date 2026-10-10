import numpy as np
import pytest

from parking_gs import feedforward
from parking_gs.feedforward import load_scene_artifact, transform_scene
from parking_gs.scene import Scene, rotation


def sample_scene():
    return Scene(np.array([[1., 2., 3.], [-2., 0., 1.]]),
                 np.array([[.1, .3, .7], [.4, .2, .6]]),
                 np.array([[1., 2., -3., 4.], [2., -1., 3., 1.]]),
                 np.full((2, 3), .4), np.array([.7, .9]),
                 dict(units='meters', up_axis='z', source='original')).validate()


def test_direct_raw_npz_uses_caller_metadata(tmp_path):
    scene = sample_scene()
    np.savez(tmp_path / 'scene.npz', means=scene.means, scales=scene.scales,
             quats=scene.quats, colors=scene.colors, opacities=scene.opacities)
    metadata = dict(units='meters', up_axis='z', source='caller')
    loaded = load_scene_artifact(tmp_path, metadata)
    assert loaded.metadata == metadata
    np.testing.assert_allclose(loaded.covariance(), scene.covariance())


def test_direct_complete_scene_preserves_metadata_and_values(tmp_path):
    scene = sample_scene()
    scene.scales[0, 0] = 1e-5
    scene.metadata['experiment'] = dict(name='saved', bounds=[-1, -2, 3, 4])
    scene.save(tmp_path / 'scene.npz')
    loaded = load_scene_artifact(tmp_path, dict(units='meters', up_axis='z', source='caller'))
    assert loaded.metadata == scene.metadata
    for name in ('means', 'scales', 'quats', 'colors', 'opacities'):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(scene, name))


def test_reconstruct_video_uses_transform_without_model(tmp_path, monkeypatch):
    scene = sample_scene()
    scene.save(tmp_path / 'scene.npz')
    manifest = feedforward.VideoManifest('video.mp4', str(tmp_path), ['a.jpg', 'b.jpg'], 30., 8, 6)
    monkeypatch.setattr(feedforward, 'extract_video', lambda *args: manifest)
    monkeypatch.setattr(feedforward, '_run', lambda *args, **kwargs: None)
    world_rotation = rotation([1., 2., 3., -4.])
    config = dict(metric_scale=2., world_rotation=world_rotation.tolist(),
                  world_translation=[1., -2., 3.])
    transformed, report = feedforward.reconstruct_video(
        'video.mp4', tmp_path, backend='external', command='unused', config=config)
    expected = transform_scene(scene, 2., world_rotation, config['world_translation'])
    np.testing.assert_allclose(transformed.means, expected.means)
    np.testing.assert_allclose(transformed.covariance(), expected.covariance())
    np.testing.assert_allclose(Scene.load(tmp_path / 'scene.npz').covariance(), expected.covariance())
    assert report['gaussians'] == 2


@pytest.mark.parametrize('world_quat', [[1., -2., 3., 4.], [0., 1., -2., 3.],
                                       [1., 0., 0., 0.]])
def test_transform_rotates_nontrivial_covariance(world_quat):
    scene = sample_scene()
    original_means = scene.means.copy()
    original_quats = scene.quats.copy()
    world_rotation = rotation(world_quat)
    scale, translation = 2.5, np.array([4., -5., 6.])
    transformed = transform_scene(scene, scale, world_rotation, translation)
    np.testing.assert_allclose(transformed.means, scale * (scene.means @ world_rotation.T) + translation)
    np.testing.assert_allclose(transformed.scales, scale * scene.scales)
    np.testing.assert_allclose(transformed.covariance(),
                               scale**2 * (world_rotation @ scene.covariance() @ world_rotation.T),
                               atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(transformed.quats, axis=1), 1.)
    np.testing.assert_array_equal(transformed.colors, scene.colors)
    np.testing.assert_array_equal(transformed.opacities, scene.opacities)
    assert transformed.metadata == scene.metadata
    assert transformed.metadata is not scene.metadata
    np.testing.assert_array_equal(scene.means, original_means)
    np.testing.assert_array_equal(scene.quats, original_quats)


@pytest.mark.parametrize('world_rotation', [np.diag([-1., 1., 1.]), np.eye(2),
                                           np.diag([2., 1., 1.]),
                                           np.full((3, 3), np.nan),
                                           np.full((3, 3), np.inf)])
def test_transform_rejects_invalid_rotation(world_rotation):
    with pytest.raises(ValueError, match='world_rotation'):
        transform_scene(sample_scene(), 1., world_rotation, [0., 0., 0.])


@pytest.mark.parametrize('scale', [0., -1., np.nan, np.inf])
def test_transform_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match='metric_scale'):
        transform_scene(sample_scene(), scale, np.eye(3), [0., 0., 0.])


@pytest.mark.parametrize('translation', [[0., 0.], [0., np.nan, 0.], [0., 0., np.inf]])
def test_transform_rejects_invalid_translation(translation):
    with pytest.raises(ValueError, match='world_translation'):
        transform_scene(sample_scene(), 1., np.eye(3), translation)
