import hashlib
import json

import numpy as np
import pytest

from gaussian_park_world.gwm_adapter.gaussians import (
    frustum_candidates, main, pack_gaussians, sample_gaussians, unpack_gaussians,
)
from gaussian_park_world.scenes import make_fixture


def arrays_for(n):
    rng = np.random.default_rng(492)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(means=rng.normal(size=(n, 3)), scales=rng.uniform(.01, 2, (n, 3)),
                quats=q, colors=rng.random((n, 3)), opacities=rng.random(n))


def test_fixture_pack_roundtrip_and_original_sidecars(tmp_path):
    make_fixture(tmp_path / "scene", "fixture")
    with np.load(tmp_path / "scene" / "scene.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    before = {k: v.copy() for k, v in arrays.items()}
    packed = pack_gaussians(arrays)
    assert packed.shape == (24, 14) and packed.dtype == np.float32
    np.testing.assert_allclose(packed[:, 10:13], arrays["colors"], rtol=1e-7)
    for dtype in (np.float32, np.float64):
        restored = unpack_gaussians(pack_gaussians(arrays, dtype=dtype), covariances=arrays["covariances"])
        for key in arrays:
            if dtype == np.float64 or key == "covariances":
                np.testing.assert_array_equal(restored[key], arrays[key])
            else:
                np.testing.assert_allclose(restored[key], arrays[key], rtol=1e-7, atol=1e-8)
    result = sample_gaussians(arrays, allow_padding=True)
    assert result["features"].shape == (2048, 14)
    assert result["valid_mask"].dtype == bool
    np.testing.assert_array_equal(result["valid_mask"], np.arange(2048) < 24)
    assert not result["features"][24:].any()
    assert result["metadata"]["source_count"] == result["metadata"]["selected_count"] == 24
    assert result["metadata"]["padding_count"] == 2024
    for key in arrays:
        np.testing.assert_array_equal(result["canonical_arrays"][key], arrays[key][result["audit_indices"]])
        np.testing.assert_array_equal(arrays[key], before[key])
        assert not np.shares_memory(result["canonical_arrays"][key], arrays[key])


@pytest.mark.parametrize("n,count,pad", [(24, 2048, True), (2000, 512, False)])
def test_permutation_qsign_and_id_independence(n, count, pad):
    arrays = arrays_for(n)
    arrays["quats"][0] = [0, -1, 0, 0]
    arrays["ids"] = np.arange(n)
    expected = sample_gaussians(arrays, count, pad, dtype=np.float64)
    permutation = np.random.default_rng(12).permutation(n)
    other = {k: v[permutation].copy() for k, v in arrays.items()}
    other["quats"] *= -1
    other["ids"] = np.arange(n)[::-1] + 999
    actual = sample_gaussians(other, count, pad, dtype=np.float64)
    np.testing.assert_array_equal(actual["features"], expected["features"])
    np.testing.assert_array_equal(actual["valid_mask"], expected["valid_mask"])
    assert actual["metadata"] == expected["metadata"]
    assert not expected["metadata"]["ids_used"]
    assert not expected["metadata"]["is_fps"]
    assert not expected["metadata"]["geometry_coverage_preserved"]
    assert not expected["metadata"]["opacity_preserved"]
    if n == 2000:
        assert np.max(expected["audit_indices"]) >= count


def test_smallest_hashes_and_duplicate_multiset():
    arrays = arrays_for(24)
    for key in arrays:
        arrays[key][1:4] = arrays[key][0]
    rows = pack_gaussians(arrays, dtype=np.float64)
    for row in rows:
        q = row[6:10]
        if q[np.flatnonzero(q)[0]] < 0:
            q *= -1
    rows[rows == 0] = 0.
    ranked = sorted((hashlib.sha256(row.astype("<f8").tobytes()).digest(),
                     row.astype("<f8").tobytes()) for row in rows)
    result = sample_gaussians(arrays, 12, dtype=np.float64)
    expected = np.stack([np.frombuffer(content, dtype="<f8") for _, content in ranked[:12]])
    np.testing.assert_array_equal(result["features"], expected)
    reverse = {k: v[::-1] for k, v in arrays.items()}
    np.testing.assert_array_equal(sample_gaussians(reverse, 12, dtype=np.float64)["features"], expected)


def test_production_size_content_sampling():
    arrays = arrays_for(120_000)
    result = sample_gaussians(arrays)
    assert result["features"].shape == (2048, 14)
    assert result["valid_mask"].all()
    assert result["metadata"]["source_count"] == 120_000
    assert result["metadata"]["selected_count"] == 2048
    assert len(np.unique(result["audit_indices"])) == 2048


@pytest.mark.parametrize("sample", [False, True])
def test_float32_scale_underflow_is_not_a_valid_gaussian(sample):
    arrays = arrays_for(1)
    arrays["scales"][:] = 1e-50
    arrays["covariances"] = np.eye(3)[None] * 1e-100
    with pytest.raises(ValueError, match="positive scales not representable"):
        if sample:
            sample_gaussians(arrays, 1)
        else:
            pack_gaussians(arrays)
    assert np.all(pack_gaussians(arrays, dtype=np.float64)[:, 3:6] > 0)


def test_sampling_report_and_unknown_time_frustum_cli(tmp_path):
    scene = tmp_path / "scene"
    make_fixture(scene, "diagnostic-only")
    report = tmp_path / "sample.json"
    main(["sample", "--input", str(scene / "scene.npz"), "--output", str(tmp_path / "sample.npz"),
          "--count", "12", "--report", str(report)])
    metadata = json.loads(report.read_text())
    assert metadata["source_count"] == 24 and metadata["selected_count"] == 12
    assert metadata["elapsed_seconds"] >= 0 and metadata["process_peak_rss_bytes"] > 0
    assert len(metadata["source_sha256"]) == len(metadata["output_sha256"]) == 64
    diagnostic = tmp_path / "frustum.json"
    main(["diagnose-frustum", "--scene", str(scene), "--output", str(diagnostic)])
    result = json.loads(diagnostic.read_text())
    assert result["counts"] == dict(frustum_mask=24, temporal_mask=0, candidate_mask=0)
    assert result["occlusion"] == "UNKNOWN" and not result["causal_verified_model_input"]
    with pytest.raises(FileExistsError):
        main(["diagnose-frustum", "--scene", str(scene), "--output", str(diagnostic)])


def test_signed_zero_does_not_change_hash_selection():
    arrays = arrays_for(24)
    arrays["quats"][:] = [0, -1, 0, 0]
    arrays["means"][:, 0] = 0.
    expected = sample_gaussians(arrays, 12, dtype=np.float64)
    arrays["quats"] *= -1
    arrays["means"][:, 0] = -0.
    actual = sample_gaussians(arrays, 12, dtype=np.float64)
    np.testing.assert_array_equal(actual["features"], expected["features"])
    assert actual["features"].tobytes() == expected["features"].tobytes()


def test_strict_insufficient_empty_and_bounds():
    with pytest.raises(ValueError, match="insufficient"):
        sample_gaussians(arrays_for(24))
    empty = sample_gaussians(arrays_for(0), 5, True)
    assert not empty["valid_mask"].any() and not empty["features"].any()
    for count in (0, -1, 8193, True, 2.5):
        with pytest.raises(ValueError, match="count"):
            sample_gaussians(arrays_for(24), count, True)
    with pytest.raises(ValueError, match="200000"):
        pack_gaussians({"means": np.broadcast_to([0., 0., 1.], (200001, 3))})
    assert sample_gaussians(arrays_for(24), 8192, True)["features"].shape == (8192, 14)


@pytest.mark.parametrize("field,value", [("means", np.nan), ("scales", np.inf),
    ("quats", np.nan), ("colors", -1), ("opacities", 2), ("scales", 0)])
def test_bad_values(field, value):
    arrays = arrays_for(24)
    arrays[field][0] = value
    with pytest.raises(ValueError):
        sample_gaussians(arrays, 5)


def test_float32_overflow_and_bad_covariance():
    arrays = arrays_for(24)
    arrays["means"][0, 0] = 1e100
    with pytest.raises(ValueError, match="finite"):
        pack_gaussians(arrays)
    assert np.isfinite(pack_gaussians(arrays, dtype=np.float64)).all()
    arrays["covariances"] = np.full((24, 3, 3), np.nan)
    with pytest.raises(ValueError, match="covariances"):
        pack_gaussians(arrays, dtype=np.float64)


def diagnostic(source=None, available=None, query=5., **overrides):
    arrays = arrays_for(8)
    # Pixel coordinates: center, right edge, left-outside, bottom edge,
    # behind, camera plane, overlapping center, top-left inclusive.
    arrays["means"] = np.array([[0, 0, 2], [2, 0, 2], [-2.1, 0, 2],
                                [0, 2, 2], [0, 0, -1], [0, 0, 0],
                                [0, 0, 3], [-2, -2, 2]], dtype=float)
    t = np.eye(4)
    t[:3, 3] = [3, -4, 2]
    arrays["means"] += t[:3, 3]
    kwargs = dict(arrays=arrays, K=[[10, 0, 10], [0, 10, 10], [0, 0, 1]],
                  T_world_from_optical=t, image_size=(20, 20), source_time_s=source,
                  available_at_s=available, query_time_s=query)
    kwargs.update(overrides)
    return frustum_candidates(**kwargs)


def test_frustum_clipping_future_source_and_availability_masks():
    source = np.ones(8)
    source[0] = 6  # Future source despite already-available timestamp.
    available = np.ones(8)
    available[6] = 6
    result = diagnostic(source, available)
    np.testing.assert_array_equal(result["frustum_mask"], [True, False, False, False, False, False, True, True])
    np.testing.assert_array_equal(result["source_time_mask"], source <= 5)
    np.testing.assert_array_equal(result["availability_mask"], available <= 5)
    np.testing.assert_array_equal(result["candidate_mask"], [False] * 7 + [True])
    assert result["metadata"]["status"] == "PARTIAL"
    assert result["metadata"]["occlusion"] == "UNKNOWN"
    assert not result["metadata"]["causal_verified_model_input"]
    assert "features" not in result and "observed_mask" not in result
    assert diagnostic(np.full(8, 5.), np.full(8, 5.))["candidate_mask"].sum() == 3


def test_rotated_world_from_optical_and_occlusion_stays_unknown():
    arrays = arrays_for(3)
    optical = np.array([[0, 0, 2], [0, 0, 4], [0, 0, -2.]])
    t = np.array([[0., 0, 1, 3], [0, 1, 0, -4], [-1, 0, 0, 2], [0, 0, 0, 1]])
    arrays["means"] = optical @ t[:3, :3].T + t[:3, 3]
    result = frustum_candidates(arrays, np.eye(3), t, (20, 20), np.ones(3), np.ones(3), 5.)
    np.testing.assert_array_equal(result["candidate_mask"], [True, True, False])
    assert result["metadata"]["occlusion"] == "UNKNOWN"
    assert not result["metadata"]["causal_verified_model_input"]


@pytest.mark.parametrize("unknown", [None, "unknown"])
def test_unknown_times_never_produce_candidates(unknown):
    for result in (diagnostic(unknown, np.ones(8)), diagnostic(np.ones(8), unknown)):
        assert not result["candidate_mask"].any()
        assert result["metadata"]["diagnostic_only"]


@pytest.mark.parametrize("name,value", [("source_time_s", np.full(8, np.nan)),
    ("available_at_s", np.full(8, np.inf)), ("source_time_s", 1.),
    ("query_time_s", np.nan), ("query_time_s", True), ("image_size", (0, 20)),
    ("T_world_from_optical", np.diag([2, 1, 1, 1]))])
def test_invalid_diagnostic_inputs(name, value):
    with pytest.raises(ValueError):
        diagnostic(np.ones(8), np.ones(8), **{name: value})


def test_cli_sample_is_strict_and_auditable(tmp_path, capsys):
    source, output = tmp_path / "scene.npz", tmp_path / "sample.npz"
    np.savez(source, **arrays_for(24))
    args = ["sample", "--input", str(source), "--output", str(output)]
    with pytest.raises(ValueError, match="insufficient"):
        main(args)
    assert not output.exists()
    main(args + ["--allow-padding"])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["runtime_status"] == "PENDING" and not metadata["official_sh_feed"]
    with np.load(output, allow_pickle=False) as archive:
        assert archive["valid_mask"].sum() == 24
        assert archive["features"].dtype == np.float32
        assert archive["audit_canonical_means"].dtype == np.float64
        assert json.loads(str(archive["metadata_json"])) == metadata
    with pytest.raises(FileExistsError):
        main(args + ["--allow-padding"])
