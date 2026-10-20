"""Offline scene import and validation; no training or renderer dependencies."""

from .contract import (
    assert_same_gauge,
    make_fixture,
    normalize_gaussians,
    prepare_dggt,
    sha256,
    validate_scene,
    write_json,
)

__all__ = [
    "assert_same_gauge", "make_fixture", "normalize_gaussians", "prepare_dggt",
    "sha256", "validate_scene", "write_json",
]
