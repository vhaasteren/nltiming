"""Engine selection vocabulary tests."""

from __future__ import annotations

import pytest

from nltiming.engine_config import _IMPL_FAMILY, normalize_engines


def test_normalize_engines_accepts_vela_for_pint_family():
    engines = normalize_engines({"pint": "vela", "tempo2": "jug"})
    assert engines == {"pint": "vela", "tempo2": "jug"}
    with pytest.raises(ValueError, match="must be one of"):
        normalize_engines({"tempo2": "vela"})


def test_vela_jax_serves_both_native_packages():
    """vela-jax separates the host from the physics: tempo2 or PINT may read
    the files, and Vela's chain evaluates the delay either way."""
    assert normalize_engines("vela_jax") == {"pint": "vela_jax", "tempo2": "vela_jax"}
    assert normalize_engines({"tempo2": "vela_jax", "pint": "pint"}) == {
        "tempo2": "vela_jax",
        "pint": "pint",
    }


def test_vela_jax_is_pint_family():
    """The host may be tempo2; the residual, design matrix and units are not."""
    assert _IMPL_FAMILY["vela_jax"] == "pint"
