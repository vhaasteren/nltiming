"""Coordinate policy and expansion-spec validation (§4.4).

Chart/prior resolution tests (affine_normal vs prior_pit) are added with the
model-wiring stage; this file pins the pure dataclass validation now so
``coordinates.py`` is not untested.
"""

from __future__ import annotations

import pytest

from nltiming.coordinates import TimingCoordinatePolicy, TimingExpansionSpec


def test_coordinate_policy_defaults_and_as_dict():
    pol = TimingCoordinatePolicy()
    assert pol.linear_scale == 50.0
    assert pol.nonlinear_scale == 50.0
    assert pol.sigma_source == "parfile_then_wls"
    assert pol.nonaffine_identically_linear == "warn"
    d = pol.as_dict()
    assert d["nonidentically_linear_marginalization"] == "warn"


def test_coordinate_policy_validation_is_strict():
    with pytest.raises(ValueError, match="linear_scale"):
        TimingCoordinatePolicy(linear_scale=0.0)
    with pytest.raises(ValueError, match="nonlinear_scale"):
        TimingCoordinatePolicy(nonlinear_scale=-1.0)
    with pytest.raises(ValueError, match="sigma_source"):
        TimingCoordinatePolicy(sigma_source="something_else")
    with pytest.raises(ValueError, match="nonaffine_identically_linear"):
        TimingCoordinatePolicy(nonaffine_identically_linear="boom")
    with pytest.raises(ValueError, match="nonidentically_linear_marginalization"):
        TimingCoordinatePolicy(nonidentically_linear_marginalization="boom")


def test_expansion_spec_factories_and_validation():
    assert TimingExpansionSpec.engine_reference().mode == "engine_reference"
    assert TimingExpansionSpec.prior_center().mode == "prior_center"
    spec = TimingExpansionSpec.explicit_delta({"F0": 1.0})
    assert spec.mode == "explicit_delta"
    assert spec.delta == {"F0": 1.0}

    with pytest.raises(ValueError, match="non-empty delta"):
        TimingExpansionSpec.explicit_delta({})
    with pytest.raises(ValueError, match="takes no delta"):
        TimingExpansionSpec(mode="engine_reference", delta={"F0": 1.0})
    with pytest.raises(ValueError, match="unsupported expansion mode"):
        TimingExpansionSpec(mode="nope")


def test_model_fingerprint_changes_with_coordinate_policy():
    from nltiming import NonLinearTimingModel, TimingInference

    base = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all()
    )._config_fingerprint()
    scaled = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        coordinate_policy=TimingCoordinatePolicy(linear_scale=25.0),
    )._config_fingerprint()
    assert base != scaled


def test_expansion_spec_as_dict_roundtrips_mode():
    assert TimingExpansionSpec.prior_center().as_dict() == {
        "mode": "prior_center",
        "delta": None,
    }
    assert TimingExpansionSpec.explicit_delta({"F0": 2.0}).as_dict() == {
        "mode": "explicit_delta",
        "delta": {"F0": 2.0},
    }
