"""The hybrid vocabulary and the binary-axis registry (nltiming owns both)."""

from __future__ import annotations

import pytest

from nltiming.hybrid import (
    BINARY_AXES,
    hybrid_linearized_fitpars,
    is_binary_axis,
    is_hybrid_engine_axis,
    resolve_hybrid_partition,
    validate_nonlinear_params,
)


@pytest.mark.parametrize(
    "name", ["A1", "PB", "T0", "TASC", "ECC", "OM", "SINI", "M2", "STIGMA", "LNEDOT"]
)
def test_binary_axes_are_binary(name):
    assert is_binary_axis(name)


@pytest.mark.parametrize("name,canonical", [("XDOT", "A1DOT"), ("E", "ECC"), ("STIG", "STIGMA")])
def test_tempo2_spellings_resolve_before_classification(name, canonical):
    """The registry is keyed on PINT names; a par may use either spelling."""
    assert canonical in BINARY_AXES
    assert is_binary_axis(name)


@pytest.mark.parametrize("name", ["FB0", "FB1", "FB3"])
def test_the_orbital_frequency_series_is_binary(name):
    """``FBk`` is ``PB`` by another parametrisation, and prefixed, so it needs
    a pattern rather than a set entry."""
    assert is_binary_axis(name)


@pytest.mark.parametrize("name", ["PX", "F0", "F1", "DM", "DMX_0001", "JUMP1", "RAJ"])
def test_non_binary_axes_are_not(name):
    assert not is_binary_axis(name)


def test_px_is_the_whole_difference_between_the_two_modes():
    assert is_hybrid_engine_axis("PX", "binary+")
    assert not is_hybrid_engine_axis("PX", "binary")
    for mode in ("binary", "binary+"):
        assert is_hybrid_engine_axis("A1", mode)
        assert not is_hybrid_engine_axis("F0", mode)


def test_no_mode_keeps_every_axis_on_the_engine():
    assert validate_nonlinear_params(None) is None
    assert is_hybrid_engine_axis("F0", None)


@pytest.mark.parametrize("value", ["binary ", "BINARY", "binary+"])
def test_known_modes_normalize(value):
    assert validate_nonlinear_params(value) in {"binary", "binary+"}


@pytest.mark.parametrize("value", ["binaries", "", 3, "binary++"])
def test_unknown_modes_are_refused(value):
    with pytest.raises(ValueError, match="nonlinear_params"):
        validate_nonlinear_params(value)


def test_linearized_fitpars_use_the_engine_spelling():
    """A PTA-suffixed fitpar is classified through its engine name."""
    fitpars = ("A1_epta", "F0_epta", "PX_epta")
    mapping = {"A1_epta": "A1", "F0_epta": "F0", "PX_epta": "PX"}
    assert hybrid_linearized_fitpars(fitpars, mapping, "binary") == {"F0_epta", "PX_epta"}
    assert hybrid_linearized_fitpars(fitpars, mapping, "binary+") == {"F0_epta"}
    assert hybrid_linearized_fitpars(fitpars, mapping, None) == frozenset()


def test_partition_defaults_to_the_mode():
    engine, exact = resolve_hybrid_partition(
        fitpars=("A1", "F0", "PHOFF"),
        param_mapping={},
        mode="binary",
        engine_fitpars=None,
        exact_linear_fitpars=None,
    )
    assert engine == ("A1",)
    assert exact == {"F0", "PHOFF"}


def test_an_engine_list_that_contradicts_the_mode_is_refused():
    """A stamped mode must match the partition the engine actually evaluates."""
    with pytest.raises(ValueError, match="linearizes"):
        resolve_hybrid_partition(
            fitpars=("A1", "F0"),
            param_mapping={},
            mode="binary",
            engine_fitpars=("A1", "F0"),
            exact_linear_fitpars=None,
        )


def test_an_axis_on_neither_path_is_refused():
    """Its delta would be dropped from the residual without a word."""
    with pytest.raises(ValueError, match="silently dropped"):
        resolve_hybrid_partition(
            fitpars=("A1", "PB"),
            param_mapping={},
            mode="binary",
            engine_fitpars=("A1",),
            exact_linear_fitpars=None,
        )
