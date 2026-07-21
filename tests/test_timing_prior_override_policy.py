"""Tests for prior_override_policy on NonLinearTimingModel."""

from __future__ import annotations

import warnings

import pytest

from nltiming import TimingInference
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from _planhelp import plan_for


class _StubBackend:
    def reference_theta_exact(self):
        return {"PMRA": "1.0", "PMDEC": "2.0"}


def test_prior_override_warn_skips_unknown_fitpar():
    ntm = NonLinearTimingModel(
        prior_override_policy="warn",
        inference=TimingInference.sample_all(),
    )
    ntm.set_prior_delta("ECC", "uniform", lower=0.0, upper=0.9)

    class _Pulsar:
        fitpars = ("PMRA", "PMDEC")
        name = "stub"

    pulsar = _Pulsar()
    partition = plan_for(pulsar, sample_all=True)
    engine = _StubBackend()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = ntm._resolve_prior_overrides(
            pulsar=pulsar, engine=engine, partition=partition,
            charts=(), chart_resolutions=(),
        )

    assert resolved == {}
    assert any("ECC" in str(w.message) for w in caught)


def test_prior_override_strict_raises_unknown_fitpar():
    ntm = NonLinearTimingModel(
        prior_override_policy="strict",
        inference=TimingInference.sample_all(),
    )
    ntm.set_prior_delta("ECC", "uniform", lower=0.0, upper=0.9)

    class _Pulsar:
        fitpars = ("PMRA", "PMDEC")
        name = "stub"

    pulsar = _Pulsar()
    partition = plan_for(pulsar, sample_all=True)
    engine = _StubBackend()

    with pytest.raises(ValueError, match="unknown fit parameters"):
        ntm._resolve_prior_overrides(
            pulsar=pulsar, engine=engine, partition=partition,
            charts=(), chart_resolutions=(),
        )
