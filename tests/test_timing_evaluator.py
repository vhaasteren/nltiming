"""Tests for the interactive, engine-independent timing evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming import TimingEvaluator


class LinearJaxBackend:
    backend_name = "jug"
    fitpars = ("F0", "F1")
    native_units = {"F0": "Hz", "F1": "Hz / s"}

    def __init__(self, design):
        self._design = np.asarray(design, dtype=float)

    def reference_theta(self):
        return np.asarray([10.0, -1.0e-15])

    def reference_theta_exact(self):
        return {"F0": "10.0000000000000001", "F1": "-1e-15"}

    def residual_delta(self, delta_theta):
        return self._design @ np.asarray(delta_theta)

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        return jnp.asarray(self._design) @ jnp.asarray(delta_theta)

    def design_matrix(self, params=None):
        return self._design

    def precision_critical_fitpars(self):
        return frozenset({"F0"})


class EvaluatorHost:
    name = "J1234+5678"
    fitpars = ["F0", "F1"]

    def __init__(self):
        self.Mmat = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        self.toas = np.arange(4.0)
        self.toaerrs = np.ones(4)
        self.freqs = np.full(4, 1400.0)
        self.flags = {"pta": np.asarray(["a"] * 4)}
        self.backend_flags = np.asarray(["a"] * 4)
        self.residuals = self.Mmat @ np.asarray([0.2, -0.05])
        self._backend = LinearJaxBackend(self.Mmat)

    def pint_model(self):
        return None

    def timing_backend(self, engines="jug", **kwargs):
        return self._backend

    def timing_parameter_mapping(self):
        return {"F0": {"pta_a": "F0"}, "F1": {"pta_a": "F1"}}


def test_evaluate_partial_delta_and_absolute_mapping():
    timing = TimingEvaluator(EvaluatorHost())

    delta = timing.evaluate({"F0": 0.25})
    absolute = timing.evaluate({"F0": 10.25}, frame="absolute")

    np.testing.assert_allclose(delta.delta, [0.25, 0.0])
    np.testing.assert_allclose(absolute.delta, delta.delta)
    np.testing.assert_allclose(delta.residuals, timing.pulsar.residuals + 0.25)
    np.testing.assert_allclose(delta.delay, -delta.residual_delta)
    assert timing.reference_exact["F0"] == "10.0000000000000001"
    assert timing.parameters["F0"].sessions == ("pta_a",)


def test_capabilities_and_autodiff_jacobian():
    timing = TimingEvaluator(EvaluatorHost())

    assert timing.capabilities.jax
    assert timing.capabilities.autodiff_jacobian
    assert timing.capabilities.session_engines == {"J1234+5678": "jug"}
    np.testing.assert_allclose(timing.jacobian(method="reference"), timing.pulsar.Mmat)
    np.testing.assert_allclose(
        timing.jacobian({"F0": 0.2}, method="autodiff"), timing.pulsar.Mmat
    )
    with pytest.raises(ValueError, match="reference-point"):
        timing.jacobian({"F0": 0.2}, method="analytic")


def test_scan_exposes_residual_and_white_statistics():
    timing = TimingEvaluator(EvaluatorHost())
    scan = timing.scan("F0", [-0.2, 0.0, 0.2])

    assert scan.residuals.shape == (3, 4)
    assert scan.residual_deltas.shape == (3, 4)
    assert scan.rms.shape == (3,)
    assert scan.weighted_rms.shape == (3,)
    assert scan.white_chi2.shape == (3,)


def test_fit_is_immutable_and_recovers_linear_solution():
    host = EvaluatorHost()
    original = host.residuals.copy()
    timing = TimingEvaluator(host)

    supplied_initial = np.zeros(2)
    result = timing.fit(
        ["F0", "F1"], initial=supplied_initial, jacobian_method="reference"
    )

    assert result.converged
    np.testing.assert_allclose(result.best_fit.delta, [-0.2, 0.05], atol=1e-12)
    np.testing.assert_allclose(result.best_fit.residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(host.residuals, original)
    np.testing.assert_allclose(supplied_initial, 0.0)
    assert result.covariance.shape == (2, 2)
    assert not result.best_fit.delta.flags.writeable
    assert not result.covariance.flags.writeable
    with pytest.raises(TypeError):
        result.uncertainties["F0"] = 1.0


def test_unknown_parameter_and_wrong_vector_shape_are_clear():
    timing = TimingEvaluator(EvaluatorHost())
    with pytest.raises(KeyError, match="matches no fitpar"):
        timing.evaluate({"NOPE": 1.0})
    with pytest.raises(ValueError, match="length 2"):
        timing.evaluate([1.0])
