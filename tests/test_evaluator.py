"""TimingEvaluator jacobian method table."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming import TimingEvaluator
from nltiming.protocols import GaugeProvenance


class _LinearBackend:
    backend_name = "jug"
    fitpars = ("F0", "Offset")
    native_units = {"F0": "Hz", "Offset": "s"}

    def __init__(self, design):
        self._design = np.asarray(design, dtype=float)

    def reference_theta(self):
        return np.asarray([10.0, 0.0])

    def reference_theta_exact(self):
        return {"F0": "10.0", "Offset": "0.0"}

    def residual_delta(self, delta_theta):
        return -(self._design @ np.asarray(delta_theta))

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        return -(jnp.asarray(self._design) @ jnp.asarray(delta_theta))

    def residual_jacobian(self):
        return -self._design

    def design_matrix(self, params=None):
        return self._design

    def precision_critical_fitpars(self):
        return frozenset()

    def gauge_provenance(self):
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self):
        return False


class _Pulsar:
    name = "J1234+5678"
    fitpars = ["F0", "Offset"]

    def __init__(self):
        self.Mmat = np.asarray(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=float
        )
        self.toas = np.arange(4.0)
        self.toaerrs = np.ones(4)
        self.freqs = np.full(4, 1400.0)
        self.flags = {"pta": np.asarray(["a"] * 4)}
        self.backend_flags = np.asarray(["a"] * 4)
        self.residuals = np.zeros(4)
        self._backend = _LinearBackend(self.Mmat)

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


def test_reference_returns_minus_m():
    timing = TimingEvaluator(_Pulsar())
    J = timing.jacobian(method="reference")
    np.testing.assert_allclose(J, -_Pulsar().Mmat)


def test_exact_returns_residual_jacobian():
    timing = TimingEvaluator(_Pulsar())
    J = timing.jacobian(method="exact")
    np.testing.assert_allclose(J, timing.engine.residual_jacobian())


def test_analytic_alias_removed():
    timing = TimingEvaluator(_Pulsar())
    with pytest.raises(ValueError, match="exact|reference|autodiff"):
        timing.jacobian(method="analytic")


def test_auto_prefers_autodiff_when_jax():
    pytest.importorskip("jax")
    timing = TimingEvaluator(_Pulsar())
    assert timing.capabilities.autodiff_jacobian
    assert timing.capabilities.exact_jacobian
    J = timing.jacobian(method="auto")
    np.testing.assert_allclose(J, timing.jacobian(method="autodiff"), atol=1e-12)


def test_exact_rejects_nonzero_at():
    timing = TimingEvaluator(_Pulsar())
    with pytest.raises(ValueError, match="reference-point"):
        timing.jacobian({"F0": 0.1}, method="exact")
