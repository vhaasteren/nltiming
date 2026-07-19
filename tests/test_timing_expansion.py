"""Expansion-spec resolution and prior-interior guard (§5.3, §14.3)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("jug")

from nltiming import (  # noqa: E402
    ExpansionOutsidePriorInteriorError,
    TimingExpansionSpec,
    TimingInference,
)
from nltiming import priors as P  # noqa: E402
from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402


class _Pulsar:
    def __init__(self):
        self.name = "J1234+5678"
        self.fitpars = ("F0", "F1", "DM")
        n = 12
        t = np.linspace(0.0, 1.0, n)
        design = np.column_stack([np.ones(n), t - 0.5, np.sin(3.0 * t)])
        self._toas = t * 3.15e7 + 5.3e4
        self._residuals = 1e-6 * np.sin(5.0 * t)
        self._toaerrs = np.full(n, 1.0e-6)
        self._freqs = np.full(n, 1400.0)
        self._backend_flags = np.array(["demo"] * n, dtype="U8")
        self._flags = {"pta": self._backend_flags}
        model = LinearModel.from_design(
            fitpars=self.fitpars, design=design,
            theta_exact={"F0": "100.0", "F1": "-1e-15", "DM": "10.0"})
        self._backend = LinearizedJugEngine.from_linear_model(model)

    @property
    def toas(self): return self._toas
    @property
    def residuals(self): return self._residuals
    @property
    def toaerrs(self): return self._toaerrs
    @property
    def freqs(self): return self._freqs
    @property
    def Mmat(self): return self._backend.design_matrix()
    @property
    def flags(self): return self._flags
    @property
    def backend_flags(self): return self._backend_flags
    def state_id(self): return "exp-token"
    def pint_model(self): return None
    def timing_engine(self, engines="jug", **kwargs): return self._backend


def _model(**kw):
    return NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(), name="timing", **kw)


def test_default_is_engine_reference():
    lin = _model().for_pulsar(_Pulsar(), condition=False).linearization
    assert lin.source == "engine_reference"
    np.testing.assert_allclose(lin.delta_expansion, np.zeros(3))


def test_explicit_prior_center_expansion_is_opt_in():
    lin = _model(expansion=TimingExpansionSpec.prior_center()).for_pulsar(
        _Pulsar(), condition=False).linearization
    assert lin.source == "prior_center"
    # z_e = 0 at the prior center by construction.
    np.testing.assert_allclose(lin.z_expansion, np.zeros(3), atol=1e-10)


def test_engine_reference_on_prior_boundary_raises_without_clipping():
    # A uniform DM prior that excludes delta=0 (engine reference) must raise,
    # not silently clip into the box.
    model = _model(priors={"DM": P.delta_uniform(1.0e-3, 2.0e-3)})
    with pytest.raises(ExpansionOutsidePriorInteriorError, match="DM"):
        model.for_pulsar(_Pulsar(), condition=False)


def test_prior_center_recovers_when_engine_reference_is_outside():
    # The same excluding prior is fine under prior_center (z_e = 0 is interior).
    model = _model(
        priors={"DM": P.delta_uniform(1.0e-3, 2.0e-3)},
        expansion=TimingExpansionSpec.prior_center(),
    )
    lin = model.for_pulsar(_Pulsar(), condition=False).linearization
    assert lin.source == "prior_center"


def test_large_affine_normal_expansion_is_not_boundary_failure():
    # Gaussian (affine_normal) charts are unbounded: a large explicit delta is
    # representable and never a boundary failure.
    ctx = _model().for_pulsar(_Pulsar(), condition=False)
    big = ctx.with_expansion(delta={"F0": 5.0e-11, "F1": 0.0, "DM": 3.0e-2})
    assert np.all(np.isfinite(big.linearization.z_expansion))
    # 3e-2 DM delta at ~1e-3 std -> |z| well above any PIT interior limit, yet fine.
    assert np.max(np.abs(big.linearization.z_expansion)) > 5.0


def test_refinement_improves_exact_objective_and_re_expands():
    import jax.numpy as jnp

    from nltiming.expansion import refine_timing_expansion

    ctx = _model().for_pulsar(_Pulsar(), condition=False)
    z_star = jnp.array([1.5, -0.8, 0.6])

    def obj(z):  # exact conditional target incl. its own 0.5 z@z-style curvature
        return 0.5 * jnp.sum((z - z_star) ** 2)

    res = refine_timing_expansion(ctx, negative_log_target_z=obj)
    assert res.converged
    np.testing.assert_allclose(res.z_final, np.asarray(z_star), atol=1e-4)
    assert res.objective_final <= res.objective_initial
    assert res.context.linearization.source == "refined"
    np.testing.assert_allclose(
        res.context.linearization.z_expansion, np.asarray(z_star), atol=1e-4)


def test_refinement_returns_original_context_on_nonconvergence():
    import jax.numpy as jnp

    from nltiming.expansion import refine_timing_expansion

    ctx = _model().for_pulsar(_Pulsar(), condition=False)

    def bad(z):  # gradient is ones everywhere -> never meets gtol
        return jnp.sum(z)

    res = refine_timing_expansion(ctx, negative_log_target_z=bad, max_iterations=3)
    assert not res.converged
    assert res.context is ctx


def test_refinement_has_no_jaxopt_dependency():
    import inspect

    import nltiming.expansion as ex

    src = inspect.getsource(ex)
    assert "import jaxopt" not in src
    assert "from jaxopt" not in src
