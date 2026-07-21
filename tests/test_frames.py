"""PR-1 (§12.1): the single EngineDeltaMap seam is bit-identical to the
legacy per-call scatter idiom it replaces, in both modes and both backends."""

import numpy as np
import pytest

from nltiming.coordinates import TimingCoordinatePolicy
from nltiming.frames import EngineDeltaMap
from nltiming.inference import (
    ResolvedTimingAxis,
    TimingInference,
    TimingParameterPlan,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _plan(fitpars, dispositions):
    axes = tuple(
        ResolvedTimingAxis(
            name=name, fitpar_index=i, disposition=disp, linearity_sources=()
        )
        for i, (name, disp) in enumerate(zip(fitpars, dispositions))
    )
    return TimingParameterPlan(
        fitpars=tuple(fitpars),
        axes=axes,
        inference=TimingInference.sample_all(),
        coordinate_policy=TimingCoordinatePolicy(),
    )


class _Lin:
    def __init__(self, delta_expansion):
        self.delta_expansion = np.asarray(delta_expansion, dtype=float)


def test_engine_delta_map_matches_legacy_scatter():
    fitpars = ["A", "B", "C", "D", "E"]
    disp = [
        "sample",
        "marginalize_z_prior",
        "sample",
        "marginalize_delta_flat",
        "marginalize_z_prior",
    ]
    plan = _plan(fitpars, disp)
    nfit = len(fitpars)
    # proper order (sample ∪ z-marg, fitpar order): A(0), B(1), C(2), E(4).
    delta_expansion = [0.11, 0.22, 0.33, 0.44]
    lin = _Lin(delta_expansion)

    # --- proper mode: input covers every proper axis; delta-flat D stays 0 ---
    m_proper = EngineDeltaMap.for_proper(plan, ())
    proper_vals = np.array([1.0, 2.0, 3.0, 5.0])  # A, B, C, E
    oracle_p = np.zeros(nfit)
    oracle_p[[0, 1, 2, 4]] = proper_vals
    got_np = np.asarray(m_proper.full_engine_delta(proper_vals, np))
    got_jax = np.asarray(m_proper.full_engine_delta(jnp.asarray(proper_vals), jnp))
    assert np.array_equal(got_np, oracle_p)
    assert np.array_equal(got_jax, oracle_p)

    # --- sampled mode: input is sampled axes only; z-marg pinned at z_m,e;
    #     delta-flat stays 0 ---
    m_sampled = EngineDeltaMap.for_sampled(plan, (), lin)
    sampled_vals = np.array([7.0, 9.0])  # A, C
    oracle_s = np.zeros(nfit)
    oracle_s[[0, 2]] = sampled_vals
    # B is proper position 1, E is proper position 3 in delta_expansion.
    oracle_s[[1, 4]] = [delta_expansion[1], delta_expansion[3]]
    got_np = np.asarray(m_sampled.full_engine_delta(sampled_vals, np))
    got_jax = np.asarray(m_sampled.full_engine_delta(jnp.asarray(sampled_vals), jnp))
    assert np.array_equal(got_np, oracle_s)
    assert np.array_equal(got_jax, oracle_s)


# ---------------------------------------------------------------------------
# Backend bit-identity: the same seam feeding a real JAX engine reproduces the
# legacy scatter + residual_delta exactly, at random sampling points.
# ---------------------------------------------------------------------------

pytest.importorskip("jug")

from nltiming import TimingInference as _TI  # noqa: E402
from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402


class _LinearPulsar:
    """A linear JAX-differentiable pulsar duck (F0, F1, DM)."""

    def __init__(self):
        self.name = "J0000+0000"
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
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "100.0", "F1": "-1e-15", "DM": "10.0"},
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

    @property
    def toas(self):
        return self._toas

    @property
    def residuals(self):
        return self._residuals

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return self._freqs

    @property
    def Mmat(self):
        return self._backend.design_matrix()

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def state_id(self):
        return "frames-token"

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


def _legacy_full_delta(ctx, sampled_vals):
    """The pre-refactor scatter: sampled at their slots, z-marg pinned at
    z_m,e, delta-flat at zero."""
    nfit = len(ctx.plan.fitpars)
    full = np.zeros(nfit, dtype=float)
    for i, col in enumerate(ctx.plan.idx_sampled):
        full[col] = sampled_vals[i]
    proper_axes = [
        a for a in ctx.plan.axes if a.disposition in ("sample", "marginalize_z_prior")
    ]
    for pos, a in enumerate(proper_axes):
        if a.disposition == "marginalize_z_prior":
            full[a.fitpar_index] = float(ctx.linearization.delta_expansion[pos])
    return full


def test_likelihood_bit_identity_after_seam_migration():
    # Mixed plan exercises all three dispositions through the seam.
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=_TI.groups(z_prior=["F1"], delta_flat=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(_LinearPulsar())
    engine = ctx.engine
    emap = ctx.engine_delta_map
    k = len(ctx.plan.sampled)
    assert ctx.plan.sampled == ("F0",)
    assert ctx.plan.marginalized_z == ("F1",)
    assert ctx.plan.marginalized_delta == ("DM",)

    rng = np.random.default_rng(20260721)
    for _ in range(10):
        vals = rng.normal(size=k) * 1e-9
        legacy = _legacy_full_delta(ctx, vals)
        via_seam = np.asarray(emap.full_engine_delta(jnp.asarray(vals), jnp))
        assert np.array_equal(via_seam, legacy)
        # The delay the likelihood emits is -residual_delta_jax(full_delta).
        r_seam = np.asarray(
            engine.residual_delta_jax(emap.full_engine_delta(jnp.asarray(vals), jnp))
        )
        r_legacy = np.asarray(engine.residual_delta_jax(legacy))
        assert np.array_equal(r_seam, r_legacy)
