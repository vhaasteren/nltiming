"""Fixed-expansion timing linearization record (§5.2, §5.3, §14.3)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

pytest.importorskip("jug")

from nltiming import TimingInference  # noqa: E402
from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402


class _Pulsar:
    """Linear JAX-differentiable pulsar (F0, F1, DM)."""

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
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "100.0", "F1": "-1e-15", "DM": "10.0"},
        )
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
    def state_id(self): return "lin-token"
    def pint_model(self): return None
    def timing_engine(self, engines="jug", **kwargs): return self._backend


def _ctx(**kw):
    ntm = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(), name="timing", **kw
    )
    return ntm.for_pulsar(_Pulsar(), condition=False)


def test_default_expansion_is_engine_reference_delta_zero():
    lin = _ctx().linearization
    assert lin.source == "engine_reference"
    assert lin.sampled_names == ("F0", "F1", "DM")
    np.testing.assert_allclose(lin.delta_expansion, np.zeros(3))
    assert np.all(np.isfinite(lin.z_expansion))


def test_expansion_waveform_and_autodiff_basis_match_finite_difference():
    ctx = _ctx()
    lin = ctx.linearization
    space, engine, plan = ctx.space, ctx.engine, ctx.plan
    idx = np.asarray(plan.idx_sampled, dtype=int)
    nfit = len(plan.fitpars)

    def d_np(z):
        delta = np.asarray(space.delta_from_z(z, np), dtype=float)
        full = np.zeros(nfit)
        full[idx] = delta
        return -np.asarray(engine.residual_delta(full), dtype=float)

    z_e = np.asarray(lin.z_expansion, dtype=float)
    np.testing.assert_allclose(lin.sampled_waveform_expansion, d_np(z_e), atol=1e-12)
    h = 1e-4
    fd = np.stack(
        [(d_np(z_e + h * np.eye(3)[j]) - d_np(z_e - h * np.eye(3)[j])) / (2 * h)
         for j in range(3)], axis=1)
    np.testing.assert_allclose(lin.sampled_basis, fd, rtol=1e-6, atol=1e-9)


def test_sampled_basis_matches_local_timing_block_at_engine_reference():
    ctx = _ctx()
    blk = ctx.local_timing_block()
    np.testing.assert_allclose(ctx.linearization.sampled_basis, blk.basis,
                               rtol=1e-10, atol=1e-12)


def test_local_timing_block_projects_refined_expansion():
    # local_timing_block is a projection of ctx.linearization, so a refined
    # expansion is reflected (z_ref tracks the expansion), not silently at delta=0.
    base = _ctx()
    z0 = base.local_timing_block().z_ref
    refined = base.with_expansion(delta={"F0": 3e-13, "F1": 0.0, "DM": 5e-4})
    blk = refined.local_timing_block()
    np.testing.assert_allclose(blk.z_ref, refined.linearization.sampled_z_expansion)
    np.testing.assert_allclose(blk.basis, refined.linearization.sampled_basis)
    assert not np.allclose(blk.z_ref, z0)  # genuinely moved off the reference


def test_joint_transport_uses_expansion_effective_residual(monkeypatch):
    # The joint transport's reference residual is the linearization's effective
    # residual, so a non-zero expansion changes what the transport centers on.
    import discovery as ds

    ds.config(kernels="metamath")
    from discovery import transport as dst
    from nltiming.sampling.numpyro import build_joint_transport

    captured = {}
    real_transport = dst.Transport

    def spy(blocks, **kw):
        captured["reference_residual"] = kw.get("reference_residual")
        return real_transport(blocks, **kw)

    monkeypatch.setattr(dst, "Transport", spy)

    class _Likelihood:
        sampled_gps: list = []

    # A uniform DM prior gives a non-affine (prior_pit) chart, so delta(z) is
    # nonlinear and the effective residual genuinely departs from y off-zero
    # (an affine-normal chart on a linear engine would leave it equal to y).
    from nltiming import priors as P
    from nltiming.coordinates import TimingCoordinatePolicy

    ntm = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(),
        priors={"DM": P.delta_uniform(-1e-2, 1e-2)},
        coordinate_policy=TimingCoordinatePolicy(nonaffine_identically_linear="ignore"),
        name="timing")
    base = ntm.for_pulsar(_Pulsar(), condition=False)
    refined = base.with_expansion(delta={"F0": 3e-13, "F1": 0.0, "DM": 5e-3})
    build_joint_transport(_Likelihood(), refined, center=False)

    y = np.asarray(refined.pulsar.residuals, dtype=float)
    expected = refined.linearization.transport_effective_residual(y)
    np.testing.assert_allclose(
        np.asarray(captured["reference_residual"]), expected, atol=1e-12)
    assert not np.allclose(expected, y)  # nonlinear chart: genuinely moved
    ds.config(kernels="matrix")


def test_effective_residual_reconstructs_local_surrogate_with_correct_sign():
    # Non-zero expansion so the W_s @ z_e term is exercised (§14.3 note).
    base = _ctx()
    delta = {"F0": 3.0e-13, "F1": 2.0e-21, "DM": 5.0e-4}
    ctx = base.with_expansion(delta=delta, source="explicit_delta")
    lin = ctx.linearization
    y = np.asarray(ctx.pulsar.residuals, dtype=float)

    z_e = np.asarray(lin.z_expansion, dtype=float)
    assert np.linalg.norm(z_e) > 0  # genuinely off-zero
    expected = y - lin.sampled_waveform_expansion + lin.sampled_basis @ z_e
    np.testing.assert_allclose(lin.transport_effective_residual(y), expected, atol=1e-12)
    # The W_s @ z_e term is present (a zero-only test would miss it).
    assert not np.allclose(
        lin.transport_effective_residual(y), y - lin.sampled_waveform_expansion)


def test_with_expansion_is_immutable_and_before_conditioning_only():
    base = _ctx()
    ctx = base.with_expansion(delta={"F0": 1e-13, "F1": 0.0, "DM": 1e-4})
    assert ctx.linearization.source == "explicit_delta"
    # arrays are read-only
    with pytest.raises(ValueError):
        ctx.linearization.sampled_basis[0, 0] = 1.0
    # conditioned context rejects re-expansion
    conditioned = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(), name="timing"
    ).for_pulsar(_Pulsar())  # condition=True default
    with pytest.raises(ValueError, match="before static conditioning"):
        conditioned.with_expansion(delta={"F0": 0.0, "F1": 0.0, "DM": 0.0})


def test_fingerprint_changes_with_expansion():
    a = _ctx().linearization.fingerprint()
    b = _ctx().with_expansion(delta={"F0": 1e-13, "F1": 0.0, "DM": 1e-4}).linearization
    assert a != b.fingerprint()


def test_with_expansion_rejects_wrong_axis_set():
    ctx = _ctx()
    with pytest.raises(ValueError, match="proper axes"):
        ctx.with_expansion(delta={"F0": 0.0})  # missing F1, DM
