"""Joint full-basis timing (Track J / §6): three-way partition, the local
timing block, and the joint model's one-affine-layer guard.

These are the engine-neutral unit tests. The full joint NumPyro run (density
exactness, whitening geometry, decode, run manifest) is a pulsar-integration
test in the metapulsar repo, where a discovery-native pulsar and likelihood are
available.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# LinearizedJugEngine import requires the jug extra.
pytest.importorskip("jug")

from nltiming import TimingInference, WhiteningConfig  # noqa: E402
from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402


class _Pulsar:
    """A linear JAX-differentiable pulsar duck (F0, F1, DM)."""

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
        return "joint-token"

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


def _joint_ctx():
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        name="timing",
    )
    return ntm, ntm.for_pulsar(_Pulsar())


def test_sample_all_plan_via_model():
    _, ctx = _joint_ctx()
    # Joint full-basis: every timing axis sampled, nothing marginalized.
    assert ctx.plan.sampled == ("F0", "F1", "DM")
    assert ctx.plan.marginalized_delta == ()
    assert ctx.plan.marginalized_z == ()
    assert ctx.marginalized == ()
    assert ctx.sampled_all == ctx.sampled
    # A linear JUG engine declares every fitpar identically linear; DM is also
    # in the fallback registry.
    assert set(ctx.identically_linear) == {"F0", "F1", "DM"}
    assert "fallback" in ctx.linearity_sources_for("DM")
    assert "engine" in ctx.linearity_sources_for("F0")


def test_chart_records_tag_proper_axes_affine_normal():
    _, ctx = _joint_ctx()
    summary = {d["name"]: d for d in ctx.chart_summary()}
    assert set(summary) == {"F0", "F1", "DM"}
    # The linear JUG engine declares every axis identically linear -> Gaussian
    # delta prior -> affine_normal chart on every proper axis.
    assert all(d["prior_chart"] == "affine_normal" for d in summary.values())
    assert ctx.plan.axis("DM").prior is not None
    assert ctx.plan.axis("DM").prior.family == "normal"
    assert ctx.nonaffine_identically_linear == ()


def test_nonlinear_axis_uses_prior_pit_chart():
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        identically_linear=[],  # assert none linear -> uniform cheat prior -> PIT
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())
    charts = {d["name"]: d["prior_chart"] for d in ctx.chart_summary()}
    assert charts == {"F0": "prior_pit", "F1": "prior_pit", "DM": "prior_pit"}


def test_uniform_override_on_identically_linear_axis_is_reported_nonaffine():
    from nltiming import priors as P
    from nltiming.coordinates import NonAffineIdenticallyLinearWarning

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        priors={"DM": P.delta_uniform(-1e-3, 1e-3)},
        name="timing",
    )
    with pytest.warns(NonAffineIdenticallyLinearWarning, match="DM"):
        ctx = ntm.for_pulsar(_Pulsar())
    # DM is identically linear but the explicit uniform prior makes its chart PIT.
    assert ctx.plan.axis("DM").prior_chart == "prior_pit"
    assert "DM" in ctx.nonaffine_identically_linear


def test_nonaffine_identically_linear_warning_can_be_suppressed(recwarn):
    from nltiming import priors as P
    from nltiming.coordinates import (
        NonAffineIdenticallyLinearWarning,
        TimingCoordinatePolicy,
    )

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        priors={"DM": P.delta_uniform(-1e-3, 1e-3)},
        coordinate_policy=TimingCoordinatePolicy(nonaffine_identically_linear="ignore"),
        name="timing",
    )
    ntm.for_pulsar(_Pulsar())
    assert not [w for w in recwarn if issubclass(
        w.category, NonAffineIdenticallyLinearWarning)]


def test_prior_on_delta_flat_axis_raises_and_names_z_prior_remedy():
    from nltiming import priors as P

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["DM"]),
        priors={"DM": P.delta_uniform(-1e-3, 1e-3)},
        prior_override_policy="warn",
        name="timing",
    )
    with pytest.raises(ValueError, match="delta-flat.*z-prior"):
        ntm.for_pulsar(_Pulsar())


def test_whitening_none_is_identity_static_layer():
    from nltiming.metric import assert_static_layer_identity

    _, ctx = _joint_ctx()  # whitening omitted -> None -> identity
    assert ctx.model.static_layer == "identity"
    assert ctx.coord == "z"
    assert_static_layer_identity(ctx.space)  # must not raise


def test_whitening_config_is_nonidentity_static_layer():
    from nltiming.metric import OneAffineLayerError, assert_static_layer_identity

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        whitening=WhiteningConfig(),
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())
    assert ctx.model.static_layer == "whitening"
    assert ctx.coord == "x"
    with pytest.raises(OneAffineLayerError):
        assert_static_layer_identity(ctx.space)


def test_transform_keyword_is_rejected():
    with pytest.raises(TypeError):
        NonLinearTimingModel(
            engines="jug",
            transform="none",
            inference=TimingInference.sample_all(),
        )


def test_z_prior_context_builds_with_wm_block_and_discovery_gp():
    """A marginalize_z_prior axis is live for Discovery: the context builds, the
    linearization carries a W_m block, and discovery_signals emits the proper
    unit-normal (standard-normal) GP for it."""
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(z_prior=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())
    assert ctx.plan.marginalized_z == ("DM",)
    assert ctx.plan.sampled == ("F0", "F1")
    assert ctx.plan.proper == ("F0", "F1", "DM")
    lin = ctx.linearization
    assert lin.marginalized_z_names == ("DM",)
    assert lin.marginalized_z_basis.shape[1] == 1
    assert lin.sampled_basis.shape[1] == 2
    # the z subspace is a real, separate ParameterSpace
    assert ctx.marginal_z_space.names == ("DM",)
    assert ctx.space.names == ("F0", "F1")
    # discovery emits a standard-normal GP for the z-prior block
    sigs = ctx.discovery_signals()
    gpnames = [getattr(s, "gpname", "") for s in sigs]
    assert any("zprior" in (gn or "") for gn in gpnames)


def test_discovery_signals_joint_rejects_any_marginalization():
    # joint=True must sample every timing direction; a z-prior (or delta-flat)
    # marginal block would be the wrong model, so it fails loudly.
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(z_prior=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())
    with pytest.raises(ValueError, match="sample_all"):
        ctx.discovery_signals(joint=True)


def test_z_prior_enterprise_assembly_builds_and_evaluates():
    """Enterprise consumes the z-prior block: the exact sampled delay (z-marg
    fixed), the proper unit-normal W_m GP, and the fixed c_m intercept assemble
    into a full PTA whose likelihood evaluates (DM analytically marginalized)."""
    from enterprise.signals import parameter, signal_base, white_signals

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(z_prior=["DM"]),
        whitening=WhiteningConfig(),
        name="timing",
    )
    white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
    pta = signal_base.PTA([(white + ntm.enterprise_signal())(_Pulsar())])
    # only F0/F1 are sampled; DM is marginalized via the W_m GP (no DM param).
    assert not any("DM" in p for p in pta.param_names)
    x0 = np.hstack(
        [np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
    assert np.isfinite(pta.get_lnlikelihood(x0))
    assert np.isfinite(pta.get_lnprior(x0))


def test_z_prior_enterprise_can_sample_wm_coefficients():
    """``enterprise_signal(sample_z_coefficients=True)`` promotes the W_m block
    from an integrated GP to a sampled ``GPCoefficients`` parameter whose prior
    is the exact unit normal ``-1/2 c^T c`` and whose delay is ``W_m @ c``."""
    from enterprise.signals import parameter, signal_base, white_signals

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(z_prior=["DM"]),
        whitening=WhiteningConfig(),
        name="timing",
    )
    white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
    pta = signal_base.PTA(
        [(white + ntm.enterprise_signal(sample_z_coefficients=True))(_Pulsar())]
    )
    names = list(pta.param_names)
    coeff = [i for i, n in enumerate(names) if "zprior_coefficients" in n]
    # exactly one z-marginalized axis (DM) -> one sampled coefficient.
    assert len(coeff) == 1

    # The coefficient prior is the exact unit normal: moving c 0 -> a shifts the
    # log-prior by -1/2 a^2, independent of the other (whitened x) coordinates.
    base = np.zeros(len(names))
    lp0 = pta.get_lnprior(base)
    for a in (1.0, 2.0, -1.5):
        x = base.copy()
        x[coeff[0]] = a
        assert np.isclose(pta.get_lnprior(x) - lp0, -0.5 * a * a, atol=1e-9)

    # The sampled-coefficient likelihood evaluates (the W_m @ c delay path).
    assert np.isfinite(pta.get_lnlikelihood(base))


def test_local_timing_block_is_negative_autodiff_jacobian():
    _, ctx = _joint_ctx()
    blk = ctx.local_timing_block()
    assert blk.dimension == 3
    assert blk.names == ("F0", "F1", "DM")
    assert blk.prior_precision == 1.0
    assert blk.joint_site == ctx.joint_site

    # W_z == -∂(residual_delta_jax(δ(z)))/∂z at z_ref (the exact engine path).
    idx = jnp.asarray(ctx.plan.idx_sampled)
    nfit = len(ctx.plan.fitpars)
    z_ref = jnp.asarray(blk.z_ref)

    def residual_of_z(z):
        delta = ctx.space.delta_from_z(z, jnp)
        full = jnp.zeros((nfit,)).at[idx].set(delta)
        return ctx.engine.residual_delta_jax(full)

    W_expected = -np.asarray(jax.jacfwd(residual_of_z)(z_ref))
    assert np.allclose(blk.basis, W_expected, rtol=1e-10, atol=1e-12)


def test_cross_term_sign(monkeypatch):
    """On a toy where a GP column equals the raw timing waveform ``M·J``, the
    transport's timing↔GP cross block is MINUS the GP–GP block iff W_z carries
    the ``-`` sign (§6.3). This is the single most load-bearing sign in J1.
    """
    import discovery as ds

    ds.config(kernels="metamath")
    from discovery import transport as dst

    _, ctx = _joint_ctx()
    blk = ctx.local_timing_block()  # W_z = -(M·J)
    F_gp = -np.asarray(blk.basis)  # a GP column bank equal to +(M·J)

    ref = dst.reference_noise(ctx.pulsar)  # N0 = toaerrs**2
    t_block = dst.array_block(
        blk.basis,
        index={ctx.joint_site: slice(0, blk.dimension)},
        conditioner_precision=1.0,
        name="timing",
    )
    g_block = dst.array_block(
        F_gp,
        index={"gp": slice(0, F_gp.shape[1])},
        conditioner_precision=1.0,
        name="gp",
    )
    tr = dst.Transport([t_block, g_block], reference_noise=ref, center=False)

    G0 = np.asarray(tr._G0)
    k = blk.dimension
    cross = G0[:k, k:]
    gpgp = G0[k:, k:]
    assert np.allclose(cross, -gpgp, rtol=1e-10, atol=1e-12)

    ds.config(kernels="matrix")


def test_joint_model_requires_identity_static_layer():
    """joint_model rejects a conditioned (non-identity) whitening layer before
    it touches the likelihood (the one-affine-layer invariant, §5.5)."""
    from nltiming.metric import OneAffineLayerError
    from nltiming.sampling import numpyro as N

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        whitening=WhiteningConfig(),
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())  # conditioned => non-identity (C, c)

    class _FakeLikelihood:
        sampled_gps: list = []

        class clogL:
            params: list = []

    with pytest.raises(OneAffineLayerError):
        N.joint_model(_FakeLikelihood(), ctx)
