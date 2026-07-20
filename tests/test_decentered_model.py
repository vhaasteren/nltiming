"""Marginalized dynamic decentering: the ``decentered_model`` builder and its
geometry certifier (feature_marginalized_dynamic_decentering.md §5, §7, §11.2).

These are the engine-neutral nltiming unit tests (T-N1..T-N4). A discovery-
native pulsar and MARGINALIZED likelihood are built over the linear
``LinearizedJugEngine`` duck of ``test_joint_model``; the full real-pulsar
identity / cross-mode gate is a metapulsar integration test (T-M1..T-M3).

The exact-identity gate (T-N1) is the marginalized analogue of the joint
full-basis identity: because the engine is exactly linear, the target is
Gaussian in the sampled timing coordinate ``z`` and

    log_density(xi, eta) + 1/2 ||xi||^2  -  ln p_marg(eta)   ==   const

for every ``(xi, eta)``. Here ``ln p_marg(eta)`` is an INDEPENDENT dense
Woodbury oracle over the same ``y_t`` and ``W_s`` (it never reuses the joint
mode's frozen-``N0`` transport-internal formula, which is wrong under live
``C(eta)``); ``center=True`` fixes ``d(eta) = 0``.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("jug")
pytest.importorskip("discovery")
pytest.importorskip("numpyro")

import discovery as ds  # noqa: E402
from discovery import metamatrix as mm  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from nltiming import TimingInference, WhiteningConfig  # noqa: E402
from nltiming.metric import OneAffineLayerError  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402
from nltiming.sampling import numpyro as N  # noqa: E402
from nltiming import (  # noqa: E402
    GeometryThresholds,
    certify_decentered_geometry,
    read_geometry_report,
    write_geometry_report,
)

from test_joint_model import _Pulsar  # noqa: E402


@pytest.fixture(autouse=True)
def _metamath():
    """The marginal transport / PulsarLikelihood.N require the metamath path.

    Set it for every test in this module and leave it there (metamath is the
    discovery default these suites assume); do NOT reset to ``matrix``, which
    would pollute any file that runs after this one.
    """
    ds.config(kernels="metamath")


class _DiscoveryPulsar(_Pulsar):
    """The linear JUG duck plus the sky position ``makegp_fourier`` needs."""

    @property
    def pos(self):
        return np.array([1.0, 0.0, 0.0])


_NOISE = {"J1234+5678_efac": 1.0, "J1234+5678_log10_t2equad": -8.0}
_COMPONENTS = 3


def _ctx_and_likelihood(*, inference=None):
    """Build (ntm, ctx, marginalized likelihood) without the model.

    Sampled timing block is ``(F0, F1)`` with ``DM`` marginalized (delta-flat);
    a single variable red-noise GP supplies the live ``C(eta)`` dependence.
    """
    if inference is None:
        inference = TimingInference.groups(delta_flat=["DM"])
    ntm = NonLinearTimingModel(engines="jug", inference=inference, name="timing")
    ctx = ntm.for_pulsar(_DiscoveryPulsar())
    likelihood = ds.PulsarLikelihood(
        [
            ctx.pulsar.residuals,
            ds.makenoise_measurement_simple(ctx.pulsar, _NOISE),
            ds.makegp_fourier(ctx.pulsar, ds.powerlaw, _COMPONENTS, name="red_noise"),
            *ctx.discovery_signals(),
        ]
    )
    return ntm, ctx, likelihood


def _decentered_setup(*, center=True, inference=None, priors=None):
    """Build (ntm, ctx, marginalized likelihood, decentered model)."""
    priors = priors if priors is not None else _PRIORS
    ntm, ctx, likelihood = _ctx_and_likelihood(inference=inference)
    model = N.decentered_model(
        likelihood, ctx, center=center, priors=priors, fixed=_NOISE
    )
    return ntm, ctx, likelihood, model


_PRIORS = {
    r".*red_noise_log10_A.*": [-18.0, -12.0],
    r".*red_noise_gamma.*": [1.0, 7.0],
}
_ETA_MPE = {
    "J1234+5678_red_noise_log10_A": -14.5,
    "J1234+5678_red_noise_gamma": 3.2,
}


def _log_marginal_oracle(likelihood, transport, y_t):
    """Independent marginal log-evidence f(eta) over the same (y_t, W_s).

    Completing the square in the sampled timing ``z`` (unit-normal prior) against
    the marginalized Gaussian ``logL_marg(z, eta) = -1/2 (y_t - W_s z)^T C(eta)^-1
    (y_t - W_s z) - 1/2 log|2 pi C(eta)|`` gives

        f(eta) = 1/2 b^T A^-1 b + logL_marg(z=0, eta) - 1/2 log|A|

    with ``A = W_s^T C^-1 W_s + I`` and ``b = W_s^T C^-1 y_t``. Both ``(b, G)``
    (via the transport's own ``make_ks`` graph) and ``logL_marg(z=0)`` (via the
    kernel's ``make_kernelproduct`` — equal to ``likelihood.logL`` at zero sampled
    delay) come from the STABLE Woodbury inner form, never a dense inverse of the
    1e40-conditioned ``C``. The ``1/2 b^T A^-1 b`` and ``1/2 log|A|`` terms are
    computed here from the transport's ``G`` (independent of the transport's own
    ``apply``/``ldJ``), so the eta-dependence is genuinely cross-checked. The
    additive framework constant is eta-independent and cancels in the identity.
    """
    y_t = np.asarray(y_t, dtype=float)
    kp = mm.func(likelihood.N.make_kernelproduct(jnp.asarray(y_t)))
    k = int(transport.dimension)

    def f(eta):
        b, G = transport._ks(dict(eta))
        b = np.asarray(b, dtype=float)
        A = np.asarray(G, dtype=float) + np.eye(k)
        mu = np.linalg.solve(A, b)
        ld_A = np.linalg.slogdet(A)[1]
        logL0 = float(
            kp(params=dict(eta))
        )  # -1/2 y_t^T C^-1 y_t - 1/2 log|2piC| + const
        return 0.5 * float(b @ mu) + logL0 - 0.5 * ld_A

    return f


# ---------------------------------------------------------------------------
# T-N1: exact identity on the linear duck
# ---------------------------------------------------------------------------


def test_decentered_exact_identity_linear_duck():
    """T-N1: log_density(xi, eta) + 1/2||xi||^2 - ln p_marg(eta) is constant to
    1e-8 over 3 eta x 3 xi draws (center=True => d(eta)=0)."""
    _, ctx, likelihood, model = _decentered_setup(center=True)
    lin = ctx.linearization
    y_t = np.asarray(
        lin.transport_effective_residual(np.asarray(ctx.pulsar.residuals)), dtype=float
    )
    f_marg = _log_marginal_oracle(likelihood, model.transport, y_t)

    xi_site = model.xi_site
    k = int(model.transport.dimension)

    rng = np.random.default_rng(20260720)
    log10A = np.array([-15.5, -14.5, -13.5])
    gammas = np.array([2.0, 3.2, 5.0])

    residuals = []
    for a, g in zip(log10A, gammas):
        eta = {
            "J1234+5678_red_noise_log10_A": float(a),
            "J1234+5678_red_noise_gamma": float(g),
        }
        f_eta = f_marg(eta)
        for _ in range(3):
            xi = rng.standard_normal(k)
            point = {xi_site: jnp.asarray(xi), **eta}
            lp, _ = log_density(model, (), {}, point)
            residuals.append(float(lp) + 0.5 * float(xi @ xi) - f_eta)

    residuals = np.asarray(residuals)
    assert np.ptp(residuals) < 1e-8, residuals - residuals[0]


# ---------------------------------------------------------------------------
# T-N2: one-affine-layer guard
# ---------------------------------------------------------------------------


def test_decentered_requires_identity_static_layer():
    """T-N2: a WhiteningConfig static layer is rejected before the likelihood is
    touched (the one-affine-layer invariant)."""
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["DM"]),
        whitening=WhiteningConfig(),
        name="timing",
    )
    ctx = ntm.for_pulsar(_DiscoveryPulsar())

    class _FakeLikelihood:
        class logL:
            params: list = []

        N = None

    with pytest.raises(OneAffineLayerError):
        N.decentered_model(_FakeLikelihood(), ctx)


# ---------------------------------------------------------------------------
# T-N3: accounting and binding
# ---------------------------------------------------------------------------


def test_decentered_accounting_and_binding(monkeypatch):
    """T-N3: xi-site name, joint-site block key, delay keys carry no prior, the
    builder binds to lin.sampled_basis / transport_effective_residual (not raw
    residuals), fixed cannot pin a timing key, and empty plan.sampled raises."""
    _, ctx, likelihood = _ctx_and_likelihood()
    lin = ctx.linearization

    # Spy on transport_effective_residual to prove the builder routes the RAW
    # residuals through the linearization (digest/monkeypatch binding, T-N3).
    calls = []
    orig_ter = type(lin).transport_effective_residual

    def _spy(self, r):
        calls.append(np.asarray(r, dtype=float))
        return orig_ter(self, r)

    monkeypatch.setattr(type(lin), "transport_effective_residual", _spy)

    model = N.decentered_model(likelihood, ctx, priors=_PRIORS, fixed=_NOISE)

    # xi site name is the decentered-specific site.
    assert model.xi_site == f"{ctx.name_stem}_timing_xi"
    assert model.xi_site.endswith("_timing_xi")

    # Transport block key is exactly ctx.joint_site (single external block).
    assert list(model.transport.index) == [ctx.joint_site]

    # Delay keys are model-owned: they never appear as sampled hyperparameters.
    assert not (set(ctx.delay_keys) & set(model.hyper_sites))
    # ... and hyper_sites is sorted (D20).
    assert list(model.hyper_sites) == sorted(model.hyper_sites)

    # The builder called transport_effective_residual on the RAW residuals and
    # bound the transport to that output (not the raw residuals) and to
    # sampled_basis.
    assert calls, "decentered_model did not route residuals through the linearization"
    assert np.allclose(calls[-1], np.asarray(ctx.pulsar.residuals, dtype=float))
    y_t = np.asarray(orig_ter(lin, np.asarray(ctx.pulsar.residuals)), dtype=float)
    assert np.allclose(np.asarray(model.transport._y, dtype=float), y_t)
    assert np.allclose(
        np.asarray(model.transport._W, dtype=float),
        np.asarray(lin.sampled_basis, dtype=float),
    )

    # fixed cannot pin a model-owned timing (delay) key.
    delay_key = ctx.delay_keys[0]
    with pytest.raises(ValueError, match="cannot pin"):
        N.decentered_model(likelihood, ctx, fixed={delay_key: 0.0})

    # Empty plan.sampled (everything marginalized) raises.
    ntm2 = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["F0", "F1", "DM"]),
        name="timing",
    )
    ctx2 = ntm2.for_pulsar(_DiscoveryPulsar())

    class _FakeLikelihood:
        class logL:
            params: list = []

        N = None

    with pytest.raises(ValueError, match="sampled"):
        N.decentered_model(_FakeLikelihood(), ctx2)


def test_decentered_model_fingerprint_is_stable_and_scoped():
    """model_fingerprint() is deterministic and changes with center."""
    _, _, _, model = _decentered_setup(center=True)
    fp = model.model_fingerprint()
    assert isinstance(fp, str) and len(fp) == 64
    assert model.model_fingerprint() == fp  # deterministic

    _, _, _, model_uncentered = _decentered_setup(center=False)
    assert model_uncentered.model_fingerprint() != fp


def test_decentered_init_values_zeroes_the_xi_site():
    """T-N3 (init): decentered_init_values keys the xi site with a zero vector of
    the transport dimension."""
    _, ctx, _, model = _decentered_setup()
    init = N.decentered_init_values(ctx, model.transport)
    assert set(init) == {model.xi_site}
    vec = np.asarray(init[model.xi_site])
    assert vec.shape == (int(model.transport.dimension),)
    assert np.all(vec == 0.0)


# ---------------------------------------------------------------------------
# T-N4: certifier wiring
# ---------------------------------------------------------------------------


def test_certify_decentered_geometry_passes_on_linear_duck(tmp_path):
    """T-N4: the decentered certifier passes at the hyper MPE on the exact-linear
    duck, its fingerprints bind to ctx, and the report round-trips through I/O."""
    _, ctx, _, model = _decentered_setup()

    report = certify_decentered_geometry(
        model, ctx, hyper_points=[_ETA_MPE], thresholds=GeometryThresholds()
    )
    assert report.passed, report.failures
    # Linear engine => exact: zero remainder, unit xi-Hessian, no cross term.
    assert report.max_residual_remainder_rms < 1e-6
    assert abs(report.xi_hessian_eigen_min - 1.0) < 1e-6
    assert abs(report.xi_hessian_eigen_max - 1.0) < 1e-6
    assert report.max_xi_eta_cross_operator_norm < 1e-6
    assert report.max_conditional_identity_spread < 1e-6

    # The report binds to the certified context and to the geometry structure
    # digest (xi/hyper/dim/index/linearization). That digest is a DIFFERENT
    # schema from the model's own nlt-decentered-model-v1 fingerprint
    # (ctx + transport + free hypers + center) — assert both explicitly rather
    # than a tautology.
    from nltiming.geometry import _model_fingerprint as _geom_model_fingerprint

    assert report.context_fingerprint == ctx.fingerprint()
    assert report.model_fingerprint == _geom_model_fingerprint(model, ctx)
    model_fp = model.model_fingerprint()
    assert isinstance(model_fp, str) and len(model_fp) == 64
    assert model.model_fingerprint() == model_fp  # deterministic
    assert report.model_fingerprint != model_fp  # distinct schemas, by design

    # Standalone report persistence round-trips.
    out = tmp_path / "j1234_decentered_geometry"
    write_geometry_report(report, out)
    loaded = read_geometry_report(out)
    assert loaded.passed == report.passed
    assert loaded.context_fingerprint == report.context_fingerprint
    assert loaded.model_fingerprint == report.model_fingerprint
