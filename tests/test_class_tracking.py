"""White-noise tracking in the joint transport (`class_tracking_reference`).

The duck pulsar of `test_decentered_model` (linear JUG engine, 12 TOAs, one backend)
with a free-EFAC measurement kernel; ECORR is absent (no repeated epochs),
so the tracker is the diagonal class Gram. Density parity with the frozen
model at the bake point and the per-pulsar (no broadcast) contract are the
two things this module pins.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import discovery as ds  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from nltiming import TimingInference  # noqa: E402
from nltiming.nonlinear_timing_model import TimingSpec  # noqa: E402
import nltiming.sampling as nlts  # noqa: E402

from test_decentered_model import _DiscoveryPulsar  # noqa: E402


@pytest.fixture(autouse=True)
def _metamath():
    ds.config(kernels="metamath")


_COMPONENTS = 3
_PRIORS = {
    r".*red_noise_log10_A.*": [-18.0, -12.0],
    r".*red_noise_gamma.*": [1.0, 7.0],
    r".*_efac$": [0.3, 10.0],
    r".*_log10_t2equad$": [-9.0, -5.0],
}


def _joint(*, free_white=True, ecorr_gp=False):
    ntm = TimingSpec(
        engines="jug", inference=TimingInference.sample_all(), name="timing"
    )
    ctx = ntm.for_pulsar(_DiscoveryPulsar())
    psr = ctx.pulsar
    params0 = {f"{psr.name}_demo_efac": 1.2, f"{psr.name}_demo_log10_t2equad": -7.0}
    signals = [
        psr.residuals,
        ds.makenoise_measurement(psr, {} if free_white else params0),
    ]
    if ecorr_gp:
        signals.append(ds.makegp_ecorr(psr, {f"{psr.name}_demo_log10_ecorr": -7.5}))
    signals += [
        ds.makegp_fourier(psr, ds.powerlaw, _COMPONENTS, name="red_noise"),
        *ctx.discovery_signals(joint=True),
    ]
    return ctx, ds.PulsarLikelihood(signals), params0


def _eta():
    return {"J1234+5678_red_noise_log10_A": -14.0, "J1234+5678_red_noise_gamma": 3.0}


def test_tracked_joint_model_matches_frozen_at_bake_point():
    ctx, like, params0 = _joint()
    ref = nlts.numpyro.class_tracking_reference(like, params0)
    tracked = nlts.numpyro.joint_model(like, ctx, reference_noise=ref, priors=_PRIORS)
    from discovery import transport as dst

    frozen_ref = dst.reference_noise_frozen(like.white_noise_kernel, params0=params0)
    frozen = nlts.numpyro.joint_model(
        like, ctx, reference_noise=frozen_ref, priors=_PRIORS
    )

    assert set(params0) <= set(tracked.hyper_sites)
    assert set(params0) <= set(frozen.hyper_sites)  # free in the likelihood too
    assert set(params0) <= set(tracked.transport.params)
    assert not (set(params0) & set(frozen.transport.params))

    site = {
        **_eta(),
        **params0,
        tracked.xi_site: jnp.zeros(tracked.transport.dimension),
    }
    lp_t, _ = log_density(tracked, (), {}, site)
    lp_f, _ = log_density(frozen, (), {}, site)
    assert abs(float(lp_t) - float(lp_f)) < 1e-9 * max(1.0, abs(float(lp_f)))

    # away from the bake point the two charts differ (same posterior, different map)
    site2 = dict(site)
    site2[f"{ctx.pulsar.name}_demo_efac"] = 1.6
    lp_t2, _ = log_density(tracked, (), {}, site2)
    lp_f2, _ = log_density(frozen, (), {}, site2)
    assert abs(float(lp_t2) - float(lp_f2)) > 1e-6
    assert tracked.transport.fingerprint() != frozen.transport.fingerprint()


def test_ecorr_gp_and_sm_forms_give_the_same_tracked_transport():
    ctx, like_gp, params0 = _joint(ecorr_gp=True)
    psr = ctx.pulsar
    like_sm = ds.PulsarLikelihood(
        [
            psr.residuals,
            ds.makenoise_measurement(
                psr, {f"{psr.name}_demo_log10_ecorr": -7.5}, ecorr=True
            ),
            ds.makegp_fourier(psr, ds.powerlaw, _COMPONENTS, name="red_noise"),
            *ctx.discovery_signals(joint=True),
        ]
    )
    # the SM form keeps ECORR free when EFAC/EQUAD are free: pin it at the bake point
    params0_sm = {**params0, f"{psr.name}_demo_log10_ecorr": -7.5}
    ref_gp = nlts.numpyro.class_tracking_reference(like_gp, params0)
    ref_sm = nlts.numpyro.class_tracking_reference(like_sm, params0_sm)
    t_gp = nlts.numpyro.build_joint_transport(like_gp, ctx, reference_noise=ref_gp)
    t_sm = nlts.numpyro.build_joint_transport(like_sm, ctx, reference_noise=ref_sm)
    assert set(params0_sm) <= set(t_sm.params) and not (
        set(t_gp.params) & {f"{psr.name}_demo_log10_ecorr"}
    )
    params = {**_eta(), **params0_sm}
    params[f"{psr.name}_demo_efac"] = 0.9
    xi = jnp.asarray(np.random.default_rng(0).standard_normal(t_gp.dimension))
    q_gp, ld_gp = t_gp.apply(params, xi)
    q_sm, ld_sm = t_sm.apply(params, xi)
    assert np.allclose(np.asarray(q_gp), np.asarray(q_sm), rtol=1e-12, atol=0)
    assert abs(float(ld_gp) - float(ld_sm)) < 1e-10


def test_joint_model_multi_requires_one_tracker_per_pulsar():
    ctx1, like1, params0 = _joint()
    ctx2, like2, _ = _joint()
    ref = nlts.numpyro.class_tracking_reference(like1, params0)
    with pytest.raises(TypeError, match="per pulsar"):
        nlts.numpyro.joint_model_multi(
            [like1, like2], [ctx1, ctx2], reference_noise=ref, priors=_PRIORS
        )
    with pytest.raises(ValueError, match="one entry per pulsar"):
        nlts.numpyro.joint_model_multi(
            [like1, like2], [ctx1, ctx2], reference_noise=[ref], priors=_PRIORS
        )


class _SecondPulsar(_DiscoveryPulsar):
    """Same duck, distinct name (per-pulsar xi sites must not collide)."""

    def __init__(self):
        super().__init__()
        self.name = "J0000+0001"


def test_joint_model_multi_with_per_pulsar_trackers_builds_and_evaluates():
    import jax.random as jr
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    ntm = TimingSpec(
        engines="jug", inference=TimingInference.sample_all(), name="timing"
    )
    ctxs, likes, refs_t, refs_f = [], [], [], []
    from discovery import transport as dst

    for duck in (_DiscoveryPulsar(), _SecondPulsar()):
        ctx = ntm.for_pulsar(duck)
        psr = ctx.pulsar
        params0 = {f"{psr.name}_demo_efac": 1.2, f"{psr.name}_demo_log10_t2equad": -7.0}
        like = ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, {}),
                ds.makegp_fourier(psr, ds.powerlaw, _COMPONENTS, name="red_noise"),
                *ctx.discovery_signals(joint=True),
            ]
        )
        ctxs.append(ctx)
        likes.append(like)
        refs_t.append(nlts.numpyro.class_tracking_reference(like, params0))
        refs_f.append(
            dst.reference_noise_frozen(like.white_noise_kernel, params0=params0)
        )

    tracked = nlts.numpyro.joint_model_multi(
        likes, ctxs, reference_noise=refs_t, priors=_PRIORS
    )
    frozen = nlts.numpyro.joint_model_multi(
        likes, ctxs, reference_noise=refs_f, priors=_PRIORS
    )
    assert len(tracked.transports) == 2
    for t_t, t_f in zip(tracked.transports, frozen.transports):
        assert "tracking" in t_t.diagnostics() and "tracking" not in t_f.diagnostics()
        assert t_t.fingerprint() != t_f.fingerprint()
    assert tracked.transports[0].fingerprint() != tracked.transports[1].fingerprint()

    init = {
        f"{ctx.name_stem}_joint_xi": jnp.zeros(tr.dimension)
        for ctx, tr in zip(ctxs, tracked.transports)
    }
    mi = initialize_model(
        jr.PRNGKey(0), tracked, init_strategy=init_to_value(values=init)
    )
    assert np.isfinite(float(mi.potential_fn(mi.param_info.z)))
