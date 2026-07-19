"""Cross-likelihood agreement gate (proposal §14.5, acceptance criterion #9).

For a small deterministic fixture, Discovery's transformed NumPyro log
density and Enterprise's flat-vector likelihood + prior must describe the
same physical posterior, for every transform mode. This is the regression
gate that proves native sampler execution (NumPyro NUTS / Enterprise
PTMCMC-or-anything) has not changed model semantics relative to each other.

Comparisons use log-density *differences* between two coordinate points
rather than absolute values, since Discovery's and Enterprise's likelihood
objects may carry different framework-level additive normalization
constants; the physically meaningful quantity — how the log posterior
changes as the timing coordinate moves — must still agree exactly.
"""

import numpy as np
import pytest

import jax.numpy as jnp
import jax.random as jr
from numpyro import handlers

from nltiming import WhiteningConfig
from nltiming import TimingInference
from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.engines.pint import LinearizedPintEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.sampling import numpyro as nlt_numpyro

pytest.importorskip("discovery")
pytest.importorskip("enterprise")

import discovery as ds  # noqa: E402
from enterprise.signals import parameter, signal_base, white_signals  # noqa: E402


class _Pulsar:
    """Small deterministic fixture: a real (linear) JAX-backed timing model
    usable by both the Discovery and Enterprise frontends via the same
    TimingContext."""

    def __init__(self):
        self.name = "J0000+0000"
        self.fitpars = ("F0", "F1", "DM")
        self._toas = np.linspace(0.0, 1.0, 8)
        self._residuals = np.linspace(-2e-6, 2e-6, 8)
        self._toaerrs = np.full(8, 1.0e-6, dtype=float)
        self._freqs = np.full(8, 1400.0, dtype=float)
        self._flags = {"pta": np.array(["demo"] * 8, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 8, dtype="U8")
        self._state_id = "agreement-token"
        design = np.array(
            [
                [1.0, 0.0, 0.2],
                [1.0, 0.1, 0.3],
                [1.0, 0.2, -0.1],
                [1.0, -0.3, 0.4],
                [1.0, 0.4, -0.2],
                [1.0, -0.5, 0.1],
                [1.0, 0.6, -0.2],
                [1.0, -0.7, 0.3],
            ],
            dtype=float,
        )
        self._design = design
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "10.0", "F1": "1.0", "DM": "5.0"},
        )
        self._jug_backend = LinearizedJugEngine.from_linear_model(model)
        self._pint_backend = LinearizedPintEngine.from_linear_model(model)

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
        return self._design

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def state_id(self):
        return self._state_id

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        if isinstance(engines, dict) and engines.get("pint") == "pint":
            return self._pint_backend
        return self._jug_backend


@pytest.fixture
def pulsar():
    return _Pulsar()


# identity (whitening=None) samples the prior-normal z coordinate (O(1)), so its
# cheat-prior box is comparatively tight; the other coordinates are O(1) by
# construction.
# identity samples in z (O(1)); whitening samples in x (O(1)).
_OFFSETS = {"identity": 0.05, "whitening": 0.05}


@pytest.mark.parametrize(
    "layer", [None, WhiteningConfig()], ids=["identity", "whitening"]
)
def test_discovery_and_enterprise_log_density_differences_agree(pulsar, layer):
    noisedict = {f"{pulsar.name}_efac": 1.0, f"{pulsar.name}_log10_t2equad": -8.0}
    ntm = NonLinearTimingModel(
        engines="jug",
        whitening=layer,
        inference=TimingInference.groups(delta_flat=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar)
    ndim = len(ctx.sampled)
    offset = _OFFSETS["identity" if layer is None else "whitening"]

    # --- Discovery side: trace the NumPyro model at two coordinate points ---
    likelihood = ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement_simple(pulsar, noisedict),
            *ctx.discovery_signals(),
        ]
    )
    numpyro_model = nlt_numpyro.model(likelihood, ctx, fixed=noisedict)
    site = ctx.latent_name_for_coord()

    q1 = jnp.zeros(ndim)
    q2 = jnp.full((ndim,), offset)

    def _trace(q):
        substituted = handlers.substitute(numpyro_model, data={site: q})
        return handlers.trace(handlers.seed(substituted, jr.PRNGKey(0))).get_trace()

    tr1, tr2 = _trace(q1), _trace(q2)
    # The coordinate site's own log_prob is the prior contribution; "ll" is
    # the numpyro.factor site carrying the likelihood. Summing over every
    # "sample"-typed site would double-count "ll" (numpyro.factor is
    # implemented as a sample site too), so extract each explicitly.
    disc_prior_diff = float(
        tr2[site]["fn"].log_prob(tr2[site]["value"]).sum()
        - tr1[site]["fn"].log_prob(tr1[site]["value"]).sum()
    )
    # For coord="delta" the site is an ImproperUniform placeholder and the real
    # prior (uniform box or Gaussian, per the axis chart) lives in the
    # "{site}_logprior" factor; include it so the prior difference is complete
    # whatever the prior family.
    lp_site = f"{site}_logprior"
    if lp_site in tr1:
        disc_prior_diff += float(
            tr2[lp_site]["fn"].log_factor - tr1[lp_site]["fn"].log_factor
        )
    disc_ll_diff = float(tr2["ll"]["fn"].log_factor - tr1["ll"]["fn"].log_factor)

    # --- Enterprise side: the same two points through the full PTA ---
    # Noise pinned to the same values as Discovery's `fixed=noisedict`, via
    # Constant parameters (not sampled), so only the timing coordinate moves.
    white = white_signals.MeasurementNoise(
        efac=parameter.Constant(1.0)
    ) + white_signals.TNEquadNoise(log10_tnequad=parameter.Constant(-8.0))
    pta = signal_base.PTA([(white + ntm.enterprise_signal())(pulsar)])

    x1 = np.zeros(ndim)
    x2 = np.full(ndim, offset)
    ent_ll_diff = pta.get_lnlikelihood(x2) - pta.get_lnlikelihood(x1)
    ent_prior_diff = pta.get_lnprior(x2) - pta.get_lnprior(x1)

    np.testing.assert_allclose(disc_ll_diff, ent_ll_diff, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(disc_prior_diff, ent_prior_diff, rtol=1e-6, atol=1e-10)


@pytest.mark.slow
@pytest.mark.requires_enterprise
def test_discovery_nuts_and_enterprise_ptmcmc_recover_the_same_posterior(
    pulsar, tmp_path
):
    """Short real chains through both native sampler stacks (NumPyro NUTS on
    the Discovery model; enterprise_extensions.sampler.setup_sampler on the
    full Enterprise PTA — the canonical, non-timing-only workflow) must
    decode to the same one-dimensional posterior, within Monte Carlo noise."""
    import jax

    pytest.importorskip("enterprise_extensions")
    from enterprise_extensions import sampler as ee_sampler

    noisedict = {f"{pulsar.name}_efac": 1.0, f"{pulsar.name}_log10_t2equad": -8.0}
    ntm = NonLinearTimingModel(
        engines="jug",
        whitening=WhiteningConfig(),
        inference=TimingInference.groups(delta_flat=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar)

    nlt_numpyro.ensure_x64()
    likelihood = ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement_simple(pulsar, noisedict),
            *ctx.discovery_signals(),
        ]
    )
    numpyro_model = nlt_numpyro.model(likelihood, ctx, fixed=noisedict)
    mcmc = nlt_numpyro.nuts(
        numpyro_model,
        ctx,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))
    disc_samples = nlt_numpyro.timing_draws(mcmc.get_samples(), ctx)

    white = white_signals.MeasurementNoise(
        efac=parameter.Constant(1.0)
    ) + white_signals.TNEquadNoise(log10_tnequad=parameter.Constant(-8.0))
    pta = signal_base.PTA([(white + ntm.enterprise_signal())(pulsar)])

    np.random.seed(0)
    sampler = ee_sampler.setup_sampler(pta, outdir=str(tmp_path), resume=False)
    x0 = np.zeros(len(pta.param_names))
    sampler.sample(
        x0,
        Niter=6000,
        isave=1000,
        thin=1,
        burn=1000,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
        writeHotChains=False,
    )
    chain = np.loadtxt(tmp_path / "chain_1.txt")
    ent_samples = chain[1000::2, : len(pta.param_names)]

    # Loose tolerance: this compares two independent short MCMC runs (NUTS
    # vs adaptive-Metropolis/DE), not a deterministic density evaluation —
    # the point is gross posterior agreement, not numerical precision.
    np.testing.assert_allclose(
        disc_samples.mean(axis=0), ent_samples.mean(axis=0), atol=0.5
    )
    assert np.all(
        np.abs(np.log(disc_samples.std(axis=0) / ent_samples.std(axis=0))) < 0.7
    )


def test_affine_normal_z_prior_marginalization_is_expansion_independent(pulsar):
    """For a globally-linear (affine_normal) z-prior axis, the analytical
    marginalization is exact, so the marginal log-likelihood is identical whether
    the block is linearized at the engine reference or at a shifted expansion
    (geometry §14.4). DM here is identically linear with a Gaussian delta prior."""
    noisedict = {f"{pulsar.name}_efac": 1.0, f"{pulsar.name}_log10_t2equad": -8.0}
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(z_prior=["DM"]),
        name="timing",
    )

    def _logL_at(ctx, values):
        like = ds.PulsarLikelihood(
            [pulsar.residuals,
             ds.makenoise_measurement_simple(pulsar, noisedict),
             *ctx.discovery_signals()])
        params = dict(noisedict)
        for key, v in zip(ctx.delay_keys, values):
            params[key] = float(v)
        return float(like.logL(params))

    base = ntm.for_pulsar(pulsar, condition=False)
    shifted = base.with_expansion(
        delta={"F0": 0.0, "F1": 0.0, "DM": 5.0e-4}, source="explicit_delta")

    # F0/F1 sampled point (engine-native delta); DM is analytically marginalized.
    values = [1.0e-13, 2.0e-21]
    assert base.plan.sampled == ("F0", "F1")
    l0 = _logL_at(base, values)
    l1 = _logL_at(shifted, values)
    assert np.isfinite(l0)
    np.testing.assert_allclose(l0, l1, rtol=1e-8, atol=1e-8)
