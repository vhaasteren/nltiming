"""Tests for the sampler glue in nltiming.sampling."""

import numpy as np
import pytest
from _engine_stubs import JaxLinearTestEngine
from numpyro import handlers

import nltiming.sampling as nlts
from nltiming import TimingInference, WhiteningConfig
from nltiming.engine_support import LinearModel
from nltiming.nonlinear_timing_model import TimingSpec


class _Pulsar:
    def __init__(self):
        self.name = "J1111+1111"
        self.fitpars = ("Offset", "F1")
        self._toas = np.linspace(0.0, 1.0, 5)
        self._residuals = np.zeros(5)
        self._toaerrs = np.full(5, 1.0e-6)
        self._freqs = np.full(5, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 5, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 5, dtype="U8")
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(5), np.linspace(-0.5, 0.5, 5)]),
            theta_exact={"Offset": "0.0", "F1": "1.0"},
        )
        self._backend = JaxLinearTestEngine.from_linear_model(model)

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

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def pulsar():
    return _Pulsar()


_UNSET = object()


def _binding(whitening=_UNSET, **kwargs):
    ntm = TimingSpec(
        engines="jug",
        whitening=WhiteningConfig() if whitening is _UNSET else whitening,
        inference=TimingInference.groups(delta_flat=["Offset"]),
        name="timing",
        **kwargs,
    )
    return ntm


class _FakeLogL:
    def __init__(self, params):
        self.params = list(params)

    def __call__(self, params):
        return -0.5 * float(sum(float(v) ** 2 for v in params.values()))


class _FakeLikelihood:
    def __init__(self, params):
        self.logL = _FakeLogL(params)


# ---------------------------------------------------------------------------
# numpyro glue


def _trace_model(model_fn, ctx):
    """Trace a model with the improper timing site pinned at the reference."""
    import jax.numpy as jnp
    import jax.random as jr

    init = {ctx.latent_name_for_coord(): jnp.zeros(len(ctx.sampled))}
    with handlers.seed(rng_seed=jr.PRNGKey(0)), handlers.substitute(data=init):
        return handlers.trace(model_fn).get_trace()


def test_model_traces_timing_site_and_ll_factor(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    # Discovery consumes the derived delay keys, never the joint latent site.
    likelihood = _FakeLikelihood([*ctx.delay_keys, "efac"])
    model_fn = nlts.numpyro.model(likelihood, ctx, fixed={"efac": 1.0})

    trace = _trace_model(model_fn, ctx)

    assert ctx.latent_name_for_coord() in trace
    # whitening's x-coordinate MVN carries its own log_prob (§6.2); no extra
    # prior-factor site is added for it
    assert f"{ctx.latent_name_for_coord()}_logprior" not in trace
    assert "ll" in trace
    # JAX-safe per-parameter delta deterministic (§6.3)
    assert f"{ctx.name_stem}_F1_delta" in trace
    # efac is fixed, not sampled
    assert "efac" not in trace


def test_model_free_params_use_priordict_bounds(pulsar):
    pytest.importorskip("discovery")
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([*ctx.delay_keys, "J1111+1111_efac"])
    model_fn = nlts.numpyro.model(
        likelihood,
        ctx,
        priors={"J1111+1111_efac": [0.5, 1.5]},
    )

    trace = _trace_model(model_fn, ctx)

    assert "J1111+1111_efac" in trace
    value = float(trace["J1111+1111_efac"]["value"])
    assert 0.5 <= value <= 1.5


def test_model_rejects_latent_site_in_likelihood_params(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([ctx.latent_name_for_coord(), *ctx.delay_keys])
    with pytest.raises(ValueError, match="joint latent timing site"):
        nlts.numpyro.model(likelihood, ctx)


def test_model_rejects_missing_delay_keys(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood(["efac"])  # missing ctx.delay_keys entirely
    with pytest.raises(ValueError, match="missing delay keys"):
        nlts.numpyro.model(likelihood, ctx)


def test_model_rejects_duplicate_likelihood_param_names(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([*ctx.delay_keys, "efac", "efac"])
    with pytest.raises(ValueError, match="duplicate likelihood parameter names"):
        nlts.numpyro.model(likelihood, ctx)


def test_model_rejects_fixed_timing_parameter(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([*ctx.delay_keys, "efac"])
    with pytest.raises(ValueError, match="cannot pin timing parameters"):
        nlts.numpyro.model(likelihood, ctx, fixed={ctx.latent_name_for_coord(): 0.0})


def test_model_rejects_non_numeric_fixed_value(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([*ctx.delay_keys, "efac"])
    with pytest.raises(TypeError, match="efac.*must be numeric"):
        nlts.numpyro.model(likelihood, ctx, fixed={"efac": "not-a-number"})


def test_timing_init_values_zero_at_reference(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    init = nlts.numpyro.timing_init_values(ctx)
    assert set(init) == {ctx.latent_name_for_coord()}
    np.testing.assert_array_equal(
        np.asarray(init[ctx.latent_name_for_coord()]),
        np.zeros(len(ctx.sampled)),
    )


def test_timing_draws_flattens_chains(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    ndim = len(ctx.sampled)
    flat = nlts.numpyro.timing_draws({site: np.zeros((7, ndim))}, ctx)
    assert flat.shape == (7, ndim)
    stacked = nlts.numpyro.timing_draws({site: np.zeros((2, 7, ndim))}, ctx)
    assert stacked.shape == (14, ndim)


# ---------------------------------------------------------------------------
# samples_to_frame / model().to_df


def test_samples_to_frame_ungrouped_columns_and_naming(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    n = 5
    samples = {
        site: np.linspace(0.0, 0.4, n).reshape(n, 1),
        f"{ctx.name_stem}_F1_delta": np.full(n, 0.01),
        "red_noise_gamma": np.linspace(1.0, 2.0, n),
        "red_noise_log10_rho": np.stack([np.array([1.0, 2.0])] * n),
    }

    df = nlts.numpyro.samples_to_frame(samples, ctx)

    assert len(df) == n
    assert f"{site}[0]" in df.columns
    assert "red_noise_gamma" in df.columns
    assert "red_noise_log10_rho[0]" in df.columns
    assert "red_noise_log10_rho[1]" in df.columns
    assert f"{ctx.name_stem}_F1_delta" in df.columns
    assert f"{ctx.name_stem}_F1_theta_native" in df.columns
    assert f"{ctx.name_stem}_F1_theta_display" in df.columns


def test_samples_to_frame_flattens_grouped_chain_major(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    n_chains, n_draws = 2, 3
    x = np.arange(n_chains * n_draws, dtype=float).reshape(n_chains, n_draws, 1)
    samples = {site: x, f"{ctx.name_stem}_F1_delta": x[..., 0]}

    df = nlts.numpyro.samples_to_frame(samples, ctx)

    assert len(df) == n_chains * n_draws
    np.testing.assert_allclose(df[f"{site}[0]"].to_numpy(), x.reshape(-1))


def test_samples_to_frame_recomputes_delta_when_absent(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    q = np.array([[0.2]])
    samples = {site: q}

    df = nlts.numpyro.samples_to_frame(samples, ctx)

    expected_delta = np.asarray(ctx.space.delta_from_coord(q[0], np, coord="x"))
    np.testing.assert_allclose(
        df[f"{ctx.name_stem}_F1_delta"].to_numpy(), expected_delta
    )


def test_samples_to_frame_recomputes_theta_ignoring_stray_values(pulsar):
    ctx = _binding(whitening=None).for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    q = np.array([[0.05]])
    samples = {
        site: q,
        f"{ctx.name_stem}_F1_theta_native": np.array([999999.0]),
    }

    df = nlts.numpyro.samples_to_frame(samples, ctx)

    # The identity static layer samples the prior-normal z coordinate.
    delta = np.asarray(ctx.space.delta_from_coord(q[0], np, coord=ctx.coord))
    expected_native = ctx.space.to_physical(
        delta[None, :], units="native", coord="delta"
    )["F1"][0]
    native = df[f"{ctx.name_stem}_F1_theta_native"].to_numpy()[0]
    assert native != 999999.0
    np.testing.assert_allclose(native, expected_native)


def test_samples_to_frame_missing_pandas_raises_actionable_error(pulsar, monkeypatch):
    import sys

    ctx = _binding().for_pulsar(pulsar)
    monkeypatch.setitem(sys.modules, "pandas", None)
    site = ctx.latent_name_for_coord()
    with pytest.raises(ImportError, match="discovery"):
        nlts.numpyro.samples_to_frame({site: np.zeros((1, 1))}, ctx)


def test_posterior_returns_short_physical_variables_with_chains(pulsar):
    pytest.importorskip("arviz")
    ctx = _binding(whitening=None).for_pulsar(pulsar)
    delta = np.array([0.01, 0.02, 0.03])

    class _FakeMCMC:
        def get_samples(self, *, group_by_chain):
            assert group_by_chain is True
            return {f"{ctx.name_stem}_F1_delta": delta[None, :]}

    post = nlts.numpyro.posterior(_FakeMCMC(), ctx)
    expected = ctx.space.to_physical(delta[:, None], units="display", coord="delta")

    assert list(post.posterior.data_vars) == ["F1"]
    assert post.posterior["F1"].dims == ("chain", "draw")
    np.testing.assert_allclose(post.posterior["F1"][0], expected["F1"])


def test_posterior_requires_recorded_timing_deltas(pulsar):
    pytest.importorskip("arviz")
    ctx = _binding(whitening=None).for_pulsar(pulsar)

    class _FakeMCMC:
        def get_samples(self, *, group_by_chain):
            assert group_by_chain is True
            return {}

    with pytest.raises(KeyError, match="timing deterministics"):
        nlts.numpyro.posterior(_FakeMCMC(), ctx)


def test_model_to_df_delegates_to_samples_to_frame(pulsar):
    pytest.importorskip("pandas")
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    likelihood = _FakeLikelihood([*ctx.delay_keys, "efac"])
    model_fn = nlts.numpyro.model(likelihood, ctx, fixed={"efac": 1.0})
    assert hasattr(model_fn, "to_df")

    site = ctx.latent_name_for_coord()
    samples = {
        site: np.array([[0.1], [0.2]]),
        f"{ctx.name_stem}_F1_delta": np.array([0.01, 0.02]),
    }
    df = model_fn.to_df(samples)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# nuts() convenience recipe


def test_nuts_defaults_pass_expected_kernel_and_mcmc_settings(pulsar, monkeypatch):
    import numpyro.infer as numpyro_infer

    ctx = _binding().for_pulsar(pulsar)
    captured = {}

    class _FakeKernel:
        def __init__(
            self,
            model_fn,
            *,
            dense_mass,
            target_accept_prob,
            max_tree_depth,
            init_strategy,
        ):
            captured["target_accept_prob"] = target_accept_prob
            captured["max_tree_depth"] = max_tree_depth
            captured["init_strategy"] = init_strategy

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured["mcmc_kwargs"] = kwargs

    monkeypatch.setattr(numpyro_infer, "NUTS", _FakeKernel)
    monkeypatch.setattr(numpyro_infer, "MCMC", _FakeMCMC)

    def model_fn():
        pass

    nlts.numpyro.nuts(model_fn, ctx)

    assert captured["target_accept_prob"] == 0.8
    assert captured["max_tree_depth"] == 10
    assert captured["mcmc_kwargs"]["chain_method"] == "vectorized"
    assert captured["mcmc_kwargs"]["progress_bar"] is True


def test_nuts_explicit_init_strategy_wins(pulsar, monkeypatch):
    import numpyro.infer as numpyro_infer

    ctx = _binding().for_pulsar(pulsar)
    sentinel = object()
    captured = {}

    class _FakeKernel:
        def __init__(
            self,
            model_fn,
            *,
            dense_mass,
            target_accept_prob,
            max_tree_depth,
            init_strategy,
        ):
            captured["init_strategy"] = init_strategy

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

    monkeypatch.setattr(numpyro_infer, "NUTS", _FakeKernel)
    monkeypatch.setattr(numpyro_infer, "MCMC", _FakeMCMC)

    def model_fn():
        pass

    nlts.numpyro.nuts(model_fn, ctx, init_strategy=sentinel)

    assert captured["init_strategy"] is sentinel


def test_nuts_attaches_to_df_when_model_has_it(pulsar, monkeypatch):
    import numpyro.infer as numpyro_infer

    ctx = _binding().for_pulsar(pulsar)
    calls = []

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

        def get_samples(self):
            return {"marker": "samples"}

    monkeypatch.setattr(numpyro_infer, "MCMC", _FakeMCMC)

    def model_fn():
        pass

    def _to_df(samples):
        calls.append(samples)
        return "a-dataframe"

    model_fn.to_df = _to_df

    mcmc = nlts.numpyro.nuts(model_fn, ctx)

    assert mcmc.to_df() == "a-dataframe"
    assert calls == [{"marker": "samples"}]


def test_nuts_no_to_df_when_model_lacks_it(pulsar, monkeypatch):
    import numpyro.infer as numpyro_infer

    ctx = _binding().for_pulsar(pulsar)

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

    monkeypatch.setattr(numpyro_infer, "MCMC", _FakeMCMC)

    def model_fn():
        pass

    mcmc = nlts.numpyro.nuts(model_fn, ctx)

    assert not hasattr(mcmc, "to_df")


# ---------------------------------------------------------------------------
# save_samples


def test_save_samples_wraps_timing_draws_and_checkpoint(tmp_path, pulsar, monkeypatch):
    import nltiming.run_io as run_io_mod

    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    site = ctx.latent_name_for_coord()
    samples = {site: np.array([[0.1], [0.2], [0.3]])}
    captured = {}

    def fake_save_discovery_checkpoint(run_dir, x, manifest, *, final, n_target=None):
        captured.update(
            run_dir=run_dir, x=x, manifest=manifest, final=final, n_target=n_target
        )
        return "the-path"

    monkeypatch.setattr(
        run_io_mod, "save_discovery_checkpoint", fake_save_discovery_checkpoint
    )

    result = nlts.numpyro.save_samples(
        tmp_path, samples, ctx, manifest="fake-manifest", final=True, n_target=5
    )

    assert result == "the-path"
    np.testing.assert_array_equal(captured["x"], np.array([[0.1], [0.2], [0.3]]))
    assert captured["manifest"] == "fake-manifest"
    assert captured["final"] is True
    assert captured["n_target"] == 5


def test_ensure_x64_enables_float64():
    nlts.numpyro.ensure_x64()
    import jax.numpy as jnp

    assert jnp.zeros(1).dtype == jnp.float64


# ---------------------------------------------------------------------------
# ptmcmc glue


def test_eval_params_whitening_uses_joint_site(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    vec = np.array([0.25])
    params = nlts.ptmcmc.eval_params(ctx, vec, fixed={"efac": 1.0})
    assert params["efac"] == 1.0
    np.testing.assert_array_equal(params[ctx.latent_name_for_coord()], vec)


def test_eval_params_identity_uses_scalar_delay_keys(pulsar):
    ctx = _binding(whitening=None).for_pulsar(pulsar)
    vec = np.array([0.25])
    params = nlts.ptmcmc.eval_params(ctx, vec)
    assert params == {ctx.delay_keys[0]: 0.25}


def test_eval_params_rejects_wrong_length(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    with pytest.raises(ValueError, match="expected vector of length 1"):
        nlts.ptmcmc.eval_params(ctx, np.zeros(3))


def test_initial_point_is_zero_reference(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    np.testing.assert_array_equal(
        nlts.ptmcmc.initial_point(ctx), np.zeros(len(ctx.sampled))
    )


def test_initial_cov_matches_wls_in_sampling_coords(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    cov = nlts.ptmcmc.initial_cov(ctx, nsamples=4000, seed=1)
    assert cov.shape == (1, 1)
    # positive definite
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # whitening scales the WLS posterior to roughly unit coordinates, so the
    # sampled-coordinate variance must be O(1), not the raw delta variance
    assert 0.1 < float(cov[0, 0]) < 10.0


def test_timing_param_names_layouts(pulsar):
    whitening = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    site = whitening.latent_name_for_coord()
    assert nlts.ptmcmc.timing_param_names(whitening) == (f"{site}_0",)

    identity = _binding(whitening=None).for_pulsar(pulsar)
    assert nlts.ptmcmc.timing_param_names(identity) == identity.delay_keys


def test_chain_layout_locates_timing_columns(pulsar):
    ctx = _binding(whitening=WhiteningConfig()).for_pulsar(pulsar)
    names = ["noise_param", *nlts.ptmcmc.timing_param_names(ctx)]
    layout = nlts.ptmcmc.chain_layout(ctx, names)
    assert layout == {"kind": "ptmcmc", "file": "chain_1.txt", "columns": [1]}


def test_chain_layout_missing_key_raises(pulsar):
    ctx = _binding().for_pulsar(pulsar)
    with pytest.raises(ValueError, match="not found in sampler param names"):
        nlts.ptmcmc.chain_layout(ctx, ["something_else"])


@pytest.mark.parametrize("whitening", [None, WhiteningConfig()])
def test_chain_layout_locates_columns_in_real_enterprise_pta(pulsar, whitening):
    """§12/§14.6: chain_layout must locate timing columns in a full PTA vector
    with free noise parameters interleaved, for both scalar-standardized and
    joint-whitened layouts, using Enterprise's own param_names ordering."""
    from enterprise.signals import parameter, signal_base, white_signals

    efac = parameter.Uniform(0.1, 5.0)
    white = white_signals.MeasurementNoise(efac=efac)
    ntm = TimingSpec(
        engines="jug",
        whitening=whitening,
        inference=TimingInference.groups(delta_flat=["Offset"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar)
    pta = signal_base.PTA([(white + ntm.enterprise_signal())(pulsar)])

    layout = nlts.ptmcmc.chain_layout(ctx, pta.param_names)

    expected_names = nlts.ptmcmc.timing_param_names(ctx)
    assert len(layout["columns"]) == len(expected_names)
    for name, col in zip(expected_names, layout["columns"]):
        assert pta.param_names[col] == name


# ---------------------------------------------------------------------------
# §10 block mass default and chain-preserving diagnostics


def _model_with_sites(hyper_sites, xi_site="timing_joint_xi"):
    def model_fn():
        pass

    model_fn.hyper_sites = tuple(hyper_sites)
    model_fn.xi_site = xi_site
    return model_fn


def test_dense_mass_auto_resolves_to_hyper_block():
    model_fn = _model_with_sites(("log10_A", "gamma"))
    assert nlts.numpyro.resolve_dense_mass(model_fn, "auto") == [("log10_A", "gamma")]


def test_dense_mass_auto_single_or_no_hyper_is_false():
    assert (
        nlts.numpyro.resolve_dense_mass(_model_with_sites(("log10_A",)), "auto")
        is False
    )

    def bare():
        pass

    assert nlts.numpyro.resolve_dense_mass(bare, "auto") is False


def test_dense_mass_auto_never_includes_xi():
    model_fn = _model_with_sites(("a", "b"), xi_site="the_xi_vector")
    resolved = nlts.numpyro.resolve_dense_mass(model_fn, "auto")
    flat = [site for group in resolved for site in group]
    assert "the_xi_vector" not in flat


def test_dense_mass_explicit_is_forwarded_unchanged():
    model_fn = _model_with_sites(("a", "b"))
    assert nlts.numpyro.resolve_dense_mass(model_fn, True) is True
    assert nlts.numpyro.resolve_dense_mass(model_fn, False) is False
    explicit = [("a",)]
    assert nlts.numpyro.resolve_dense_mass(model_fn, explicit) is explicit


def test_nuts_passes_resolved_auto_block_mass_to_kernel(pulsar):
    ctx = _binding().for_pulsar(pulsar)

    def model_fn():
        pass

    model_fn.hyper_sites = ("log10_A", "gamma")
    model_fn.xi_site = ctx.latent_name_for_coord()
    mcmc = nlts.numpyro.nuts(model_fn, ctx)
    assert mcmc.sampler._dense_mass == [("log10_A", "gamma")]


def test_nuts_does_not_override_requested_warmup(pulsar):
    ctx = _binding().for_pulsar(pulsar)

    def model_fn():
        pass

    mcmc = nlts.numpyro.nuts(model_fn, ctx, num_warmup=2000, num_samples=5000)
    assert mcmc.num_warmup == 2000
    assert mcmc.num_samples == 5000


def test_tree_depth_saturation_fraction():
    # Depth-10 cap is 2**10 - 1 = 1023 leapfrog steps.
    num_steps = np.array([[1, 3, 1023], [7, 1023, 1023]])
    frac = nlts.numpyro.tree_depth_saturation_fraction(num_steps, max_tree_depth=10)
    assert frac == pytest.approx(3 / 6)
    assert nlts.numpyro.tree_depth_saturation_fraction(np.array([]), 10) == 0.0


def test_chain_diagnostics_group_by_chain_and_extra_fields():
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        numpyro.sample("a", dist.Uniform(-1.0, 1.0))
        numpyro.sample("b", dist.Uniform(-1.0, 1.0))
        numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([3]).to_event(1))

    mcmc = MCMC(
        NUTS(model),
        num_warmup=20,
        num_samples=15,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0), extra_fields=nlts.numpyro.NUTS_EXTRA_FIELDS)

    diag = nlts.numpyro.chain_diagnostics(mcmc, max_tree_depth=10)
    # Chains are preserved, never pooled: shape is (n_chains, n_samples, ...).
    for field in nlts.numpyro.NUTS_EXTRA_FIELDS:
        assert diag[field].shape == (2, 15)
    assert diag["samples"]["a"].shape == (2, 15)
    assert diag["samples"]["xi"].shape == (2, 15, 3)
    assert 0.0 <= diag["tree_depth_saturation_fraction"] <= 1.0
    assert isinstance(diag["tree_depth_saturated"], bool)
    assert diag["max_tree_depth"] == 10


def test_chain_diagnostics_requires_extra_fields():
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        numpyro.sample("a", dist.Uniform(-1.0, 1.0))

    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))  # no extra_fields collected
    with pytest.raises(ValueError, match="extra_fields"):
        nlts.numpyro.chain_diagnostics(mcmc)


def test_save_chain_diagnostics_roundtrip_preserves_chains(tmp_path):
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        numpyro.sample("a", dist.Uniform(-1.0, 1.0))
        numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([2]).to_event(1))

    mcmc = MCMC(
        NUTS(model),
        num_warmup=20,
        num_samples=15,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0), extra_fields=nlts.numpyro.NUTS_EXTRA_FIELDS)

    out = nlts.numpyro.save_chain_diagnostics(tmp_path / "diag", mcmc)
    assert out.suffix == ".npz"
    loaded = np.load(out)
    # Samples keep chains; extra fields present.
    assert loaded["sample__a"].shape == (2, 15)
    assert loaded["sample__xi"].shape == (2, 15, 2)
    for field in nlts.numpyro.NUTS_EXTRA_FIELDS:
        assert loaded[field].shape == (2, 15)
    assert int(loaded["max_tree_depth"]) == 10


# ---------------------------------------------------------------------------
# DiscoveryTarget (derivative-free Discovery / PTMCMC)


def _discovery_likelihood(pulsar, ctx, noisedict, *, add_equad=False):
    ds = pytest.importorskip("discovery")
    return ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement_simple(pulsar, noisedict, add_equad=add_equad),
            *ctx.discovery_signals(),
        ]
    )


@pytest.mark.parametrize(
    "whitening", [None, WhiteningConfig()], ids=["identity", "whitening"]
)
def test_discovery_target_writes_delta_not_sampler_coord(pulsar, whitening):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding(whitening=whitening).for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, noisedict)
    target = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    q = np.array([0.4])
    delta = np.asarray(ctx.space.delta_from_coord(q, np, coord=ctx.coord), dtype=float)
    params = {**noisedict, ctx.delay_keys[0]: float(delta[0])}
    np.testing.assert_allclose(
        target.loglikelihood(q),
        float(likelihood.logL(params)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        target.logprior(q),
        float(ctx.space.logprior_coord(q, np, coord=ctx.coord)),
        rtol=1e-12,
    )


def test_discovery_target_hyper_bounds_and_prior(pulsar):
    nlts.numpyro.ensure_x64()
    ctx = _binding(whitening=None).for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, {})
    efac = f"{pulsar.name}_efac"
    target = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed={})
    assert target.hyperparameter_names == (efac,)
    assert target.dimension == 2
    layout = target.chain_layout()
    assert layout["hyperparameter_names"] == [efac]
    assert layout["hyper_columns"] == [1]
    assert layout["parameter_names"] == list(target.parameter_names)
    assert layout["coord"] == ctx.coord
    lo, hi = 0.1, 10.0
    logwidth = float(np.log(hi - lo))
    q = np.array([0.0, 1.0])
    np.testing.assert_allclose(
        target.logprior(q),
        float(ctx.space.logprior_coord(q[:1], np, coord=ctx.coord)) - logwidth,
        rtol=1e-12,
    )
    assert target.logprior(np.array([0.0, 0.05])) == -np.inf
    assert target.loglikelihood(np.array([0.0, 0.05])) == -np.inf
    p0 = target.initial_point({efac: 1.0})
    np.testing.assert_array_equal(p0, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="exactly the free"):
        target.initial_point()
    with pytest.raises(ValueError, match="exactly the free"):
        target.initial_point({efac: 1.0, "extra": 0.0})


def test_discovery_target_rejects_fixed_timing_and_bad_shape(pulsar):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, noisedict)
    with pytest.raises(ValueError, match="nltiming-owned"):
        nlts.ptmcmc.discovery_target(
            likelihood, ctx, fixed={**noisedict, ctx.delay_keys[0]: 0.0}
        )
    target = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    with pytest.raises(ValueError, match="expected vector shape"):
        target.loglikelihood(np.zeros(3))
    layout = target.chain_layout()
    assert layout["columns"] == [0]
    assert layout["parameter_names"] == list(target.parameter_names)
    assert layout["timing_parameter_names"] == list(target.timing_parameter_names)
    assert layout["hyperparameter_names"] == []
    assert layout["hyper_columns"] == []
    assert layout["coord"] == ctx.coord


def test_discovery_target_jit_matches_eager(pulsar):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding(whitening=None).for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, noisedict)
    compiled = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    eager = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict, jit=False)
    for q in (np.zeros(1), np.array([0.3]), np.array([-0.2])):
        np.testing.assert_allclose(
            compiled.loglikelihood(q),
            eager.loglikelihood(q),
            rtol=1e-10,
            atol=1e-12,
        )


def test_discovery_sampler_covariance_guards(pulsar, tmp_path):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, {})
    free = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed={})
    with pytest.raises(ValueError, match="covariance is required"):
        nlts.ptmcmc.discovery_sampler(free, tmp_path)
    pinned = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    with pytest.raises(ValueError, match="covariance has shape"):
        nlts.ptmcmc.discovery_sampler(pinned, tmp_path, covariance=np.eye(3))


def test_discovery_target_requires_conditioned_context(pulsar):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding().for_pulsar(pulsar, condition=False)
    likelihood = _discovery_likelihood(pulsar, ctx, noisedict)
    with pytest.raises(ValueError, match="requires a conditioned TimingSignal"):
        nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)


def test_discovery_target_rejects_latent_site(pulsar):
    nlts.numpyro.ensure_x64()
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _FakeLikelihood([ctx.latent_name_for_coord(), *ctx.delay_keys])
    with pytest.raises(ValueError, match="joint latent timing site"):
        nlts.ptmcmc.discovery_target(likelihood, ctx)


def test_discovery_target_calls_ensure_x64(pulsar, monkeypatch):
    calls = []
    monkeypatch.setattr(nlts.numpyro, "ensure_x64", lambda: calls.append(True))
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _binding().for_pulsar(pulsar)
    likelihood = _discovery_likelihood(pulsar, ctx, noisedict)
    nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    assert calls == [True]
