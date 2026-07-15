"""Tests for the sampler glue in nltiming.sampling."""

import numpy as np
import pytest
from numpyro import handlers

from nltiming.backends.base import LinearModel
from nltiming.backends.jug import LinearizedJugEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.sampling import numpyro as nlt_numpyro
from nltiming.sampling import ptmcmc as nlt_ptmcmc


class _Host:
    def __init__(self):
        self.name = "J1111+1111"
        self.fitpars = ("F0", "F1")
        self._toas = np.linspace(0.0, 1.0, 5)
        self._residuals = np.zeros(5)
        self._toaerrs = np.full(5, 1.0e-6)
        self._freqs = np.full(5, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 5, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 5, dtype="U8")
        self._cache_token = "sampling-token"
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(5), np.linspace(-0.5, 0.5, 5)]),
            theta_exact={"F0": "100.0", "F1": "1.0"},
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

    def cache_token(self):
        return self._cache_token

    def pint_model(self):
        return object()

    def timing_backend(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def host():
    return _Host()


def _binding(transform="whitening", **kwargs):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform=transform,
        analytically_marginalize=["F0"],
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


def _trace_model(model_fn, binding):
    """Trace a model with the improper timing site pinned at the reference."""
    import jax.numpy as jnp
    import jax.random as jr

    init = {binding.coord_site_name(): jnp.zeros(len(binding.sampled))}
    with handlers.seed(rng_seed=jr.PRNGKey(0)):
        with handlers.substitute(data=init):
            return handlers.trace(model_fn).get_trace()


def test_model_traces_timing_site_and_ll_factor(host):
    binding = _binding().bind(host)
    # Discovery consumes the derived delay keys, never the joint latent site.
    likelihood = _FakeLikelihood([*binding.delay_keys, "efac"])
    model_fn = nlt_numpyro.model(likelihood, binding, fixed={"efac": 1.0})

    trace = _trace_model(model_fn, binding)

    assert binding.coord_site_name() in trace
    # whitening's x-coordinate MVN carries its own log_prob (§6.2); no extra
    # prior-factor site is added for it
    assert f"{binding.coord_site_name()}_logprior" not in trace
    assert "ll" in trace
    # JAX-safe per-parameter delta deterministic (§6.3)
    assert f"{binding.prefix}_F1_delta" in trace
    # efac is fixed, not sampled
    assert "efac" not in trace


def test_model_free_params_use_priordict_bounds(host):
    pytest.importorskip("discovery")
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood([*binding.delay_keys, "J1111+1111_efac"])
    model_fn = nlt_numpyro.model(
        likelihood,
        binding,
        priors={"J1111+1111_efac": [0.5, 1.5]},
    )

    trace = _trace_model(model_fn, binding)

    assert "J1111+1111_efac" in trace
    value = float(trace["J1111+1111_efac"]["value"])
    assert 0.5 <= value <= 1.5


def test_model_rejects_latent_site_in_likelihood_params(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood([binding.coord_site_name(), *binding.delay_keys])
    with pytest.raises(ValueError, match="joint latent timing site"):
        nlt_numpyro.model(likelihood, binding)


def test_model_rejects_missing_delay_keys(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood(["efac"])  # missing binding.delay_keys entirely
    with pytest.raises(ValueError, match="missing delay keys"):
        nlt_numpyro.model(likelihood, binding)


def test_model_rejects_duplicate_likelihood_param_names(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood([*binding.delay_keys, "efac", "efac"])
    with pytest.raises(ValueError, match="duplicate likelihood parameter names"):
        nlt_numpyro.model(likelihood, binding)


def test_model_rejects_fixed_timing_parameter(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood([*binding.delay_keys, "efac"])
    with pytest.raises(ValueError, match="cannot pin timing parameters"):
        nlt_numpyro.model(likelihood, binding, fixed={binding.coord_site_name(): 0.0})


def test_model_rejects_non_numeric_fixed_value(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood([*binding.delay_keys, "efac"])
    with pytest.raises(TypeError, match="efac.*must be numeric"):
        nlt_numpyro.model(likelihood, binding, fixed={"efac": "not-a-number"})


def test_timing_init_values_zero_at_reference(host):
    binding = _binding().bind(host)
    init = nlt_numpyro.timing_init_values(binding)
    assert set(init) == {binding.coord_site_name()}
    np.testing.assert_array_equal(
        np.asarray(init[binding.coord_site_name()]),
        np.zeros(len(binding.sampled)),
    )


def test_timing_draws_flattens_chains(host):
    binding = _binding().bind(host)
    site = binding.coord_site_name()
    ndim = len(binding.sampled)
    flat = nlt_numpyro.timing_draws({site: np.zeros((7, ndim))}, binding)
    assert flat.shape == (7, ndim)
    stacked = nlt_numpyro.timing_draws({site: np.zeros((2, 7, ndim))}, binding)
    assert stacked.shape == (14, ndim)


# ---------------------------------------------------------------------------
# samples_to_frame / model().to_df


def test_samples_to_frame_ungrouped_columns_and_naming(host):
    binding = _binding(transform="whitening").bind(host)
    site = binding.coord_site_name()
    n = 5
    samples = {
        site: np.linspace(0.0, 0.4, n).reshape(n, 1),
        f"{binding.prefix}_F1_delta": np.full(n, 0.01),
        "red_noise_gamma": np.linspace(1.0, 2.0, n),
        "red_noise_log10_rho": np.stack([np.array([1.0, 2.0])] * n),
    }

    df = nlt_numpyro.samples_to_frame(samples, binding)

    assert len(df) == n
    assert f"{site}[0]" in df.columns
    assert "red_noise_gamma" in df.columns
    assert "red_noise_log10_rho[0]" in df.columns
    assert "red_noise_log10_rho[1]" in df.columns
    assert f"{binding.prefix}_F1_delta" in df.columns
    assert f"{binding.prefix}_F1_theta_native" in df.columns
    assert f"{binding.prefix}_F1_theta_display" in df.columns


def test_samples_to_frame_flattens_grouped_chain_major(host):
    binding = _binding(transform="whitening").bind(host)
    site = binding.coord_site_name()
    n_chains, n_draws = 2, 3
    x = np.arange(n_chains * n_draws, dtype=float).reshape(n_chains, n_draws, 1)
    samples = {site: x, f"{binding.prefix}_F1_delta": x[..., 0]}

    df = nlt_numpyro.samples_to_frame(samples, binding)

    assert len(df) == n_chains * n_draws
    np.testing.assert_allclose(df[f"{site}[0]"].to_numpy(), x.reshape(-1))


def test_samples_to_frame_recomputes_delta_when_absent(host):
    binding = _binding(transform="standardized").bind(host)
    site = binding.coord_site_name()
    q = np.array([[0.2]])
    samples = {site: q}

    df = nlt_numpyro.samples_to_frame(samples, binding)

    expected_delta = np.asarray(binding.space.delta_from_coord(q[0], np, coord="x"))
    np.testing.assert_allclose(
        df[f"{binding.prefix}_F1_delta"].to_numpy(), expected_delta
    )


def test_samples_to_frame_recomputes_theta_ignoring_stray_values(host):
    binding = _binding(transform="none").bind(host)
    site = binding.coord_site_name()
    q = np.array([[0.05]])
    samples = {
        site: q,
        f"{binding.prefix}_F1_theta_native": np.array([999999.0]),
    }

    df = nlt_numpyro.samples_to_frame(samples, binding)

    delta = np.asarray(binding.space.delta_from_coord(q[0], np, coord="delta"))
    expected_native = binding.space.to_physical(
        delta[None, :], units="native", coord="delta"
    )["F1"][0]
    native = df[f"{binding.prefix}_F1_theta_native"].to_numpy()[0]
    assert native != 999999.0
    np.testing.assert_allclose(native, expected_native)


def test_samples_to_frame_missing_pandas_raises_actionable_error(host, monkeypatch):
    import sys

    binding = _binding().bind(host)
    monkeypatch.setitem(sys.modules, "pandas", None)
    site = binding.coord_site_name()
    with pytest.raises(ImportError, match="discovery"):
        nlt_numpyro.samples_to_frame({site: np.zeros((1, 1))}, binding)


def test_model_to_df_delegates_to_samples_to_frame(host):
    pytest.importorskip("pandas")
    binding = _binding(transform="standardized").bind(host)
    likelihood = _FakeLikelihood([*binding.delay_keys, "efac"])
    model_fn = nlt_numpyro.model(likelihood, binding, fixed={"efac": 1.0})
    assert hasattr(model_fn, "to_df")

    site = binding.coord_site_name()
    samples = {
        site: np.array([[0.1], [0.2]]),
        f"{binding.prefix}_F1_delta": np.array([0.01, 0.02]),
    }
    df = model_fn.to_df(samples)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# nuts() convenience recipe


def test_nuts_defaults_pass_expected_kernel_and_mcmc_settings(host, monkeypatch):
    import numpyro.infer as numpyro_infer

    binding = _binding().bind(host)
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

    nlt_numpyro.nuts(model_fn, binding)

    assert captured["target_accept_prob"] == 0.8
    assert captured["max_tree_depth"] == 10
    assert captured["mcmc_kwargs"]["chain_method"] == "vectorized"
    assert captured["mcmc_kwargs"]["progress_bar"] is True


def test_nuts_explicit_init_strategy_wins(host, monkeypatch):
    import numpyro.infer as numpyro_infer

    binding = _binding().bind(host)
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

    nlt_numpyro.nuts(model_fn, binding, init_strategy=sentinel)

    assert captured["init_strategy"] is sentinel


def test_nuts_attaches_to_df_when_model_has_it(host, monkeypatch):
    import numpyro.infer as numpyro_infer

    binding = _binding().bind(host)
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

    mcmc = nlt_numpyro.nuts(model_fn, binding)

    assert mcmc.to_df() == "a-dataframe"
    assert calls == [{"marker": "samples"}]


def test_nuts_no_to_df_when_model_lacks_it(host, monkeypatch):
    import numpyro.infer as numpyro_infer

    binding = _binding().bind(host)

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

    monkeypatch.setattr(numpyro_infer, "MCMC", _FakeMCMC)

    def model_fn():
        pass

    mcmc = nlt_numpyro.nuts(model_fn, binding)

    assert not hasattr(mcmc, "to_df")


# ---------------------------------------------------------------------------
# save_samples


def test_save_samples_wraps_timing_draws_and_checkpoint(tmp_path, host, monkeypatch):
    import nltiming.artifacts as artifacts_mod

    binding = _binding(transform="whitening").bind(host)
    site = binding.coord_site_name()
    samples = {site: np.array([[0.1], [0.2], [0.3]])}
    captured = {}

    def fake_save_discovery_checkpoint(run_dir, x, artifact, *, final, n_target=None):
        captured.update(
            run_dir=run_dir, x=x, artifact=artifact, final=final, n_target=n_target
        )
        return "the-path"

    monkeypatch.setattr(
        artifacts_mod, "save_discovery_checkpoint", fake_save_discovery_checkpoint
    )

    result = nlt_numpyro.save_samples(
        tmp_path, samples, binding, artifact="fake-artifact", final=True, n_target=5
    )

    assert result == "the-path"
    np.testing.assert_array_equal(captured["x"], np.array([[0.1], [0.2], [0.3]]))
    assert captured["artifact"] == "fake-artifact"
    assert captured["final"] is True
    assert captured["n_target"] == 5


def test_ensure_x64_enables_float64():
    nlt_numpyro.ensure_x64()
    import jax.numpy as jnp

    assert jnp.zeros(1).dtype == jnp.float64


# ---------------------------------------------------------------------------
# ptmcmc glue


def test_eval_params_whitening_uses_joint_site(host):
    binding = _binding(transform="whitening").bind(host)
    vec = np.array([0.25])
    params = nlt_ptmcmc.eval_params(binding, vec, fixed={"efac": 1.0})
    assert params["efac"] == 1.0
    np.testing.assert_array_equal(params[binding.coord_site_name()], vec)


def test_eval_params_standardized_uses_scalar_delay_keys(host):
    binding = _binding(transform="standardized").bind(host)
    vec = np.array([0.25])
    params = nlt_ptmcmc.eval_params(binding, vec)
    assert params == {binding.delay_keys[0]: 0.25}


def test_eval_params_rejects_wrong_length(host):
    binding = _binding().bind(host)
    with pytest.raises(ValueError, match="expected vector of length 1"):
        nlt_ptmcmc.eval_params(binding, np.zeros(3))


def test_initial_point_is_zero_reference(host):
    binding = _binding().bind(host)
    np.testing.assert_array_equal(
        nlt_ptmcmc.initial_point(binding), np.zeros(len(binding.sampled))
    )


def test_initial_cov_matches_wls_in_sampling_coords(host):
    binding = _binding(transform="whitening").bind(host)
    cov = nlt_ptmcmc.initial_cov(binding, nsamples=4000, seed=1)
    assert cov.shape == (1, 1)
    # positive definite
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # whitening scales the WLS posterior to roughly unit coordinates, so the
    # sampled-coordinate variance must be O(1), not the raw delta variance
    assert 0.1 < float(cov[0, 0]) < 10.0


def test_timing_param_names_layouts(host):
    whitening = _binding(transform="whitening").bind(host)
    site = whitening.coord_site_name()
    assert nlt_ptmcmc.timing_param_names(whitening) == (f"{site}_0",)

    standardized = _binding(transform="standardized").bind(host)
    assert nlt_ptmcmc.timing_param_names(standardized) == standardized.delay_keys


def test_chain_layout_locates_timing_columns(host):
    binding = _binding(transform="standardized").bind(host)
    names = ["noise_param", *nlt_ptmcmc.timing_param_names(binding)]
    layout = nlt_ptmcmc.chain_layout(binding, names)
    assert layout == {"kind": "ptmcmc", "file": "chain_1.txt", "columns": [1]}


def test_chain_layout_missing_key_raises(host):
    binding = _binding().bind(host)
    with pytest.raises(ValueError, match="not found in sampler param names"):
        nlt_ptmcmc.chain_layout(binding, ["something_else"])


@pytest.mark.parametrize("transform", ["standardized", "whitening"])
def test_chain_layout_locates_columns_in_real_enterprise_pta(host, transform):
    """§12/§14.6: chain_layout must locate timing columns in a full PTA vector
    with free noise parameters interleaved, for both scalar-standardized and
    joint-whitened layouts, using Enterprise's own param_names ordering."""
    from enterprise.signals import parameter, signal_base, white_signals

    efac = parameter.Uniform(0.1, 5.0)
    white = white_signals.MeasurementNoise(efac=efac)
    ntm = NonLinearTimingModel(
        engines="jug",
        transform=transform,
        analytically_marginalize=["F0"],
        name="timing",
    )
    binding = ntm.bind(host)
    pta = signal_base.PTA([(white + ntm.enterprise_signal())(host)])

    layout = nlt_ptmcmc.chain_layout(binding, pta.param_names)

    expected_names = nlt_ptmcmc.timing_param_names(binding)
    assert len(layout["columns"]) == len(expected_names)
    for name, col in zip(expected_names, layout["columns"]):
        assert pta.param_names[col] == name
