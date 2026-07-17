"""Slice-5 tests for NonLinearTimingModel component behavior."""

import numpy as np
import pytest

import jax.numpy as jnp
import jax.random as jr
from numpyro import handlers

from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.engines.pint import LinearizedPintEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.whitening import normalized_basis
from nltiming.sampling.numpyro import _sample_timing_coord, sample_timing
from nltiming.partition import resolve_partition
from nltiming.whitening import schur_delta_wls


class _Pulsar:
    def __init__(self):
        self.name = "J0000+0000"
        self.fitpars = ("F0", "F1", "DM")
        self._toas = np.linspace(0.0, 1.0, 8)
        self._residuals = np.linspace(-2e-6, 2e-6, 8)
        self._toaerrs = np.full(8, 1.0e-6, dtype=float)
        self._freqs = np.full(8, 1400.0, dtype=float)
        self._flags = {"pta": np.array(["demo"] * 8, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 8, dtype="U8")
        self._state_id = "token-v1"
        self.backend_calls = []
        self.default_analytically_marginalize = ["F0", "DM"]

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
        self.backend_calls.append((engines, dict(kwargs)))
        if isinstance(engines, dict) and engines.get("pint") == "pint":
            return self._pint_backend
        return self._jug_backend


@pytest.fixture
def pulsar():
    return _Pulsar()


def _monkeypatch_numpyro(monkeypatch, sample_value):
    calls = {"sample": [], "factor": [], "deterministic": []}

    def _sample(name, dist):
        calls["sample"].append((name, dist))
        return sample_value

    def _factor(name, value):
        calls["factor"].append((name, value))

    def _deterministic(name, value):
        calls["deterministic"].append((name, value))

    monkeypatch.setattr("numpyro.sample", _sample)
    monkeypatch.setattr("numpyro.factor", _factor)
    monkeypatch.setattr("numpyro.deterministic", _deterministic)
    return calls


def _schur_fisher(pulsar, *, analytically_marginalize, variance):
    part = resolve_partition(pulsar, analytically_marginalize=analytically_marginalize)
    return schur_delta_wls(pulsar=pulsar, partition=part, variance=variance).fisher


def _autodiff_test_matrix(pulsar):
    design = np.asarray(pulsar.Mmat, dtype=float).copy()
    design[:, 0] *= 2.0
    design[:, 1] *= 3.0
    design[:, 2] *= 4.0
    return design


def _covariance_from_fisher(fisher: np.ndarray) -> np.ndarray:
    import scipy.linalg as sl

    cf = sl.cho_factor(fisher)
    return sl.cho_solve(cf, np.eye(fisher.shape[0], dtype=float))


def _z_space_fisher(space, delta_fisher):
    jac = np.asarray(
        space.prior_bijector.jacobian_diag_delta_from_z(space.linear.z0, np),
        dtype=float,
    )
    J = np.diag(jac)
    return J @ delta_fisher @ J


def test_component_config_only_build_and_with_engines():
    from nltiming.metric import WhiteningConfig

    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0"],
        prior_policy="wide_default",
        whitening=WhiteningConfig(),
        name="timing",
    )
    swapped = ntm.with_engines({"tempo2": "jug", "pint": "pint"})

    assert ntm.engines == {"tempo2": "jug", "pint": "jug"}
    assert swapped.engines == {"tempo2": "jug", "pint": "pint"}
    assert swapped.transform == ntm.transform
    assert swapped.analytically_marginalize == ntm.analytically_marginalize
    assert swapped.prior_policy == ntm.prior_policy
    assert swapped.whitening == ntm.whitening


def test_space_cached_and_invalidated_by_state_id(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    first = ntm.for_pulsar(pulsar).space
    second = ntm.for_pulsar(pulsar).space
    assert first is second

    pulsar._state_id = "token-v2"
    third = ntm.for_pulsar(pulsar).space
    assert third is not first


def test_whitening_config_roundtrip_and_space_shape(pulsar):
    """WhiteningConfig is the serializable config; dict whitening_config is gone."""
    from nltiming.metric import WhiteningConfig

    ntm_default = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
    )
    ntm_toa = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        whitening=WhiteningConfig(reference_noise="toa_errors"),
    )
    sp_default = ntm_default.for_pulsar(pulsar).space
    sp_toa = ntm_toa.for_pulsar(pulsar).space

    assert sp_default.linear.C.shape == (1, 1)
    assert sp_toa.linear.C.shape == (1, 1)
    assert ntm_default.whitening.as_dict() == ntm_toa.whitening.as_dict()
    # Default and explicit toa_errors configs produce the same linear map.
    np.testing.assert_allclose(sp_default.linear.C, sp_toa.linear.C)


def test_whitening_builders_condition_fisher_to_unit_scale(pulsar):
    analytically_marginalize_cfg = None
    ntm_default = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=analytically_marginalize_cfg,
    )

    default_fisher = _schur_fisher(
        pulsar,
        analytically_marginalize=analytically_marginalize_cfg,
        variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
    )

    for space, fisher in ((ntm_default.for_pulsar(pulsar).space, default_fisher),):
        fisher_z = _z_space_fisher(space, fisher)
        # Posterior metric (§5.3): C whitens F_z + I (the PIT-prior curvature),
        # not the likelihood Fisher alone.
        posterior = fisher_z + np.eye(len(space.names))
        conditioned = space.linear.C.T @ posterior @ space.linear.C
        np.testing.assert_allclose(
            conditioned,
            np.eye(len(space.names)),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_autodiff_design_matrix_method_feeds_whitening(pulsar):
    autodiff_matrix = _autodiff_test_matrix(pulsar)
    pulsar._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    resolved = ntm.for_pulsar(pulsar)
    np.testing.assert_allclose(resolved.design_matrix, autodiff_matrix)

    part = resolve_partition(pulsar, analytically_marginalize=["F0", "DM"])
    expected_fisher = schur_delta_wls(
        pulsar=pulsar,
        partition=part,
        variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
        design_matrix=autodiff_matrix,
    ).fisher
    fisher_z = _z_space_fisher(resolved.space, expected_fisher)
    posterior = fisher_z + np.eye(len(resolved.space.names))
    conditioned = resolved.space.linear.C.T @ posterior @ resolved.space.linear.C
    np.testing.assert_allclose(
        conditioned,
        np.eye(len(resolved.space.names)),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_standardized_builder_uses_z_space_marginal_scales(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    space = ntm.for_pulsar(pulsar).space
    fisher = _schur_fisher(
        pulsar,
        analytically_marginalize=None,
        variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
    )
    fisher_z = _z_space_fisher(space, fisher)
    # Standardized posterior scales are the marginal sigmas of (F_z + I)^-1.
    covariance_z = _covariance_from_fisher(fisher_z + np.eye(len(space.names)))

    assert np.allclose(space.linear.C, np.diag(np.diag(space.linear.C)))
    np.testing.assert_allclose(
        np.diag(space.linear.C),
        np.sqrt(np.diag(covariance_z)),
    )


def test_cheat_wls_prior_is_wide_uniform_box(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    block = ntm.for_pulsar(pulsar).priors
    fisher = _schur_fisher(
        pulsar,
        analytically_marginalize=None,
        variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
    )
    expected_stds = np.sqrt(np.diag(_covariance_from_fisher(fisher)))

    assert set(block.sources.values()) == {"cheat_wls"}
    assert all(prior.family == "uniform" for prior in block.priors)
    # Flat box of half-width scale * sigma, centered on the par-file value
    # (F0/F1/DM are physically unbounded here, so no clipping occurs).
    half_widths = np.array(
        [(prior.upper - prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    centers = np.array(
        [(prior.upper + prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    np.testing.assert_allclose(
        half_widths, ntm.cheat_prior_scale * expected_stds, rtol=1e-6
    )
    np.testing.assert_allclose(centers, 0.0, atol=1e-12)


def test_autodiff_design_matrix_method_feeds_cheat_prior_widths(pulsar):
    autodiff_matrix = _autodiff_test_matrix(pulsar)
    pulsar._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )

    block = ntm.for_pulsar(pulsar).priors
    part = resolve_partition(pulsar, analytically_marginalize=None)
    expected_fisher = schur_delta_wls(
        pulsar=pulsar,
        partition=part,
        variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
        design_matrix=autodiff_matrix,
    ).fisher
    expected_stds = np.sqrt(np.diag(_covariance_from_fisher(expected_fisher)))
    half_widths = np.array(
        [(prior.upper - prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    np.testing.assert_allclose(
        half_widths, ntm.cheat_prior_scale * expected_stds, rtol=1e-6
    )


def test_cheat_prior_box_clipped_to_physical_bounds():
    class _BoundedHost(_Pulsar):
        def __init__(self):
            super().__init__()
            self.fitpars = ("ECC", "F1", "DM")
            self._ecc_ref = 1.0e-5
            model = LinearModel.from_design(
                fitpars=self.fitpars,
                design=self._design,
                theta_exact={"ECC": repr(self._ecc_ref), "F1": "1.0", "DM": "5.0"},
            )
            self._jug_backend = LinearizedJugEngine.from_linear_model(model)

    bounded = _BoundedHost()
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F1", "DM"],
        name="timing",
    )
    block = ntm.for_pulsar(bounded).priors
    fisher = _schur_fisher(
        bounded,
        analytically_marginalize=["F1", "DM"],
        variance=np.asarray(bounded.toaerrs, dtype=float) ** 2,
    )
    sigma_ecc = float(np.sqrt(_covariance_from_fisher(fisher)[0, 0]))
    half = ntm.cheat_prior_scale * sigma_ecc

    ecc_prior = block.priors[block.names.index("ECC")]
    assert ecc_prior.family == "uniform"
    # 50 sigma extends below 0, so the lower edge is clipped to ECC >= 0.
    assert half > bounded._ecc_ref
    np.testing.assert_allclose(ecc_prior.lower, -bounded._ecc_ref, rtol=1e-9)
    np.testing.assert_allclose(ecc_prior.upper, half, rtol=1e-6)


def test_discovery_signals_delta_only_and_jax_gate(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    signals = ntm.for_pulsar(pulsar).discovery_signals()
    delay = signals[-1]
    output = np.asarray(delay({f"{pulsar.name}_timing_F1": 0.25}), dtype=float)
    expected = -pulsar._jug_backend.residual_delta(np.array([0.0, 0.25, 0.0]))
    np.testing.assert_allclose(output, expected)

    ntm_nonjax = NonLinearTimingModel(
        engines={"tempo2": "jug", "pint": "pint"},
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    with pytest.raises(ValueError, match="JAX-capable engine"):
        ntm_nonjax.for_pulsar(pulsar).discovery_signals()


def test_autodiff_design_matrix_method_feeds_discovery_gp_basis(pulsar, monkeypatch):
    autodiff_matrix = _autodiff_test_matrix(pulsar)
    pulsar._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    captured = {}

    def fake_makegp_improper(psr, basis, *, constant, name):
        captured["basis"] = np.asarray(basis, dtype=float)
        captured["constant"] = constant
        captured["name"] = name
        return "gp"

    monkeypatch.setattr("discovery.signals.makegp_improper", fake_makegp_improper)
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    signals = ntm.for_pulsar(pulsar).discovery_signals()

    assert signals[0] == "gp"
    np.testing.assert_allclose(
        captured["basis"], normalized_basis(autodiff_matrix[:, [0, 2]])
    )
    assert not hasattr(pulsar, "iisort")


def test_all_analytically_marginalized_paths(pulsar):
    ntm = NonLinearTimingModel(
        engines={"tempo2": "jug", "pint": "pint"},
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    assert len(ntm.for_pulsar(pulsar).discovery_signals()) == 1
    ent = ntm.enterprise_signal()
    bound = ent(pulsar)
    assert hasattr(bound, "get_basis")
    assert ntm.for_pulsar(pulsar).timing_param_keys() == ()


def test_non_timing_params_and_timing_param_keys_are_plain_set_subtraction(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    keys = ntm.for_pulsar(pulsar).timing_param_keys()
    assert keys[0] == f"{pulsar.name}_timing_x"
    assert keys[1:] == (f"{pulsar.name}_timing_F1",)
    params = ("efac", keys[0], "gamma", keys[1], "log10_A")
    assert ntm.for_pulsar(pulsar).non_timing_params(params) == (
        "efac",
        "gamma",
        "log10_A",
    )

    ntm_all_marg = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    plain = ("efac", "gamma", "log10_A")
    assert ntm_all_marg.for_pulsar(pulsar).timing_param_keys() == ()
    assert ntm_all_marg.for_pulsar(pulsar).non_timing_params(plain) == plain


def test_sample_timing_x_site_samples_and_injects_delta_deterministic(
    pulsar, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([0.2]))

    out = sample_timing(ntm.for_pulsar(pulsar), {"efac": 1.0})

    assert f"{pulsar.name}_timing_F1" in out
    assert out["efac"] == 1.0
    assert f"{pulsar.name}_timing_x" not in out
    assert calls["sample"][0][0] == f"{pulsar.name}_timing_x"
    # the x-coordinate MVN's own log_prob equals space.logprior_coord, so no
    # second prior factor is added for this coord
    assert calls["factor"] == []
    assert calls["deterministic"][0][0] == f"{pulsar.name}_timing_F1_delta"
    np.testing.assert_allclose(
        calls["deterministic"][0][1], out[f"{pulsar.name}_timing_F1"]
    )


def test_sample_timing_delta_coord_has_exactly_one_prior_factor(pulsar, monkeypatch):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([0.0]))

    sample_timing(ntm.for_pulsar(pulsar), {})

    assert len(calls["factor"]) == 1
    assert calls["factor"][0][0] == f"{pulsar.name}_timing_delta_logprior"


def test_sample_timing_noop_when_no_sampled(pulsar, monkeypatch):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([]))
    params = {"efac": 1.0}
    out = sample_timing(ntm.for_pulsar(pulsar), params)
    assert out is params
    assert calls["sample"] == []
    assert calls["factor"] == []


def test_set_prior_validated_against_sampled_partition(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        prior_override_policy="strict",
        name="timing",
    )
    ntm.set_prior("F0", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="non-sampled"):
        ntm.for_pulsar(pulsar).space


def test_set_prior_unknown_name_raises(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        prior_override_policy="strict",
        name="timing",
    )
    ntm.set_prior("F11", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="unknown fit parameters"):
        ntm.for_pulsar(pulsar).space


def test_enterprise_signal_forwards_engines(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    ent = ntm.enterprise_signal()
    _ = ent(pulsar)
    # The Enterprise likelihood interface consumes the ctx, so its engine is built
    # with the model's full timing_engine kwargs (not a bare re-query).
    engines_seen, kwargs_seen = pulsar.backend_calls[-1]
    assert engines_seen == {"tempo2": "jug", "pint": "jug"}
    assert kwargs_seen["design_matrix_method"] == "analytic"
    assert kwargs_seen["subtract_tzr"] is False


def test_autodiff_design_matrix_method_feeds_enterprise_gp_basis(pulsar):
    autodiff_matrix = _autodiff_test_matrix(pulsar)
    pulsar._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    ent = ntm.enterprise_signal()
    bound = ent(pulsar)

    np.testing.assert_allclose(
        bound.get_basis(), normalized_basis(autodiff_matrix[:, [0, 2]])
    )


def test_sample_timing_x_site_has_vector_event_shape(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    def model():
        sample_timing(ntm.for_pulsar(pulsar), {})

    substituted = handlers.substitute(
        model,
        data={f"{pulsar.name}_timing_x": np.array([0.0])},
    )
    trace = handlers.trace(handlers.seed(substituted, jr.PRNGKey(0))).get_trace()
    site = trace[f"{pulsar.name}_timing_x"]
    assert tuple(site["fn"].batch_shape) == ()
    assert tuple(site["fn"].event_shape) == (1,)


def test_timing_coord_distribution_log_prob_matches_logprior_coord(pulsar):
    """The x/z coordinate distributions built by _sample_timing_coord must have
    log_prob equal to space.logprior_coord, including normalization (§6.2) —
    across diagonal standardization and non-diagonal whitening."""
    cases = [("standardized", "x"), ("whitening", "x"), ("none", "z")]
    for transform, coord in cases:
        ntm = NonLinearTimingModel(
            engines="jug",
            transform=transform,
            analytically_marginalize=["DM"],
            name="timing",
        )
        ctx = ntm.for_pulsar(pulsar)

        def model():
            _sample_timing_coord(ctx, coord=coord)

        trace = handlers.trace(handlers.seed(model, jr.PRNGKey(0))).get_trace()
        site_name = ctx.latent_name_for_coord(coord)
        distribution = trace[site_name]["fn"]
        q = trace[site_name]["value"]

        got = float(distribution.log_prob(q))
        expected = float(ctx.space.logprior_coord(jnp.asarray(q), jnp, coord=coord))
        assert np.isclose(got, expected, rtol=1e-6, atol=1e-8), (transform, coord)


def test_timing_coord_x_and_z_add_no_extra_prior_factor(pulsar, monkeypatch):
    for transform, coord in [("standardized", "x"), ("whitening", "x"), ("none", "z")]:
        ntm = NonLinearTimingModel(
            engines="jug",
            transform=transform,
            analytically_marginalize=["DM"],
            name="timing",
        )
        calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.zeros(2))
        _sample_timing_coord(ntm.for_pulsar(pulsar), coord=coord)
        assert calls["factor"] == [], (transform, coord)


def _enterprise_pta(pulsar, *, transform, analytically_marginalize):
    from enterprise.signals import parameter, signal_base, white_signals

    efac = parameter.Uniform(0.1, 5.0)
    white = white_signals.MeasurementNoise(efac=efac)
    ntm = NonLinearTimingModel(
        engines="jug",
        transform=transform,
        analytically_marginalize=analytically_marginalize,
        name="timing",
    )
    return signal_base.PTA([(white + ntm.enterprise_signal())(pulsar)])


@pytest.mark.parametrize(
    ("transform", "analytically_marginalize"),
    [
        ("none", ["DM"]),
        ("standardized", ["DM"]),
        ("whitening", ["F0", "DM"]),  # ndim=1: size-1 block must stay a vector
        ("whitening", ["DM"]),  # ndim=2, non-diagonal C from the pulsar design matrix
    ],
)
def test_enterprise_parameters_sample_and_evaluate_full_pta(
    pulsar, transform, analytically_marginalize
):
    """p.sample() works for every NLT Enterprise parameter in every transform
    mode, over the complete PTA vector (noise + timing sampled jointly), and
    dict/flat-vector evaluations agree."""
    pta = _enterprise_pta(
        pulsar, transform=transform, analytically_marginalize=analytically_marginalize
    )
    x0 = np.hstack(
        [np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params]
    )
    assert x0.shape == (len(pta.param_names),)
    assert np.isfinite(pta.get_lnprior(x0))
    assert np.isfinite(pta.get_lnlikelihood(x0))

    mapped = pta.map_params(x0)
    assert pta.get_lnprior(x0) == pta.get_lnprior(mapped)
    assert pta.get_lnlikelihood(x0) == pta.get_lnlikelihood(mapped)


def test_scalar_timing_parameter_prior_draw_mode_is_component(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["DM"],
        name="timing",
    )
    bound = ntm.enterprise_signal()(pulsar)
    (param,) = [p for p in bound.params if p.name.endswith("_timing_F1")]

    assert param.size is None
    assert param.prior_draw_mode == "component"
    assert isinstance(param.sample(), float)


def test_whitening_vector_parameter_is_joint_and_size_one_stays_a_vector(pulsar):
    """A one-dimensional full-whitening block remains a vector end to end
    (acceptance criterion #5): size, sample(), and prior_draw_mode all agree
    it is a length-1 array, not a bare scalar."""
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    bound = ntm.enterprise_signal()(pulsar)
    (param,) = [p for p in bound.params if p.name.endswith("_timing_x")]

    assert param.size == 1
    assert param.prior_draw_mode == "joint"

    sample = param.sample()
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (1,)


def test_whitening_vector_sample_and_ppf_describe_the_same_distribution(pulsar):
    """PPF draws and direct sampler draws must describe the same distribution
    (they compose through the same coord_from_cube map), and the declared
    log density must be finite and consistent for both."""
    from scipy import stats

    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["DM"],
        name="timing",
    )
    bound = ntm.enterprise_signal()(pulsar)
    (param,) = [p for p in bound.params if p.name.endswith("_timing_x")]
    assert param.size == 2

    rng = np.random.default_rng(0)
    n = 500
    direct_samples = np.array([param.sample() for _ in range(n)])
    ppf_samples = np.array(
        [param.get_ppf(rng.uniform(1e-6, 1 - 1e-6, size=2)) for _ in range(n)]
    )

    for axis in range(2):
        _, pvalue = stats.ks_2samp(direct_samples[:, axis], ppf_samples[:, axis])
        assert pvalue > 0.01

    assert np.isfinite(param.get_logpdf(direct_samples[0]))
    assert np.isfinite(param.get_logpdf(ppf_samples[0]))
