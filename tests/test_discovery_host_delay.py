"""Host-callback Discovery delays (Vela/PINT stand-in: LinearTestEngine)."""

from __future__ import annotations

import numpy as np
import pytest

from _engine_stubs import JaxLinearTestEngine, LinearTestEngine
from nltiming import TimingInference, WhiteningConfig
from nltiming.engine_support import LinearModel
from nltiming.nonlinear_timing_model import TimingSpec
import nltiming.sampling as nlts

pytest.importorskip("discovery")
pytest.importorskip("jax")

import discovery as ds  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


class _Pulsar:
    def __init__(self):
        self.name = "J0000+0000"
        self.fitpars = ("Offset", "F1", "DM")
        self._toas = np.linspace(0.0, 1.0, 8)
        self._residuals = np.linspace(-2e-6, 2e-6, 8)
        self._toaerrs = np.full(8, 1.0e-6, dtype=float)
        self._freqs = np.full(8, 1400.0, dtype=float)
        self._flags = {"pta": np.array(["demo"] * 8, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 8, dtype="U8")
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
            theta_exact={"Offset": "0.0", "F1": "1.0", "DM": "5.0"},
        )
        self._jug_backend = JaxLinearTestEngine.from_linear_model(model)
        self._pint_backend = LinearTestEngine.from_linear_model(model)

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

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        if isinstance(engines, dict) and engines.get("pint") == "pint":
            return self._pint_backend
        return self._jug_backend


@pytest.fixture
def pulsar():
    return _Pulsar()


def _host_spec(**kwargs):
    return TimingSpec(
        engines={"tempo2": "jug", "pint": "pint"},
        name="timing",
        **kwargs,
    )


def _jax_spec(**kwargs):
    return TimingSpec(engines="jug", name="timing", **kwargs)


def _inference():
    return TimingInference.groups(delta_flat=["Offset", "DM"])


def test_host_callback_delay_matches_residual_delta(pulsar):
    ctx = _host_spec(inference=_inference()).for_pulsar(pulsar)
    delay = ctx.discovery_signals()[-1]
    params = {ctx.delay_keys[0]: 0.25}
    output = np.asarray(delay(params), dtype=float)
    full = ctx.engine_delta_map.full_engine_delta(np.array([0.25]), np)
    expected = -np.asarray(ctx.engine.residual_delta(full), dtype=float)
    np.testing.assert_allclose(output, expected)


def test_host_callback_runs_inside_jitted_discovery_logl(pulsar):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _host_spec(inference=_inference()).for_pulsar(pulsar)
    likelihood = ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement_simple(pulsar, noisedict, add_equad=False),
            *ctx.discovery_signals(),
        ]
    )
    params = {**noisedict, ctx.delay_keys[0]: 0.25}
    eager = float(likelihood.logL(params))
    compiled = jax.jit(likelihood.logL)
    jitted = float(compiled(params))
    assert np.isfinite(eager)
    np.testing.assert_allclose(jitted, eager, rtol=1e-10, atol=1e-12)


def test_host_callback_has_no_jvp(pulsar):
    ctx = _host_spec(inference=_inference()).for_pulsar(pulsar)
    delay = ctx.discovery_signals()[-1]
    key = ctx.delay_keys[0]

    def f(x):
        return jnp.sum(delay({key: x}))

    with pytest.raises((TypeError, ValueError, jax.errors.JaxRuntimeError)) as excinfo:
        jax.jacfwd(f)(jnp.float64(0.25))
    message = str(excinfo.value).lower()
    assert any(
        token in message
        for token in ("jvp", "jacobian", "differenti", "pure_callback", "callback")
    )


def test_jax_engine_does_not_use_pure_callback(pulsar, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("pure_callback should not be used")

    monkeypatch.setattr(jax, "pure_callback", _boom)
    ctx = _jax_spec(inference=_inference()).for_pulsar(pulsar)
    delay = ctx.discovery_signals()[-1]
    assert delay.nltiming_execution == "jax"
    value = np.asarray(delay({ctx.delay_keys[0]: 0.25}), dtype=float)
    assert value.shape == (len(pulsar.residuals),)

    def f(x):
        return jnp.sum(delay({ctx.delay_keys[0]: x}))

    grad = jax.jacfwd(f)(jnp.float64(0.25))
    assert np.isfinite(float(grad))


def test_host_callback_applies_engine_delta_map_on_host(pulsar):
    ntm = _host_spec(
        inference=TimingInference.groups(delta_flat=["Offset"], z_prior=["DM"]),
    )
    base = ntm.for_pulsar(pulsar, condition=False)
    ctx = base.with_expansion(
        delta={"F1": 0.0, "DM": 1.0e-3},
        source="explicit_delta",
    )
    captured = {}
    original = pulsar._pint_backend.residual_delta

    def spy(delta):
        captured["delta"] = np.asarray(delta, dtype=float)
        return original(delta)

    pulsar._pint_backend.residual_delta = spy
    delay = ctx.discovery_signals()[-1]
    delay({ctx.delay_keys[0]: 0.25})
    # Offset (flat) stays 0; F1 sampled; DM pinned at the expansion delta.
    np.testing.assert_allclose(captured["delta"], [0.0, 0.25, 1.0e-3])


def test_host_callback_rejects_wrong_residual_shape(pulsar):
    ctx = _host_spec(inference=_inference()).for_pulsar(pulsar)
    pulsar._pint_backend.residual_delta = lambda delta: np.zeros(3)
    delay = ctx.discovery_signals()[-1]
    with pytest.raises(
        (ValueError, jax.errors.JaxRuntimeError), match="residual shape"
    ):
        np.asarray(delay({ctx.delay_keys[0]: 0.25}))


@pytest.mark.parametrize(
    "layer", [None, WhiteningConfig()], ids=["identity", "whitening"]
)
def test_host_discovery_target_matches_direct_logl(pulsar, layer):
    nlts.numpyro.ensure_x64()
    noisedict = {f"{pulsar.name}_efac": 1.0}
    ctx = _host_spec(whitening=layer, inference=_inference()).for_pulsar(pulsar)
    likelihood = ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement_simple(pulsar, noisedict, add_equad=False),
            *ctx.discovery_signals(),
        ]
    )
    target = nlts.ptmcmc.discovery_target(likelihood, ctx, fixed=noisedict)
    q = np.full(target.dimension, 0.4)
    delta = np.asarray(ctx.space.delta_from_coord(q, np, coord=ctx.coord), dtype=float)
    params = dict(noisedict)
    params[ctx.delay_keys[0]] = float(delta[0])
    direct = float(likelihood.logL(params))
    np.testing.assert_allclose(target.loglikelihood(q), direct, rtol=1e-10)
    np.testing.assert_allclose(
        target.logprior(q),
        float(ctx.space.logprior_coord(q, np, coord=ctx.coord)),
        rtol=1e-12,
    )
    # Delay keys are delta, not the sampler coordinate.
    if abs(float(delta[0]) - 0.4) > 1e-12:
        wrong = dict(noisedict)
        wrong[ctx.delay_keys[0]] = 0.4
        assert float(likelihood.logL(wrong)) != pytest.approx(direct, rel=1e-8)
