"""Slice-5 tests for record_physical_postprocess timing physical-value output."""

import numpy as np
import pytest

from nltiming.backends.base import LinearModel
from nltiming.backends.jug import LinearizedJugEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.sampling.numpyro import contribute_timing, record_physical_postprocess


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
        self._cache_token = "record-token"
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


def _patch_numpyro(monkeypatch, sample_value):
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


def test_record_physical_postprocess_timing_scope_returns_prefixed_theta_values(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    _patch_numpyro(monkeypatch, sample_value=np.array([0.25]))
    params = contribute_timing(ntm.bind(host), {})

    out = record_physical_postprocess(ntm.bind(host), params, scope="timing")

    assert f"{host.name}_timing_F1_theta_native" in out
    assert f"{host.name}_timing_F1_theta_display" in out
    expected_theta = ntm.bind(host).space.theta_from_delta(
        np.array([params[f"{host.name}_timing_F1"]])
    )
    np.testing.assert_allclose(
        out[f"{host.name}_timing_F1_theta_native"], expected_theta[0]
    )


def test_record_physical_postprocess_scope_all_raises(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    with pytest.raises(NotImplementedError, match="scope='all'"):
        record_physical_postprocess(ntm.bind(host), {}, scope="all")


def test_record_physical_postprocess_touches_no_numpyro_state(host, monkeypatch):
    """record_physical_postprocess is pure post-processing: it must not call
    numpyro.sample/factor/deterministic, so it works outside any trace."""
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    calls = _patch_numpyro(monkeypatch, sample_value=np.array([0.1]))
    params = contribute_timing(ntm.bind(host), {})
    n_sample = len(calls["sample"])
    n_factor = len(calls["factor"])
    n_deterministic = len(calls["deterministic"])

    out = record_physical_postprocess(ntm.bind(host), params, scope="timing")

    assert len(calls["sample"]) == n_sample
    assert len(calls["factor"]) == n_factor
    assert len(calls["deterministic"]) == n_deterministic
    assert len(out) == 2


def test_record_physical_postprocess_explicit_coord_handles_standardized_scalar_params(
    host,
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    x_value = 0.25
    params = {f"{host.name}_timing_F1": x_value}

    out = record_physical_postprocess(ntm.bind(host), params, scope="timing", coord="x")

    space = ntm.bind(host).space
    delta = space.delta_from_coord(np.array([x_value], dtype=float), np, coord="x")
    expected_theta = space.theta_from_delta(delta)
    assert f"{host.name}_timing_F1_theta_native" in out
    assert f"{host.name}_timing_F1_theta_display" in out
    np.testing.assert_allclose(
        out[f"{host.name}_timing_F1_theta_native"], expected_theta[0]
    )


def test_record_physical_postprocess_implicit_coord_handles_standardized_contribute_output(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    _patch_numpyro(monkeypatch, sample_value=np.array([0.2]))
    params = contribute_timing(ntm.bind(host), {})

    out = record_physical_postprocess(ntm.bind(host), params, scope="timing")

    # contribute_timing injects backend-facing delta keys; implicit standardized
    # record_physical_postprocess should interpret those as delta, not x.
    expected_theta = ntm.bind(host).space.theta_from_delta(
        np.array([params[f"{host.name}_timing_F1"]], dtype=float)
    )
    assert f"{host.name}_timing_F1_theta_native" in out
    assert f"{host.name}_timing_F1_theta_display" in out
    np.testing.assert_allclose(
        out[f"{host.name}_timing_F1_theta_native"], expected_theta[0]
    )


def test_record_physical_postprocess_invalid_coord_raises(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    with pytest.raises(ValueError, match="coord must be one of"):
        record_physical_postprocess(ntm.bind(host), {}, scope="timing", coord="invalid")
