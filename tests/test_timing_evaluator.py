"""Tests for the interactive, engine-independent timing evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming import TimingEvaluator
from nltiming.bijectors import PriorBijector
from nltiming.space import ParameterSpace


class LinearJaxBackend:
    backend_name = "jug"
    fitpars = ("F0", "F1")
    native_units = {"F0": "Hz", "F1": "Hz / s"}

    def __init__(self, design):
        self._design = np.asarray(design, dtype=float)

    def reference_theta(self):
        return np.asarray([10.0, -1.0e-15])

    def reference_theta_exact(self):
        return {"F0": "10.0000000000000001", "F1": "-1e-15"}

    def residual_delta(self, delta_theta):
        return self._design @ np.asarray(delta_theta)

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        return jnp.asarray(self._design) @ jnp.asarray(delta_theta)

    def design_matrix(self, params=None):
        return self._design

    def precision_critical_fitpars(self):
        return frozenset({"F0"})


class NonlinearJaxBackend(LinearJaxBackend):
    def residual_delta(self, delta_theta):
        delta = np.asarray(delta_theta)
        linear = self._design @ delta
        curved = np.array([0.5 * delta[0] ** 2, 0.0, delta[0] ** 2, 0.0])
        return linear + curved

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        linear = jnp.asarray(self._design) @ delta
        curved = jnp.asarray([0.5 * delta[0] ** 2, 0.0, delta[0] ** 2, 0.0])
        return linear + curved


class EvaluatorPulsar:
    name = "J1234+5678"
    fitpars = ["F0", "F1"]

    def __init__(self):
        self.Mmat = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        self.toas = np.arange(4.0)
        self.toaerrs = np.ones(4)
        self.freqs = np.full(4, 1400.0)
        self.flags = {"pta": np.asarray(["a"] * 4)}
        self.backend_flags = np.asarray(["a"] * 4)
        self.residuals = self.Mmat @ np.asarray([0.2, -0.05])
        self._backend = LinearJaxBackend(self.Mmat)

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend

    def timing_parameter_mapping(self):
        return {"F0": {"pta_a": "F0"}, "F1": {"pta_a": "F1"}}


class NonlinearEvaluatorPulsar(EvaluatorPulsar):
    def __init__(self):
        super().__init__()
        self._backend = NonlinearJaxBackend(self.Mmat)


class RankDeficientEvaluatorPulsar(EvaluatorPulsar):
    def __init__(self):
        super().__init__()
        self.Mmat = np.asarray(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=float
        )
        self.residuals = self.Mmat @ np.asarray([0.2, 0.0])
        self._backend = LinearJaxBackend(self.Mmat)


def test_evaluate_partial_delta_and_absolute_mapping():
    timing = TimingEvaluator(EvaluatorPulsar())

    delta = timing.evaluate({"F0": 0.25})
    absolute = timing.evaluate({"F0": 10.25}, frame="absolute")

    np.testing.assert_allclose(delta.delta, [0.25, 0.0])
    np.testing.assert_allclose(absolute.delta, delta.delta)
    np.testing.assert_allclose(delta.residuals, timing.pulsar.residuals + 0.25)
    np.testing.assert_allclose(delta.delay, -delta.residual_delta)
    assert timing.reference_exact["F0"] == "10.0000000000000001"
    assert timing.parameters["F0"].sessions == ("pta_a",)


def test_capabilities_and_autodiff_jacobian():
    timing = TimingEvaluator(EvaluatorPulsar())

    assert timing.capabilities.jax
    assert timing.capabilities.autodiff_jacobian
    assert timing.capabilities.session_engines == {"J1234+5678": "jug"}
    np.testing.assert_allclose(timing.jacobian(method="reference"), timing.pulsar.Mmat)
    np.testing.assert_allclose(
        timing.jacobian({"F0": 0.2}, method="autodiff"), timing.pulsar.Mmat
    )
    with pytest.raises(ValueError, match="reference-point"):
        timing.jacobian({"F0": 0.2}, method="analytic")


def test_scan_exposes_residual_and_white_statistics():
    timing = TimingEvaluator(EvaluatorPulsar())
    scan = timing.scan("F0", [-0.2, 0.0, 0.2])

    assert scan.residuals.shape == (3, 4)
    assert scan.residual_deltas.shape == (3, 4)
    assert scan.rms.shape == (3,)
    assert scan.weighted_rms.shape == (3,)
    assert scan.white_chi2.shape == (3,)


def test_fit_is_immutable_and_recovers_linear_solution():
    pulsar = EvaluatorPulsar()
    original = pulsar.residuals.copy()
    timing = TimingEvaluator(pulsar)

    supplied_initial = np.zeros(2)
    result = timing.fit(
        ["F0", "F1"], initial=supplied_initial, jacobian_method="reference"
    )

    assert result.converged
    np.testing.assert_allclose(result.best_fit.delta, [-0.2, 0.05], atol=1e-12)
    np.testing.assert_allclose(result.best_fit.residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(pulsar.residuals, original)
    np.testing.assert_allclose(supplied_initial, 0.0)
    assert result.covariance.shape == (2, 2)
    assert not result.best_fit.delta.flags.writeable
    assert not result.covariance.flags.writeable
    with pytest.raises(TypeError):
        result.uncertainties["F0"] = 1.0


def test_jacobian_z_scales_physical_columns_by_prior_derivative():
    timing = TimingEvaluator(EvaluatorPulsar())
    space = ParameterSpace.build(
        theta_ref_mapping=timing.reference_exact,
        prior_bijector=PriorBijector.from_normal(
            names=timing.fitpars,
            means=np.zeros(2, dtype=float),
            stds=np.array([2.0, 0.5], dtype=float),
        ),
    )

    jacobian_z = timing.jacobian_z(space, method="reference")

    np.testing.assert_allclose(
        jacobian_z,
        timing.pulsar.Mmat * np.array([2.0, 0.5], dtype=float)[None, :],
    )


def test_fit_z_recovers_linear_solution_and_covariances():
    pulsar = EvaluatorPulsar()
    timing = TimingEvaluator(pulsar)
    space = ParameterSpace.build(
        theta_ref_mapping=timing.reference_exact,
        prior_bijector=PriorBijector.from_normal(
            names=timing.fitpars,
            means=np.zeros(2, dtype=float),
            stds=np.array([2.0, 0.5], dtype=float),
        ),
    )

    result = timing.fit_z(space, jacobian_method="reference")
    expected_weighted_jacobian = pulsar.Mmat * np.array([2.0, 0.5])[None, :]
    expected_covariance_z = np.linalg.pinv(
        expected_weighted_jacobian.T @ expected_weighted_jacobian
    )
    derivative = np.diag([2.0, 0.5])

    assert result.parameters == ("F0", "F1")
    assert result.iterations == 1
    assert result.covariance_coord == "z"
    assert result.rank == 2
    np.testing.assert_allclose(result.z_best, [-0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(result.best_fit.delta, [-0.2, 0.05], atol=1e-12)
    np.testing.assert_allclose(result.best_fit.residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.covariance, expected_covariance_z)
    np.testing.assert_allclose(
        result.covariance_delta, derivative @ expected_covariance_z @ derivative.T
    )
    assert result.singular_values.shape == (2,)
    assert not result.z_best.flags.writeable
    assert not result.covariance.flags.writeable
    with pytest.raises(TypeError):
        result.uncertainties["F0"] = 1.0


def test_fit_z_can_fit_subset_without_moving_other_space_axes():
    timing = TimingEvaluator(EvaluatorPulsar())
    space = ParameterSpace.build(
        theta_ref_mapping=timing.reference_exact,
        prior_bijector=PriorBijector.from_normal(
            names=timing.fitpars,
            means=np.zeros(2, dtype=float),
            stds=np.ones(2, dtype=float),
        ),
    )

    result = timing.fit_z(
        space,
        parameters=["F0"],
        initial={"F1": 0.25},
        jacobian_method="reference",
    )

    np.testing.assert_allclose(result.z_best[1], 0.25)
    assert result.parameters == ("F0",)
    assert result.covariance.shape == (1, 1)
    assert result.covariance_delta.shape == (1, 1)


def test_fit_z_reports_rank_and_singular_values_from_final_jacobian():
    timing = TimingEvaluator(NonlinearEvaluatorPulsar())
    space = ParameterSpace.build(theta_ref_mapping=timing.reference_exact)

    result = timing.fit_z(space, jacobian_method="autodiff")
    final_weighted = (
        timing.jacobian_z(space, result.z_best, method="autodiff")
        / timing.pulsar.toaerrs[:, None]
    )
    pre_step_weighted = (
        timing.jacobian_z(space, result.z_initial, method="autodiff")
        / timing.pulsar.toaerrs[:, None]
    )

    np.testing.assert_allclose(
        result.singular_values, np.linalg.svd(final_weighted, compute_uv=False)
    )
    assert not np.allclose(
        np.linalg.svd(final_weighted, compute_uv=False),
        np.linalg.svd(pre_step_weighted, compute_uv=False),
    )
    assert result.rank == np.linalg.matrix_rank(final_weighted)


def test_fit_z_rank_deficient_covariance_and_fixed_iteration_count(monkeypatch):
    timing = TimingEvaluator(RankDeficientEvaluatorPulsar())
    space = ParameterSpace.build(theta_ref_mapping=timing.reference_exact)
    calls = 0
    original_lstsq = np.linalg.lstsq

    def counting_lstsq(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_lstsq(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counting_lstsq)
    result = timing.fit_z(space, jacobian_method="reference", iterations=3)

    assert calls == 3
    assert result.iterations == 3
    assert result.rank == 1
    assert result.singular_values[-1] < 1e-12
    assert np.linalg.matrix_rank(result.covariance) == 1


def test_unknown_parameter_and_wrong_vector_shape_are_clear():
    timing = TimingEvaluator(EvaluatorPulsar())
    with pytest.raises(KeyError, match="matches no fitpar"):
        timing.evaluate({"NOPE": 1.0})
    with pytest.raises(ValueError, match="length 2"):
        timing.evaluate([1.0])
    space = ParameterSpace.build(theta_ref_mapping=timing.reference_exact)
    with pytest.raises(KeyError, match="matches no ParameterSpace"):
        timing.fit_z(space, initial={"NOPE": 1.0})
