"""Stage 1 internal geometry metric kernel (feature §8.3, §16.1).

These tests exercise the target-only gradient / conditional-Hessian /
cross-Hessian / conditional-identity kernel against pure-NumPyro oracle models,
independent of any timing engine. The public certifier is layered on this kernel
in a later stage; the oracles here pin the kernel's behavior so the refactor
cannot silently change it.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("numpyro")

from nltiming.geometry import (  # noqa: E402
    conditional_identity_spread,
    deterministic_xi_probes,
    target_metrics_at,
)


def _oracle(dim, *, curvature=0.0, cross=0.0):
    """A NumPyro model whose total log density is

        log p(xi, eta) = -1/2 ||xi||^2 - curvature * sum(xi^4)
                         - cross * eta * sum(xi) + const(eta)

    The ``xi`` Normal site supplies the ``-1/2||xi||^2``; the factor adds the
    rest. ``curvature=cross=0`` is the exactly-whitened linear-Gaussian oracle.
    """
    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist

    def model():
        eta = numpyro.sample("eta", dist.Uniform(-5.0, 5.0))
        xi = numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([dim]).to_event(1))
        numpyro.factor(
            "extra",
            -curvature * jnp.sum(xi**4) - cross * eta * jnp.sum(xi),
        )

    model.xi_site = "xi"
    model.hyper_sites = ("eta",)
    return model


def test_linear_gaussian_oracle_is_unit_whitened():
    dim = 4
    model = _oracle(dim)
    m = target_metrics_at(model, xi=np.zeros(dim), hyper={"eta": 0.0})
    assert m.xi_gradient_inf_norm < 1e-8
    assert m.xi_hessian_eigen_min == pytest.approx(1.0, abs=1e-6)
    assert m.xi_hessian_eigen_max == pytest.approx(1.0, abs=1e-6)
    assert m.xi_eta_cross_operator_norm < 1e-8
    assert abs(m.conditional_identity) < 1e-8

    # And the conditional identity holds across the full deterministic probe set.
    probes = deterministic_xi_probes(dim)
    assert conditional_identity_spread(model, hyper={"eta": 0.0}, xi_points=probes) < 1e-6


def test_zero_slice_identity_cannot_hide_nonzero_xi_curvature():
    """Regression for the J1640 diagnostic gap (§14.5): a target equal to the
    whitened surrogate at ``xi=0`` but with quartic curvature away from zero.
    The zero-slice gradient and Hessian look perfectly whitened; the off-zero
    conditional-identity metric exposes the curvature."""
    dim = 3
    model = _oracle(dim, curvature=0.1)

    at_zero = target_metrics_at(model, xi=np.zeros(dim), hyper={"eta": 0.0})
    # Zero slice is indistinguishable from the whitened oracle.
    assert at_zero.xi_gradient_inf_norm < 1e-8
    assert at_zero.xi_hessian_eigen_min == pytest.approx(1.0, abs=1e-6)
    assert at_zero.xi_hessian_eigen_max == pytest.approx(1.0, abs=1e-6)
    assert abs(at_zero.conditional_identity) < 1e-8

    # Off zero, the quartic term is revealed: D = -curvature * sum(xi^4).
    xi = np.array([1.0, -1.0, 0.5])
    off = target_metrics_at(model, xi=xi, hyper={"eta": 0.0})
    expected = -0.1 * float(np.sum(xi**4))
    assert off.conditional_identity == pytest.approx(expected, rel=1e-6)

    probes = deterministic_xi_probes(dim)
    spread = conditional_identity_spread(model, hyper={"eta": 0.0}, xi_points=probes)
    assert spread > 0.1


def test_cross_hessian_detects_xi_eta_coupling():
    dim = 2
    decoupled = target_metrics_at(_oracle(dim), xi=np.zeros(dim), hyper={"eta": 0.0})
    coupled = target_metrics_at(
        _oracle(dim, cross=0.7), xi=np.zeros(dim), hyper={"eta": 0.0}
    )
    assert decoupled.xi_eta_cross_operator_norm < 1e-8
    assert coupled.xi_eta_cross_operator_norm > 1e-2


def test_probe_set_is_deterministic_and_sized():
    a = deterministic_xi_probes(5)
    b = deterministic_xi_probes(5)
    assert len(a) == 2 * 5 + 9
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert np.array_equal(a[0], np.zeros(5))


def test_kernel_requires_site_metadata():
    def bare_model():
        import numpyro
        from numpyro import distributions as dist

        numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([2]).to_event(1))

    with pytest.raises(ValueError, match="xi_site"):
        target_metrics_at(bare_model, xi=np.zeros(2), hyper={})
