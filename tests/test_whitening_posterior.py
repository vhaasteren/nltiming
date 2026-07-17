"""Posterior-metric whitening (§5.3) and JAX-safe static pullback (§3).

These are unit tests over ``nltiming.whitening.posterior_linear_transform`` and
the custom NumPyro pullback distribution. They pin the correct local
Gauss-Newton/Fisher posterior geometry ``C^T (F_z + I) C = I`` and the
JAX-native, no-``C.T@C`` induced-``x`` prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.bijectors import PriorBijector, WhiteningLinear
from nltiming.space import ParameterSpace
from nltiming.whitening import posterior_linear_transform


def _normal_bijector(names, stds):
    return PriorBijector.from_normal(
        names=tuple(names),
        means=np.zeros(len(names), dtype=float),
        stds=np.asarray(stds, dtype=float),
    )


def _random_spd(rng, ndim, *, scale=1.0):
    a = rng.standard_normal((ndim, ndim))
    return scale * (a @ a.T + ndim * np.eye(ndim))


# --------------------------------------------------------------------------
# Whitening metric (§10 whitening tests 1, 2, 3, 5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["whitening", "standardized"])
def test_posterior_metric_whitens_fz_plus_identity(mode):
    """§10.1: ``C^T (F_z + I) C = I`` for the full posterior metric."""
    rng = np.random.default_rng(0)
    names = ("a", "b", "c")
    stds = np.array([1.0, 2.5, 0.4])
    bij = _normal_bijector(names, stds)
    fisher_delta = _random_spd(rng, 3, scale=1e3)

    linear, diag = posterior_linear_transform(
        fisher_delta, prior_bijector=bij, mode=mode
    )
    # z_e = 0 and jac = std for a zero-mean normal prior, so F_z = S F_delta S.
    jac = stds
    fisher_z = (jac[:, None] * jac[None, :]) * fisher_delta
    posterior = fisher_z + np.eye(3)
    posterior_cov = np.linalg.inv(posterior)
    if mode == "whitening":
        conditioned = linear.C.T @ posterior @ linear.C
        np.testing.assert_allclose(conditioned, np.eye(3), atol=1e-9)
    else:
        # Diagonal standardization sets each axis to its posterior marginal
        # sigma; it does not diagonalize the full metric.
        assert np.allclose(linear.C, np.diag(np.diag(linear.C)))
        np.testing.assert_allclose(
            np.diag(linear.C), np.sqrt(np.diag(posterior_cov)), rtol=1e-9
        )
    assert diag["expansion_point"] == "reference"
    assert diag["origin"] == "reference"


def test_null_likelihood_direction_gives_prior_scale():
    """§10.2: a null likelihood (F_delta = 0) leaves prior-scale geometry."""
    names = ("a", "b")
    bij = _normal_bijector(names, [1.0, 1.0])
    linear, _ = posterior_linear_transform(
        np.zeros((2, 2)), prior_bijector=bij, mode="whitening"
    )
    # H = 0 + I = I  ->  C = I, origin at the reference.
    np.testing.assert_allclose(linear.C, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(linear.z0, np.zeros(2), atol=1e-12)


def test_parameter_unit_rescaling_leaves_geometry_invariant():
    """§10.3: rescaling a parameter's native unit does not change the target.

    delta' = delta / a  =>  F_delta' = a^2 F_delta and prior std' = std / a,
    so F_z = std^2 F_delta is invariant and the transform is identical.
    """
    rng = np.random.default_rng(3)
    fisher = _random_spd(rng, 2, scale=10.0)
    std = np.array([1.3, 0.7])
    a = np.array([1e6, 1e-3])

    base, _ = posterior_linear_transform(
        fisher, prior_bijector=_normal_bijector(("a", "b"), std), mode="whitening"
    )
    scaled_fisher = (a[:, None] * a[None, :]) * fisher
    scaled, _ = posterior_linear_transform(
        scaled_fisher,
        prior_bijector=_normal_bijector(("a", "b"), std / a),
        mode="whitening",
    )
    np.testing.assert_allclose(base.C, scaled.C, rtol=1e-9, atol=1e-12)


def test_local_posterior_origin_is_guarded_and_reference_is_deterministic():
    """§10.5: local-posterior origin damps/guards; reference is deterministic."""
    names = ("a", "b")
    bij = _normal_bijector(names, [1.0, 1.0])
    fisher = np.diag([4.0, 9.0])
    # Huge score would push the raw Newton center far outside PIT support.
    score = np.array([1e6, -1e6])

    guarded, diag = posterior_linear_transform(
        fisher,
        prior_bijector=bij,
        mode="whitening",
        score_delta=score,
        origin="local_posterior",
    )
    assert diag["origin"] == "local_posterior"
    assert diag["guard_engaged"] is True
    assert np.all(np.abs(guarded.z0) < 7.0)  # strictly inside the PIT edge

    # auto with a score behaves like local_posterior; without one, reference.
    auto_with, d1 = posterior_linear_transform(
        fisher, prior_bijector=bij, mode="whitening", score_delta=score, origin="auto"
    )
    assert d1["origin"] == "local_posterior"
    auto_without, d2 = posterior_linear_transform(
        fisher, prior_bijector=bij, mode="whitening", origin="auto"
    )
    assert d2["origin"] == "reference"
    np.testing.assert_allclose(auto_without.z0, np.zeros(2))

    with pytest.raises(ValueError, match="requires a score_delta"):
        posterior_linear_transform(
            fisher, prior_bijector=bij, mode="whitening", origin="local_posterior"
        )


# --------------------------------------------------------------------------
# JAX-safe static pullback (§3, §10 JAX tests 1, 2, 3)
# --------------------------------------------------------------------------


def _highly_conditioned_C():
    # Lower-triangular factor spanning ~1e8 in scale (cond(C) ~ 1e8, so a
    # naive C.T @ C would square that to ~1e16).
    return np.array(
        [
            [1e4, 0.0, 0.0],
            [3.0, 1e-4, 0.0],
            [-2.0, 5.0, 1e-2],
        ],
        dtype=float,
    )


def test_pullback_logprob_matches_direct_and_space_density():
    """§10.6: induced-x log density matches the direct N(Cx+z0) pullback and
    the space's own ``logprior_coord`` for a highly conditioned C."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from nltiming.sampling.numpyro import ensure_x64, _static_pullback_distribution_cls

    ensure_x64()
    C = _highly_conditioned_C()
    z0 = np.array([0.2, -0.5, 1.1])
    linear = WhiteningLinear(C=C, z0=z0)
    names = ("a", "b", "c")
    space = ParameterSpace.build(
        {"a": "0.0", "b": "0.0", "c": "0.0"},
        prior_bijector=_normal_bijector(names, [1.0, 1.0, 1.0]),
        transform="whitening",
        linear_transform=linear,
    )

    dist = _static_pullback_distribution_cls()(C, z0, linear.logabsdet)
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.standard_normal(3)
        # Direct pullback, formed WITHOUT C.T @ C.
        z = C @ x + z0
        direct = -0.5 * float(z @ z) - 0.5 * 3 * np.log(2 * np.pi) + linear.logabsdet
        got = float(dist.log_prob(jnp.asarray(x)))
        assert got == pytest.approx(direct, rel=1e-9, abs=1e-6)
        # The space's PIT-based density agrees for every prior family.
        space_lp = float(space.logprior_coord(jnp.asarray(x), jnp, coord="x"))
        assert got == pytest.approx(space_lp, rel=1e-9, abs=1e-6)


def test_pullback_sample_roundtrips_and_is_jax_transformable():
    """§10 JAX 1: sample/log_prob pass jit, grad, jacfwd, vmap; sampled draws
    map back to standard-normal z."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from nltiming.sampling.numpyro import ensure_x64, _static_pullback_distribution_cls

    ensure_x64()
    C = _highly_conditioned_C()
    z0 = jnp.array([0.2, -0.5, 1.1])
    dist = _static_pullback_distribution_cls()(C, z0, WhiteningLinear(C=C).logabsdet)

    lp = jax.jit(dist.log_prob)
    g = jax.grad(lambda x: dist.log_prob(x))
    x0 = jnp.array([0.1, 0.2, -0.3])
    assert np.isfinite(float(lp(x0)))
    assert np.all(np.isfinite(np.asarray(g(x0))))
    jac = jax.jacfwd(lambda x: dist.log_prob(x))(x0)
    assert np.all(np.isfinite(np.asarray(jac)))

    xs = jnp.stack([x0, x0 + 0.5, x0 - 0.7])
    vals = jax.vmap(dist.log_prob)(xs)
    assert vals.shape == (3,)

    # Draws map back to unit normal z = C x + z0.
    draws = dist.sample(jax.random.PRNGKey(0), (2048,))
    z = np.asarray(draws) @ C.T + np.asarray(z0)
    np.testing.assert_allclose(z.mean(axis=0), np.zeros(3), atol=0.1)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(3), atol=0.15)


def test_pullback_gradients_finite_under_high_condition_without_ctc():
    """§10 JAX 3: highly conditioned C keeps log_prob gradients finite and the
    density never materializes C.T @ C (cond would be squared)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from nltiming.sampling.numpyro import ensure_x64, _static_pullback_distribution_cls

    ensure_x64()
    C = _highly_conditioned_C()
    z0 = jnp.zeros(3)
    dist = _static_pullback_distribution_cls()(C, z0, WhiteningLinear(C=C).logabsdet)
    grad = jax.grad(lambda x: dist.log_prob(x))(jnp.array([1e3, -2e2, 4.0]))
    assert np.all(np.isfinite(np.asarray(grad)))
