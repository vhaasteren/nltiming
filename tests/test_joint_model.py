"""Joint full-basis timing (Track J / §6): three-way partition, the local
timing block, and the joint model's one-affine-layer guard.

These are the engine-neutral unit tests. The full joint NumPyro run (density
exactness, whitening geometry, decode, run manifest) is a pulsar-integration
test in the metapulsar repo, where a discovery-native pulsar and likelihood are
available.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# LinearizedJugEngine import requires the jug extra.
pytest.importorskip("jug")

from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402


class _Pulsar:
    """A linear JAX-differentiable pulsar duck (F0, F1, DM)."""

    def __init__(self):
        self.name = "J1234+5678"
        self.fitpars = ("F0", "F1", "DM")
        n = 12
        t = np.linspace(0.0, 1.0, n)
        design = np.column_stack([np.ones(n), t - 0.5, np.sin(3.0 * t)])
        self._toas = t * 3.15e7 + 5.3e4
        self._residuals = 1e-6 * np.sin(5.0 * t)
        self._toaerrs = np.full(n, 1.0e-6)
        self._freqs = np.full(n, 1400.0)
        self._backend_flags = np.array(["demo"] * n, dtype="U8")
        self._flags = {"pta": self._backend_flags}
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "100.0", "F1": "-1e-15", "DM": "10.0"},
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

    def state_id(self):
        return "joint-token"

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


def _joint_ctx():
    ntm = NonLinearTimingModel(
        engines="jug",
        sample=["F0", "F1"],
        sample_linear="remaining",
        transform="none",
        name="timing",
    )
    return ntm, ntm.for_pulsar(_Pulsar())


def test_three_way_partition_via_model():
    _, ctx = _joint_ctx()
    assert ctx.partition.nonlinear_sampled == ("F0", "F1")
    assert ctx.partition.linear_sampled == ("DM",)
    assert ctx.sampled == ("F0", "F1", "DM")
    assert ctx.marginalized == ()
    # sampled_all is an alias of sampled (nonlinear + linear).
    assert ctx.sampled_all == ctx.sampled


def test_local_timing_block_is_negative_autodiff_jacobian():
    _, ctx = _joint_ctx()
    blk = ctx.local_timing_block()
    assert blk.dimension == 3
    assert blk.names == ("F0", "F1", "DM")
    assert blk.prior_precision == 1.0
    assert blk.joint_site == ctx.joint_site

    # W_z == -∂(residual_delta_jax(δ(z)))/∂z at z_ref (the exact engine path).
    idx = jnp.asarray(ctx.partition.idx_sampled)
    nfit = len(ctx.partition.fitpars)
    z_ref = jnp.asarray(blk.z_ref)

    def residual_of_z(z):
        delta = ctx.space.delta_from_z(z, jnp)
        full = jnp.zeros((nfit,)).at[idx].set(delta)
        return ctx.engine.residual_delta_jax(full)

    W_expected = -np.asarray(jax.jacfwd(residual_of_z)(z_ref))
    assert np.allclose(blk.basis, W_expected, rtol=1e-10, atol=1e-12)


def test_cross_term_sign(monkeypatch):
    """On a toy where a GP column equals the raw timing waveform ``M·J``, the
    transport's timing↔GP cross block is MINUS the GP–GP block iff W_z carries
    the ``-`` sign (§6.3). This is the single most load-bearing sign in J1.
    """
    import discovery as ds

    ds.config(kernels="metamath")
    from discovery import transport as dst

    _, ctx = _joint_ctx()
    blk = ctx.local_timing_block()  # W_z = -(M·J)
    F_gp = -np.asarray(blk.basis)  # a GP column bank equal to +(M·J)

    ref = dst.reference_noise(ctx.pulsar)  # N0 = toaerrs**2
    t_block = dst.array_block(
        blk.basis,
        index={ctx.joint_site: slice(0, blk.dimension)},
        conditioner_precision=1.0,
        name="timing",
    )
    g_block = dst.array_block(
        F_gp,
        index={"gp": slice(0, F_gp.shape[1])},
        conditioner_precision=1.0,
        name="gp",
    )
    tr = dst.Transport([t_block, g_block], reference_noise=ref, center=False)

    G0 = np.asarray(tr._G0)
    k = blk.dimension
    cross = G0[:k, k:]
    gpgp = G0[k:, k:]
    assert np.allclose(cross, -gpgp, rtol=1e-10, atol=1e-12)

    ds.config(kernels="matrix")


def test_joint_model_requires_identity_static_layer():
    """joint_model rejects a conditioned (non-identity) whitening layer before
    it touches the likelihood (the one-affine-layer invariant, §5.5)."""
    from nltiming.metric import OneAffineLayerError
    from nltiming.sampling import numpyro as N

    ntm = NonLinearTimingModel(
        engines="jug",
        sample=["F0", "F1"],
        sample_linear="remaining",
        transform="whitening",
        name="timing",
    )
    ctx = ntm.for_pulsar(_Pulsar())  # conditioned => non-identity (C, c)

    class _FakeLikelihood:
        sampled_gps: list = []

        class clogL:
            params: list = []

    with pytest.raises(OneAffineLayerError):
        N.joint_model(_FakeLikelihood(), ctx)
