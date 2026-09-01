"""Discovery likelihood interface for nonlinear timing.

This module builds Discovery-native likelihood signals from a bound
``TimingSpec`` pulsar: an optional improper GP for analytically
marginalized linear fit parameters, plus a nonlinear delay for the
numerically sampled block.

JAX timing engines emit a traced delay suitable for gradient-based sampling.
Host timing engines emit a value-only delay through ``jax.pure_callback``.
The latter supports a JAX-compiled Discovery marginal likelihood with
derivative-free samplers, but not NUTS or nltiming's joint and decentered
NumPyro models.

Priors
------
Discovery delay keys carry timing-engine-native ``delta_theta`` values; this module
does **not** attach timing priors to the likelihood. Sampled-parameter priors
live in ``ParameterSpace`` (from ``TimingSpec.space``) and are
added separately—for example via ``sample_timing``, which evaluates
``space.logprior_coord`` as a NumPyro factor.

With ``prior_policy="fallback"``, unresolved sampled priors use the reference-stack
*cheat* prior convention: a flat ``uniform`` on ``[center ± coordinate_policy.nonlinear_scale · σ]`` in
delta space (center = par-file reference, ``σ`` = par-file uncertainty with
WLS fallback), clipped to ``native_physical_bounds`` when applicable. The
whitening/standardized linear layer is a sampler reparameterization only; it
does not change the physical prior measure.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from nltiming.inference import TimingParameterPlan
from nltiming.protocols import JaxTimingEngine


def _sample_key(pulsar_name: str, name: str, fitpar: str) -> str:
    return f"{pulsar_name}_{name}_{fitpar}"


def _sample_keys(pulsar, name: str, partition: TimingParameterPlan) -> list[str]:
    return [_sample_key(pulsar.name, name, fitpar) for fitpar in partition.sampled]


def _build_jax_delay_callable(
    *,
    pulsar,
    engine: JaxTimingEngine,
    partition: TimingParameterPlan,
    name: str,
    engine_map,
) -> Callable[[dict[str, object]], object]:
    """Differentiable exact sampled delay ``d_anchor(z_s) = d(z_s, z_m,e)``.

    The ``engine_map`` (sampled-mode ``EngineDeltaMap``) scatters the sampled
    deltas, pins z-marginalized slots at their fixed expansion delta ``z_m,e``,
    holds delta-flat slots at zero, and applies any physical charts. The
    remaining z-marginal variation is carried by the separate ``W_m``
    standard-normal GP block.
    """
    keys = _sample_keys(pulsar, name, partition)

    def delay(params):
        import jax.numpy as jnp

        delta_sampled = jnp.asarray([params[key] for key in keys], dtype=float)
        return -engine.residual_delta_jax(
            engine_map.full_engine_delta(delta_sampled, jnp)
        )

    delay.params = list(keys)
    delay.nltiming_execution = "jax"
    delay.nltiming_differentiable = True
    return delay


def _build_host_delay_callable(
    *,
    pulsar,
    engine,
    partition: TimingParameterPlan,
    name: str,
    engine_map,
) -> Callable[[dict[str, object]], object]:
    """Value-only exact timing delay evaluated through ``jax.pure_callback``.

    The sampling-frame-to-engine map, physical charts, and ``residual_delta``
    all run on NumPy inside the callback so the numeric path matches Enterprise
    and no JVP is defined.
    """
    keys = _sample_keys(pulsar, name, partition)
    n_toa = len(np.asarray(pulsar.residuals))
    callback_lock = threading.Lock()

    def host_delay(delta_sampled):
        delta = np.asarray(delta_sampled, dtype=np.float64)
        if delta.shape != (len(keys),):
            raise ValueError(
                "host timing callback expected sampled delta shape "
                f"{(len(keys),)}, got {delta.shape}"
            )
        full_delta = engine_map.full_engine_delta(delta, np)
        # A pyvela session is treated as read-only but is not assumed to be
        # safe for concurrent callback execution within one process.
        with callback_lock:
            result = -np.asarray(engine.residual_delta(full_delta), dtype=np.float64)
        if result.shape != (n_toa,):
            raise ValueError(
                "host timing callback returned residual shape "
                f"{result.shape}, expected {(n_toa,)}"
            )
        return result

    def delay(params):
        import jax
        import jax.numpy as jnp

        delta_sampled = jnp.asarray(
            [params[key] for key in keys],
            dtype=jnp.float64,
        )
        output = jax.ShapeDtypeStruct((n_toa,), jnp.float64)
        return jax.pure_callback(host_delay, output, delta_sampled)

    delay.params = list(keys)
    delay.nltiming_execution = "host_callback"
    delay.nltiming_differentiable = False
    return delay


def _build_delay_callable(
    *,
    pulsar,
    engine,
    partition: TimingParameterPlan,
    name: str,
    engine_map,
) -> Callable[[dict[str, object]], object]:
    """Exact sampled delay ``d_anchor(z_s) = d(z_s, z_m,e)``."""
    if isinstance(engine, JaxTimingEngine):
        return _build_jax_delay_callable(
            pulsar=pulsar,
            engine=engine,
            partition=partition,
            name=name,
            engine_map=engine_map,
        )
    return _build_host_delay_callable(
        pulsar=pulsar,
        engine=engine,
        partition=partition,
        name=name,
        engine_map=engine_map,
    )


def _build_cm_delay(pulsar, name: str, c_m: np.ndarray):
    """Parameter-free deterministic delay carrying the fixed z-marginal intercept
    ``c_m = -W_m z_m,e`` (§5.5)."""
    c = np.asarray(c_m, dtype=float)

    def delay(params: dict[str, object]):
        import jax.numpy as jnp

        return jnp.asarray(c)

    delay.params = []
    delay.gpname = f"{name}_zprior_intercept"
    return delay


def discovery_signals(
    *,
    pulsar,
    space,
    engine,
    partition: TimingParameterPlan,
    name: str,
    engine_map,
    design_matrix=None,
    linearization=None,
) -> list:
    """Return the Discovery-native timing signals for the plan (§5.5).

    Up to four pieces: the exact sampled nonlinear delay ``d_anchor(z_s)`` (with
    z-marginalized axes held at their fixed expansion delta), the delta-flat
    improper ``M_f`` GP, the proper unit-normal z-prior ``W_m`` GP
    (``makegp_standard_normal``), and the parameter-free ``c_m`` intercept delay.

    Parameters
    ----------
    pulsar
        Timing pulsar with residuals, design matrix, and TOA metadata.
    space
        ``ParameterSpace`` for the sampled block. Passed for API symmetry with
        ``TimingSpec.discovery_signals``; delay keys already use
        engine-native ``delta_theta``, so the likelihood path here does not
        consume ``space`` directly. Priors from ``space`` are applied outside
        this builder (see module docstring).
    engine
        Timing engine. A ``JaxTimingEngine`` yields a traced delay; any other
        ``TimingEngine`` yields a host-callback delay. The latter is valid for
        ``PulsarLikelihood.logL`` with a derivative-free sampler only.
    partition
        Numerically sampled vs analytically marginalized fit-parameter partition in pulsar
        column order.
    name
        Component name prefix for emitted signal keys.

    Returns
    -------
    list
        Discovery signal factories: ``makegp_improper`` for analytically marginalized
        columns (when present) and a delay callable keyed by
        ``{pulsar.name}_{name}_{fitpar}`` for each sampled parameter.

    Notes
    -----
    Discovery uses ``detres = residuals - delay``, so the emitted delay is
    ``-residual_delta(full_delta)``. Analytically marginalized linear parameters are
    represented with a flat improper GP (``constant=1e40``), matching the
    Woodbury/Schur analytical-marginalization path used elsewhere in the stack.
    """
    from discovery import signals as discovery_signals

    _ = space  # Discovery delay keys are already engine-facing delta_theta values.

    if tuple(partition.fitpars) != tuple(pulsar.fitpars):
        raise ValueError(
            "partition.fitpars must match pulsar.fitpars in canonical order"
        )

    signals: list = []
    if partition.idx_analytically_marginalized:
        from nltiming.whitening import normalized_basis

        if design_matrix is None:
            raise ValueError(
                "discovery_signals requires the context design matrix; "
                "call ctx.discovery_signals()"
            )
        # Column-normalized: span-preserving under the improper prior, and
        # required for float64 conditioning with constant=1e40.
        basis = normalized_basis(
            np.asarray(design_matrix, dtype=float)[
                :, list(partition.idx_analytically_marginalized)
            ]
        )
        signals.append(
            discovery_signals.makegp_improper(
                pulsar,
                basis,
                constant=1.0e40,
                name=f"{name}_timingmodel",
            )
        )

    # z-prior marginal block: proper unit-normal coefficients on W_m, plus the
    # fixed c_m intercept as a parameter-free delay (§5.5). The exact sampled
    # delay pins z-marg axes at z_m,e inside the engine_map (sampled mode).
    if partition.marginalized_z:
        if linearization is None:
            raise ValueError(
                "discovery_signals requires the context linearization to emit the "
                "z-prior W_m block; call ctx.discovery_signals()"
            )
        from discovery.signals import makegp_standard_normal

        W_m = np.asarray(linearization.marginalized_z_basis, dtype=float)
        signals.append(makegp_standard_normal(pulsar, W_m, name=f"{name}_zprior"))
        c_m = np.asarray(linearization.marginalized_z_intercept, dtype=float)
        if np.any(c_m != 0.0):
            signals.append(_build_cm_delay(pulsar, name, c_m))

    if not partition.sampled:
        return signals

    signals.append(
        _build_delay_callable(
            pulsar=pulsar,
            engine=engine,
            partition=partition,
            name=name,
            engine_map=engine_map,
        )
    )
    return signals
