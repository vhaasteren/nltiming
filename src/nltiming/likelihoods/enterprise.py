"""Enterprise likelihood interface for nonlinear timing.

This module wires ``NonLinearTimingModel`` into Enterprise's signal graph:
a deterministic nonlinear delay for numerically sampled fit parameters and an optional
``TimingModel`` GP basis for analytically marginalized linear nuisances.

Priors
------
Sampled-parameter priors come from the bound ``ParameterSpace`` and are
evaluated through Enterprise ``UserParameter`` hooks that call
``PriorBijector.logprior_physical`` / ``ParameterSpace.logprior_coord``,
including the probability-integral-transform (PIT) Jacobian for bounded
families (``uniform``, truncated normal, etc.).

With ``prior_policy="fallback"``, unresolved sampled priors use the reference-stack
*cheat* prior convention—not Gaussians at the WLS scale. Each axis is a flat
``uniform`` on ``[center ± coordinate_policy.nonlinear_scale · σ]`` in delta space
(``center`` = par-file reference, ``σ`` = par-file uncertainty with WLS
fallback), clipped to ``native_physical_bounds`` (e.g. ``ECC ∈ [0, 1]``,
``M2 ≥ 0``). Over the typical posterior support these boxes are
effectively flat. The whitening coordinate map is for sampler
preconditioning only and does not alter the physical prior density.
"""

from __future__ import annotations

import numpy as np

from nltiming.bijectors import PriorBijector
from nltiming.space import coord_for_static_layer


def _residual_delta(engine, full_delta: np.ndarray) -> np.ndarray:
    """Evaluate residual delta, preferring the JAX path when available.

    The JUG NumPy residual path is deprecated and less accurate at large
    deltas; using the JAX path keeps the Enterprise likelihood consistent
    with the Discovery likelihood interface on JAX-capable engines.
    """
    fn = getattr(engine, "residual_delta_jax", None)
    if fn is not None:
        return np.asarray(fn(full_delta), dtype=float)
    return np.asarray(engine.residual_delta(full_delta), dtype=float)


def _axis_bijector(*, space, idx: int) -> PriorBijector:
    return PriorBijector(
        names=(space.names[idx],),
        priors=(space.prior_bijector.priors[idx],),
    )


def _scalar_user_parameter(*, space, coord: str, idx: int):
    from enterprise.signals import parameter

    axis = _axis_bijector(space=space, idx=idx)

    def _logprior(value):
        q = np.asarray([value], dtype=float)
        if coord == "delta":
            return float(axis.logprior_physical(q, np))
        if coord == "z":
            delta = axis.delta_from_z(q, np)
            return float(
                axis.logprior_physical(delta, np) + axis.logabsdet_delta_from_z(q, np)
            )
        raise ValueError(f"Scalar Enterprise parameters do not support coord={coord!r}")

    def _ppf(u):
        cube = np.asarray([u], dtype=float)
        delta = axis.delta_from_u(cube, np)
        if coord == "delta":
            return float(delta[0])
        if coord == "z":
            return float(axis.z_from_delta(delta, np)[0])
        raise ValueError(f"Scalar Enterprise parameters do not support coord={coord!r}")

    def _sampler(size=None):
        # Enterprise passes size for scalar Parameter.sample(); ignore it and
        # return a scalar. A scalar timing parameter is one prior axis.
        u = np.random.uniform(1e-12, 1.0 - 1e-12)
        return _ppf(u)

    return parameter.UserParameter(
        logprior=parameter.Function(_logprior),
        sampler=_sampler,
        ppf=parameter.Function(_ppf),
        prior_draw_mode="component",
    )


def _vector_user_parameter(*, space):
    from enterprise.signals import parameter

    def _logprior(value):
        q = np.asarray(value, dtype=float)
        return float(space.logprior_coord(q, np, coord="x"))

    def _ppf(u):
        cube = np.asarray(u, dtype=float)
        return np.asarray(space.coord_from_cube(cube, np, coord="x"), dtype=float)

    def _sampler(size=None):
        if size is not None and size != space.ndim:
            raise ValueError(f"expected sampler size {space.ndim}, received {size}")
        u = np.random.uniform(1e-12, 1.0 - 1e-12, size=space.ndim)
        return np.asarray(space.coord_from_cube(u, np, coord="x"), dtype=float)

    return parameter.UserParameter(
        logprior=parameter.Function(_logprior),
        sampler=_sampler,
        ppf=parameter.Function(_ppf),
        size=space.ndim,
        prior_draw_mode="joint",
    )


def _validate_kwarg_name(name: str) -> None:
    if not name.isidentifier():
        raise ValueError(
            f"Fit parameter {name!r} cannot be used as an Enterprise waveform keyword"
        )


def _explicit_scalar_delay_function(sampled_names: tuple[str, ...], evaluator):
    for fitpar in sampled_names:
        _validate_kwarg_name(fitpar)
    signature = ", ".join(f"{fitpar}=None" for fitpar in sampled_names)
    call = ", ".join(f"{fitpar}={fitpar}" for fitpar in sampled_names)
    source = (
        f"def _delay_body(toas, psr=None, mask=None, {signature}):\n"
        f"    return _evaluator({call})\n"
    )
    namespace = {"_evaluator": evaluator}
    exec(source, namespace)
    return namespace["_delay_body"]


def _make_waveform(
    *,
    ctx_fn,
    coord: str,
):
    from enterprise.signals import parameter

    def waveform(signal_name, psr=None):
        if psr is None:
            raise ValueError("enterprise waveform requires a pulsar")
        ctx = ctx_fn(psr)
        space = ctx.space
        partition = ctx.plan
        engine = ctx.engine
        sampled_names = tuple(partition.sampled)
        sampled_indices = tuple(partition.idx_sampled)
        ndim = len(partition.fitpars)
        zm_indices, zm_fixed = _zmarg_fixed(ctx)

        if coord in {"delta", "z"}:

            def _evaluate(**coord_values):
                q = np.asarray(
                    [coord_values[param] for param in sampled_names],
                    dtype=float,
                )
                space_coord = coord
                delta_sampled = np.asarray(
                    space.delta_from_coord(q, np, coord=space_coord)
                )
                full_delta = np.zeros((ndim,), dtype=float)
                for i, col in enumerate(sampled_indices):
                    full_delta[col] = delta_sampled[i]
                for i, col in enumerate(zm_indices):  # z-marg fixed at z_m,e
                    full_delta[col] = zm_fixed[i]
                return -_residual_delta(engine, full_delta)

            delay_body = _explicit_scalar_delay_function(sampled_names, _evaluate)
            kwargs = {
                param: _scalar_user_parameter(
                    space=space,
                    coord=coord,
                    idx=i,
                )
                for i, param in enumerate(sampled_names)
            }
            return parameter.Function(delay_body, **kwargs)(signal_name, psr=psr)

        if coord != "x":
            raise ValueError(f"Unsupported enterprise timing coord: {coord}")

        def _delay_body(toas, psr=None, mask=None, x=None):
            q = np.asarray(x, dtype=float)
            delta_sampled = np.asarray(space.delta_from_coord(q, np, coord="x"))
            full_delta = np.zeros((ndim,), dtype=float)
            for i, col in enumerate(sampled_indices):
                full_delta[col] = delta_sampled[i]
            for i, col in enumerate(zm_indices):
                full_delta[col] = zm_fixed[i]
            return -_residual_delta(engine, full_delta)

        kwargs = {"x": _vector_user_parameter(space=space)}
        return parameter.Function(_delay_body, **kwargs)(signal_name, psr=psr)

    return waveform


def _make_marginalizing_signal(
    *,
    ctx_fn,
    name: str,
):
    from enterprise.signals import gp_signals, signal_base

    class MarginalizingTimingModel(
        signal_base.Signal, metaclass=signal_base.MetaSignal
    ):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = f"{name}_timingmodel"

        def __init__(self, psr):
            super().__init__(psr)
            from nltiming.whitening import normalized_basis

            ctx = ctx_fn(psr)
            partition = ctx.plan
            # Column-normalized: span-preserving under the improper prior, and
            # required for float64 conditioning with the 1e40 prior weight.
            self._basis = normalized_basis(
                ctx.design_matrix[:, list(partition.idx_analytically_marginalized)]
            )
            base = gp_signals.TimingModel(
                name=f"{name}_timingmodel",
                idx_exclude=partition.idx_sampled,
            )
            self._inner = base(psr)
            self.name = self._inner.name
            self._params = self._inner._params
            self.basis_params = list(self._inner.basis_params)
            self.prior_params = list(getattr(self._inner, "prior_params", []))
            self.delay_params = list(getattr(self._inner, "delay_params", []))
            self.basis_combine = getattr(self._inner, "basis_combine", False)

        def get_basis(self, params=None):
            return self._basis

        def get_phi(self, params):
            return self._inner.get_phi(params)

        def get_phiinv(self, params):
            return self._inner.get_phiinv(params)

        def get_delay(self, params):
            return self._inner.get_delay(params)

        def get_logsignalprior(self, params):
            return self._inner.get_logsignalprior(params)

        def set_default_params(self, params):
            self._inner.set_default_params(params)

    return MarginalizingTimingModel


def _zmarg_fixed(ctx):
    """z-marginalized fitpar indices and their fixed expansion deltas ``z_m,e``."""
    proper_axes = [
        a for a in ctx.plan.axes
        if a.disposition in ("sample", "marginalize_z_prior")
    ]
    zm_indices = tuple(
        a.fitpar_index for a in proper_axes if a.disposition == "marginalize_z_prior"
    )
    zm_fixed = np.asarray(
        [ctx.linearization.delta_expansion[i]
         for i, a in enumerate(proper_axes)
         if a.disposition == "marginalize_z_prior"],
        dtype=float,
    )
    return zm_indices, zm_fixed


def _make_zprior_signal(*, ctx_fn, name: str, coefficients: bool = False):
    """Proper unit-normal ``W_m`` block, ``c ~ Normal(0, I)`` (§5.6).

    Two modes, matching Enterprise's basis-GP convention:

    - ``coefficients=False`` (default) — analytically marginalized: ``get_phi``
      returns ``ones`` (identity coefficient prior; ``log|Phi| = 0`` retained),
      and the Enterprise normal machinery integrates the coefficients out.
    - ``coefficients=True`` — the coefficients are sampled as a ``GPCoefficients``
      parameter; the delay is ``W_m @ c`` and the unit-normal prior
      ``-1/2 c^T c`` lives on that parameter. Use this when the ``W_m`` block is
      sampled jointly (e.g. a dynamic decentering transport) rather than
      integrated.

    The basis ``W_m`` is passed unnormalized (its unit coefficient variance is
    the physical z prior).
    """
    from enterprise.signals import parameter, signal_base

    class ZPriorTimingGP(signal_base.Signal, metaclass=signal_base.MetaSignal):
        signal_type = "basis"
        signal_name = "z-prior timing"
        signal_id = f"{name}_zprior"

        def __init__(self, psr):
            super().__init__(psr)
            ctx = ctx_fn(psr)
            self._basis = np.asarray(
                ctx.linearization.marginalized_z_basis, dtype=float
            )
            self._k = int(self._basis.shape[1])
            self._phi = np.ones(self._k)
            self.name = f"{psr.name}_{name}_zprior"
            self.basis_params = []
            self.prior_params = []
            self.basis_combine = False
            if coefficients:
                k = self._k

                def _coeff_logprior(c, **params):
                    c = np.asarray(c, dtype=float)
                    return -0.5 * np.sum(c * c) - 0.5 * k * np.log(2.0 * np.pi)

                cpar = parameter.GPCoefficients(
                    logprior=parameter.Function(_coeff_logprior),
                    size=self._k,
                )(f"{self.name}_coefficients")
                self._coeff = cpar
                self._params = {cpar.name: cpar}
                self.delay_params = [cpar.name]
            else:
                self._coeff = None
                self._params = {}
                self.delay_params = []

        def get_basis(self, params=None):
            return None if coefficients else self._basis

        def get_phi(self, params):
            return None if coefficients else self._phi

        def get_phiinv(self, params):
            return None if coefficients else self._phi  # 1 / ones == ones

        def get_delay(self, params=None):
            if not coefficients:
                return np.zeros(self._basis.shape[0], dtype=float)
            params = params or {}
            c = (
                np.asarray(params[self._coeff.name], dtype=float)
                if self._coeff.name in params
                else np.zeros(self._k)
            )
            return self._basis @ c

        def get_logsignalprior(self, params):
            # Marginalized: log|Phi| = log|I| = 0. Sampled: the coefficient prior
            # lives on the GPCoefficients parameter, not here.
            return 0.0

        def set_default_params(self, params):
            pass

    return ZPriorTimingGP


def _make_cm_signal(*, ctx_fn, name: str):
    """Parameter-free deterministic ``c_m = -W_m z_m,e`` intercept (§5.6)."""
    from enterprise.signals import deterministic_signals, parameter

    def waveform(signal_name, psr=None):
        ctx = ctx_fn(psr)
        c_m = np.asarray(ctx.linearization.marginalized_z_intercept, dtype=float)

        def _body(toas, psr=None, mask=None):
            return c_m

        return parameter.Function(_body)(signal_name, psr=psr)

    return deterministic_signals.Deterministic(
        waveform, name=f"{name}_zprior_intercept"
    )


def enterprise_signal(
    *,
    ctx_fn,
    name: str,
    static_layer: str,
    has_delta_flat: bool = True,
    has_z_prior: bool = False,
    sample_z_coefficients: bool = False,
):
    """Return a deferred Enterprise signal with deterministic delay + timing GP.

    Parameters
    ----------
    ctx_fn
        Callable ``pulsar -> TimingContext`` (typically
        ``NonLinearTimingModel.for_pulsar``). All pulsar-bound state — parameter
        space, partition, timing engine, design matrix — comes from the
        ctx, so the Enterprise likelihood shares the exact engine
        configuration used by the Discovery likelihood interface and the run products.
    name
        Enterprise signal / component name prefix.
    static_layer
        ``NonLinearTimingModel`` static layer: ``"identity"`` or ``"whitening"``
        or ``"whitening"``. Selects the Enterprise sampling coordinate
        (``z`` under the identity layer, or joint ``x`` under whitening).

    Returns
    -------
    type
        ``MetaSignal`` subclass that materializes on ``(psr)`` into either
        ``Deterministic(delay)``, ``TimingModel`` GP, or their sum.

    Notes
    -----
    Delay parameters are mapped from the sampling coordinate back to native
    ``delta_theta`` via ``space.delta_from_coord`` before evaluating the
    engine residual delta (JAX path when available). Prior terms follow
    ``space`` exactly, so fallback cheat priors are the wide uniform boxes
    described in the module docstring—not informative Gaussians tied to the
    WLS covariance.
    """
    from enterprise.signals import deterministic_signals

    coord = coord_for_static_layer(static_layer)
    waveform = _make_waveform(ctx_fn=ctx_fn, coord=coord)
    signal = deterministic_signals.Deterministic(waveform, name=name)
    # A GP block is only composed when its axes exist: an empty basis breaks
    # Enterprise's SignalCollection (``Fmat[:, []]`` treats [] as float indices).
    if has_delta_flat:  # improper delta-flat M_f TimingModel
        signal = signal + _make_marginalizing_signal(ctx_fn=ctx_fn, name=name)
    if has_z_prior:
        # Four-piece z-prior assembly (§5.6): proper unit-normal W_m GP plus the
        # fixed c_m intercept; the exact sampled delay holds z-marg at z_m,e.
        signal = signal + _make_zprior_signal(
            ctx_fn=ctx_fn, name=name, coefficients=sample_z_coefficients
        )
        signal = signal + _make_cm_signal(ctx_fn=ctx_fn, name=name)
    return signal


def enterprise_marginal_products(pta, ctx, *, fixed_wn_params):
    """``products_fn`` for ``NumpyMarginalTransport`` from a single-pulsar PTA.

    ``pta`` must be the post-geometry Enterprise assembly for ``ctx`` (exact
    sampled delay + ``c_m`` + normalized improper ``M_f`` block + unit-normal
    ``W_m`` block + RN/DM; geometry plan §5.6). ``ctx`` supplies the sealed
    transport inputs: ``W_s = ctx.linearization.sampled_basis`` and
    ``y_t = ctx.linearization.transport_effective_residual(ctx.pulsar.residuals)``.
    Raw residuals and caller-supplied bases are deliberately not accepted
    (D-INV; marginalized D19).

    ``fixed_wn_params`` pins every white-noise parameter so the N-side products
    are cached once (E7). Returns a callable ``products(params) ->
    MarginalProducts`` with attribute ``params`` (sorted hyperparameter names,
    delay keys excluded — E8) and a required one-slot memo on the last eta point
    (E9).
    """
    import scipy.linalg as sl

    from nltiming.decentering import MarginalProducts
    from nltiming.metric import assert_static_layer_identity

    if len(pta.pulsars) != 1:
        raise ValueError(
            f"enterprise_marginal_products requires a single-pulsar PTA; "
            f"got {len(pta.pulsars)} pulsars"
        )
    assert_static_layer_identity(ctx.space)
    lin = ctx.linearization
    if lin is None:
        raise RuntimeError(
            "enterprise_marginal_products requires ctx.linearization "
            "(TimingLinearization from the geometry plan)"
        )

    y_t = np.asarray(
        lin.transport_effective_residual(np.asarray(ctx.pulsar.residuals, dtype=float)),
        dtype=float,
    )
    W = np.asarray(lin.sampled_basis, dtype=float)
    if W.shape[1] != len(ctx.plan.sampled):
        raise RuntimeError(
            f"linearization.sampled_basis has {W.shape[1]} columns but "
            f"ctx.plan.sampled has {len(ctx.plan.sampled)} names"
        )

    p0 = dict(fixed_wn_params)
    # White-noise parameter names come from the Enterprise signal collections
    # (E7). The combined ``pta.get_ndiag(p0)[0]`` is an ``ndarray_alt`` with no
    # ``.params`` attribute, and ``get_TNT`` does not raise on a missing white
    # param (it defaults), so the pinned-WN check must enumerate them explicitly.
    wn_names = set().union(
        *(set(getattr(sc, "white_params", []) or []) for sc in pta._signalcollections)
    )
    missing = sorted(wn_names - set(p0))
    if missing:
        raise ValueError(
            f"enterprise_marginal_products requires fixed white noise; "
            f"missing from fixed_wn_params: {missing}"
        )
    ndiag = pta.get_ndiag(p0)[0]
    T = np.asarray(pta.get_basis(p0)[0], dtype=float)
    if T.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"basis rows {T.shape[0]} != n_toa {y_t.shape[0]}; "
            f"the PTA and ctx describe different pulsars"
        )

    # Cached constant products (WN fixed; basis constant): E7.
    WNW = np.asarray(ndiag.solve(W, left_array=W), dtype=float)  # (k, k)
    TNW = np.asarray(ndiag.solve(W, left_array=T), dtype=float)  # (m, k)
    WNy = np.asarray(ndiag.solve(y_t, left_array=W), dtype=float)  # (k,)
    TNy = np.asarray(ndiag.solve(y_t, left_array=T), dtype=float)  # (m,)
    TNT = np.asarray(pta.get_TNT(p0)[0], dtype=float)  # (m, m)

    # E8: delay keys are NOT hyperparameters (rev-1 defect fixed here).
    hyper_names = tuple(sorted(set(pta.param_names) - set(p0) - set(ctx.delay_keys)))

    memo = {"key": None, "value": None}  # E9

    def products(params):
        key = tuple(float(params[n]) for n in hyper_names)
        if memo["key"] == key:
            return memo["value"]
        full = {**p0, **{n: v for n, v in zip(hyper_names, key)}}
        phiinv = np.asarray(pta.get_phiinv(full, logdet=False)[0], dtype=float)
        # get_phiinv may be a (m,) diagonal or a dense (m, m) matrix
        # (common-signal / non-diagonal Phi). Match Enterprise's own branch.
        sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        cf = sl.cho_factor(sigma, lower=True)
        SW = sl.cho_solve(cf, TNW)  # (m, k)
        out = MarginalProducts(G=WNW - TNW.T @ SW, b=WNy - SW.T @ TNy)
        memo["key"], memo["value"] = key, out
        return out

    products.params = hyper_names
    return products
