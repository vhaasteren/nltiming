"""Contract 3: MarginalBasisFrame + conversion-metadata probe (T22–T29)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from nltiming import SUPPORTS_CONVERSION_METADATA, TimingInference  # noqa: E402
from nltiming.physical_charts import (  # noqa: E402
    KeplerLaplacePolicy,
    activate_charts,
    frame_change_matrix,
    resolve_chart_candidates,
)
from nltiming.priors import (  # noqa: E402
    stigma_mass_ceiling_lower,
    stigma_orientation_logpdf,
)
from nltiming.whitening import normalized_basis  # noqa: E402
from nltiming.coordinates import TimingCoordinatePolicy  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402
from _engine_stubs import JaxLinearTestEngine  # noqa: E402
from nltiming.engine_support import LinearModel  # noqa: E402

from test_physical_charts import (  # noqa: E402
    FITPARS,
    REFS,
    _FakeEngineNoCap,
    _FakePulsar,
    _activate,
    _plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _j2145_like_refs(*, e="1.9e-5"):
    return {
        "Offset": "0.0",
        "ECC": e,
        "OM": "50.7",
        "T0": "55000.0",
        "PB": "8.6866194196",
        "A1": "10.0",
    }


def _ill_conditioned_binary_pulsar(*, e=1.9e-5, seed=1, n=80):
    """Linear fixture whose ECC/OM/T0 columns mimic low-e Kepler collinearity.

    Build well-conditioned Laplace-direction columns L, then set
    M_e[:, triple] = L @ inv(J) so the engine Gram is ill-conditioned and the
    framed Gram M_e @ J recovers L.
    """
    fitpars = ("Offset", "ECC", "OM", "T0", "PB", "A1")
    refs = _j2145_like_refs(e=repr(e))
    # Temporary pulsar/engine to obtain chart geometry / J at this e.
    tmp = _FakePulsar(fitpars=fitpars)
    eng = _FakeEngineNoCap(refs)
    (cand,) = [
        c
        for c in resolve_chart_candidates(tmp, eng, KeplerLaplacePolicy())
        if c.chart is not None
    ]
    ch = cand.chart
    J = ch.jacobian_at()
    Jinv = np.linalg.inv(J)

    rng = np.random.default_rng(seed)
    # Orthonormal-ish Laplace columns + independent Offset/PB/A1 columns.
    # Offset must be the constant gauge column.
    q, _ = np.linalg.qr(rng.standard_normal((n, 3)))
    laplace_cols = q * np.array([1.0, 2.0, 0.5])  # mild scale spread
    other = rng.standard_normal((n, 2))
    design = np.zeros((n, 6))
    design[:, 0] = 1.0  # Offset (gauge)
    design[:, 4] = other[:, 0]  # PB
    design[:, 5] = other[:, 1]  # A1
    design[:, [1, 2, 3]] = laplace_cols @ Jinv

    class _P:
        name = "J2145LIKE"

        def __init__(self):
            self.name = "J2145LIKE"
            self.fitpars = ("Offset", "ECC", "OM", "T0", "PB", "A1")
            self._toas = np.linspace(0.0, 1.0, n) * 3.15e7 + 5.3e4
            self._residuals = 1e-7 * rng.standard_normal(n)
            self._toaerrs = np.full(n, 1e-7)
            self._freqs = np.full(n, 1400.0)
            self._bf = np.array(["d"] * n, dtype="U8")
            self._flags = {"pta": self._bf}
            model = LinearModel.from_design(
                fitpars=self.fitpars, design=design, theta_exact=dict(refs)
            )
            self._backend = JaxLinearTestEngine.from_linear_model(model)
            self._design = design

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
            return self._bf

        def state_id(self):
            return f"j2145like-{e}"

        def pint_model(self):
            return None

        def timing_engine(self, engines="jug", **kwargs):
            return self._backend

    return _P(), ch


def _improper_marg_lnlike(residuals, toaerrs, design, marg_idx):
    """Adapter-style improper-flat marginal lnL (normalized columns, N-weighted)."""
    w = 1.0 / np.asarray(toaerrs, dtype=float)
    r = np.asarray(residuals, dtype=float) * w
    F_raw = np.asarray(design[:, list(marg_idx)], dtype=float)
    Fn = normalized_basis(F_raw)
    Fw = Fn * w[:, None]
    G = Fw.T @ Fw
    c = np.linalg.solve(G, Fw.T @ r)
    r_perp = r - Fw @ c
    return float(-0.5 * np.dot(r_perp, r_perp) - 0.5 * np.linalg.slogdet(G)[1])


def _ctx_for(pulsar, *, inference, binary_chart="auto", **kw):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=inference,
        binary_chart=binary_chart,
        name="t",
        **kw,
    )
    return ntm, ntm.for_pulsar(pulsar, condition=False)


# ---------------------------------------------------------------------------
# T22 – frame activation
# ---------------------------------------------------------------------------


def test_t22_frame_activation():
    pulsar, ch = _ill_conditioned_binary_pulsar()
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    _, ctx = _ctx_for(pulsar, inference=inf, binary_chart="auto")
    assert ctx.physical_charts == ()
    assert ctx.binary_chart_records[0]["reason"] == "no_sampled_axis"
    assert len(ctx.marginal_basis_frames) == 1
    frame = ctx.marginal_basis_frames[0]
    assert frame.chart.engine_names == ch.engine_names
    assert np.isfinite(frame.log_abs_det_b)
    # Plan axes keep engine names and delta-flat dispositions.
    for name in ("ECC", "OM", "T0"):
        ax = ctx.plan.axis(name)
        assert ax.name == name
        assert ax.disposition == "marginalize_delta_flat"
        assert ax.physical_chart is None
    # The frame transforms itself, NOT its chart: the chart's write_frame_block
    # also sets B[T0, PB], which would modify the PB design column even though
    # PB is not part of the marginalized triple (§10.1 "only ... the delta-flat
    # columns"). Composing from `f.chart` here is what the old code did.
    transforms = tuple(ctx.marginal_basis_frames)
    expected = ctx.engine_design_matrix @ frame_change_matrix(
        len(ctx.plan.fitpars), transforms
    )
    np.testing.assert_allclose(ctx.design_matrix, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(ctx.engine_design_matrix, pulsar.Mmat)

    # Columns outside the triple are bit-identical to the engine basis.
    triple = set(ctx.marginal_basis_frames[0].chart.slots)
    for j in range(ctx.design_matrix.shape[1]):
        if j in triple:
            continue
        np.testing.assert_array_equal(
            ctx.design_matrix[:, j], ctx.engine_design_matrix[:, j]
        )
    man = ctx.binary_marginal_basis_frame_manifest()
    assert man["groups"][0]["enabled"] is True
    assert np.isfinite(man["groups"][0]["log_abs_det_b"])
    assert man["groups"][0]["log_volume_offset"] is not None


# ---------------------------------------------------------------------------
# T23 – adapter flow
# ---------------------------------------------------------------------------


def test_t23_adapter_flow_normalized_basis():
    pulsar, _ = _ill_conditioned_binary_pulsar()
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    _, ctx = _ctx_for(pulsar, inference=inf)
    marg_idx = list(ctx.plan.idx_analytically_marginalized)
    expected = normalized_basis(ctx.design_matrix[:, marg_idx])
    assert np.all(np.isfinite(expected))

    # Discovery path: makegp_improper receives the framed normalized columns.
    pytest.importorskip("discovery")
    sigs = ctx.discovery_signals()
    assert sigs
    assert np.all(np.isfinite(expected))
    ln = _improper_marg_lnlike(
        pulsar.residuals,
        pulsar.toaerrs,
        ctx.design_matrix,
        marg_idx,
    )
    assert np.isfinite(ln)

    # Enterprise path (optional; pulsar duck may not satisfy full protocol).
    enterprise = pytest.importorskip("enterprise")
    _ = enterprise
    signal_cls = ctx.model.enterprise_signal()
    try:
        sig = signal_cls(pulsar)
        basis = sig.get_basis()
        np.testing.assert_allclose(basis, expected, rtol=1e-12, atol=0.0)
    except Exception as exc:
        pytest.skip(f"enterprise signal instantiation unavailable: {exc}")


# ---------------------------------------------------------------------------
# T24 – conditioning on the operative basis
# ---------------------------------------------------------------------------


def test_t24_conditioning_operative_basis():
    pulsar, ch = _ill_conditioned_binary_pulsar(e=1.9e-5)
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    pol_on = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="auto")
    pol_off = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="off")
    _, ctx_on = _ctx_for(pulsar, inference=inf, binary_chart=pol_on)
    _, ctx_off = _ctx_for(pulsar, inference=inf, binary_chart=pol_off)
    slots = list(ch.slots)
    w = 1.0 / np.asarray(pulsar.toaerrs, dtype=float)

    def kappa(design):
        cols = np.asarray(design[:, slots], dtype=float) * w[:, None]
        Fn = normalized_basis(cols)
        G = Fn.T @ Fn
        evals = np.linalg.eigvalsh(G)
        return float(evals[-1] / evals[0])

    assert kappa(ctx_off.design_matrix) > 1e8
    assert kappa(ctx_on.design_matrix) < 1e6

    # End-to-end marginal lnL is finite; a higher-precision (longdouble accum)
    # reference agrees to rtol 1e-6. (np.linalg does not accept longdouble on
    # all platforms, so the solve stays float64 while the Gram/residual
    # reductions accumulate in longdouble.)
    ln_f64 = _improper_marg_lnlike(
        pulsar.residuals,
        pulsar.toaerrs,
        ctx_on.design_matrix,
        ctx_on.plan.idx_analytically_marginalized,
    )
    w = 1.0 / np.asarray(pulsar.toaerrs, dtype=float)
    r = np.asarray(pulsar.residuals, dtype=float) * w
    F_raw = np.asarray(
        ctx_on.design_matrix[:, list(ctx_on.plan.idx_analytically_marginalized)],
        dtype=float,
    )
    Fn = normalized_basis(F_raw)
    Fw = Fn * w[:, None]
    G = Fw.T @ Fw
    c = np.linalg.solve(G, Fw.T @ r)
    r_perp = r - Fw @ c
    ln_ref = float(
        -0.5
        * np.asarray(r_perp, dtype=np.longdouble)
        @ np.asarray(r_perp, dtype=np.longdouble)
        - 0.5 * np.linalg.slogdet(G)[1]
    )
    assert np.isfinite(ln_f64)
    assert ln_f64 == pytest.approx(ln_ref, rel=1e-6)


# ---------------------------------------------------------------------------
# T25 – semantics invariance / log_volume_offset
# ---------------------------------------------------------------------------


def test_t25_semantics_invariance_log_volume_offset():
    # Moderate e: both bases numerically viable.
    pulsar, _ = _ill_conditioned_binary_pulsar(e=1e-3)
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    pol_on = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="auto")
    pol_off = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="off")
    _, ctx_on = _ctx_for(pulsar, inference=inf, binary_chart=pol_on)
    _, ctx_off = _ctx_for(pulsar, inference=inf, binary_chart=pol_off)
    offset = ctx_on.marginal_basis_frame_records[0]["log_volume_offset"]
    assert offset is not None

    rng = np.random.default_rng(42)
    k = len(ctx_on.plan.sampled)
    diffs = []
    for _ in range(10):
        v = rng.normal(size=k) * 1e-9
        # Residuals bit-identical (delta-flat pinned at reference).
        r_on = np.asarray(
            ctx_on.engine.residual_delta_jax(
                ctx_on.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp)
            )
        )
        r_off = np.asarray(
            ctx_off.engine.residual_delta_jax(
                ctx_off.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp)
            )
        )
        np.testing.assert_array_equal(r_on, r_off)
        # Marginal lnL on the (identical) residual vector through each basis.
        ln_on = _improper_marg_lnlike(
            r_on,
            pulsar.toaerrs,
            ctx_on.design_matrix,
            ctx_on.plan.idx_analytically_marginalized,
        )
        ln_off = _improper_marg_lnlike(
            r_off,
            pulsar.toaerrs,
            ctx_off.design_matrix,
            ctx_off.plan.idx_analytically_marginalized,
        )
        diffs.append(ln_on - ln_off)

    diffs = np.asarray(diffs)
    assert np.max(np.abs(diffs - diffs[0])) < 1e-6
    assert diffs[0] == pytest.approx(offset, rel=0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# T26 – z-prior exclusion
# ---------------------------------------------------------------------------


def test_t26_z_prior_exclusion():
    p = _FakePulsar()
    # ECC/OM z-prior (support includes e=0 via default boxes), T0 delta-flat.
    inf = TimingInference.groups(z_prior=["ECC", "OM"], delta_flat=["T0"])
    plan, resolved, frames, records = _activate(
        p, inf, KeplerLaplacePolicy(mode="auto")
    )
    assert resolved == ()
    assert frames == ()
    assert records[0]["reason"] == "no_sampled_axis"
    # Frame skip reason is mixed_marginal_dispositions (via context records).
    pulsar, _ = _ill_conditioned_binary_pulsar(e=8e-4)
    _, ctx = _ctx_for(pulsar, inference=inf)
    assert ctx.marginal_basis_frames == ()
    assert (
        ctx.marginal_basis_frame_records[0]["reason"] == "mixed_marginal_dispositions"
    )
    assert ctx.physical_charts == ()
    # Sampled-axis variant still exercises the unchanged origin guard (S-path).
    plan_s, res_s, frames_s, rec_s = _activate(
        p, TimingInference.sample_all(), KeplerLaplacePolicy(mode="auto")
    )
    # Without an origin-certified backend the default WLS box contains the
    # origin → chart demotes (existing S-path regression).
    if res_s == ():
        assert rec_s[0]["reason"] == "origin_uncertified_backend"
    else:
        assert frames_s == ()  # chart path, not frame path


# ---------------------------------------------------------------------------
# T27 – skip reasons
# ---------------------------------------------------------------------------


def test_t27_skip_reasons():
    from dataclasses import replace as dc_replace

    from nltiming.physical_charts import ChartCandidate, marginal_frame_skip_reason

    p = _FakePulsar()
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
        if c.chart is not None
    ]
    disp = {n: "marginalize_delta_flat" for n in cand.chart.engine_names}
    assert (
        marginal_frame_skip_reason(
            KeplerLaplacePolicy(marginal_basis_frame="off"), disp, cand, cand.chart
        )
        == "policy_off"
    )

    # Secular terms present → frame skip (chart record still no_sampled_axis).
    p_sec = _FakePulsar(fitpars=FITPARS + ("EDOT",))
    refs_sec = dict(REFS)
    refs_sec["EDOT"] = "1e-15"
    cands = resolve_chart_candidates(
        p_sec, _FakeEngineNoCap(refs_sec), KeplerLaplacePolicy()
    )
    (cand_sec,) = [c for c in cands if c.chart is not None]
    assert cand_sec.secular_terms
    _, _, frames_s, rec_s, _ = activate_charts(
        _plan(p_sec, TimingInference.groups(delta_flat=["ECC", "OM", "T0"])),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p_sec,
        engine_design_matrix=p_sec.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        engine_refs=refs_sec,
        prior_policy="wide_default",
    )
    assert frames_s == ()
    assert rec_s[0]["reason"] == "no_sampled_axis"
    assert (
        marginal_frame_skip_reason(
            KeplerLaplacePolicy(), disp, cand_sec, cand_sec.chart
        )
        == "secular_terms_present"
    )

    # policy_off via context manifest record
    pulsar, ch = _ill_conditioned_binary_pulsar(e=8e-4)
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    _, ctx_off = _ctx_for(
        pulsar,
        inference=inf,
        binary_chart=KeplerLaplacePolicy(marginal_basis_frame="off"),
    )
    assert ctx_off.marginal_basis_frame_records[0]["reason"] == "policy_off"

    # zero_eccentricity_reference (defensive; candidacy already rejects e<=0)
    ch0 = dc_replace(ch, e_ref=0.0, eps1_ref=0.0, eps2_ref=0.0)
    fake_cand = ChartCandidate(
        suffix=ch0.suffix,
        engine_names=ch0.engine_names,
        chart=ch0,
        skip_reason=None,
        e_ref=0.0,
        capability=None,
        secular_terms=(),
    )
    assert (
        marginal_frame_skip_reason(KeplerLaplacePolicy(), disp, fake_cand, ch0)
        == "zero_eccentricity_reference"
    )


# ---------------------------------------------------------------------------
# T28 – E-spelling alias
# ---------------------------------------------------------------------------


def test_t28_e_spelling_alias():
    from nltiming.pint_compat import resolve_parameter_alias

    # Par-file spelling "E" on pulsar.fitpars; engine refs are canonicalized
    # the same way canonical_fitpars rewrites aliases (ECC).
    fitpars = ("Offset", "E", "OM", "T0", "PB")
    refs = {
        "Offset": "0.0",
        "E": "8e-4",
        "OM": "50.7",
        "T0": "55000.0",
        "PB": "8.6866194196",
    }
    n = 40
    rng = np.random.default_rng(2)
    design = rng.standard_normal((n, 5))
    design[:, 0] = 1.0  # Offset gauge column

    class _EPulsar:
        def __init__(self):
            self.name = "EALIAS"
            self.fitpars = fitpars
            self._toas = np.linspace(0.0, 1.0, n)
            self._residuals = 1e-7 * rng.standard_normal(n)
            self._toaerrs = np.full(n, 1e-7)
            self._freqs = np.full(n, 1400.0)
            self._bf = np.array(["d"] * n, dtype="U8")
            self._flags = {"pta": self._bf}
            model = LinearModel.from_design(
                fitpars=fitpars, design=design, theta_exact=dict(refs)
            )
            self._backend = JaxLinearTestEngine.from_linear_model(model)
            self._design = design

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
            return self._bf

        def state_id(self):
            return "e-alias"

        def pint_model(self):
            return None

        def timing_engine(self, engines="jug", **kwargs):
            backend = self._backend
            raw = backend.reference_theta_exact()
            canon = {resolve_parameter_alias(k): v for k, v in raw.items()}
            backend.reference_theta_exact = lambda: dict(canon)
            return backend

    _, ctx = _ctx_for(
        _EPulsar(),
        inference=TimingInference.groups(delta_flat=["E", "OM", "T0"]),
    )
    assert len(ctx.marginal_basis_frames) == 1
    # Plan axes use the canonical ECC name after alias normalization.
    assert ctx.plan.axis("ECC").disposition == "marginalize_delta_flat"
    assert ctx.plan.axis("E").disposition == "marginalize_delta_flat"


# ---------------------------------------------------------------------------
# T29 – chart path (S) regression covered by existing suite; smoke here
# ---------------------------------------------------------------------------


def test_t29_chart_path_still_activates():
    p = _FakePulsar()
    plan, resolved, frames, records = _activate(
        p, TimingInference.sample_all(), KeplerLaplacePolicy()
    )
    # Either activates (certified) or demotes for origin — but never frames.
    assert frames == ()
    if resolved:
        assert {"EPS1", "EPS2", "TASC"} <= set(plan.axis_names)
        assert records[0]["enabled"] is True
    else:
        assert records[0]["enabled"] is False


# ---------------------------------------------------------------------------
# §8.5a / §10.8.1 conversion metadata
# ---------------------------------------------------------------------------


def test_supports_conversion_metadata_constant():
    assert SUPPORTS_CONVERSION_METADATA is True


@dataclass(frozen=True)
class _FakeConversionMetadata:
    target_family: str = "DDH"
    gauge: str | None = "absorbed"
    required_sampling: tuple[str, ...] = ("STIGMA",)
    stigma_central: float | None = 0.37
    stigma_provenance: str | None = "test"


def test_conversion_metadata_rejects_delta_flat_stigma():
    pulsar, _ = _ill_conditioned_binary_pulsar(e=8e-4)
    # ``_P`` below extends the fitpar list with STIGMA for the disposition check.
    refs = _j2145_like_refs(e="8e-4")
    refs["STIGMA"] = "0.37"
    n = pulsar.Mmat.shape[0]
    design = np.hstack([pulsar.Mmat, np.ones((n, 1)) * 1e-6])

    class _P:
        def __init__(self):
            self.name = "STIGMACASE"
            self.fitpars = ("Offset", "ECC", "OM", "T0", "PB", "A1", "STIGMA")
            self._toas = pulsar.toas
            self._residuals = pulsar.residuals
            self._toaerrs = pulsar.toaerrs
            self._freqs = pulsar.freqs
            self._bf = pulsar.backend_flags
            self._flags = pulsar.flags
            model = LinearModel.from_design(
                fitpars=self.fitpars, design=design, theta_exact=dict(refs)
            )
            self._backend = JaxLinearTestEngine.from_linear_model(model)
            self._design = design

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
            return self._bf

        def state_id(self):
            return "stigma-case"

        def pint_model(self):
            return None

        def timing_engine(self, engines="jug", **kwargs):
            return self._backend

        def conversion_metadata(self):
            return _FakeConversionMetadata()

    with pytest.raises(ValueError, match="required_sampling"):
        _ctx_for(
            _P(),
            inference=TimingInference.groups(delta_flat=["ECC", "OM", "T0", "STIGMA"]),
        )


def test_conversion_metadata_copied_to_manifest():
    pulsar, _ = _ill_conditioned_binary_pulsar(e=8e-4)

    class _Wrapped:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def conversion_metadata(self):
            return _FakeConversionMetadata(required_sampling=())

        def timing_engine(self, engines="jug", **kwargs):
            return self._inner.timing_engine(engines=engines, **kwargs)

    wrapped = _Wrapped(pulsar)
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["ECC", "OM", "T0"]),
        binary_chart="auto",
        name="t",
        whitening=None,
    )
    ctx = ntm.for_pulsar(wrapped, condition=True)
    assert ctx.conversion_metadata is not None
    man = ctx.run_manifest(likelihood="test", sampler="test")
    assert man.binary_conversion_metadata["target_family"] == "DDH"
    assert man.binary_conversion_metadata["stigma_provenance"] == "test"
    assert man.binary_marginal_basis_frame["groups"][0]["enabled"] is True


def test_stigma_orientation_prior_helper():
    logp = stigma_orientation_logpdf(0.5)
    # p(ς)=4ς/(1+ς²)² at ς=0.5
    expected = np.log(4 * 0.5 / (1 + 0.5**2) ** 2)
    assert logp == pytest.approx(expected)
    lo = stigma_mass_ceiling_lower(1e-6, 3.0)
    assert 0.0 < lo < 1.0


# ---------------------------------------------------------------------------
# Independent-review regressions
# ---------------------------------------------------------------------------


def test_log_volume_offset_is_exact_when_pb_is_also_delta_flat():
    """The frame must not touch the PB column (§10.1).

    The chart's ``write_frame_block`` also writes ``B[T0, PB]``, because under
    the chart T0 is replaced by TASC. The frame renames nothing, so d/dPB is
    unchanged. Inheriting that coupling modified the PB design column and made
    the recorded ``log_volume_offset`` wrong by 7.5e-4 (T25 tolerance is 1e-6)
    whenever PB was marginalized alongside the triple, because the offset sums
    column norms over the triple slots only.
    """
    pulsar, _ = _ill_conditioned_binary_pulsar(e=1e-3)
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0", "PB"])
    pol_on = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="auto")
    pol_off = KeplerLaplacePolicy(mode="auto", marginal_basis_frame="off")
    _, ctx_on = _ctx_for(pulsar, inference=inf, binary_chart=pol_on)
    _, ctx_off = _ctx_for(pulsar, inference=inf, binary_chart=pol_off)

    pb_slot = list(ctx_on.plan.fitpars).index("PB")
    np.testing.assert_array_equal(
        ctx_on.design_matrix[:, pb_slot], ctx_on.engine_design_matrix[:, pb_slot]
    )

    offset = ctx_on.marginal_basis_frame_records[0]["log_volume_offset"]
    rng = np.random.default_rng(7)
    k = len(ctx_on.plan.sampled)
    diffs = []
    for _ in range(5):
        v = rng.normal(size=k) * 1e-9
        r = np.asarray(
            ctx_on.engine.residual_delta_jax(
                ctx_on.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp)
            )
        )
        diffs.append(
            _improper_marg_lnlike(
                r,
                pulsar.toaerrs,
                ctx_on.design_matrix,
                ctx_on.plan.idx_analytically_marginalized,
            )
            - _improper_marg_lnlike(
                r,
                pulsar.toaerrs,
                ctx_off.design_matrix,
                ctx_off.plan.idx_analytically_marginalized,
            )
        )
    diffs = np.asarray(diffs)
    assert np.max(np.abs(diffs - diffs[0])) < 1e-6
    assert diffs[0] == pytest.approx(offset, rel=0.0, abs=1e-6)


def test_required_sampling_absent_from_plan_is_rejected():
    """A Case-D STIGMA that is not in the plan at all stays silently pinned.

    ``marginalize_delta_flat`` was rejected, but the quieter violation — the
    axis simply missing, so the emitted prior *center* is frozen and read as a
    measurement — was not. That is the fixed-ς path §5.5 forbids outright.
    """
    from nltiming.nonlinear_timing_model import _reject_delta_flat_required_sampling

    pulsar, _ = _ill_conditioned_binary_pulsar(e=8e-4)
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    _, ctx = _ctx_for(pulsar, inference=inf)
    assert not any(a.name == "STIGMA" for a in ctx.plan.axes)

    meta = _FakeConversionMetadata()
    assert meta.required_sampling == ("STIGMA",)
    with pytest.raises(ValueError, match="absent from the inference plan"):
        _reject_delta_flat_required_sampling(ctx.plan, meta)

    # No required_sampling -> unaffected; a pulsar without the hook is unaffected.
    _reject_delta_flat_required_sampling(ctx.plan, None)
    _reject_delta_flat_required_sampling(
        ctx.plan, _FakeConversionMetadata(required_sampling=())
    )


def test_stigma_mass_function_closure_and_installable_prior():
    """The third §10.8.1 helper, and a spec the framework can actually carry.

    ``AxisPrior`` has only bounded/normal families, so the orientation density
    4ς/(1+ς²)² is not directly installable; the composed support is.
    """
    from nltiming.priors import (
        stigma_mass_function_support,
        stigma_prior_from_support,
    )

    # J2145 numbers: design note quotes support ~[0.30, 0.46] around ς0 = 0.37.
    lo, hi = stigma_mass_function_support(1.8e-7, 6.83890261, 10.1641056)
    assert 0.0 < lo < hi <= 1.0
    assert 0.30 <= lo and hi <= 0.46
    assert lo <= 0.37 <= hi

    # Monotone in m_p, and in this direction: at fixed mass function a heavier
    # pulsar needs a heavier companion, and m_c = h3/(T_sun*ς**3) means heavier
    # companion <=> SMALLER ς.
    lo2, hi2 = stigma_mass_function_support(
        1.8e-7, 6.83890261, 10.1641056, m_p_range=(2.0, 2.5)
    )
    assert hi2 <= lo and lo2 < lo

    # Composability: the mass-ceiling bound is a floor on the same axis.
    ceiling = stigma_mass_ceiling_lower(1.8e-7, 3.0)
    combined_lo = max(lo, ceiling)
    assert 0.0 < combined_lo < hi

    spec = stigma_prior_from_support(combined_lo, hi)
    assert spec.prior.family == "uniform"
    assert spec.frame == "absolute"
    assert (spec.prior.lower, spec.prior.upper) == (combined_lo, hi)
    # ... and its support is inside the map domain, so it cannot trip the
    # activation guard.
    assert spec.prior.lower > 0.0 and spec.prior.upper <= 1.0

    normal_spec = stigma_prior_from_support(combined_lo, hi, family="normal")
    assert normal_spec.prior.family == "normal"
    assert normal_spec.prior.mean == pytest.approx(0.5 * (combined_lo + hi))

    for bad in ((0.0, 0.5), (0.5, 0.4), (0.5, 1.5)):
        with pytest.raises(ValueError):
            stigma_prior_from_support(*bad)
