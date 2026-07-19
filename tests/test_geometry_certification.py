"""Public geometry certifier and standalone report I/O (feature §8, §14.5).

The certifier differentiates the *actual* NumPyro joint model's unconstrained
potential at the deterministic ``2K+9`` probe set. These tests build small
linear-Gaussian oracle models (a real ``discovery.transport.Transport`` plus a
hand-written NumPyro factor replicating ``joint_model``'s density) so the exact
geometry is known, then confirm each metric catches its intended defect.
"""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("jug")
pytest.importorskip("discovery")
numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402

from discovery import transport as tr  # noqa: E402

from nltiming import (  # noqa: E402
    GeometryCertificationError,
    GeometryDiagnosticWarning,
    GeometryThresholds,
    JointGeometryReport,
    TimingInference,
    TransportCenterAxis,
    box_hyper_probe_points,
    certify_joint_geometry,
    read_geometry_report,
    transport_center_report,
    write_geometry_report,
)
from nltiming.geometry import target_metrics_at  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402

from test_joint_model import _Pulsar  # noqa: E402


class _NonlinearEngine:
    """Linear engine plus a quadratic bump on one TOA (zero value+gradient at
    the delta=0 expansion, so the linearization is unchanged but the exact
    waveform departs from the linear surrogate away from zero)."""

    def __init__(self, base, *, row, scale, direction):
        self._base = base
        self._row = int(row)
        self._scale = float(scale)
        self._v = np.asarray(direction, dtype=float)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def residual_delta_jax(self, delta):
        base = self._base.residual_delta_jax(delta)
        proj = jnp.dot(jnp.asarray(delta, float), jnp.asarray(self._v))
        return base.at[self._row].add(self._scale * proj * proj)

    def residual_delta(self, delta):
        base = np.asarray(self._base.residual_delta(delta), dtype=float).copy()
        proj = float(np.dot(np.asarray(delta, float), self._v))
        base[self._row] += self._scale * proj * proj
        return base


class _BigPulsar(_Pulsar):
    """Same linear timing model as ``_Pulsar`` with many TOAs, so a single-TOA
    remainder spike averages out of the global RMS."""

    def __init__(self, n=400):
        from nltiming.engines.base import LinearModel
        from nltiming.engines.jug import LinearizedJugEngine

        self.name = "J0000+0000"
        self.fitpars = ("F0", "F1", "DM")
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


def _oracle(
    *,
    pulsar=None,
    identically_linear=None,
    engine_wrap=None,
    wrong_W=1.0,
    xi_quartic=0.0,
    eta_coupling=0.0,
    center_mult=1.0,
):
    """Build a linear-Gaussian oracle joint model plus its context.

    The model is exactly ``N(0, I)`` in ``xi`` for the defaults; each keyword
    injects one controlled defect used by a single test.
    """
    kwargs = {"engines": "jug", "inference": TimingInference.sample_all(), "name": "timing"}
    if identically_linear is not None:
        kwargs["identically_linear"] = identically_linear
    ntm = NonLinearTimingModel(**kwargs)

    psr = pulsar if pulsar is not None else _Pulsar()
    if engine_wrap is not None:
        psr._backend = engine_wrap(psr._backend)
    ctx = ntm.for_pulsar(psr)

    block = ctx.local_timing_block()
    k = block.dimension
    W = np.asarray(block.basis, dtype=float)
    z_e = np.asarray(block.z_ref, dtype=float)
    d_e = np.asarray(ctx.linearization.sampled_waveform_expansion, dtype=float)
    y = np.asarray(psr.residuals, dtype=float)
    n0 = np.asarray(psr.toaerrs, dtype=float) ** 2

    rn = tr.reference_noise(psr)
    r_eff = ctx.linearization.transport_effective_residual(y) * center_mult

    class _Prec:
        params = []

        def __call__(self, params):
            return jnp.ones(k)

    blk = tr.TransportBlock(
        name="timing",
        index={ctx.joint_site: slice(0, k)},
        F=W,
        conditioner_precision=_Prec(),
    )
    transport = tr.Transport([blk], reference_noise=rn, reference_residual=r_eff)
    xi_site = ctx.joint_site + "_xi"
    Wmodel = W * wrong_W

    def model():
        eta = numpyro.sample("timing_eta", dist.Uniform(-1.0, 1.0))
        xi = numpyro.sample(xi_site, dist.Normal(0.0, 1.0).expand([k]).to_event(1))
        q, ldj = transport.apply({}, xi)
        z = q
        resid = y - (d_e + Wmodel @ (z - z_e))
        logL = -0.5 * jnp.sum(resid * resid / n0)
        extra = -xi_quartic * jnp.sum(xi ** 4) + eta_coupling * eta * jnp.sum(z)
        numpyro.factor(
            "f", logL - 0.5 * jnp.sum(z * z) + ldj + 0.5 * jnp.sum(xi * xi) + extra
        )

    model.transport = transport
    model.xi_site = xi_site
    model.hyper_sites = ("timing_eta",)
    return model, ctx, transport


_CENTER = [{"timing_eta": 0.0}]


# ---------------------------------------------------------------------------
# Metric behavior
# ---------------------------------------------------------------------------


def test_linear_gaussian_oracle_passes_all_metrics():
    model, ctx, _ = _oracle()
    report = certify_joint_geometry(
        model, ctx, hyper_points=[{"timing_eta": 0.0}, {"timing_eta": 0.5}]
    )
    assert report.passed, report.failures
    assert report.max_residual_remainder_rms < 1e-6
    assert report.max_residual_remainder_standardized_toa < 1e-6
    assert report.max_xi_gradient_inf_norm < 1e-6
    assert abs(report.xi_hessian_eigen_min - 1.0) < 1e-6
    assert abs(report.xi_hessian_eigen_max - 1.0) < 1e-6
    assert report.max_xi_eta_cross_operator_norm < 1e-6
    assert report.max_conditional_identity_spread < 1e-6


def test_zero_slice_identity_cannot_hide_nonzero_xi_curvature():
    # A quartic-in-xi term vanishes in value, gradient, and Hessian at xi=0, so
    # the old single-point xi=0 identity scan passes; the 2K+9 spread catches it.
    model, ctx, _ = _oracle(xi_quartic=3.0)
    at_zero = target_metrics_at(
        model, xi=np.zeros(model.transport.dimension), hyper={"timing_eta": 0.0}
    )
    assert abs(at_zero.conditional_identity) < 1e-6  # the old scan is fooled
    assert at_zero.xi_gradient_inf_norm < 1e-6
    assert abs(at_zero.xi_hessian_eigen_max - 1.0) < 1e-6

    report = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    assert not report.passed
    assert any("conditional_identity_spread" in f for f in report.failures)


def test_wrong_timing_basis_fails_hessian_and_identity():
    model, ctx, _ = _oracle(wrong_W=2.5)
    report = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    assert not report.passed
    assert report.xi_hessian_eigen_max > 2.0
    assert any("xi_hessian_eigen_max" in f for f in report.failures)
    assert any("conditional_identity_spread" in f for f in report.failures)


def test_localized_toa_remainder_fails_when_global_rms_passes():
    wrap = lambda base: _NonlinearEngine(  # noqa: E731
        base, row=200, scale=2.0e6, direction=[1.0, 0.0, 0.0]
    )
    model, ctx, _ = _oracle(pulsar=_BigPulsar(), engine_wrap=wrap)
    report = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    # Global RMS averages the single-TOA spike away, but the standardized
    # per-TOA metric refuses to.
    assert report.max_residual_remainder_rms <= 0.10
    assert report.max_residual_remainder_standardized_toa > 1.0
    assert not report.passed
    assert any("standardized_toa" in f for f in report.failures)
    assert not any("residual_remainder_rms" in f for f in report.failures)


def test_wrong_cross_term_sign_fails_cross_hessian():
    model, ctx, _ = _oracle(eta_coupling=40.0)
    report = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    assert not report.passed
    assert report.max_xi_eta_cross_operator_norm > 0.25
    assert any("xi_eta_cross_operator_norm" in f for f in report.failures)
    # The cross term does not disturb the diagonal geometry at zero.
    assert report.max_xi_gradient_inf_norm < 1e-6
    assert abs(report.xi_hessian_eigen_max - 1.0) < 1e-6


def test_pit_boundary_fails_center_interior():
    # A bounded (uniform -> prior_pit) chart whose transport centers far outside
    # the PIT interior limit.
    model, ctx, _ = _oracle(identically_linear=[], center_mult=500.0)
    assert ctx.space.prior_bijector.chart_kinds() == ("prior_pit",) * 3
    report = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    assert not report.passed
    assert any("center_interior" in f for f in report.failures)
    assert any(not a.interior and a.chart == "prior_pit" for a in report.center_axes)


# ---------------------------------------------------------------------------
# Transport-center report (§7)
# ---------------------------------------------------------------------------


def test_transport_center_report_distinguishes_affine_and_pit_axes():
    # Affine-normal chart: interior for every finite center, no chart ratio.
    _, ctx_a, transport_a = _oracle(center_mult=500.0)
    axes_a = transport_center_report(ctx_a, transport_a, {})
    assert all(a.chart == "affine_normal" for a in axes_a)
    assert all(a.interior for a in axes_a)
    assert all(a.local_chart_ratio is None for a in axes_a)

    # prior_pit chart: interior tracks |center_z|, and carries a chart ratio.
    _, ctx_p, transport_p = _oracle(identically_linear=[], center_mult=500.0)
    axes_p = transport_center_report(ctx_p, transport_p, {})
    assert all(a.chart == "prior_pit" for a in axes_p)
    assert all(a.local_chart_ratio is not None for a in axes_p)
    assert any(not a.interior for a in axes_p)


# ---------------------------------------------------------------------------
# Box hyper probe construction (§8.4)
# ---------------------------------------------------------------------------


def test_box_hyper_probe_points_quantiles_no_cartesian_product():
    center = {"a": 0.0, "b": 10.0}
    bounds = {"a": (-1.0, 1.0), "b": (0.0, 20.0)}
    points = box_hyper_probe_points(center, bounds)
    # center + 4 quantiles per parameter, no product.
    assert len(points) == 1 + 4 * 2
    assert points[0] == {"a": 0.0, "b": 10.0}
    # Each non-center point moves exactly one parameter.
    for p in points[1:]:
        moved = [k for k in center if p[k] != center[k]]
        assert len(moved) == 1
    a_vals = sorted({p["a"] for p in points})
    assert a_vals == pytest.approx([-0.9, -0.5, 0.0, 0.5, 0.9])


def test_box_hyper_probe_points_rejects_missing_or_degenerate_bounds():
    with pytest.raises(ValueError, match="no bounds"):
        box_hyper_probe_points({"a": 0.0}, {})
    with pytest.raises(ValueError, match="lo<hi"):
        box_hyper_probe_points({"a": 0.0}, {"a": (1.0, 1.0)})


# ---------------------------------------------------------------------------
# Report semantics and determinism
# ---------------------------------------------------------------------------


def test_report_is_deterministic_and_fingerprinted():
    model, ctx, _ = _oracle()
    r1 = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    model2, ctx2, _ = _oracle()
    r2 = certify_joint_geometry(model2, ctx2, hyper_points=_CENTER)
    assert r1.xi_points_digest == r2.xi_points_digest
    assert r1.context_fingerprint == r2.context_fingerprint
    assert r1.model_fingerprint == r2.model_fingerprint
    assert r1.context_fingerprint  # non-empty, stable across identical builds
    assert len(r1.model_fingerprint) == 64  # bare sha256 hex digest
    assert r1.max_conditional_identity_spread == r2.max_conditional_identity_spread
    assert r1.xi_hessian_eigen_max == r2.xi_hessian_eigen_max


def test_report_warn_and_require_passed_are_explicit():
    ok, ctx_ok, _ = _oracle()
    good = certify_joint_geometry(ok, ctx_ok, hyper_points=_CENTER)
    good.require_passed()  # no raise
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeometryDiagnosticWarning)
        good.warn()  # no concern -> nothing emitted

    bad_model, bad_ctx, _ = _oracle(wrong_W=2.5)
    bad = certify_joint_geometry(bad_model, bad_ctx, hyper_points=_CENTER)
    with pytest.raises(GeometryCertificationError):
        bad.require_passed()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad.warn()
    emitted = [w for w in caught if issubclass(w.category, GeometryDiagnosticWarning)]
    assert len(emitted) == len(bad.failures) > 0


def test_threshold_defaults_are_recorded_and_user_overrides_win():
    model, ctx, _ = _oracle(eta_coupling=40.0)
    default = certify_joint_geometry(model, ctx, hyper_points=_CENTER)
    assert not default.passed
    assert default.thresholds == GeometryThresholds()

    m2, ctx2, _ = _oracle(eta_coupling=40.0)
    loose = certify_joint_geometry(
        m2,
        ctx2,
        hyper_points=_CENTER,
        thresholds=GeometryThresholds(xi_eta_cross_operator_norm=10.0),
    )
    assert loose.passed  # the relaxed cross threshold now admits the model
    assert loose.thresholds.xi_eta_cross_operator_norm == 10.0


# ---------------------------------------------------------------------------
# Standalone report persistence (§8.5)
# ---------------------------------------------------------------------------


def _sample_report(**over):
    axis = TransportCenterAxis(
        name="F0",
        chart="prior_pit",
        expansion_z=0.0,
        center_z=6.0,
        center_delta=1e-9,
        local_chart_ratio=0.2,
        interior=False,
    )
    base = dict(
        passed=False,
        failures=("center_interior[point 0]: axis F0",),
        hyper_points=({"log10_A": -14.0},),
        xi_points_digest="deadbeef",
        center_axes=(axis,),
        max_residual_remainder_rms=0.02,
        max_residual_remainder_standardized_toa=0.3,
        max_xi_gradient_inf_norm=0.01,
        xi_hessian_eigen_min=0.9,
        xi_hessian_eigen_max=1.1,
        max_xi_eta_cross_operator_norm=0.05,
        max_conditional_identity_spread=0.04,
        per_point=(
            {
                "hyper": {"log10_A": -14.0},
                "residual_remainder_rms": 0.02,
                "residual_remainder_standardized_toa": 0.3,
                "xi_gradient_inf_norm": 0.01,
                "xi_hessian_eigen_min": 0.9,
                "xi_hessian_eigen_max": 1.1,
                "xi_eta_cross_operator_norm": 0.05,
                "conditional_identity_spread": 0.04,
            },
        ),
        thresholds=GeometryThresholds(),
        context_fingerprint="c" * 64,
        model_fingerprint="d" * 64,
    )
    base.update(over)
    return JointGeometryReport(**base)


def test_standalone_report_json_npz_roundtrip_and_digest_validation(tmp_path):
    report = _sample_report()
    stem = tmp_path / "geom"
    json_path, npz_path = write_geometry_report(report, stem)
    assert json_path.exists() and npz_path.exists()

    loaded = read_geometry_report(stem)
    assert loaded.passed == report.passed
    assert loaded.failures == report.failures
    assert loaded.center_axes == report.center_axes
    assert loaded.thresholds == report.thresholds
    assert loaded.context_fingerprint == report.context_fingerprint
    assert loaded.model_fingerprint == report.model_fingerprint
    assert loaded.max_conditional_identity_spread == pytest.approx(
        report.max_conditional_identity_spread
    )
    assert loaded.per_point[0]["hyper"] == report.per_point[0]["hyper"]

    # Corrupting the NPZ payload is caught by the recorded digest.
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_geometry_report(stem)


def test_standalone_report_atomic_write_refuses_overwrite_by_default(tmp_path):
    report = _sample_report()
    stem = tmp_path / "geom"
    write_geometry_report(report, stem)
    with pytest.raises(FileExistsError):
        write_geometry_report(report, stem)
    # Explicit overwrite succeeds.
    write_geometry_report(report, stem, overwrite=True)


# ---------------------------------------------------------------------------
# No sampler / run-writer coupling (§8.5, §12)
# ---------------------------------------------------------------------------


def test_geometry_diagnostic_is_never_called_by_nuts():
    import inspect

    from nltiming.sampling import numpyro as nlt_numpyro

    sig = inspect.signature(nlt_numpyro.nuts)
    assert not any("geometry" in p or "report" in p for p in sig.parameters)
    src = inspect.getsource(nlt_numpyro)
    assert "certify_joint_geometry" not in src
    assert "write_geometry_report" not in src


def test_run_writer_has_no_geometry_report_api():
    import inspect

    from nltiming import run_io

    sig = inspect.signature(run_io.build_run_manifest)
    assert not any("geometry" in p for p in sig.parameters)
    src = inspect.getsource(run_io)
    assert "geometry_certification" not in src
    assert "certify_joint_geometry" not in src
