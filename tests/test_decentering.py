"""NumPy dynamic-decentering transport twin (feature_enterprise_dynamic_parity.md
§5, §10.1): the frontend-agnostic ``NumpyMarginalTransport``, its live marginal
products, and the Enterprise products builder.

T-E2 (dense oracle) and T-E3 (finite-difference Jacobian) are pure NumPy/SciPy
against an independent Woodbury oracle; T-E5 (products accounting) exercises
``enterprise_marginal_products`` on the linear JUG duck. T-E1/T-E4/T-E6/T-E7 land
in PR-E2.
"""

import numpy as np
import pytest

from nltiming.decentering import MarginalProducts, NumpyMarginalTransport


# ---------------------------------------------------------------------------
# Toy live kernel: C(eta) = diag(n0) + F Phi(eta) F^T + M (1e40) M^T + W_m W_m^T
# ---------------------------------------------------------------------------


def _toy(rng, n=40, k=3, comps=4, n_improper=2):
    return {
        "n0": rng.uniform(0.5, 1.5, n),
        "W_s": rng.standard_normal((n, k)),
        "F": rng.standard_normal((n, 2 * comps)),  # RN Fourier (sin+cos)
        "M": rng.standard_normal((n, n_improper)),  # improper timing model
        "W_m": rng.standard_normal((n, 1)),  # z-prior unit-normal block
        "y_t": rng.standard_normal(n),
        "k": k,
        "comps": comps,
        "n_improper": n_improper,
    }


def _phi(eta, comps):
    """A simple positive power-law spectrum, repeated over sin/cos."""
    f = np.arange(1, comps + 1, dtype=float)
    p = 10.0 ** (2.0 * eta["log10_A"]) * f ** (-eta["gamma"])
    return np.repeat(p, 2)


def _f_all(toy):
    return np.hstack([toy["F"], toy["M"], toy["W_m"]])


def _phi_all(eta, toy):
    return np.concatenate(
        [
            _phi(eta, toy["comps"]),
            np.full(toy["n_improper"], 1.0e40),  # improper timing model
            np.full(1, 1.0),  # unit-normal z-prior
        ]
    )


def _products_fn(toy):
    """Enterprise-style Woodbury projection: C^-1 = N^-1 - N^-1 F Sigma^-1 F^T N^-1
    with Sigma = diag(Phi^-1) + F^T N^-1 F. Never forms a dense C^-1 (the 1e40
    block stays as 1e-40 in Phi^-1)."""
    W_s, y_t = toy["W_s"], toy["y_t"]
    Ninv = 1.0 / toy["n0"]
    F_all = _f_all(toy)

    def products(params):
        phi_all_inv = 1.0 / _phi_all(params, toy)
        FtNi = (F_all * Ninv[:, None]).T
        sigma = np.diag(phi_all_inv) + FtNi @ F_all
        FtNiW = FtNi @ W_s
        FtNiy = FtNi @ y_t
        WtNiW = (W_s * Ninv[:, None]).T @ W_s
        WtNiy = (W_s * Ninv[:, None]).T @ y_t
        SinvFW = np.linalg.solve(sigma, FtNiW)
        G = WtNiW - FtNiW.T @ SinvFW
        b = WtNiy - SinvFW.T @ FtNiy
        return MarginalProducts(G=G, b=b)

    return products


def _oracle(toy, eta):
    """Independent Woodbury oracle that materializes C^-1 as a matrix, then
    forms A = W_s^T C^-1 W_s + I and mu = A^-1 W_s^T C^-1 y_t."""
    W_s, y_t = toy["W_s"], toy["y_t"]
    Ninv = np.diag(1.0 / toy["n0"])
    F_all = _f_all(toy)
    phi_all = _phi_all(eta, toy)
    inner = np.diag(1.0 / phi_all) + F_all.T @ Ninv @ F_all
    Cinv = Ninv - Ninv @ F_all @ np.linalg.solve(inner, F_all.T @ Ninv)
    G = W_s.T @ Cinv @ W_s
    b = W_s.T @ Cinv @ y_t
    A = G + np.eye(toy["k"])
    mu = np.linalg.solve(A, b)
    return A, mu


_ETAS = [
    {"log10_A": -14.5, "gamma": 2.0},
    {"log10_A": -14.0, "gamma": 3.2},
    {"log10_A": -13.5, "gamma": 4.5},
    {"log10_A": -15.0, "gamma": 1.5},
    {"log10_A": -13.0, "gamma": 5.5},
]


def test_te2_matches_dense_oracle():
    """T-E2: A = G + I and the centering mu = A^-1 b match the independent
    Woodbury oracle to rtol=1e-8 at 5 eta draws, center on and off."""
    rng = np.random.default_rng(20260720)
    toy = _toy(rng)
    products = _products_fn(toy)

    tr = NumpyMarginalTransport(
        products,
        dimension=toy["k"],
        key="timing",
        params=("gamma", "log10_A"),
        center=True,
    )
    tr0 = NumpyMarginalTransport(
        products,
        dimension=toy["k"],
        key="timing",
        params=("gamma", "log10_A"),
        center=False,
    )
    for eta in _ETAS:
        A_oracle, mu_oracle = _oracle(toy, eta)
        L, _ = tr._factor(eta)
        np.testing.assert_allclose(L @ L.T, A_oracle, rtol=1e-8)
        # center=True at xi=0 gives the GLS centering mu.
        z, _ = tr.apply(eta, np.zeros(toy["k"]))
        np.testing.assert_allclose(z, mu_oracle, rtol=1e-8)
        # center=False at xi=0 gives 0 (same A factor).
        z0, _ = tr0.apply(eta, np.zeros(toy["k"]))
        np.testing.assert_allclose(z0, 0.0, atol=1e-12)


@pytest.mark.parametrize("center", [True, False])
def test_te3_jacobian_finite_difference(center):
    """T-E3: central-FD Jacobian of xi -> z has slogdet == returned ldJ to
    rtol=1e-8 (pins the trans=1 / L^-T orientation)."""
    rng = np.random.default_rng(7)
    toy = _toy(rng)
    tr = NumpyMarginalTransport(
        _products_fn(toy),
        dimension=toy["k"],
        key="timing",
        params=("gamma", "log10_A"),
        center=center,
    )
    eta = {"log10_A": -14.0, "gamma": 3.0}
    xi0 = rng.standard_normal(toy["k"])

    def z_of(xi):
        return tr.apply(eta, xi)[0]

    h = 1e-6
    J = np.zeros((toy["k"], toy["k"]))
    for j in range(toy["k"]):
        e = np.zeros(toy["k"])
        e[j] = h
        J[:, j] = (z_of(xi0 + e) - z_of(xi0 - e)) / (2.0 * h)
    _, logabsdet = np.linalg.slogdet(J)
    _, ldJ = tr.apply(eta, xi0)
    np.testing.assert_allclose(logabsdet, ldJ, rtol=1e-8)


def test_numpy_transport_rejects_bad_precision():
    """No floors: a negative or wrong-shape prior_precision raises (E12)."""
    rng = np.random.default_rng(1)
    toy = _toy(rng)
    with pytest.raises(ValueError, match="prior_precision"):
        NumpyMarginalTransport(
            _products_fn(toy),
            dimension=toy["k"],
            key="t",
            params=(),
            prior_precision=-1.0,
        )
    with pytest.raises(ValueError, match="dimension"):
        NumpyMarginalTransport(_products_fn(toy), dimension=0, key="t", params=())


def test_numpy_transport_duck_surface():
    """Duck-parity with discovery MarginalTransport (S11): blocks[0] keys and the
    distinct fingerprint schema."""
    rng = np.random.default_rng(2)
    toy = _toy(rng)
    tr = NumpyMarginalTransport(
        _products_fn(toy),
        dimension=toy["k"],
        key="timing",
        params=("gamma", "log10_A"),
    )
    d = tr.diagnostics()
    assert sorted(d["blocks"][0]) == ["conditioner_kind", "k", "keys", "name", "params"]
    assert d["reference_noise"] == "live_kernel_numpy"
    assert tr.fingerprint().startswith("sha256:")
    assert list(tr.index) == ["timing"]


# ---------------------------------------------------------------------------
# T-E5: enterprise_marginal_products accounting (needs a real PTA)
# ---------------------------------------------------------------------------


@pytest.fixture
def _enterprise_setup():
    import jax

    jax.config.update("jax_enable_x64", True)
    pytest.importorskip("jug")
    pytest.importorskip("discovery")
    pytest.importorskip("enterprise")
    import discovery as ds

    ds.config(kernels="metamath")
    from enterprise.signals import (  # noqa: E402
        gp_signals,
        parameter,
        signal_base,
        white_signals,
    )
    from enterprise.signals import utils as ent_utils

    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_decentered_model import _DiscoveryPulsar

    from nltiming import TimingInference
    from nltiming.nonlinear_timing_model import NonLinearTimingModel

    mp = _DiscoveryPulsar()
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(mp)

    efac = parameter.Uniform(0.5, 2.0)
    white = white_signals.MeasurementNoise(efac=efac)
    pl = ent_utils.powerlaw(
        log10_A=parameter.Uniform(-18, -11), gamma=parameter.Uniform(0, 7)
    )
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=5, name="rednoise")
    pta = signal_base.PTA([(white + rn + ntm.enterprise_signal())(mp)])
    return {"pta": pta, "ctx": ctx, "mp": mp}


def test_te5_products_accounting(_enterprise_setup):
    """T-E5: products.params excludes delay keys and fixed WN and is sorted (E8);
    missing WN raises listing the names; the E9 memo returns the identical object
    on a repeated eta and recomputes on any change."""
    from nltiming.likelihoods.enterprise import enterprise_marginal_products

    pta, ctx, mp = (_enterprise_setup[k] for k in ("pta", "ctx", "mp"))
    fixed_wn = {f"{mp.name}_efac": 1.0}

    products = enterprise_marginal_products(pta, ctx, fixed_wn_params=fixed_wn)

    # E8: sorted; excludes delay keys and fixed WN.
    assert list(products.params) == sorted(products.params)
    assert not (set(products.params) & set(ctx.delay_keys))
    assert f"{mp.name}_efac" not in products.params
    # Exactly the two red-noise hypers remain.
    assert set(products.params) == {
        f"{mp.name}_rednoise_log10_A",
        f"{mp.name}_rednoise_gamma",
    }

    # E7: missing white noise raises, listing the offending name.
    with pytest.raises(ValueError, match="efac"):
        enterprise_marginal_products(pta, ctx, fixed_wn_params={})

    # E9: one-slot memo returns the identical object for a repeated eta and
    # recomputes on any change.
    eta = {n: -14.0 if "log10_A" in n else 3.0 for n in products.params}
    o1 = products(dict(eta))
    o2 = products(dict(eta))
    assert o1 is o2
    changed = dict(eta)
    changed[f"{mp.name}_rednoise_gamma"] = 4.0
    o3 = products(changed)
    assert o3 is not o1
    assert isinstance(o3, MarginalProducts)


def test_products_handles_dense_phiinv(_enterprise_setup, monkeypatch):
    """P1: get_phiinv may return a dense (m, m) matrix (common-signal / non-
    diagonal Phi); Sigma must add it as a MATRIX, not misread its diagonal and
    broadcast. Mirrors Enterprise's own `phiinv.ndim == 1` branch."""
    import scipy.linalg as sl

    from nltiming.likelihoods.enterprise import enterprise_marginal_products

    pta, ctx, mp = (_enterprise_setup[k] for k in ("pta", "ctx", "mp"))
    fixed_wn = {f"{mp.name}_efac": 1.0}
    eta = {
        f"{mp.name}_rednoise_log10_A": -14.0,
        f"{mp.name}_rednoise_gamma": 3.0,
    }
    full = {**fixed_wn, **eta}

    # Real 1-D phiinv -> a genuinely dense SPD version with off-diagonal terms.
    phi1d = np.asarray(pta.get_phiinv(full, logdet=False)[0], dtype=float)
    m = phi1d.shape[0]
    rng = np.random.default_rng(0)
    off = 0.01 * rng.standard_normal((m, m))
    dense = np.diag(phi1d) + off @ off.T  # SPD, genuinely dense
    monkeypatch.setattr(pta, "get_phiinv", lambda p, logdet=False: [dense])

    products = enterprise_marginal_products(pta, ctx, fixed_wn_params=fixed_wn)
    out = products(full)

    # Independent oracle with the dense phiinv added as a MATRIX.
    ndiag = pta.get_ndiag(fixed_wn)[0]
    lin = ctx.linearization
    W = np.asarray(lin.sampled_basis, dtype=float)
    y_t = np.asarray(
        lin.transport_effective_residual(np.asarray(mp.residuals)), dtype=float
    )
    T = np.asarray(pta.get_basis(fixed_wn)[0], dtype=float)
    WNW = np.asarray(ndiag.solve(W, left_array=W), dtype=float)
    TNW = np.asarray(ndiag.solve(W, left_array=T), dtype=float)
    WNy = np.asarray(ndiag.solve(y_t, left_array=W), dtype=float)
    TNy = np.asarray(ndiag.solve(y_t, left_array=T), dtype=float)
    TNT = np.asarray(pta.get_TNT(fixed_wn)[0], dtype=float)
    SW = sl.cho_solve(sl.cho_factor(TNT + dense, lower=True), TNW)
    np.testing.assert_allclose(out.G, WNW - TNW.T @ SW, rtol=1e-10)
    np.testing.assert_allclose(out.b, WNy - SW.T @ TNy, rtol=1e-10)
    # The old broadcast bug (TNT + np.diag(dense), where np.diag of a 2-D array
    # returns the 1-D diagonal) is not even SPD here, so it would crash the
    # builder rather than pass this assertion — either way the branch is pinned.
