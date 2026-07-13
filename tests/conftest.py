"""Shared fixtures for nltiming tests."""

from __future__ import annotations

import numpy as np
import pytest


class FakeTimingBackend:
    """Minimal backend scaffold for timing unit tests."""

    def __init__(self, fitpars: list[str]):
        self.fitpars = tuple(fitpars)
        self.native_units = {name: "native" for name in self.fitpars}
        self._theta_exact = {name: "0.0" for name in self.fitpars}

    def reference_theta(self) -> np.ndarray:
        return np.zeros(len(self.fitpars), dtype=float)

    def reference_theta_exact(self) -> dict[str, str]:
        return dict(self._theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return np.zeros(4, dtype=float)

    def design_matrix(self) -> np.ndarray:
        return np.zeros((4, len(self.fitpars)), dtype=float)


class FakePulsarInterface:
    """Minimal host scaffold implementing the expected timing host shape."""

    def __init__(self) -> None:
        self.name = "FAKEPSR"
        self.fitpars = ["F0", "F1"]
        self._toas = np.array([3.0, 1.0, 4.0, 2.0], dtype=float)
        self._residuals = np.zeros(4, dtype=float)
        self._toaerrs = np.full(4, 1e-6, dtype=float)
        self._freqs = np.full(4, 1400.0, dtype=float)
        self._Mmat = np.zeros((4, 2), dtype=float)
        self._flags = {"pta": np.array(["fake"] * 4, dtype="U8")}
        self._backend_flags = np.array(["fake"] * 4, dtype="U8")

    @property
    def toas(self) -> np.ndarray:
        return self._toas

    @property
    def residuals(self) -> np.ndarray:
        return self._residuals

    @property
    def toaerrs(self) -> np.ndarray:
        return self._toaerrs

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    @property
    def Mmat(self) -> np.ndarray:
        return self._Mmat

    @property
    def flags(self) -> dict[str, np.ndarray]:
        return self._flags

    @property
    def backend_flags(self) -> np.ndarray:
        return self._backend_flags

    def pint_model(self):
        return None

    def timing_backend(self, engines="jug") -> FakeTimingBackend:
        return FakeTimingBackend(self.fitpars)

    def can_use_engines(self, engines="jug") -> bool:
        return True

    def cache_token(self) -> str:
        return "fake-pulsar-v1"


@pytest.fixture
def fake_pulsar_interface() -> FakePulsarInterface:
    """Provide a deterministic fake pulsar interface for timing tests."""
    return FakePulsarInterface()
