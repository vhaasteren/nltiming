"""Sampler-facing glue over a ``TimingSignal``.

- ``numpyro``: NumPyro/Discovery model builder, NUTS setup, timing sites.
- ``ptmcmc``: PTMCMC helpers for Enterprise and derivative-free Discovery
  marginal likelihoods, including host timing engines such as Vela.

Modules import their sampler dependencies lazily; importing this package
requires neither numpyro nor PTMCMCSampler.
"""

from . import numpyro, ptmcmc

__all__ = ["numpyro", "ptmcmc"]
