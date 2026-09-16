# Installation

`nltiming` is alpha software and is not on PyPI yet. Install the `v0.1.0`
git tag. Python 3.11 or newer is required. The default vela-jax engine runs on
3.11; the optional JUG engine needs 3.12.

## Typical stack

```bash
# nltiming + Discovery + NumPyro (NUTS)
pip install "nltiming[discovery,numpyro] @ git+https://github.com/vhaasteren/nltiming@v0.1.0"

# nltiming + Enterprise + PTMCMCSampler
pip install "nltiming[enterprise,ptmcmc] @ git+https://github.com/vhaasteren/nltiming@v0.1.0"

# The pulsar host (required today) and the default JAX timing engine (vela-jax)
pip install "metapulsar[vela_jax] @ git+https://github.com/vhaasteren/metapulsar"
```

Extras can be combined: `nltiming[discovery,numpyro,enterprise,ptmcmc]`.

## What each extra installs

| Extra | Installs | Notes |
|-------|----------|-------|
| `discovery` | Discovery from `vhaasteren/discovery@feat/class-tracking` | Transport / `kernels="metamath"` / `class_tracking`. Needed for NUTS. The PyPI package named `discovery` is unrelated. `@temp/nltiming` is a stale cho_solve-only tip and cannot run the notebooks. |
| `numpyro` | `jax`, `numpyro`, `arviz` | NUTS sampling of Discovery models. `arviz` 0.x and 1.x both work. |
| `enterprise` | `enterprise-pulsar` from `nanograv/enterprise@dev` | Needs `prior_draw_mode`, not yet released. Pulls `scikit-sparse`, which needs SuiteSparse/CHOLMOD headers (`libsuitesparse-dev` on Debian/Ubuntu). |
| `ptmcmc` | `ptmcmcsampler` | PTMCMC sampling of Enterprise (and Discovery) targets. |
| `enterprise_extensions` | `enterprise_extensions` from `nanograv/enterprise_extensions@dev` | Optional `JumpProposal` / `setup_sampler` helpers. See the libstempo note below. |
| `libstempo` | `libstempo` | Only with a system tempo2 install. |
| `dev` | pytest, black, ruff, mypy | Development. |

The pulsar record and its feather schema come from
[`psrdata`](https://github.com/nanograv/psrdata) tag `v0.1.0`, an
unconditional dependency.

## Timing engines

`nltiming` evaluates the timing model through an engine supplied by the
pulsar host. Today the host is [MetaPulsar](https://github.com/vhaasteren/metapulsar).

| Engine | `engines=` | Install | Gradients |
|--------|-----------|---------|-----------|
| vela-jax (default; Vela's chain in JAX, PINT or tempo2 host) | `"vela_jax"` | `metapulsar[vela_jax]`; the package is not public yet | yes (NUTS) |
| JUG (optional JAX timing package) | `"jug"` | `metapulsar[jug]`, Python 3.12 | yes (NUTS) |
| libstempo / tempo2 | `"libstempo"` | `metapulsar[libstempo]` + system tempo2 | no (PTMCMC, derivative-free) |
| PINT | `"pint"` | included with MetaPulsar | no |
| Vela.jl | `"vela"` | `metapulsar[vela]` + Julia | no |

## tempo2 / libstempo

Only install the `libstempo` extra when a real tempo2 stack is available:

```bash
# TEMPO2_PREFIX must point at the tempo2 install prefix (bin/tempo2 lives
# under $TEMPO2_PREFIX/bin). See libstempo's install docs.
export TEMPO2_PREFIX=/path/to/tempo2/prefix
pip install "nltiming[libstempo]"
```

When you use libstempo for real `tempopulsar` evaluation, prefer process
isolation (`libstempo.sandbox`, or MetaPulsar's `sandbox_tempo2`) so a tempo2
segfault cannot take down the host process. `nltiming` accepts whatever
session object the pulsar provides and never constructs libstempo itself.

`enterprise_extensions@dev` hard-depends on `libstempo>=2.4.0`. On a machine
without tempo2, install it without its dependency chain:

```bash
pip install --no-deps \
  "enterprise_extensions @ git+https://github.com/nanograv/enterprise_extensions.git@dev"
pip install healpy emcee "ptmcmcsampler>=2.1.0" "scikit-learn>=0.24" \
  ephem matplotlib pyarrow six
```

## Logging

`nltiming`, MetaPulsar, and PINT all log through `loguru`, whose built-in
handler prints every `DEBUG` line. Importing `nltiming` (or `metapulsar`)
replaces that untouched default with a `WARNING`-and-above sink. Any
configuration you make yourself, before or after the import, wins:

```python
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="INFO")
```

## Check the install

```bash
python -c "import nltiming, metapulsar, vela_jax, discovery, numpyro; print(nltiming.__name__)"
cd examples/scripts && python quickstart_discovery.py
```
