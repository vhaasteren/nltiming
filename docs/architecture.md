# Architecture and scope

`nltiming` owns the nonlinear-timing math, engine-selection vocabulary,
backend-neutral engine support, and the Discovery and Enterprise likelihood
interfaces. Concrete PINT, libstempo, JUG, and Vela adapters plus composite
assembly live in MetaPulsar (`metapulsar.engines`). Pulsars (single-pulsar or
multi-PTA composites such as MetaPulsar) supply the data via the
`TimingPulsar` protocol; the JUG package owns the JAX timing-engine
primitives.

Deliberately **out of scope**: Fourier/DM/chromatic/ECORR bases, `Phi`
inference, power-law or free-spectrum projection, and correlated-noise
likelihoods. Those belong to Discovery and Enterprise. `nltiming` supplies the
timing block and prior transform they build on, and downstream quick-look GP
tooling composes `nltiming` with those likelihood interfaces rather than re-homing noise
math here.

The interactive transformed-space (`z`) timing fit (`fit_z`, `jacobian_z`,
`TimingZFitResult`) is described in [`evaluator.md`](evaluator.md).

| Layer | Owner |
|-------|-------|
| Nonlinear-timing math (`ParameterSpace`, bijectors, whitening, priors, inference plan) | **nltiming** |
| Engine protocols, selection vocabulary (`engine_config`), validators / `LinearModel` (`engine_support`) | **nltiming** |
| Backend adapters (PINT, libstempo, JUG, Vela) + multi-PTA composite | **MetaPulsar** (`metapulsar.engines`) |
| Discovery + Enterprise likelihood interfaces, model helpers, sampler recipes, run products | **nltiming** |
| JAX / nonlinear timing-engine primitives | **JUG** |
| Multi-PTA pulsar, session construction, data combination | **MetaPulsar** |
| GP bases, `Phi` inference, spectra, correlated-noise likelihoods | **Discovery / Enterprise** |

`nltiming` never imports the `metapulsar` package. At runtime it calls a supplied
`TimingPulsar.timing_engine(...)` which returns a MetaPulsar-owned engine.


## Ownership: `nltiming` owns model semantics, not sampler execution

`nltiming` supplies native objects at the two likelihood interfaces:

- **Discovery:** build a NumPyro model with `sampling.numpyro.joint_model`
  (full-basis / dynamic transport, `whitening=None`),
  `sampling.numpyro.decentered_model` (marginalized dynamic decentering, the
  small sampled timing block whitened against the live `C(η)`), or
  `sampling.numpyro.model` (static whitening path). All return an ordinary
  zero-argument NumPyro model with `.to_df` for decoded timing columns. Sample
  with `sampling.numpyro.nuts`, Discovery's `makesampler_nuts` +
  `run_nuts_with_checkpoints`, or raw `numpyro.infer.NUTS`/`MCMC`.
- **Enterprise:** `spec.enterprise_signal()` returns ordinary Enterprise
  `Parameter` objects (per-axis scalars when `whitening=None`, or one joint
  vector under `WhiteningConfig`). Sample the resulting `PTA` exactly like any
  other Enterprise analysis, `enterprise_extensions.sampler.setup_sampler`
  needs no `nltiming` import.

The `xi -> z -> delta_theta` chart-and-layer, the physical prior, and the
Jacobian are model semantics and live only in `nltiming.ParameterSpace`; they are
never reimplemented in a sampler wrapper. The static-layer choice (`whitening=None`
vs `WhiteningConfig()`) changes the Enterprise parameter layout (`pta.param_names`)
and the NumPyro coordinate, never the top-level sampling script.

The pulsar object must satisfy the `TimingPulsar` protocol (also exported under
the original `TimingPulsar` name): frozen TOA arrays, `pint_model()`,
`timing_engine()`; single-pulsar and multi-PTA composite pulsars both work , 
PTA-suffixed parameter names are matched by base name.

**Today, MetaPulsar is required.** The only production `TimingPulsar`
implementation is
[MetaPulsar](https://github.com/vhaasteren/metapulsar), even for a single PTA
dataset. Examples and docs therefore build pulsars with `create_metapulsar`.
Once Discovery and/or Enterprise ship a native `TimingPulsar`, that dependency
can be dropped; the `nltiming` API does not change.


## Engine contract: phase gauge and hybrid residual linearization

**Phase gauge.** Engines export gauge-free `residual_delta` /
`residual_jacobian` (\(J=-M\)). `design_matrix` / \(M\) is the delay tangent
(fitter sign). `derivative_method="analytic"|"autodiff"` selects the route to
\(M\) (`pulsar.Mmat` vs `-residual_jacobian()`), not a different object. The
same knob selects the proper-axis `TimingLinearization` source (`M_s ∂δ/∂z`
vs `jacfwd` of `residual_delta_jax`); there is no finite-difference route.
There is no `waveform_jacobian`.

**Hybrid residual linearization.** `TimingSpec(nonlinear_params=...)`
forwards a closed mode (`None` | `"binary"` | `"binary+"`) into
`MetaPulsar.timing_engine`, and **every engine family executes it**: `None`
is the full native residual at the sampled point; `"binary"` keeps only the
binary axes nonlinear and evaluates every other fitpar (spin, astrometry,
DM, …) through its design-matrix column, `−M[:, lin] δ_lin`, with
astrometry frozen at the par-file reference inside the binary delay;
`"binary+"` additionally keeps `PX` nonlinear. JUG runs the formula inside
its residual graph; the libstempo / Vela / PINT adapters realise the same
model with their native/exact-linear split, using JUG's parameter registry
for the binary partition. nltiming does not choose a mode from the
inference plan; it refuses an engine that did not execute the requested
mode, and the run manifest records the mode the engine *executed*. Under a
hybrid mode the linearized axes are reported as identically linear by every
family, on the leaf engines and on the composite alike.

A linear-vs-nonlinear contrast is therefore a deliberate choice of mode
(`None` vs `"binary"` / `"binary+"`) by the caller: two runs that both pass
`nonlinear_params=None` are the same residual on any engine.
