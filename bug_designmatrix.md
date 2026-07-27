# Redesign: raw fitter matrices and exact residual Jacobians in nltiming

**Status:** revised redesign proposal  
**Package:** `nltiming` (`ref-packages/nltiming`)  
**Depends on JUG:** `ref-packages/jug/feature_designmatrix_naming_conventions.md`  
**Backward compatibility:** not required. JUG, nltiming, MetaPulsar, validation
code, and notebooks are updated together. Delete wrong APIs; do not shim them.

---

## Terminology lock (normative)

Locked with JUG / MetaPulsar. See
`ref-packages/jug/feature_designmatrix_naming_conventions.md` (terminology lock)
and MetaPulsar `docs/design_matrix_terminology.md`.

| Symbol | Public name | Meaning |
|---|---|---|
| \(M\) | **`design_matrix`** | Raw fitter basis (PINT/tempo2). What every other PTA package means by “design matrix”. |
| \(J\) | **`residual_jacobian`** | \(\partial(\Delta r)/\partial\theta\). Never a design matrix. |
| \(W\) | **`waveform_jacobian`** | Delay tangent; \(W=-J\). |

Hard rules: `design_matrix` means only raw \(M\); `residual_jacobian` is derived
only from the residual function; bare “Jacobian” in docs means you should
probably write **residual Jacobian** if you mean \(J\).

---

## 0. Executive decision

nltiming uses the same raw design-matrix meaning as JUG and PINT:

\[
M_\mathrm{raw}
  := \frac{\partial d(\theta)}{\partial\theta_\mathrm{fit}},
\]

in PINT–Vela public fit units and the owning engine's declared row/column order.
Leaf contribution engines use contribution-local order; the composite engine
assembles canonical global pulsar order.

The exact residual Jacobian is a different object:

\[
J_\mathrm{residual}
  := \frac{\partial\,\Delta r}{\partial\delta\theta}.
\]

For a residual projection \(P\) and the row-constant TZR-reference response
\(A_\mathrm{TZR}\):

\[
J_\mathrm{residual}
  = P(-M_\mathrm{raw}+A_\mathrm{TZR}).
\]

Therefore:

```text
weighted/unweighted mean removal:
    J_residual = -P M_raw

uncentered residuals:
    J_residual = -M_raw + A_TZR

subtract_tzr=False:
    A_TZR = 0
```

Current JUG exports use weighted or unweighted mean removal, so their ordinary
production identity remains \(J=-PM\). The uncentered case is nevertheless part
of the interface contract and prevents nltiming from defining an autodiff
design matrix as `-jac(residual_delta)`.

There is no parameter-class-specific sign policy. JUMP, DMX, PHOFF, and
synthetic Offset columns use the same raw fitter sign as all other columns.
Exact-linear is an evaluation strategy, not a second convention.

---

## 1. Current failure

### 1.1 Sign mixing

Today:

1. JUG's public matrix builder returns fitter \(M\).
2. `export_jax_timing_state` negates it and stores residual-Jacobian-like
   columns under `state.design_matrix`.
3. `JugEngine.linearized_design_matrix()` copies those columns into a matrix
   otherwise populated from `MetaPulsar.Mmat`.
4. Exact-linear JUMP/DMX/Offset columns retain the opposite sign.
5. Exact-linear residual evaluation adds `+Mδ`, whereas PINT-compatible
   residual behavior requires the projected `-Mδ`.

The upstream JUG feature fixes the state boundary first.

### 1.2 Construction and residual differentiation are conflated

`_timing_design_matrix(..., method="autodiff")` currently calls
`linearized_design_matrix()`. The earlier redesign proposed falling back to:

```python
-jax.jacfwd(engine.residual_delta_jax)(zero)
```

That fallback is rejected. It produces \(-J_\mathrm{residual}\), not necessarily
the raw fitter matrix:

```text
-J_residual = P M_raw - P A_TZR
```

It is safe only after a known projection has annihilated the TZR response, and
even then it returns a projected matrix rather than `M_raw`.

### 1.3 Unit conversion is duplicated downstream

The revised JUG state stores `state.design_matrix` directly in PINT–Vela fit
units. nltiming's current `linearized_design_matrix()` divides JUG state columns
by `_native_scale`; doing that after the JUG migration would scale RAJ/DECJ
twice.

Input deltas passed to the JAX state still require fit-to-native conversion.
Output design columns do not.

---

## 2. Objects and invariants

### 2.1 Raw design matrix

Every public nltiming method named `design_matrix` returns raw fitter
\(M_\mathrm{raw}\):

- PINT–Vela fit units;
- the owning engine's declared row order;
- its declared `fitpars` column order;
- no residual mean removal;
- no whitening or TOA-error weighting;
- no hidden TZR-reference term;
- implicit Offset only when it is explicitly part of the pulsar fitter basis.

`design_matrix()` and `linearized_design_matrix()` share meaning, sign, units,
and ordering. “Linearized” means frozen at the engine/export reference.

### 2.2 Projected fitter matrix

When an engine exports mean-removed or reordered residuals:

\[
M_\mathrm{projected}=P M_\mathrm{raw}.
\]

This is exposed only through an explicitly named operation:

```python
engine.project_fitter_matrix(matrix)
engine.projected_design_matrix()
```

Projection does not change the fitter sign. It only applies frozen mean removal
and row ordering.

### 2.3 Exact residual Jacobian

Code that needs the tangent of nonlinear residual evaluation uses:

```python
engine.residual_jacobian(...)
# or jacfwd(engine.residual_delta_jax)(...)
```

It does not call that object a design matrix.

Model D/Discovery waveform linearization remains:

```python
delay = -residual_delta
W = jacfwd(-residual_delta)
```

Thus \(W=-J_\mathrm{residual}\). With ordinary mean removal,
\(W=P M_\mathrm{raw}\); for an uncentered TZR-referenced residual it also
contains \(-A_\mathrm{TZR}\). This is correct because Model D wants the exact
waveform tangent, not merely the fitter basis.

### 2.4 TZR and Offset

nltiming follows the upstream PINT-compatible decision:

- `TZRMJD`, `TZRSITE`, and `TZRFRQ` are model state, not design-matrix fit
  coordinates;
- implicit Offset is a synthetic leading fitter coordinate;
- explicit PHOFF is an ordinary timing parameter and suppresses implicit
  Offset;
- Offset must never be implemented as a TZRMJD perturbation.

The implicit Offset metadata/value comes from the pulsar/JUG/PINT matrix oracle;
nltiming does not reconstruct it independently.

---

## 3. Target engine surface

### 3.1 Required surface

Every `TimingEngine` continues to expose:

```python
def design_matrix(self, params=None) -> np.ndarray:
    """Raw fitter M at the engine reference."""

def residual_delta(self, delta_theta) -> np.ndarray:
    """Exact exported residual change."""
```

JAX-capable engines expose `residual_delta_jax`.

### 3.2 Construction-specific optional surface

Engines may expose:

```python
def autodiff_design_matrix(self, params=None) -> np.ndarray:
    """Raw M obtained by differentiating the engine's timing-prediction graph."""

def linearized_design_matrix(self, params=None) -> np.ndarray:
    """Raw M frozen at the engine/export reference."""

def project_fitter_matrix(self, matrix) -> np.ndarray:
    """Apply this engine's frozen residual mean/order projection to raw M."""

def projected_design_matrix(self, params=None) -> np.ndarray:
    return self.project_fitter_matrix(self.design_matrix(params))
```

Rules:

- No `*design_matrix*` method returns a residual Jacobian.
- `autodiff_design_matrix()` must be implemented from the raw timing-prediction
  graph. It must not be implemented as `-jac(residual_delta)`.
- Engines without a raw autodiff implementation reject
  `design_matrix_method="autodiff"` clearly.
- A projection method is required before nltiming substitutes exact-linear
  columns into an engine whose residual output is projected.
- An identity projection is declared explicitly; it is not guessed because an
  engine happens not to expose projection metadata.

### 3.3 JUG engine

After the JUG feature lands:

```python
def design_matrix(self, params=None):
    # Canonical pulsar/Mmat raw fitter basis, normally analytic.
    return np.asarray(self._model.design, dtype=float)

def autodiff_design_matrix(self, params=None):
    design = np.asarray(self._model.design, dtype=float).copy()
    jug_raw = np.asarray(self._state.design_matrix, dtype=float)
    for local_col, model_col in enumerate(self._jug_indices):
        # State columns are already PINT–Vela fit units.
        design[:, model_col] = jug_raw[:, local_col]
    return design

def linearized_design_matrix(self, params=None):
    # Frozen JUG state M for JUG-owned columns; raw exact-linear model columns.
    return self.autodiff_design_matrix(params=params)

def project_fitter_matrix(self, matrix):
    return self._state.residual_projection.apply_matrix(matrix)
```

Important:

- delete output division by `_native_scale`;
- retain `_native_scale` (or the authoritative JUG helper) only for converting
  incoming nltiming fit-unit deltas to the state graph's native coordinates;
- preserve the state matrix's session TOA order until the contribution/composite
  layer places it in canonical global row order;
- do not reapply `ResidualProjection.row_indices` twice.

### 3.4 Composite engine

The composite engine assembles raw matrices contribution by contribution in
canonical global row order.

Projection is contribution-local:

```python
projected_local = contribution.engine.project_fitter_matrix(raw_local)
```

Do not apply one global weighted mean across PTA contributions. Each timing
engine owns its frozen residual convention and row mapping.

---

## 4. Exact-linear evaluation

### 4.1 Required sign and projection

For exact-linear raw columns \(M_E\):

\[
\Delta r_E=-P M_E\,\delta_E.
\]

JUG implementation sketch:

```python
def _exact_linear_delta(self, delta_theta):
    indices = self._exact_linear_indices
    if not indices:
        return zeros
    raw = np.asarray(self._model.design[:, list(indices)], dtype=float)
    projected = self.project_fitter_matrix(raw)
    return -(projected @ delta_theta[list(indices)])
```

The JAX twin uses the same frozen projection with JAX arrays.

This replaces the current `+raw @ delta` behavior.

### 4.2 Other engines

Apply the same rule to libstempo and Vela exact-linear fallbacks:

- use minus fitter sign;
- apply the engine's declared residual projection;
- preserve local row order;
- do not invent a JUMP/DMX sign exception.

If an engine cannot state how its residual output projects the raw fitter
matrix, it cannot safely use raw columns as exact residual deltas. Fail rather
than silently assuming identity.

### 4.3 Synthetic Offset

If Offset appears in nltiming `fitpars`, it is exact-linear:

- take the raw oracle column from `LinearModel.design`;
- apply the engine projection;
- multiply by the same explicit minus;
- never map it to TZRMJD.

After mean removal its projected direction may be zero/rank-deficient. That is
expected and must be handled by existing rank/marginalization logic.

---

## 5. Design-matrix method selection

Use:

```python
DesignMatrixMethod = Literal["analytic", "autodiff", "linearized"]
```

All three values return raw fitter \(M_\mathrm{raw}\), but use independent
construction routes:

| Method | Source |
|---|---|
| `"analytic"` | `pulsar.Mmat` / existing host analytical basis |
| `"autodiff"` | `engine.autodiff_design_matrix()` |
| `"linearized"` | `engine.linearized_design_matrix()` |

Selector:

```python
def resolve_timing_design_matrix(pulsar, engine, *, method):
    method = normalize_design_matrix_method(method)
    if method == "analytic":
        return np.asarray(pulsar.Mmat, dtype=float)
    if method == "autodiff":
        fn = getattr(engine, "autodiff_design_matrix", None)
        if fn is None:
            raise ValueError(
                "design_matrix_method='autodiff' requires a raw "
                "autodiff_design_matrix() implementation"
            )
        return np.asarray(fn(), dtype=float)
    if method == "linearized":
        fn = getattr(engine, "linearized_design_matrix", None)
        if fn is None:
            raise ValueError(
                "design_matrix_method='linearized' requires "
                "linearized_design_matrix()"
            )
        return np.asarray(fn(), dtype=float)
    raise AssertionError("unreachable")
```

Do not fall back from autodiff to:

- `linearized_design_matrix`;
- `-jacfwd(residual_delta_jax)`;
- analytic `pulsar.Mmat`.

Analytic and autodiff are intentionally independent. Their numerical equality
is not an acceptance criterion.

Export this selector as a public nltiming helper so paper validation and other
callers cannot reimplement method routing.

---

## 6. Model D, whitening, and marginalization

### 6.1 Model D / Discovery

Keep:

```python
delay = -residual_delta
W = jacfwd(delay)
```

This consumes the exact residual graph and therefore handles the TZR term
correctly. Do not replace \(W\) with a raw design matrix.

### 6.2 Fisher matrices and improper timing bases

`ctx.design_matrix`, cheat priors, Fisher metrics, and improper GP timing bases
continue to use the selected raw fitter matrix.

Most such operations depend on column span or \(M^\top N^{-1}M\), but that does
not justify mixing raw and projected matrices silently. Callers requiring the
residual-output basis must request projection explicitly.

### 6.3 WLS sign conventions

Audit `schur_delta_wls` and related routines by equation rather than relying on
the old overloaded name:

- if the routine solves in the fitter basis, pass raw/projected \(M\) as its
  documented equation requires;
- if it linearizes exact residual changes, pass `residual_jacobian`;
- do not change signs merely because Fisher widths are sign-insensitive.

---

## 7. Concrete file changes

| File | Required change |
|---|---|
| `src/nltiming/protocols.py` | Document raw `design_matrix`; add optional construction/projection protocols if useful |
| `src/nltiming/engines/jug.py` | exact-linear minus + projection; raw state columns with no output rescaling; add `autodiff_design_matrix`; fix docstrings |
| `src/nltiming/engines/composite.py` | contribution-local raw assembly and projection; avoid double row sorting |
| `src/nltiming/engines/tempo2.py` | exact-linear minus + declared projection |
| `src/nltiming/engines/vela.py` | exact-linear minus + declared projection |
| `src/nltiming/engines/pint.py` | verify raw matrix and exact residual behavior; do not infer raw autodiff from residual jacobian |
| `src/nltiming/nonlinear_timing_model.py` | replace private selector with public `resolve_timing_design_matrix`; accept three methods |
| `src/nltiming/run_io.py` | persist/validate three method values |
| `src/nltiming/whitening.py` | document whether each entry point consumes raw/projected M or residual J |
| `paper/code/validation/validation/marginalization_diagnostics.py` | use public selector, never bare backend method |

Update MetaPulsar engine construction to stop passing `design_matrix_method` into
JUG state export; method selection happens at nltiming matrix resolution, while
the JUG frozen state always owns its selected-graph raw autodiff matrix.

---

## 8. Tests

### 8.1 Naming, sign, and units

For every engine:

- every `*design_matrix*` result has fitter sign;
- raw matrices share PINT–Vela fit units and their declared ordering;
- no raw matrix has residual mean removal baked in;
- JUG RAJ/DECJ state columns are not divided by `_native_scale` on output.

Use explicit array tolerances, labels, and units rather than correlation-only
checks.

### 8.2 Method routing

Assert:

- `"analytic"` reads `pulsar.Mmat`;
- `"autodiff"` calls only `autodiff_design_matrix`;
- `"linearized"` calls only `linearized_design_matrix`;
- missing methods raise;
- autodiff never differentiates `residual_delta` as a fallback;
- analytic/autodiff equality is not required.

### 8.3 Projected residual tangent

For weighted/unweighted mean removal:

```python
J = jacfwd(engine.residual_delta_jax)(zero)
M_projected = engine.projected_design_matrix()
assert_allclose(J, -M_projected, ...)
```

For a synthetic uncentered test:

```python
J = jacfwd(engine.residual_delta_jax)(zero)
assert_row_constant(J + engine.design_matrix())
```

When `subtract_tzr=False`, assert the row-constant difference is zero.

Current production JUG sessions do not emit `mean_mode="none"`; the synthetic
test protects the reserved future contract.

### 8.4 Exact-linear columns

For JUMP, DMX, PHOFF, and Offset where supported:

```python
expected = -(engine.project_fitter_matrix(M_exact) @ delta_exact)
assert_allclose(engine_exact_delta, expected)
```

Verify contribution-local projection in a composite/multi-PTA case.

### 8.5 Model D

Verify:

```python
W = jacfwd(-engine.residual_delta_jax)(zero)
```

against direct finite differences of the waveform. Do not define success as
equality with raw \(M\) in the uncentered case.

### 8.6 Public selector consumers

Test paper validation and nltiming context construction with each method and
assert both use the public selector.

---

## 9. Implementation order

1. Land the JUG proposal first.
2. Update JUG state field/units consumption in `engines/jug.py`.
3. Add raw autodiff/linearized/projection engine methods.
4. Fix exact-linear signs and projections in JUG, tempo2, Vela, and composite
   paths.
5. Implement/export the public three-way selector.
6. Update nonlinear model, run manifests, whitening docs, and validation.
7. Rewrite fake engines/tests to distinguish raw \(M\), projected \(PM\), and
   exact \(J\).
8. Run focused nltiming tests, JUG contract tests, then `make fast`.

No intermediate dual-sign compatibility flag.

---

## 10. Clarifications

**Why not define autodiff as `-jac(residual_delta)`?**  
Because that is a projected residual object and may include the TZR-reference
term. Public autodiff means differentiation of the raw timing-prediction graph.

**Is `linearized_design_matrix` redundant?**  
Often yes at the reference point. It records the intent to use a frozen raw
matrix rather than a newly assembled analytic matrix.

**Should analytic and autodiff agree?**  
Not by contract. They are independent implementations with independent oracles.

**Where do consumer/sampling-chart transforms live?**  
In nltiming's physical-chart layer. JUG exports PINT–Vela fit coordinates only.

**What happens to TZRMJD?**  
It remains real model state but is not a PINT-compatible fit coordinate.
Implicit Offset or explicit PHOFF represents the fitted phase-offset direction.

**What if an engine cannot expose its residual projection?**  
It may still provide raw matrices, but nltiming must not use those columns as
exact residual deltas. Fail at the exact-linear boundary.

---

## 11. Acceptance criteria

- [ ] `design_matrix`, `autodiff_design_matrix`, and
      `linearized_design_matrix` all mean raw fitter \(M\).
- [ ] No design-matrix API returns or derives from a residual Jacobian.
- [ ] Raw matrices use PINT–Vela fit units and canonical ordering.
- [ ] JUG state columns receive no second output unit conversion in nltiming.
- [ ] Input nltiming deltas are still converted to JUG native graph units.
- [ ] Residual projection is explicit and contribution-local.
- [ ] Mean-removed \(J=-PM\); uncentered \(J=-M+A_\mathrm{TZR}\).
- [ ] \(A_\mathrm{TZR}=0\) when `subtract_tzr=False`.
- [ ] Exact-linear residual evaluation uses projected `-M`, including
      JUMP/DMX/PHOFF/Offset.
- [ ] Autodiff routing requires `autodiff_design_matrix`; it has no fallback.
- [ ] Analytic/autodiff equality is not an acceptance criterion.
- [ ] Model D continues to use `jac(-residual_delta)`.
- [ ] Public selector is shared by nltiming and validation.
- [ ] TZR parameters are model state but not fit coordinates.
- [ ] No backward-compatible dual-sign surface is retained.
