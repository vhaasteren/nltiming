# Interactive evaluator

The same engine interface supports engine-independent timing inspection without
constructing a likelihood:

```python
from nltiming import TimingEvaluator
from nltiming.space import ParameterSpace

timing = TimingEvaluator.from_pulsar(
    pulsar,
    engines={"pint": "jug", "tempo2": "jug"},
    derivative_method="autodiff",
)

timing.parameters["F0"]
evaluation = timing.evaluate({"F0": 1e-10}, frame="delta")
scan = timing.scan("TASC", [-0.5, 0.0, 0.5], scale="PB")
jacobian = timing.jacobian(method="autodiff")
fit = timing.fit(["F0", "F1"])

# Transformed-space (z) fit: prior-bijector-scaled Jacobian + weighted LSQ,
# returning a TimingZFitResult with z_best, covariance, rank/singular values.
space = ParameterSpace.build(theta_ref_mapping=timing.reference_exact)
jacobian_z = timing.jacobian_z(space)
zfit = timing.fit_z(space, ["F0", "F1"])
```

All operations return immutable result objects. The evaluator does not mutate
TOAs, parameter fit flags, timing sessions, or input files. `white_chi2` and
the built-in fit use diagonal TOA errors only; correlated-noise inference
remains the responsibility of the Discovery or Enterprise likelihood interface.

