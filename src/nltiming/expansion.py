"""Optional deterministic expansion refinement (§6).

``refine_timing_expansion`` moves the fixed linearization point to a better
local expansion by minimizing a caller-supplied exact conditional negative log
target in prior-normal ``z`` (at fixed hyperparameters), using SciPy's L-BFGS-B
over a jitted JAX value-and-gradient. It is never invoked automatically by model
construction, ``joint_model``, or ``nuts`` — the caller chooses and records the
fixed hyperparameters. JAXopt is deliberately not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ExpansionRefinementResult:
    context: object
    z_initial: np.ndarray
    z_final: np.ndarray
    delta_final: np.ndarray
    objective_initial: float
    objective_final: float
    gradient_inf_norm: float
    iterations: int
    converged: bool


def refine_timing_expansion(
    ctx,
    *,
    negative_log_target_z: Callable,
    max_iterations: int = 12,
    gradient_tolerance: float = 1e-5,
) -> ExpansionRefinementResult:
    """Refine the fixed expansion point by minimizing an exact conditional target.

    The objective must be the exact conditional negative log posterior in the
    proper-axis ``z`` at fixed non-timing parameters, including ``0.5 z@z`` and
    excluding constants. On success the returned ``context`` is re-linearized at
    the refined point (``source="refined"``); otherwise it is the input context.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)

    z0 = np.asarray(ctx.linearization.z_expansion, dtype=np.float64)
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_target_z))
    # Compile once before the wall-clocked optimizer.
    value_and_grad(jnp.asarray(z0))

    def fun(x):
        v, g = value_and_grad(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    obj0, _ = fun(z0)
    options = {
        "maxiter": max_iterations,
        "gtol": gradient_tolerance,
        "ftol": 0.0,
        "maxcor": 10,
        "maxls": 20,
    }
    res = minimize(fun, z0, method="L-BFGS-B", jac=True, options=options)
    z_final = np.asarray(res.x, dtype=np.float64)
    obj_final, grad_final = fun(z_final)
    grad_inf = float(np.max(np.abs(grad_final))) if grad_final.size else 0.0

    proper_space = ctx.proper_space
    delta_final = np.asarray(proper_space.delta_from_z(z_final, np), dtype=float)

    converged = bool(
        np.isfinite(obj0)
        and np.isfinite(obj_final)
        and obj_final <= obj0
        and np.all(np.isfinite(delta_final))
        and grad_inf <= gradient_tolerance
    )

    refined_context = ctx
    if converged:
        refined_context = ctx.with_expansion(
            delta=delta_final, source="refined"
        )

    return ExpansionRefinementResult(
        context=refined_context,
        z_initial=z0,
        z_final=z_final,
        delta_final=delta_final,
        objective_initial=float(obj0),
        objective_final=float(obj_final),
        gradient_inf_norm=grad_inf,
        iterations=int(res.get("nit", 0)),
        converged=converged,
    )
