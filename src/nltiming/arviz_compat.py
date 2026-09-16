"""ArviZ 0.x / 1.x ``from_dict`` adapter.

ArviZ 0.x takes group kwargs (``from_dict(posterior=..., sample_stats=...)``).
ArviZ 1.x moved that helper to a nested mapping
(``from_dict({"posterior": ..., "sample_stats": ...})``) and rejects the
``posterior=`` keyword. Both shapes are in the wild; the notebooks must run
on either.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping


def inference_data_from_dict(
    *,
    posterior: Mapping[str, Any],
    sample_stats: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
):
    """Build InferenceData / DataTree from group dictionaries.

    Raises ``ImportError`` if ArviZ is not installed, matching the previous
    call-site messages.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required; install arviz (it is also a dependency "
            "of the corner plotting package)"
        ) from exc

    from_dict = getattr(az, "from_dict", None)
    if from_dict is None:
        raise ImportError("this arviz install does not provide from_dict")

    params = inspect.signature(from_dict).parameters
    if "posterior" in params:
        kwargs: dict[str, Any] = {"posterior": posterior}
        if sample_stats is not None:
            kwargs["sample_stats"] = sample_stats
        if attrs is not None:
            kwargs["attrs"] = attrs
        return from_dict(**kwargs)

    data: dict[str, Any] = {"posterior": posterior}
    if sample_stats is not None:
        data["sample_stats"] = sample_stats
    return from_dict(data, attrs=attrs)
