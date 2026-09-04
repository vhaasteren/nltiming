"""Default loguru sink for nltiming.

loguru's built-in handler (id ``0``) writes every ``DEBUG`` record to stderr,
and PINT logs its model construction at that level. ``import nltiming``
replaces that one untouched handler with a ``WARNING``-and-above stderr sink.
Configuration performed before the import, and any ``pint.logging.setup()`` /
``logger.add`` after it, is respected.
"""

from __future__ import annotations

import sys

from loguru import logger

DEFAULT_LEVEL = "WARNING"


def configure_logging(level: str = DEFAULT_LEVEL, *, force: bool = False) -> bool:
    """Route loguru output to stderr at ``level`` and above.

    With ``force=False`` (default) only loguru's built-in handler ``0`` is
    replaced; returns ``False`` without touching anything if it is already
    gone. ``force=True`` replaces every handler.
    """
    if force:
        logger.remove()
    else:
        try:
            logger.remove(0)
        except ValueError:
            return False
    logger.add(sys.stderr, level=level)
    return True


# Installed at import so that package ``__init__`` can simply import this module
# first; see the module docstring for the back-off rule.
configure_logging()
