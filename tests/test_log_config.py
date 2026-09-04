"""Default loguru sink: WARNING and above, user configuration respected."""

import sys

from loguru import logger

from nltiming.log_config import configure_logging


def _stderr_has(capsys, level: str) -> bool:
    token = f"probe-{level.lower()}-line"
    getattr(logger, level.lower())(token)
    return token in capsys.readouterr().err


def test_backs_off_from_user_configuration():
    logger.remove()
    hid = logger.add(sys.stderr, level="DEBUG")
    try:
        assert configure_logging() is False
    finally:
        logger.remove(hid)
        configure_logging(force=True)


def test_force_installs_warning_sink(capsys):
    assert configure_logging(force=True) is True
    assert not _stderr_has(capsys, "DEBUG")
    assert not _stderr_has(capsys, "INFO")
    assert _stderr_has(capsys, "WARNING")
