from argparse import Namespace
import logging

from seasenselib.cli.router import CLIRouter


def test_cli_default_logging_suppresses_seasenselib_warnings():
    package_logger = logging.getLogger("seasenselib")
    assert package_logger.handlers == []
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate

    try:
        CLIRouter._configure_logging(
            Namespace(verbose=False, verbose_log=None, verbose_level=None)
        )

        assert package_logger.level == logging.ERROR
        assert package_logger.handlers == []
        assert package_logger.propagate is False
    finally:
        for handler in list(package_logger.handlers):
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate


def test_cli_verbose_logging_enables_requested_level_without_root_logging(monkeypatch):
    package_logger = logging.getLogger("seasenselib")
    root_logger = logging.getLogger()
    previous_level = package_logger.level
    previous_handlers = list(package_logger.handlers)
    previous_propagate = package_logger.propagate
    previous_root_level = root_logger.level
    previous_root_handlers = list(root_logger.handlers)

    def fake_basic_config(**kwargs):
        raise AssertionError("CLI logging should not configure the root logger")

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    try:
        CLIRouter._configure_logging(
            Namespace(verbose=True, verbose_log=None, verbose_level="warning")
        )

        assert package_logger.level == logging.WARNING
        assert package_logger.propagate is False
        assert len(package_logger.handlers) == 1
        assert package_logger.handlers[0].level == logging.WARNING
        assert root_logger.level == previous_root_level
        assert list(root_logger.handlers) == previous_root_handlers
    finally:
        for handler in list(package_logger.handlers):
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate

        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            if handler not in previous_root_handlers:
                handler.close()
        for handler in previous_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_root_level)
