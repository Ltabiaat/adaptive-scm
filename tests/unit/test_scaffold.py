"""Smoke tests verifying the package and all submodules import cleanly.

These are not full unit tests for any feature — they only assert that the
scaffold itself is wired correctly. Real per-module tests land alongside
each feature implementation.
"""

from __future__ import annotations


def test_package_import() -> None:
    import adaptive_scm

    assert adaptive_scm.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    # Just importing these must not raise. They are placeholder packages
    # until Phase 1+ fills them in.
    import adaptive_scm.data  # noqa: F401
    import adaptive_scm.evaluation  # noqa: F401
    import adaptive_scm.forecasting  # noqa: F401
    import adaptive_scm.policies  # noqa: F401
    import adaptive_scm.simulation  # noqa: F401
    import adaptive_scm.utils  # noqa: F401


def test_base_interfaces_importable() -> None:
    from adaptive_scm.forecasting.base import Forecaster, ForecastOutput
    from adaptive_scm.policies.base import Policy, State

    # ABCs cannot be instantiated directly
    assert Forecaster.__abstractmethods__
    assert Policy.__abstractmethods__
    assert ForecastOutput.__name__ == "ForecastOutput"
    assert State.__name__ == "State"
