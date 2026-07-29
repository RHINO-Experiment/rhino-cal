"""The package imports, and its error type is catchable as a ValueError."""

import pytest


def test_package_imports():
    import rhino_cal_jax

    assert rhino_cal_jax.__version__


def test_validation_error_is_a_value_error():
    from rhino_cal_jax.errors import RhinoCalError, ValidationError

    assert issubclass(ValidationError, RhinoCalError)
    assert issubclass(ValidationError, ValueError)
    with pytest.raises(ValueError):
        raise ValidationError("boom")


def test_numpy_reference_is_importable():
    """The consistency suite reads the numpy implementation from the repo root."""
    from simulation.radiometer_power import compute_radiometer_power

    assert callable(compute_radiometer_power)
