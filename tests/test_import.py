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


def test_installing_this_package_does_not_leak_the_repo_root():
    """An editable install must expose rhino_cal_jax and nothing else.

    Hatchling's default "loose" editable mode drops the project root onto
    sys.path, which in this flat-layout repo would make the numpy pipeline's
    `simulation`, `gcr` and `utils` directories importable process-wide from
    any directory. `dev-mode-exact = true` in pyproject.toml prevents that.

    Checked in a subprocess from an unrelated working directory, with
    PYTHONPATH cleared, because sys.path leakage cannot be observed from
    inside the interpreter that already has the repo root on it.
    """
    import os
    import subprocess
    import sys
    import tempfile

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def run_import(module: str, cwd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
        )

    with tempfile.TemporaryDirectory() as elsewhere:
        result = run_import("rhino_cal_jax", elsewhere)
        assert result.returncode == 0, (
            "rhino_cal_jax is not importable outside the repo -- is it "
            f"installed? stderr:\n{result.stderr}"
        )
        for leaked in ("simulation", "gcr", "utils", "rfi_flagging"):
            assert run_import(leaked, elsewhere).returncode != 0, (
                f"{leaked!r} is importable from outside the repository: the "
                "editable install has leaked the project root onto sys.path."
            )
