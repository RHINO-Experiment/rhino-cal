"""Differentiable JAX/Equinox implementation of the RHINO noise-wave data model.

:func:`~rhino_cal_jax.power.system_temperature` computes the bracket of the
Noise_Wave_GCR draft's Eq. 1::

    T_sys(nu, t) = T_src c_src + T_unc k_unc + T_cos k_cos + T_sin k_sin + T_rx

Multiplying by the gain and applying the fractional radiometer noise of the
draft's Eq. 8 gives the recorded spectral power::

    d(nu, t) = G(nu, t) T_sys(nu, t) (1 + w)

-- the ``(1 + w)`` convention this package implements (and the one the numpy
reference implements), rather than Eq. 1's own additive noise term inside the
bracket.

Everything to the right of a temperature depends only on the reflection
coefficients, and the bracket is *exactly linear* in the temperature vector.
That is the seam this package is built on: :mod:`rhino_cal_jax.reflection`
produces the couplings, :mod:`rhino_cal_jax.switching` gathers them onto the
time axis through the Dicke switch, and :mod:`rhino_cal_jax.power` contracts
them with the temperatures.
"""

from rhino_cal_jax.errors import RhinoCalError, ValidationError
from rhino_cal_jax.loads import SPEED_OF_LIGHT, Load, Receiver, cable_gamma, termination_gamma
from rhino_cal_jax.power import (
    add_radiometer_noise,
    design_matrix,
    radiometer_power,
    system_temperature,
)
from rhino_cal_jax.reflection import Couplings, couplings, reflection_factor
from rhino_cal_jax.sky import synchrotron_temperature
from rhino_cal_jax.switching import SwitchCycle, stack_load_gammas

# Read by hatchling via [tool.hatch.version]; keep it the single source.
__version__ = "0.1.0"

__all__ = [
    "Couplings",
    "Load",
    "Receiver",
    "RhinoCalError",
    "SPEED_OF_LIGHT",
    "SwitchCycle",
    "ValidationError",
    "add_radiometer_noise",
    "cable_gamma",
    "couplings",
    "design_matrix",
    "radiometer_power",
    "reflection_factor",
    "stack_load_gammas",
    "synchrotron_temperature",
    "system_temperature",
    "termination_gamma",
]
