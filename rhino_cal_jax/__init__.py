"""Differentiable JAX/Equinox implementation of the RHINO noise-wave data model.

The model (Noise_Wave_GCR draft, Eq. 1) is the spectral power recorded by the
spectrometer when source ``k`` is connected to the receiver::

    d(nu, t) = G(nu, t) [ T_src c_s + T_unc k_unc + T_cos k_cos
                          + T_sin k_sin + T_rx ] (1 + w)

Everything to the right of a temperature depends only on the reflection
coefficients, and the bracket is *exactly linear* in the temperature vector.
That is the seam this package is built on: :mod:`rhino_cal_jax.reflection`
produces the couplings, :mod:`rhino_cal_jax.switching` gathers them onto the
time axis through the Dicke switch, and :mod:`rhino_cal_jax.power` contracts
them with the temperatures.
"""

from rhino_cal_jax.errors import RhinoCalError, ValidationError

__version__ = "0.1.0"

__all__ = ["RhinoCalError", "ValidationError", "__version__"]
