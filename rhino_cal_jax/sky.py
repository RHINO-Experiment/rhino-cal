"""A synchrotron power law standing in for the antenna temperature.

This is not a sky model in any serious sense -- it is the smooth, bright,
spectrally simple thing you point the calibration machinery at while testing it.
Real sky models belong upstream (rheplicant's sky engines, limTOD).
"""

import jax
import jax.numpy as jnp


def synchrotron_temperature(
    freq: jax.Array,
    *,
    t_ref: float | jax.Array = 180.0,
    beta: float | jax.Array = -2.6,
    freq_ref: float | jax.Array = 210e6,
) -> jax.Array:
    """``T(nu) = t_ref (nu / freq_ref) ** beta``.

    Args:
        freq: channel frequencies [Hz].
        t_ref: brightness temperature at ``freq_ref`` [K].
        beta: spectral index (negative for synchrotron).
        freq_ref: reference frequency [Hz].

    Returns:
        Brightness temperature [K], shaped like ``freq``. Differentiable in
        ``t_ref`` and ``beta``, which is what makes it usable as a fitted
        foreground rather than a fixed backdrop.
    """
    return t_ref * jnp.power(jnp.asarray(freq) / freq_ref, beta)
